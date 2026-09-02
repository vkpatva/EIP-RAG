/**
 * Dev script: read chunks from disk, embed them, and write the result.
 *
 *   npm run embed                      provider from EMBEDDING_PROVIDER in .env
 *   npm run embed -- --provider=voyage override it for one run
 *   npm run embed -- --limit=20        embed only the first 20 chunks
 *   npm run embed -- --batch=50
 *   npm run embed -- --interval=25000   pace requests (rate-limited accounts)
 *   npm run embed -- --in=data/chunks.json --out=data/embeddings.json
 */
import { readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import type { Chunk } from "./chunker/index.js";
import {
  embedChunks,
  cosineSimilarity,
  EmbeddingError,
  OpenAIEmbeddingProvider,
  VoyageEmbeddingProvider,
} from "./embedder/index.js";
import type { EmbeddingProvider } from "./embedder/index.js";

function flag(name: string, fallback: string): string {
  return (
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ??
    fallback
  );
}

const inPath = flag("in", "data/chunks.json");
const outPath = flag("out", "data/embeddings.json");
const batchSize = Number(flag("batch", "100"));
const limit = Number(flag("limit", "0"));
const requestIntervalMs = Number(flag("interval", "0"));

/**
 * Pick a provider. Both satisfy `EmbeddingProvider`, so nothing downstream
 * changes — but their vectors are NOT interchangeable. Switching providers
 * means re-embedding the whole corpus into a fresh output file.
 */
function createProvider(name: string): EmbeddingProvider {
  switch (name) {
    case "openai":
      return new OpenAIEmbeddingProvider({
        model: process.env.OPENAI_EMBEDDING_MODEL,
      });
    case "voyage":
      return new VoyageEmbeddingProvider({
        model: process.env.VOYAGE_EMBEDDING_MODEL,
      });
    default:
      throw new Error(
        `Unknown provider "${name}". Expected "openai" or "voyage".`,
      );
  }
}

const all = JSON.parse(await readFile(inPath, "utf8")) as Chunk[];
const chunks = limit > 0 ? all.slice(0, limit) : all;

const provider = createProvider(
  flag("provider", process.env.EMBEDDING_PROVIDER ?? "openai"),
);

console.log(
  `Embedding ${chunks.length} chunks with ${provider.model} ` +
    `(${provider.dimensions} dims, batch=${batchSize})\n`,
);

try {
  const started = Date.now();
  const embedded = await embedChunks(chunks, provider, {
    batchSize,
    requestIntervalMs,
    onProgress: (done, total) =>
      process.stdout.write(`\r  ${done}/${total} chunks`),
  });
  console.log(`\n  done in ${((Date.now() - started) / 1000).toFixed(1)}s\n`);

  // A vector is opaque, so show the two things that are actually checkable:
  // the shape is right, and near-neighbours look plausible.
  const first = embedded[0]!;
  console.log(`  ${first.id}`);
  console.log(`  dims ${first.embedding.length}`);
  console.log(
    `  head [${first.embedding.slice(0, 4).map((v) => v.toFixed(4)).join(", ")}, ...]\n`,
  );

  const nearest = embedded
    .slice(1)
    .map((c) => ({ c, score: cosineSimilarity(first.embedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  console.log(`  Nearest to "${first.section ?? first.id}":`);
  for (const { c, score } of nearest) {
    console.log(`    ${score.toFixed(3)}  ${c.id}  ${c.section ?? "-"}`);
  }
  await mkdir(path.dirname(outPath), { recursive: true });
  // Compact: 1536 floats per chunk makes a pretty-printed file unusable.
  await writeFile(outPath, JSON.stringify(embedded), "utf8");
  console.log(`\nWrote ${embedded.length} embedded chunks to ${outPath}`);
} catch (error) {
  if (error instanceof EmbeddingError) {
    console.error(`\nEmbedding failed${error.status ? ` (${error.status})` : ""}: ${error.message}`);
    if (error.retryable) console.error("Transient — retries were exhausted.");
    process.exit(1);
  }
  throw error;
}
