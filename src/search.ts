/**
 * Dev script: search the embedded corpus with one or more queries.
 *
 * This is retrieval without a vector database: embed the query, compare it to
 * every stored vector, rank. Qdrant will later make the comparison step fast
 * via an approximate index — it does not change what is being computed here.
 *
 *   npm run search                          run the built-in query set
 *   npm run search -- "how are fees burned?"
 *   npm run search -- --index=data/embeddings-voyage.json
 *   npm run search -- --k=10 --text         show matched chunk text
 *   npm run search -- --interval=25000      pace queries (rate-limited accounts)
 */
import { readFile } from "node:fs/promises";

import {
  cosineSimilarity,
  embedQuery,
  EmbeddingError,
  OpenAIEmbeddingProvider,
  VoyageEmbeddingProvider,
} from "./embedder/index.js";
import type { EmbeddedChunk, EmbeddingProvider } from "./embedder/index.js";

/** Default probe set: exact title, two paraphrases, an absent concept, a topic shift. */
const DEFAULT_QUERIES = [
  "What is EIP-712?",
  "How does typed structured data signing work?",
  "Why do wallets need structured signing?",
  "What is account abstraction?",
  "How does Ethereum's fee market work?",
];

function flag(name: string, fallback: string): string {
  return (
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ??
    fallback
  );
}

const indexPath = flag("index", "data/embeddings.json");
const k = Number(flag("k", "5"));
const showText = process.argv.includes("--text");
const queryIntervalMs = Number(flag("interval", "0"));

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Positional args are queries; anything starting with -- is a flag.
const queries = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const toRun = queries.length > 0 ? queries : DEFAULT_QUERIES;

const index = JSON.parse(await readFile(indexPath, "utf8")) as EmbeddedChunk[];
if (index.length === 0) throw new Error(`${indexPath} is empty.`);

/**
 * Rebuild the provider that produced this index.
 *
 * Read from the stored `model` rather than from .env: a query embedded by a
 * different model lands in an unrelated coordinate space, and the resulting
 * scores look perfectly reasonable while being meaningless. This is the one
 * check that makes that failure impossible rather than merely unlikely.
 */
function providerFor(model: string, dimensions: number): EmbeddingProvider {
  if (model.startsWith("text-embedding-")) {
    return new OpenAIEmbeddingProvider({ model, dimensions });
  }
  if (model.startsWith("voyage-")) {
    return new VoyageEmbeddingProvider({ model, outputDimension: dimensions });
  }
  throw new Error(`Cannot rebuild a provider for model "${model}".`);
}

const { model, dimensions } = index[0]!;
// A mixed index cannot be searched coherently — fail rather than rank garbage.
const models = new Set(index.map((c) => c.model));
if (models.size > 1) {
  throw new Error(
    `${indexPath} mixes models (${[...models].join(", ")}). ` +
      `Vectors from different models are not comparable.`,
  );
}

const provider = providerFor(model, dimensions);

console.log(
  `${index.length} chunks · ${model} · ${dimensions} dims · top ${k}\n`,
);

try {
  let first = true;
  for (const query of toRun) {
    // Each query is its own API call, so a rate-limited account needs the
    // same pacing here as the bulk embed run.
    if (!first && queryIntervalMs > 0) await sleep(queryIntervalMs);
    first = false;

    // Same provider as the corpus, and flagged as a query so asymmetric
    // models (Voyage) use their query-side encoding.
    const vector = await embedQuery(query, provider, {
      // Generous: on a rate-limited account a 429 here is expected, and the
      // provider's Retry-After tells us exactly how long to wait.
      maxRetries: 6,
      retryDelayMs: 2_000,
      maxRetryDelayMs: 90_000,
    });

    const hits = index
      .map((chunk) => ({ chunk, score: cosineSimilarity(vector, chunk.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);

    console.log(`? ${query}`);
    for (const { chunk, score } of hits) {
      const doc = chunk.source.fileName.replace(/\.md$/, "");
      console.log(
        `   ${score.toFixed(3)}  ${doc.padEnd(12)} ${(chunk.section ?? "-").slice(0, 40)}`,
      );
      if (showText) {
        console.log(`          ${chunk.text.trim().replace(/\s+/g, " ").slice(0, 150)}...`);
      }
    }

    // Spread between the top hit and the k-th tells you whether the ranking
    // actually discriminated or just returned the corpus in arbitrary order.
    const spread = hits[0]!.score - hits[hits.length - 1]!.score;
    console.log(`   └ top ${hits[0]!.score.toFixed(3)} · spread ${spread.toFixed(3)}\n`);
  }
} catch (error) {
  if (error instanceof EmbeddingError) {
    console.error(`\nQuery embedding failed: ${error.message}`);
    process.exit(1);
  }
  throw error;
}
