/**
 * Dev script: load embedded chunks from disk and store them in Qdrant.
 *
 *   npm run index                                    index data/embeddings.json
 *   npm run index -- --in=data/embeddings-voyage.json
 *   npm run index -- --recreate                      drop and rebuild first
 *   npm run index -- --batch=128
 *
 * Note there is no `--collection`: the name is derived from the model stored
 * on the chunks. Pointing this at the Voyage file therefore fills a second,
 * separate collection with no code change and no risk of mixing coordinate
 * spaces.
 */
import { readFile } from "node:fs/promises";

import type { EmbeddedChunk } from "./embedder/index.js";
import { QdrantVectorRepository } from "./vectorstore/index.js";

function flag(name: string, fallback: string): string {
  return (
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ??
    fallback
  );
}

const inPath = flag("in", "data/embeddings.json");
const batchSize = Number(flag("batch", "64"));
const recreate = process.argv.includes("--recreate");

const chunks = JSON.parse(await readFile(inPath, "utf8")) as EmbeddedChunk[];
if (chunks.length === 0) throw new Error(`${inPath} is empty.`);

// A file mixing two models cannot be stored coherently in one collection:
// their vectors live in unrelated coordinate spaces. Fail rather than index.
const models = new Set(chunks.map((c) => c.model));
if (models.size > 1) {
  throw new Error(
    `${inPath} mixes models (${[...models].join(", ")}). ` +
      `Each model needs its own collection.`,
  );
}

const { model, dimensions } = chunks[0]!;
const repo = new QdrantVectorRepository({ model, dimensions, batchSize });

console.log(
  `Indexing ${chunks.length} chunks · ${model} · ${dimensions} dims\n` +
    `  collection: ${repo.collection}\n`,
);

if (recreate) {
  const before = await repo.getCollectionInfo();
  if (before.exists) {
    await repo.raw.deleteCollection(repo.collection);
    console.log(`  dropped existing collection (${before.pointsCount} points)`);
  }
}

const before = await repo.getCollectionInfo();
console.log(
  before.exists
    ? `  before: ${before.pointsCount} points`
    : `  before: collection does not exist`,
);

await repo.createCollection();

const started = Date.now();
await repo.upsertChunks(chunks);
console.log(`  upserted in ${((Date.now() - started) / 1000).toFixed(1)}s`);

const after = await repo.getCollectionInfo();
console.log(
  `  after:  ${after.pointsCount} points · ${after.vectorSize} dims · ` +
    `${after.distance} · ${after.status}`,
);

// The number that matters. `delta` is 0 on every run after the first, because
// the point ids are a pure function of the chunk ids.
const delta = after.pointsCount - before.pointsCount;
console.log(`  delta:  ${delta >= 0 ? "+" : ""}${delta}\n`);
