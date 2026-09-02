/**
 * Dev script: prove the collection holds what it should.
 *
 *   npm run verify
 *   npm run verify -- --in=data/embeddings-voyage.json
 *
 * A vector store is easy to get subtly wrong in ways nothing complains about:
 * the collection exists but is empty, points are there but the vectors were
 * dropped, the payload arrived without its text. This checks each of those
 * explicitly rather than trusting that the upsert "seemed to work".
 */
import { readFile } from "node:fs/promises";

import type { EmbeddedChunk } from "./embedder/index.js";
import { QdrantVectorRepository, chunkPointId } from "./vectorstore/index.js";

function flag(name: string, fallback: string): string {
  return (
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ??
    fallback
  );
}

const inPath = flag("in", "data/embeddings.json");
const chunks = JSON.parse(await readFile(inPath, "utf8")) as EmbeddedChunk[];
const { model, dimensions } = chunks[0]!;
const repo = new QdrantVectorRepository({ model, dimensions });

// 1. Does the collection exist, with the configuration we asked for?
const info = await repo.getCollectionInfo();
console.log(`Collection: ${info.name}`);
console.log(`  exists:      ${info.exists}`);
console.log(`  status:      ${info.status ?? "-"}`);
console.log(`  points:      ${info.pointsCount}`);
console.log(`  vector size: ${info.vectorSize ?? "-"}`);
console.log(`  distance:    ${info.distance ?? "-"}`);

if (!info.exists) {
  console.error(`\nCollection missing. Run: npm run index`);
  process.exit(1);
}

// 2. Does the stored count match the source file? A shortfall means some
//    batch failed silently; an excess means duplicate ids got in.
console.log(
  `\nSource file has ${chunks.length} chunks; collection has ` +
    `${info.pointsCount}. ${
      info.pointsCount === chunks.length ? "match" : "MISMATCH"
    }`,
);

// 3. Fetch one known point *by its deterministic id* and inspect it. This is
//    the strongest single check available: it proves the id derivation, the
//    payload mapping and the vector storage all agree at once.
const sample = chunks[Math.floor(chunks.length / 2)]!;
const [point] = await repo.raw.retrieve(repo.collection, {
  ids: [chunkPointId(sample.id)],
  with_payload: true,
  // Qdrant omits vectors from responses by default — they are large and
  // rarely wanted. Asking for it is the only way to confirm one was stored.
  with_vector: true,
});

if (!point) {
  console.error(`\nPoint for chunk ${sample.id} not found. Ids are unstable.`);
  process.exit(1);
}

const vector = point.vector as number[];
const payload = point.payload as Record<string, unknown>;

console.log(`\nSample point (chunk ${sample.id}):`);
console.log(`  id:          ${point.id}`);
console.log(`  vector dims: ${vector.length} (expected ${dimensions})`);
console.log(
  `  vector head: [${vector.slice(0, 4).map((v) => v.toFixed(4)).join(", ")}, ...]`,
);
// A vector of all zeros would pass a length check while carrying no meaning.
const magnitude = Math.hypot(...vector);
console.log(`  magnitude:   ${magnitude.toFixed(4)} (0 would mean empty)`);

console.log(`\n  payload:`);
for (const key of [
  "chunkId",
  "documentId",
  "eipNumber",
  "title",
  "section",
  "sourcePath",
  "index",
]) {
  console.log(`    ${key.padEnd(11)} ${JSON.stringify(payload[key])}`);
}
const text = String(payload.text ?? "");
console.log(`    text        ${text.length} chars`);
console.log(`      "${text.trim().replace(/\s+/g, " ").slice(0, 120)}..."`);

// 4. Round-trip check: the stored vector should be identical to the source.
const drift = Math.max(
  ...sample.embedding.map((v, i) => Math.abs(v - (vector[i] ?? 0))),
);
console.log(
  `\n  max coordinate drift vs source file: ${drift.toExponential(2)}`,
);

// 5. How many distinct documents made it in? 12 EIP files should give 12.
const docs = new Set<string>();
let offset: string | number | undefined | null = undefined;
do {
  const page = await repo.raw.scroll(repo.collection, {
    limit: 256,
    offset,
    with_payload: ["documentId"],
    with_vector: false,
  });
  for (const p of page.points) docs.add(String((p.payload as any).documentId));
  offset = page.next_page_offset as typeof offset;
} while (offset !== null && offset !== undefined);

console.log(`\nDistinct documents stored: ${docs.size}`);
for (const d of [...docs].sort()) console.log(`  ${d}`);
