/**
 * Dev script: load documents, chunk them, and write the result to disk.
 *
 *   npm run chunk                          defaults (1000 / 150)
 *   npm run chunk -- --size=500 --overlap=50
 *   npm run chunk -- --out=data/chunks.json
 *   npm run chunk -- --dry                 print a summary, write nothing
 */
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { loadDocuments } from "./loader/index.js";
import { chunkDocuments, DEFAULT_CHUNK_OPTIONS } from "./chunker/index.js";

/** Read `--key=value` from argv, falling back to a default. */
function numberFlag(name: string, fallback: number): number {
  const raw = process.argv.find((a) => a.startsWith(`--${name}=`));
  if (!raw) return fallback;
  const value = Number(raw.split("=")[1]);
  if (!Number.isFinite(value)) throw new Error(`--${name} must be a number`);
  return value;
}

const chunkSize = numberFlag("size", DEFAULT_CHUNK_OPTIONS.chunkSize);
const chunkOverlap = numberFlag("overlap", DEFAULT_CHUNK_OPTIONS.chunkOverlap);
const dryRun = process.argv.includes("--dry");
const outPath =
  process.argv.find((a) => a.startsWith("--out="))?.split("=")[1] ??
  "data/chunks.json";

const { documents, errors } = await loadDocuments();
for (const err of errors) console.error(`ERROR ${err.filePath}: ${err.message}`);

const chunks = chunkDocuments(documents, { chunkSize, chunkOverlap });

console.log(
  `${documents.length} documents -> ${chunks.length} chunks ` +
    `(size=${chunkSize}, overlap=${chunkOverlap})\n`,
);

// Per-document counts, so an unexpected split is easy to spot.
for (const doc of documents) {
  const own = chunks.filter((c) => c.documentId === doc.id);
  const sections = new Set(own.map((c) => c.section ?? "-"));
  console.log(
    `  ${doc.source.fileName.padEnd(14)} ${String(own.length).padStart(3)} chunks` +
      `  ${sections.size} sections`,
  );
}

if (dryRun) {
  console.log("\n--dry: nothing written.");
} else {
  await mkdir(path.dirname(outPath), { recursive: true });
  // Pretty-printed: this file is meant to be read while learning.
  await writeFile(outPath, JSON.stringify(chunks, null, 2), "utf8");
  console.log(`\nWrote ${chunks.length} chunks to ${outPath}`);
}
