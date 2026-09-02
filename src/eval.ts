/**
 * Retrieval eval: run a labelled query set and score the rankings.
 *
 * Measures hit@k — whether an expected document appears in the top k results.
 * Ranking, not raw score, is what matters: scores are not comparable across
 * models, but "did the right document surface" is.
 *
 *   npm run eval
 *   npm run eval -- --index=data/embeddings-voyage.json --interval=22000
 *   npm run eval -- --out=eval/results-openai.json
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";

import {
  cosineSimilarity,
  embedQuery,
  OpenAIEmbeddingProvider,
  VoyageEmbeddingProvider,
} from "./embedder/index.js";
import type { EmbeddedChunk, EmbeddingProvider } from "./embedder/index.js";

interface EvalQuery {
  id: string;
  q: string;
  /** Document basenames that count as correct. Empty = a negative case. */
  expect: string[];
  /** Original standard labels, e.g. ["EIP-721"]. */
  standards?: string[];
  difficulty?: string;
  /** product | natural | technical | indirect | comparison | negative */
  type?: string;
}

interface EvalResult extends EvalQuery {
  /** Ranked document basenames, deduped, best first. */
  ranked: string[];
  topScore: number;
  spread: number;
  hit1: boolean;
  hit3: boolean;
  hit5: boolean;
  /** For multi-answer questions: how many expected docs made the top 5. */
  covered: number;
}

function flag(name: string, fallback: string): string {
  return (
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ??
    fallback
  );
}

const indexPath = flag("index", "data/embeddings.json");
const queryPath = flag("queries", "eval/queries.json");
const outPath = flag("out", "");
const intervalMs = Number(flag("interval", "0"));
const k = Number(flag("k", "5"));

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const index = JSON.parse(await readFile(indexPath, "utf8")) as EmbeddedChunk[];
const queries = JSON.parse(await readFile(queryPath, "utf8")) as EvalQuery[];

const { model, dimensions } = index[0]!;
const provider: EmbeddingProvider = model.startsWith("voyage-")
  ? new VoyageEmbeddingProvider({ model, outputDimension: dimensions })
  : new OpenAIEmbeddingProvider({ model, dimensions });

console.log(
  `${index.length} chunks · ${model} · ${dimensions}d · ${queries.length} queries · hit@${k}\n`,
);

const results: EvalResult[] = [];

for (const [i, query] of queries.entries()) {
  if (i > 0 && intervalMs > 0) await sleep(intervalMs);

  const vector = await embedQuery(query.q, provider, {
    maxRetries: 6,
    retryDelayMs: 2_000,
    maxRetryDelayMs: 90_000,
  });

  const scored = index
    .map((c) => ({ c, score: cosineSimilarity(vector, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 20);

  // Collapse chunk hits to document hits, keeping best-first order. Several
  // chunks from one document are one retrieval result, not several.
  const ranked: string[] = [];
  for (const { c } of scored) {
    const doc = c.source.fileName.replace(/\.md$/, "");
    if (!ranked.includes(doc)) ranked.push(doc);
  }

  const rank = ranked.findIndex((d) => query.expect.includes(d));
  const found = query.expect.length > 0 && rank >= 0;

  results.push({
    ...query,
    covered: query.expect.filter((e) => ranked.slice(0, 5).includes(e)).length,
    ranked: ranked.slice(0, 5),
    topScore: scored[0]!.score,
    spread: scored[0]!.score - scored[Math.min(k, scored.length) - 1]!.score,
    hit1: found && rank === 0,
    hit3: found && rank < 3,
    hit5: found && rank < 5,
  });

  const r = results.at(-1)!;
  const mark = query.expect.length === 0 ? "n" : r.hit1 ? "✓" : r.hit3 ? "~" : "✗";
  process.stdout.write(`\r  ${i + 1}/${queries.length} ${mark}   `);
}

const scoreable = results.filter((r) => r.expect.length > 0);
const negatives = results.filter((r) => r.expect.length === 0);
const pct = (n: number, d: number) => `${((n / d) * 100).toFixed(0)}%`;

const h1 = scoreable.filter((r) => r.hit1).length;
const h3 = scoreable.filter((r) => r.hit3).length;
const h5 = scoreable.filter((r) => r.hit5).length;

console.log(`\n\n  ${scoreable.length} positive · ${negatives.length} negative\n`);
console.log(`  hit@1   ${h1}/${scoreable.length}  ${pct(h1, scoreable.length)}`);
console.log(`  hit@3   ${h3}/${scoreable.length}  ${pct(h3, scoreable.length)}`);
console.log(`  hit@5   ${h5}/${scoreable.length}  ${pct(h5, scoreable.length)}`);

/** Group a list of results by a field and report hit@1 / hit@5. */
function breakdown(label: string, key: "type" | "difficulty") {
  const groups = new Map<string, EvalResult[]>();
  for (const r of scoreable) {
    const g = r[key] ?? "?";
    groups.set(g, [...(groups.get(g) ?? []), r]);
  }
  console.log(`\n  By ${label}:`);
  for (const [g, rs] of [...groups].sort()) {
    const a = rs.filter((r) => r.hit1).length;
    const b = rs.filter((r) => r.hit5).length;
    console.log(
      `    ${g.padEnd(11)} ${String(rs.length).padStart(2)}q   ` +
        `hit@1 ${pct(a, rs.length).padStart(4)}   hit@5 ${pct(b, rs.length).padStart(4)}`,
    );
  }
}
breakdown("question type", "type");
breakdown("difficulty", "difficulty");

// Multi-answer coverage: did we surface every expected standard, or just one?
const multi = scoreable.filter((r) => r.expect.length > 1);
if (multi.length) {
  const full = multi.filter((r) => r.covered === r.expect.length).length;
  console.log(
    `\n  Multi-standard questions: ${multi.length}` +
      `  ·  all expected docs in top 5: ${full}/${multi.length}  ${pct(full, multi.length)}`,
  );
}

const misses = scoreable.filter((r) => !r.hit5);
if (misses.length) {
  console.log(`\n  Misses (${misses.length}):`);
  for (const m of misses) {
    console.log(
      `    ${m.id}  ${(m.type ?? "").padEnd(10)} want ${m.expect.join("/")} · got ${m.ranked.slice(0, 3).join(", ")}`,
    );
  }
}

// Negatives have no right answer. What matters is that confidence stays low:
// a high-scoring hit on an out-of-scope question is a false-confidence risk,
// because a RAG pipeline would feed those chunks to the LLM as if relevant.
if (negatives.length) {
  const avgNeg = negatives.reduce((s, r) => s + r.topScore, 0) / negatives.length;
  const avgPos = scoreable.reduce((s, r) => s + r.topScore, 0) / scoreable.length;
  const maxNeg = Math.max(...negatives.map((r) => r.topScore));
  const overlap = negatives.filter((r) => r.topScore > avgPos).length;
  console.log(`\n  Negative controls (no correct answer exists):`);
  console.log(`    mean top score   positive ${avgPos.toFixed(3)}  ·  negative ${avgNeg.toFixed(3)}`);
  console.log(`    highest negative ${maxNeg.toFixed(3)}`);
  console.log(`    negatives scoring above the positive mean: ${overlap}/${negatives.length}`);
  console.log(`\n    Most confident negatives (false-confidence risk):`);
  for (const n of [...negatives].sort((a, b) => b.topScore - a.topScore).slice(0, 4)) {
    console.log(`      ${n.topScore.toFixed(3)}  ${n.ranked[0]?.padEnd(9)} ${n.q.slice(0, 52)}`);
  }
}

if (outPath) {
  await mkdir(path.dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify({ model, dimensions, results }, null, 2));
  console.log(`\nWrote ${outPath}`);
}
