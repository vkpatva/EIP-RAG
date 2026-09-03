/**
 * Dev script: measure retrieval quality against the labelled query set.
 *
 *   npm run eval:retrieval
 *   npm run eval:retrieval -- --k=10
 *   npm run eval:retrieval -- --in=data/embeddings-voyage.json
 *   npm run eval:retrieval -- --limit=10 --interval=25000
 *   npm run eval:retrieval -- --out=eval/retrieval-openai.json
 *
 * The only metric here is Recall@K: did the EIP that should have answered the
 * question appear anywhere in the top K? That is deliberately the crudest
 * useful measure — it asks whether the right *document* was reachable at all,
 * and nothing about ordering quality beyond the cutoff. If Recall@5 is poor,
 * no amount of downstream cleverness can recover: the answer was never
 * retrieved. Everything more sophisticated is a refinement of this one number.
 *
 * Note the unit. Chunks are retrieved, but recall is scored on the EIP the
 * chunk came from — a question about the fee market is answered by eip-1559,
 * and which of its 30 chunks surfaced is not what is being measured.
 */
import { readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { connect } from "./vectorstore/connect.js";
import { EmbeddingError } from "./embedder/index.js";

interface EvalQuery {
  id: string;
  q: string;
  /** Document ids without the .md suffix, e.g. "erc-55". Empty = negative. */
  expect: string[];
  standards: string[];
  difficulty: string;
  type: string;
}

function flag(name: string, fallback: string): string {
  return (
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ??
    fallback
  );
}

const inPath = flag("in", "data/embeddings.json");
const hybrid = process.argv.includes("--hybrid");
const rrfK = Number(flag("rrf-k", "2"));
const bm25Weight = Number(flag("bm25-weight", "0.5"));
const queriesPath = flag("queries", "eval/queries.json");
const k = Number(flag("k", "5"));
const limit = Number(flag("limit", "0"));
const queryIntervalMs = Number(flag("interval", "0"));
const outPath = flag("out", "");

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const all = JSON.parse(await readFile(queriesPath, "utf8")) as EvalQuery[];
const queries = limit > 0 ? all.slice(0, limit) : all;

const { retriever, model, dimensions, pointsCount, repository, mode } =
  await connect(inPath, {
  hybrid,
  weights: { dense: 1, bm25: bm25Weight },
  rrfK,
});

console.log(
  `${repository.collection} · ${pointsCount} points · ${model} · ` +
    `${dimensions} dims · ${mode} · ${queries.length} queries · top ${k}\n`,
);

/** "eip-1559.md" -> "eip-1559", so hits line up with the `expect` labels. */
const docKey = (sourcePath: string) => sourcePath.replace(/\.md$/, "");

interface Row {
  id: string;
  q: string;
  type: string;
  difficulty: string;
  expected: string[];
  /** Distinct EIP documents in the top K, best-ranked first. */
  retrieved: string[];
  topScore: number;
  recall: Record<number, number>;
}

const rows: Row[] = [];
const CUTOFFS = [1, 3, 5].filter((c) => c <= k);

try {
  let first = true;
  for (const query of queries) {
    if (!first && queryIntervalMs > 0) await sleep(queryIntervalMs);
    first = false;

    const hits = await retriever.retrieve(query.q, k);

    // Collapse chunks to documents, preserving rank order. Two chunks from
    // eip-1559 at ranks 1 and 2 are one document at document-rank 1 — without
    // this, "Recall@3" would silently mean "within the top 3 *chunks*", which
    // is a much harsher bar and not what the label claims.
    const ranked: string[] = [];
    for (const hit of hits) {
      const key = docKey(hit.metadata.sourcePath ?? "");
      if (key && !ranked.includes(key)) ranked.push(key);
    }

    const recall: Record<number, number> = {};
    for (const cutoff of CUTOFFS) {
      // Cut on *chunk* rank, since that is what a consumer would actually be
      // handed, then ask whether any expected document is in that slice.
      const slice: string[] = [];
      for (const hit of hits.slice(0, cutoff)) {
        const key = docKey(hit.metadata.sourcePath ?? "");
        if (key && !slice.includes(key)) slice.push(key);
      }
      recall[cutoff] =
        query.expect.length === 0
          ? // A negative query has no right answer. Scoring it as recall would
            // be meaningless, so it is excluded from the averages below and
            // judged on its score instead.
            Number.NaN
          : query.expect.some((e) => slice.includes(e))
            ? 1
            : 0;
    }

    rows.push({
      id: query.id,
      q: query.q,
      type: query.type,
      difficulty: query.difficulty,
      expected: query.expect,
      retrieved: ranked,
      // The best *dense* score among the hits, not `hits[0].score`. Under
      // hybrid the top-ranked chunk can be a lexical-only hit with no score,
      // and coercing that to 0 would drag the score-separation stats toward a
      // floor that no retriever actually produced.
      topScore: Math.max(
        0,
        ...hits.map((h) => h.score).filter((x): x is number => x !== undefined),
      ),
      recall,
    });

    const label = (d: string) => (d ? `EIP-${d.replace(/^\D+/, "")}` : "-");
    const hitMark =
      query.expect.length === 0 ? "  " : recall[CUTOFFS[0]!] === 1 ? "  " : "! ";
    console.log(`${hitMark}${query.id}  ${query.q}`);
    console.log(
      `     expected  ${
        query.expect.length ? query.expect.map(label).join(", ") : "(none)"
      }`,
    );
    console.log(`     top ${k}     ${ranked.map(label).join(", ") || "-"}`);
    console.log(
      `     score     ${rows.at(-1)!.topScore.toFixed(4)}   ` +
        CUTOFFS.map(
          (c) =>
            `R@${c}=${Number.isNaN(recall[c]!) ? "-" : recall[c]}`,
        ).join(" "),
    );
    console.log();
  }
} catch (error) {
  if (error instanceof EmbeddingError) {
    console.error(`\nQuery embedding failed: ${error.message}`);
    process.exit(1);
  }
  throw error;
}

// ---- Summary ---------------------------------------------------------------

const scored = rows.filter((r) => r.expected.length > 0);
const negatives = rows.filter((r) => r.expected.length === 0);

console.log("=".repeat(64));
console.log(`Recall over ${scored.length} answerable queries:`);
for (const cutoff of CUTOFFS) {
  const hits = scored.filter((r) => r.recall[cutoff] === 1).length;
  const pct = ((hits / scored.length) * 100).toFixed(1);
  console.log(`  Recall@${cutoff}  ${hits}/${scored.length}  ${pct}%`);
}

// Breakdowns, because an aggregate hides where the failures live. A corpus
// that answers "What is the purpose of supportsInterface?" but not "how do I
// check what a contract can do?" has a paraphrase problem, not a coverage one.
const by = (field: "type" | "difficulty") => {
  const groups = new Map<string, Row[]>();
  for (const row of scored) {
    const list = groups.get(row[field]) ?? [];
    list.push(row);
    groups.set(row[field], list);
  }
  console.log(`\nBy ${field}:`);
  for (const [name, list] of [...groups].sort()) {
    const line = CUTOFFS.map((c) => {
      const n = list.filter((r) => r.recall[c] === 1).length;
      return `R@${c} ${((n / list.length) * 100).toFixed(0).padStart(3)}%`;
    }).join("  ");
    console.log(`  ${name.padEnd(11)} n=${String(list.length).padStart(2)}  ${line}`);
  }
};
by("type");
by("difficulty");

// The negatives do not have a right answer, so the useful question is whether
// they *look* different from the answerable ones. If the two score ranges
// overlap, no threshold can separate "found it" from "found nothing".
if (negatives.length > 0) {
  const scores = (list: Row[]) => list.map((r) => r.topScore);
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const pos = scores(scored);
  const neg = scores(negatives);
  console.log(`\nTop-1 score separation:`);
  console.log(
    `  answerable  n=${pos.length}  min ${Math.min(...pos).toFixed(4)}  ` +
      `mean ${mean(pos).toFixed(4)}  max ${Math.max(...pos).toFixed(4)}`,
  );
  console.log(
    `  negative    n=${neg.length}  min ${Math.min(...neg).toFixed(4)}  ` +
      `mean ${mean(neg).toFixed(4)}  max ${Math.max(...neg).toFixed(4)}`,
  );
  const gap = Math.min(...pos) - Math.max(...neg);
  console.log(
    gap > 0
      ? `  clean gap of ${gap.toFixed(4)} — a score threshold would separate them`
      : `  ranges overlap by ${(-gap).toFixed(4)} — no single threshold separates them`,
  );
}

if (outPath) {
  await mkdir(path.dirname(outPath), { recursive: true });
  await writeFile(
    outPath,
    JSON.stringify({ model, dimensions, mode, k, rows }, null, 1),
    "utf8",
  );
  console.log(`\nWrote ${outPath}`);
}
