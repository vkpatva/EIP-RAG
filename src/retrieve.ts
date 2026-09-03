/**
 * Dev script: run queries against Qdrant and print the Top-K chunks.
 *
 *   npm run retrieve                            the built-in probe set
 *   npm run retrieve -- "What is EIP-712?"
 *   npm run retrieve -- --k=3
 *   npm run retrieve -- --chars=600             more of each chunk's text
 *   npm run retrieve -- --in=data/embeddings-voyage.json
 *   npm run retrieve -- --interval=25000        pace queries (rate limits)
 *
 * Output is deliberately verbose: the point of this script is to *read* the
 * chunks and judge them yourself. An automated score comes later.
 */
import { connect } from "./vectorstore/connect.js";
import { EmbeddingError } from "./embedder/index.js";

/**
 * A probe set chosen to exercise different retrieval behaviours:
 * an exact title lookup, four paraphrases that share almost no vocabulary
 * with their target text, and one query about something the corpus does not
 * contain at all.
 */
const DEFAULT_QUERIES = [
  "What is EIP-712?",
  "How can I make transaction fees easier for my users?",
  "Why do some Ethereum addresses contain uppercase letters?",
  "How can I create unique digital items?",
  "How can a smart contract verify a signature?",
  "How does Ethereum's fee market work?",
  "How do I build a React application?",
];

/** Scores are absent for lexical-only hits under hybrid retrieval. */
function fmtScore(score: number | undefined): string {
  return score === undefined ? "  --  " : score.toFixed(4);
}

/**
 * Summarise the dense scores present in a result set.
 *
 * Spread over *scored* hits only. Mixing in a lexical-only hit as 0 made the
 * spread equal the top score, which looked like perfect discrimination when
 * it meant the opposite. Under hybrid the ranking is RRF's, so these numbers
 * are diagnostic rather than the ordering — a low spread still says the dense
 * half matched a theme rather than a passage.
 */
function scoreSummary(hits: Array<{ score?: number }>): string {
  const scored = hits
    .map((h) => h.score)
    .filter((s): s is number => s !== undefined);
  if (scored.length === 0) return "  no dense scores (all lexical-only hits)";

  const top = Math.max(...scored);
  const spread = top - Math.min(...scored);
  const lexical = hits.length - scored.length;
  const note = lexical > 0 ? ` · ${lexical} lexical-only` : "";
  return `  top ${top.toFixed(4)} · spread ${spread.toFixed(4)}` +
    ` (over ${scored.length} scored)${note}`;
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
const k = Number(flag("k", "3"));
const chars = Number(flag("chars", "300"));
const queryIntervalMs = Number(flag("interval", "0"));

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const queries = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const toRun = queries.length > 0 ? queries : DEFAULT_QUERIES;

const { retriever, model, dimensions, pointsCount, repository, mode } =
  await connect(inPath, {
  hybrid,
  weights: { dense: 1, bm25: bm25Weight },
  rrfK,
});

console.log(
  `${repository.collection} · ${pointsCount} points · ${model} · ` +
    `${dimensions} dims · ${mode} · top ${k}\n`,
);

try {
  let first = true;
  for (const query of toRun) {
    if (!first && queryIntervalMs > 0) await sleep(queryIntervalMs);
    first = false;

    console.log("=".repeat(76));
    console.log(`Query: ${query}\n`);

    const hits = await retriever.retrieve(query, k);
    if (hits.length === 0) {
      console.log("  (no results)\n");
      continue;
    }

    for (const [i, hit] of hits.entries()) {
      const eip = hit.metadata.eipNumber
        ? `EIP-${hit.metadata.eipNumber}`
        : "-";
      console.log(`Rank ${i + 1}`);
      console.log(`Score:    ${fmtScore(hit.score)}`);
      console.log(`EIP:      ${eip}  ${hit.metadata.title ?? ""}`);
      console.log(`Section:  ${hit.metadata.section ?? "-"}`);
      console.log(`Chunk ID: ${hit.chunkId}`);
      console.log(`Source:   ${hit.metadata.sourcePath ?? "-"}`);
      console.log(
        `Text:     ${hit.text.trim().replace(/\s+/g, " ").slice(0, chars)}...`,
      );
      console.log();
    }

    // The spread tells you whether the ranking actually discriminated or just
    // handed back the corpus in near-arbitrary order. A high top score with a
    // flat spread usually means the query matched a theme, not a passage.
    console.log(`${scoreSummary(hits)}\n`);
  }
} catch (error) {
  if (error instanceof EmbeddingError) {
    console.error(`\nQuery embedding failed: ${error.message}`);
    process.exit(1);
  }
  throw error;
}
