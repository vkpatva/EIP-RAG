/**
 * Dev script: prove that citations are not fabricated.
 *
 *   npm run verify:attribution
 *   npm run verify:attribution -- --k=5 --interval=20000
 *
 * Part 8 asks for a check "that every displayed source actually came from the
 * retrieved chunks". Reading the output by eye establishes that for one run;
 * this establishes it mechanically, and in a form that keeps working after a
 * refactor.
 *
 * Five invariants, each the negation of a specific citation failure:
 *
 *   1. Every source's chunkId appears in the retrieved set.
 *        -> no citation to a document that was never retrieved.
 *   2. Every source field equals the corresponding chunk field.
 *        -> no invented EIP number, section, path or score.
 *   3. Citation ids are exactly 1..n in order.
 *        -> numbering is the application's, and is deterministic.
 *   4. Sources are either all of the retrieved chunks or none.
 *        -> the only filtering is the documented refusal policy.
 *   5. No source id appears in the answer that isn't a real citation id.
 *        -> the model did not emit a bracketed reference of its own.
 *
 * Note what makes invariants 1 and 2 checkable at all: `mapSources` is pure,
 * so its output can be compared field-by-field against its input. If sources
 * came from the LLM there would be nothing to compare against.
 */
import { connect } from "./vectorstore/connect.js";
import { OpenAIChatProvider, RAGGenerationService } from "./generator/index.js";
import { answerQuestion, looksLikeRefusal } from "./attribution/index.js";
import type { Source } from "./attribution/index.js";
import type { RetrievedChunk } from "./vectorstore/types.js";

const QUERIES = [
  "What is EIP-712?",
  "Why do some Ethereum addresses contain uppercase letters?",
  "What functions are required by ERC-20?",
  "What's the difference between ERC-721 and ERC-1155?",
  "How do I build a React application?",
];

function flag(name: string, fallback: string): string {
  return (
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ??
    fallback
  );
}

const k = Number(flag("k", "5"));
const queryIntervalMs = Number(flag("interval", "0"));
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Check one response. Returns the invariant violations found. */
function checkInvariants(
  answer: string,
  sources: Source[],
  chunks: RetrievedChunk[],
): string[] {
  const problems: string[] = [];
  const byId = new Map(chunks.map((c) => [c.chunkId, c]));

  for (const source of sources) {
    // 1 — traceability.
    const chunk = byId.get(source.chunkId);
    if (!chunk) {
      problems.push(
        `source [${source.citationId}] cites ${source.chunkId}, ` +
          `which was not retrieved`,
      );
      continue;
    }

    // 2 — fidelity. Every field must be the chunk's own value, not a
    // near-miss: a *plausible* EIP number is the failure being excluded.
    const mismatches: string[] = [];
    if (source.documentId !== chunk.documentId) mismatches.push("documentId");
    if (source.eipNumber !== chunk.metadata.eipNumber)
      mismatches.push("eipNumber");
    if (source.title !== chunk.metadata.title) mismatches.push("title");
    if (source.section !== chunk.metadata.section) mismatches.push("section");
    if (source.sourcePath !== chunk.metadata.sourcePath)
      mismatches.push("sourcePath");
    if (source.score !== chunk.score) mismatches.push("score");
    if (mismatches.length > 0) {
      problems.push(
        `source [${source.citationId}] (${source.chunkId}) diverges from ` +
          `its chunk: ${mismatches.join(", ")}`,
      );
    }
  }

  // 3 — numbering is the application's, and deterministic.
  const expected = sources.map((_, i) => String(i + 1)).join(",");
  const actual = sources.map((s) => s.citationId).join(",");
  if (expected !== actual) {
    problems.push(`citation ids are "${actual}", expected "${expected}"`);
  }

  // 4 — the only filtering is the refusal policy.
  const refused = looksLikeRefusal(answer);
  if (!refused && sources.length !== chunks.length) {
    problems.push(
      `${chunks.length} chunks retrieved but ${sources.length} sources ` +
        `shown on a non-refusal answer`,
    );
  }
  if (refused && sources.length !== 0) {
    problems.push(`answer reads as a refusal but ${sources.length} sources shown`);
  }

  // 5 — the model did not invent its own bracketed references. A number
  // beyond the source count could only have come from the LLM, since the
  // application never emits one.
  const cited = [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
  const invented = cited.filter((n) => n < 1 || n > sources.length);
  if (invented.length > 0) {
    problems.push(
      `answer contains citation marker(s) [${invented.join("], [")}] with ` +
        `only ${sources.length} sources available`,
    );
  }

  return problems;
}

const { retriever, model, pointsCount, repository, mode } = await connect(
  "data/embeddings.json",
  { hybrid: true, weights: { dense: 1, bm25: 0.5 }, rrfK: 2 },
);

const generator = new RAGGenerationService({
  provider: new OpenAIChatProvider(),
  mode: "extraction",
});

console.log(
  `${repository.collection} · ${pointsCount} points · ${model} · ` +
    `${mode} · top ${k}\n`,
);
console.log("Verifying that every displayed source came from retrieval.\n");

let failures = 0;
let first = true;

for (const question of QUERIES) {
  if (!first && queryIntervalMs > 0) await sleep(queryIntervalMs);
  first = false;

  const { answer, sources, chunks } = await answerQuestion(question, {
    retriever,
    generator,
    k,
  });

  const problems = checkInvariants(answer, sources, chunks);
  const refused = looksLikeRefusal(answer);
  const status = problems.length === 0 ? "PASS" : "FAIL";
  if (problems.length > 0) failures++;

  console.log(
    `${status}  ${sources.length}/${chunks.length} sources` +
      `${refused ? " (refusal, suppressed)" : ""}  ${question}`,
  );
  for (const problem of problems) console.log(`      ! ${problem}`);

  // The chunk ids on both sides, so a reader can see the correspondence
  // rather than only being told it holds.
  console.log(`      chunks : ${chunks.map((c) => c.chunkId).join(", ")}`);
  console.log(
    `      sources: ${
      sources.length > 0
        ? sources.map((s) => `[${s.citationId}]${s.chunkId}`).join(", ")
        : "none"
    }`,
  );
}

console.log(
  failures === 0
    ? `\nAll ${QUERIES.length} responses passed. No source was fabricated.`
    : `\n${failures} of ${QUERIES.length} responses violated an invariant.`,
);
process.exit(failures === 0 ? 0 : 1);
