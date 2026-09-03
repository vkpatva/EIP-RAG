/**
 * Rendering: RAGResponse -> terminal text.
 *
 * Separate from `sourceMapper.ts` because these are different kinds of
 * decision. The mapper answers "what is true about this evidence" and its
 * output is data — JSON an API could return. This file answers "what should a
 * reader see", and its answers are presentational: two-space indents, a
 * dashed rule, how many decimal places on a score. A web UI would replace
 * this file entirely and keep the mapper untouched, which is the reason for
 * the seam.
 *
 * Pure, like the mapper: strings in, strings out, nothing printed. The caller
 * owns the console, so this is testable by comparing strings.
 */
import { sourceLabel } from "./sourceMapper.js";
import type { RAGResponse, Source } from "./types.js";
import type { RetrievedChunk } from "../vectorstore/types.js";

/** Scores are absent for lexical-only hits under hybrid retrieval. */
export function formatScore(score: number | undefined): string {
  return score === undefined ? "  --  " : score.toFixed(4);
}

/**
 * Summarise the dense scores in a result set.
 *
 * Spread over *scored* hits only: counting a lexical-only hit as 0 makes the
 * spread equal the top score, which reads as perfect discrimination when it
 * means the opposite. Under hybrid the ranking is RRF's, so these numbers are
 * diagnostic rather than the ordering — but a low spread on a high top score
 * still says the dense half matched a theme rather than a passage, which is
 * the single most useful thing to know when a source list looks wrong.
 */
export function scoreSummary(hits: Array<{ score?: number }>): string {
  const scored = hits
    .map((h) => h.score)
    .filter((s): s is number => s !== undefined);
  if (scored.length === 0) return "  no dense scores (all lexical-only hits)";

  const top = Math.max(...scored);
  const spread = top - Math.min(...scored);
  const lexical = hits.length - scored.length;
  const note = lexical > 0 ? ` · ${lexical} lexical-only` : "";
  return (
    `  top ${top.toFixed(4)} · spread ${spread.toFixed(4)}` +
    ` (over ${scored.length} scored)${note}`
  );
}

/**
 * The retrieval block: what the retriever returned, before generation.
 *
 * Printed above the answer because when an answer is wrong the first question
 * is always whether the evidence was wrong or the reading of it was. It is
 * also what makes the source list checkable by eye: every citation must name
 * a chunk id that appears here.
 */
export function formatRetrieval(
  chunks: RetrievedChunk[],
  chars = 160,
): string {
  const lines = [`RETRIEVED (${chunks.length} chunks):`];

  for (const [i, hit] of chunks.entries()) {
    const eip =
      hit.metadata.eipNumber !== undefined ? `EIP-${hit.metadata.eipNumber}` : "-";
    // Which retriever found a chunk is the first thing you want when a hit
    // looks wrong: "bm25" alone on an off-topic chunk means the query shared
    // a rare-looking term with it and nothing more.
    const via = hit.retrievedBy
      ? `  [${Object.keys(hit.retrievedBy).join("+")}]`
      : "";
    lines.push(
      `  [${i + 1}] ${formatScore(hit.score)}  ${eip}  ` +
        `${hit.metadata.section ?? "-"}  (${hit.chunkId})${via}`,
    );
    lines.push(
      `      ${hit.text.trim().replace(/\s+/g, " ").slice(0, chars)}...`,
    );
  }

  if (chunks.length > 0) lines.push(scoreSummary(chunks));
  return lines.join("\n");
}

/** One source block: label, then the fields that make it checkable. */
function formatSource(source: Source): string {
  const lines = [`[${source.citationId}] ${sourceLabel(source)}`];

  // Section first: in a spec corpus it is the field that says what *kind* of
  // authority the passage carries — "Methods" is normative interface text,
  // "Rationale" is the authors explaining themselves.
  if (source.section) lines.push(`    Section: ${source.section}`);

  // The chunk id is what a reader (or you, debugging) pastes back into
  // `npm run retrieve` to read the exact passage the model saw.
  lines.push(`    Chunk: ${source.chunkId}`);

  if (source.sourcePath) lines.push(`    File: ${source.sourcePath}`);

  // Printed as "--" when absent rather than 0.0000, because the two mean
  // different things: a lexical-only hit was not scored by the dense
  // retriever, and a 0 would read as "maximally dissimilar".
  lines.push(`    Score: ${formatScore(source.score).trim()}`);

  return lines.join("\n");
}

/**
 * The SOURCES section.
 *
 * The empty case is the interesting one. "None" is printed with the retrieval
 * count beside it, because those are two different facts and collapsing them
 * hides the more useful one: "Sources: None" alone reads as "retrieval found
 * nothing", while "None (5 chunks retrieved, none cited)" says what actually
 * happened — retrieval worked, the evidence did not support an answer. The
 * first sends you to debug the retriever; the second tells you the corpus has
 * a gap. That distinction is the same one rule 4 of `SYSTEM_PROMPT` asks the
 * model to make in prose.
 */
export function formatSources(response: RAGResponse): string {
  if (response.sources.length === 0) {
    const note =
      response.chunksRetrieved > 0
        ? `  (${response.chunksRetrieved} chunk${
            response.chunksRetrieved === 1 ? "" : "s"
          } retrieved; the answer did not rely on them)`
        : "  (retrieval returned nothing)";
    return `SOURCES: None\n${note}`;
  }

  return `SOURCES:\n\n${response.sources.map(formatSource).join("\n\n")}`;
}

/**
 * A full response block: question, answer, sources.
 *
 * The answer header carries the model and mode. Mode especially: with
 * `--mode=auto` it is a heuristic's guess over the question's wording, and the
 * two rule sets have opposite failure modes, so an unexpected answer is very
 * often the right rule set applied to the wrong question. Not printing it made
 * exactly that failure look like a retrieval problem.
 */
export function renderResponse(
  question: string,
  response: RAGResponse,
): string {
  return [
    "=".repeat(76),
    `QUESTION: ${question}`,
    "=".repeat(76),
    "",
    `ANSWER (${response.model} · ${response.mode}):`,
    "",
    response.answer.trim(),
    "",
    formatSources(response),
  ].join("\n");
}
