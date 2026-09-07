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
import type { CheckedSource } from "./support.js";
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

  // The matched terms, when known. This is the line that makes the citation
  // checkable without leaving the terminal: "Quotes: transferFrom, allowance"
  // says which strings in the answer came out of this chunk.
  if (isChecked(source) && source.quoted.length > 0) {
    const shown = source.quoted.slice(0, 6).join(", ");
    const more =
      source.quoted.length > 6 ? ` (+${source.quoted.length - 6} more)` : "";
    lines.push(`    Quotes: ${shown}${more}`);
  }

  return lines.join("\n");
}

/** A `Source` that has been through `checkSupport`. */
function isChecked(source: Source): source is CheckedSource {
  return "support" in source;
}

/**
 * The SOURCES section.
 *
 * Two things are separated here, because they are two different claims and
 * one heading was making the stronger one on the weaker one's evidence:
 *
 *  - **SOURCES** — chunks that share a distinctive term with the answer.
 *    These are the ones the answer demonstrably drew on, so a reader checking
 *    a signature will find it there.
 *  - **ALSO RETRIEVED** — chunks that were sent to the model with no such
 *    term found. Listed compactly, below, under a heading that does not claim
 *    they supported anything.
 *
 * The second list is kept rather than dropped, and the reason is a limit of
 * the check rather than a preference. A shared identifier is proof a chunk
 * *was* drawn on; its absence is not proof a chunk was not, because a
 * paraphrasing answer shares no exact strings with its source. Deleting those
 * chunks would replace an overclaim ("all ten supported this") with a
 * different overclaim ("only these two were involved"), and would also hide
 * the retrieval problem that put eight unhelpful chunks in the context window
 * — which is a thing worth seeing, not hiding.
 *
 * The empty case: "None" is printed with the retrieval count beside it,
 * because those are two different facts and collapsing them hides the more
 * useful one. "Sources: None" alone reads as "retrieval found nothing", while
 * "None (5 chunks retrieved, none cited)" says retrieval worked and the
 * evidence did not support an answer. The first sends you to debug the
 * retriever; the second says the corpus has a gap.
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

  const checked = response.sources.filter(isChecked);

  // No support information available — either the sources were never checked
  // or the answer had nothing checkable in it. Fall back to one flat list,
  // since splitting it would imply a distinction that was never computed.
  if (checked.length !== response.sources.length) {
    return `SOURCES:\n\n${response.sources.map(formatSource).join("\n\n")}`;
  }
  if (checked.every((source) => source.support === "unchecked")) {
    return (
      `SOURCES (support not checked — the answer quotes no identifiers):\n\n` +
      checked.map(formatSource).join("\n\n")
    );
  }

  const supported = checked.filter((source) => source.support === "quoted");
  const rest = checked.filter((source) => source.support !== "quoted");

  const blocks: string[] = [];

  if (supported.length > 0) {
    blocks.push(
      `SOURCES (${supported.length} of ${checked.length} retrieved chunks ` +
        `contain text the answer uses):\n\n` +
        supported.map(formatSource).join("\n\n"),
    );
  } else {
    // Every chunk failed the check. Worth saying loudly rather than printing
    // an empty heading: an answer full of signatures none of whose chunks
    // contain them came from training weights, which is the exact failure
    // synthesis rule 2 forbids and cannot itself enforce.
    blocks.push(
      `SOURCES: none of the ${checked.length} retrieved chunks contain text ` +
        `the answer uses.\n` +
        `  The answer may not be grounded in the retrieved evidence.`,
    );
  }

  if (rest.length > 0) {
    const lines = rest.map(
      (source) =>
        `  [${source.citationId}] ${sourceLabel(source)}` +
        `${source.section ? ` · ${source.section}` : ""}` +
        `  (${source.chunkId})`,
    );
    blocks.push(
      `ALSO RETRIEVED (${rest.length} chunk${
        rest.length === 1 ? "" : "s"
      } sent to the model; no quoted text found):\n` + lines.join("\n"),
    );
  }

  return blocks.join("\n\n");
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
