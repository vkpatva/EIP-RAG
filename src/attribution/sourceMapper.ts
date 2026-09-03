/**
 * SourceMapper: RetrievedChunk[] -> Source[].
 *
 * Deliberately pure, in the same sense as `prompt.ts`: no network, no
 * environment, no clock, no LLM. The same chunks in the same order always
 * produce the same sources, so this is testable by hand-writing three chunks
 * and asserting on the output — no API key, no Qdrant, no cost, no flake.
 *
 * That purity is not a style choice, it is the security property. Every field
 * written here is *copied* from a chunk payload that ingestion wrote from the
 * file on disk. There is no generative step, so the three highest-damage
 * citation failures are not mitigated but absent:
 *
 *   - an EIP number that does not exist        (the number is read, not chosen)
 *   - a citation to a document never retrieved (input *is* the retrieval output)
 *   - a fabricated score or chunk id           (both copied verbatim)
 *
 * The fourth failure — a claim attributed to the wrong source — is not
 * addressed here, and cannot be: it lives in the answer->chunk link, which is
 * the one link an LLM establishes. See `looksLikeRefusal` below and the
 * Version A / Version B distinction in the README.
 *
 * A function rather than a class, matching the codebase's split: classes hold
 * state or a connection (`RAGGenerationService` holds a provider, `BM25Index`
 * holds an index), pure transforms are plain exports (`buildUserPrompt`,
 * `reciprocalRankFusion`). There is no state to hold here, and a constructor
 * wrapping a `.map()` would be ceremony with no payload.
 */
import type { Source } from "./types.js";
import type { RetrievedChunk } from "../vectorstore/types.js";

/**
 * Attribute retrieved chunks, in the order given.
 *
 * Citation ids are the 1-based array index: `[1]` is the first chunk printed,
 * `[2]` the second. Three properties come out of that choice.
 *
 * It is deterministic — position in the array is the only input, so the same
 * retrieval always yields the same labels, which is what makes two runs of
 * `npm run generate` diffable.
 *
 * It carries information for free — the retriever returns strongest-first, so
 * `[1]` *is* the top-ranked evidence without anything having to say so.
 *
 * And it is scoped to one response, correctly. `[1]` here is not `[1]` in the
 * next answer, exactly as footnote 1 of one chapter is not footnote 1 of the
 * next. Stable cross-response identity is what `chunkId` is for, and it is
 * already on every `Source`.
 *
 * The index is used rather than `chunk.rank` because the label must describe
 * what the reader sees. `rank` is optional (the dense-only retriever never
 * sets it) and belongs to the retriever's own ranking, which a filtered or
 * re-sliced list would no longer match — a `[3]` appearing second in the list
 * would be worse than a label that merely repeats the position.
 *
 * Deduplication is not performed. Two chunks from the same document are two
 * distinct pieces of evidence — different sections, different text, different
 * ranks — and collapsing them to one citation would claim the answer rested
 * on one passage when it rested on two. Grouping for display is a rendering
 * concern; `documentId` is the key for it when it is wanted.
 */
export function mapSources(chunks: RetrievedChunk[]): Source[] {
  return chunks.map((chunk, index) => ({
    citationId: String(index + 1),
    chunkId: chunk.chunkId,
    documentId: chunk.documentId,
    // Spread the metadata field-by-field rather than by object spread: the
    // set of fields is then visible at the call site, so a field added to
    // `RetrievedChunk.metadata` for retrieval's own purposes does not
    // silently become part of the reader-facing contract.
    eipNumber: chunk.metadata.eipNumber,
    title: chunk.metadata.title,
    section: chunk.metadata.section,
    sourcePath: chunk.metadata.sourcePath,
    score: chunk.score,
    rank: chunk.rank,
  }));
}

/**
 * A short human label for a source, e.g. "EIP-20 — ERC-20 Token Standard".
 *
 * Falls back through eipNumber -> sourcePath -> documentId so a source is
 * never anonymous. An unlabelled citation is worse than none: it looks like
 * provenance while being unusable, which is the failure this whole module
 * exists to prevent. Mirrors `chunkHeader` in `generator/prompt.ts` on
 * purpose — the reader should see the same label the model saw.
 */
export function sourceLabel(source: Source): string {
  const base =
    source.eipNumber !== undefined
      ? `EIP-${source.eipNumber}`
      : (source.sourcePath ?? source.documentId);

  return source.title ? `${base} — ${source.title}` : base;
}

/**
 * Phrases that mark an answer as declining for want of evidence.
 *
 * These are not guesses about model behaviour: they are the wordings rule 4
 * of `SYSTEM_PROMPT` asks for ("say so plainly and state what the evidence
 * does cover"). Matching the prompt's own instruction is what keeps this
 * honest — if the rule is reworded, this list is what must be reworded with
 * it, and the coupling is deliberate rather than accidental.
 *
 * Kept as a small literal list, not a grammar. Detecting refusal properly is
 * a classification problem, and solving it properly means a second LLM call
 * to judge the first — an evaluation concern, out of scope here.
 */
const REFUSAL_MARKERS = [
  "contain enough information",
  "provide enough information",
  "contain any information",
  "contain information",
  "does not address",
  "do not address",
  "does not answer",
  "do not answer",
  "does not mention",
  "do not mention",
  "does not discuss",
  "do not discuss",
  // "does not cover" was a real miss: the negative-control answer opened
  // "The evidence provided does not cover how to build a React application"
  // and kept all five citations — precisely the misleading output this
  // function exists to prevent. Added from observed behaviour, which is how
  // this list is meant to grow.
  "does not cover",
  "do not cover",
  "does not contain",
  "do not contain",
  "does not include",
  "do not include",
  "does not relate",
  "does not pertain",
  // Observed misses, both from a synthesis request answered under extraction
  // rules ("can you write me an ERC20 interface"). The answer opened "The
  // evidence does not provide a specific ERC20 interface or its function
  // signatures" and closed "Therefore, I cannot provide you with the ERC20
  // interface" — a refusal that kept all ten citations, which is the
  // misleading output this function exists to prevent.
  "does not provide",
  "do not provide",
  "does not specify",
  "do not specify",
  "does not define",
  "do not define",
  "cannot provide",
  "can't provide",
  "unable to provide",
  "no information",
  "sufficient to answer",
  "insufficient information",
  "cannot be answered",
  "cannot answer",
  "can't answer",
  "unable to answer",
  "unrelated to the question",
  "not related to the question",
];

/**
 * Words naming the *evidence* rather than a specification.
 *
 * The distinction this draws is the one that makes the heuristic safe.
 * "The evidence does not mention royalties" is a refusal. "ERC-20 does not
 * mention royalties" is a grounded claim about a standard — a good answer,
 * and one that must keep its citations. The marker phrase is identical in
 * both; only the subject differs, so the subject is what gets checked.
 */
const EVIDENCE_SUBJECT =
  /\b(evidence|excerpts?|context|passages?|documents?|sources?|provided|supplied|retrieved|information)\b/;

/**
 * A first-person decline: the model saying it will not or cannot act.
 *
 * Distinct from `EVIDENCE_SUBJECT` because the two identify different things.
 * That one asks "is this sentence about the evidence"; this asks "is the model
 * refusing". A closing sentence needs the second, since a good answer's final
 * caveat is also about the evidence.
 */
const FIRST_PERSON_DECLINE = /\bi (cannot|can't|am unable to|will not|won't)\b/;

/**
 * Whether an answer declined to use the retrieved evidence.
 *
 * Used to decide between "SOURCES: [1] [2] ..." and "SOURCES: None". The
 * problem it solves: on a refusal, listing the retrieved chunks under a
 * "SOURCES" heading asserts something false. The heading means "the evidence
 * behind this answer", and there is no evidence behind a refusal. A reader
 * scanning the output sees a confident-looking citation list beside "I cannot
 * answer" and reasonably concludes the sources *were* relevant and the model
 * was merely being cautious — the opposite of what happened.
 *
 * A heuristic, and only defensible because of what it is *not*. It does not
 * score relevance, does not threshold on `score` (which is `undefined` for
 * lexical-only hits, so any cutoff would be both a magic number and wrong on
 * a real case), and does not ask a model to judge. It reads the answer the
 * generator produced and takes it at its word.
 *
 * Two tests, and each exists because the other alone gave a false positive on
 * a real answer:
 *
 *  1. The marker must appear in the answer's *first sentence*. Rule 4 of
 *     `SYSTEM_PROMPT` asks the model to decline up front, so a refusal states
 *     itself immediately, whereas an answer that notes a gap does so after
 *     establishing what it can say. "EIP-1559 introduces a base fee. The
 *     evidence does not mention gas refunds" is an answer; only position
 *     separates it from a refusal, since the phrase itself is the same.
 *  2. The marker's subject must be the evidence. Without this, "ERC-20 does
 *     not mention royalties" — a perfectly good grounded claim — reads as a
 *     refusal and loses its citations.
 *
 * The remaining failure modes, since a heuristic whose errors are unknown is
 * not a heuristic but a hope:
 *
 *  - A partial answer that opens with the gap ("the evidence does not cover
 *    X, but it does define Y") is read as a refusal and loses its sources.
 *    Wrong, and wrong in the safe direction: understating support is
 *    recoverable, overstating it is not, and `chunksRetrieved` keeps the
 *    retrieval visible either way.
 *  - A refusal phrased in some unlisted way keeps its sources — i.e. the
 *    misleading case this exists to prevent. Which is why the fix for a
 *    missed case is to add the phrasing here, not to reach for scoring.
 */
export function looksLikeRefusal(answer: string): boolean {
  const text = answer.trim().toLowerCase();

  const sentences = text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  if (sentences.length === 0) return false;

  // The first sentence, where rule 4 asks a refusal to declare itself. The
  // subject must be the evidence, or the sentence must be a first-person
  // decline — otherwise "ERC-20 does not mention royalties", a good grounded
  // claim about a *standard*, would read as a refusal.
  const opening = sentences[0]!;
  if (
    (EVIDENCE_SUBJECT.test(opening) || FIRST_PERSON_DECLINE.test(opening)) &&
    REFUSAL_MARKERS.some((marker) => opening.includes(marker))
  ) {
    return true;
  }

  // The last sentence, where a refusal that reasoned first states its
  // conclusion ("Therefore, I cannot provide you with the ERC20 interface").
  //
  // This position takes a *stricter* test — a first-person decline only, with
  // no evidence-subject fallback. The reason is that the closing sentence is
  // genuinely ambiguous: a refusal's conclusion and a good answer's closing
  // caveat are the same shape. "The evidence does not mention permit, which
  // is ERC-2612" ends a correct nine-function answer and must keep its
  // citations, while "I cannot provide it from this evidence" ends a refusal.
  // What separates them is not the evidence being named but the model
  // declining to act, so that is what is matched.
  const closing = sentences[sentences.length - 1]!;
  return (
    FIRST_PERSON_DECLINE.test(closing) &&
    REFUSAL_MARKERS.some((marker) => closing.includes(marker))
  );
}
