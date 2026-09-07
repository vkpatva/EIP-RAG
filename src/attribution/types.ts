/**
 * Types for the attribution stage.
 *
 * The boundary this file draws: attribution turns *retrieval output* into
 * *reader-facing provenance*. It sits beside generation, not inside it, and
 * the reason is a guarantee rather than a preference — a `Source` is derived
 * from a database read, so there is no step at which a wrong EIP number could
 * be produced. The prompt in `generator/prompt.ts` *asks* the model not to
 * invent an EIP number (rule 2); this stage makes inventing one impossible,
 * because nothing here generates.
 *
 * That difference is the whole module. A citation is a trust signal: bare
 * prose invites scepticism, prose with a bracketed reference invites belief.
 * So a fabricated citation does not merely add one error, it lends
 * credibility to every claim beside it. The only safe place for citations is
 * therefore code that cannot fabricate.
 *
 * A second property falls out of the split, and it is the one you notice in
 * production: sources are computable with the API key removed. If the LLM
 * call times out, the evidence is already in hand and can be shown to the
 * reader instead of a blank screen.
 */
import type { RetrievedChunk } from "../vectorstore/types.js";

/**
 * One attributed piece of evidence: a retrieved chunk, labelled for a reader.
 *
 * Every field except `citationId` is copied verbatim from the chunk that
 * Qdrant returned, and every one of those was written during ingestion from
 * the file on disk. Nothing here is inferred, scored, or predicted.
 *
 * The fields form a chain, and the chain is the point — each link is what
 * makes the next one checkable:
 *
 *   citationId  [1]                       the label the reader sees
 *     -> chunkId      eip-20.md:3         the passage the model actually read
 *     -> section      "Methods"           what role that passage plays
 *     -> documentId   eip-20.md           which document holds it
 *     -> sourcePath   eip-20.md           the exit back to the original file
 *
 * Break any link and the rest proves nothing: "somewhere in EIP-721" is not a
 * verifiable claim, and a bare `chunkId` is an opaque token a reader cannot
 * act on. Both ends are needed, which is why both are here.
 */
export interface Source {
  /**
   * The reader-facing label, e.g. "1", rendered as `[1]`.
   *
   * The one invented field, and inventing it is the purpose: numbering is an
   * application decision, so the application assigns it. Scoped to a single
   * response — `[1]` means nothing outside the answer it appears in.
   */
  citationId: string;

  /**
   * `RetrievedChunk.chunkId`, e.g. "eip-20.md:3".
   *
   * The precision link. Without it a citation names a whole document, and
   * "it is somewhere in EIP-721" cannot be checked in any useful time. Also
   * the debugging handle: this id appears in `npm run retrieve` output, so a
   * suspect citation can be traced to the exact passage.
   */
  chunkId: string;

  /**
   * `RetrievedChunk.documentId`, e.g. "eip-20.md".
   *
   * The grouping key — "these three citations are all EIP-20" is a comparison
   * on this field. Also the stable identity of the two: chunk ids are
   * positional (`file.md:7`), so they shift when a document is re-chunked
   * while the document id does not.
   */
  documentId: string;

  /**
   * EIP/ERC number, e.g. 20. A `number`, matching `ChunkPayload.eipNumber`.
   *
   * Not a string: the eval computes Recall@K on the numeric field, and
   * stringifying it here would only make the two halves of the pipeline
   * disagree about the type of the same value. Optional because a document
   * with absent or malformed front matter legitimately has none, and a
   * citation must still be renderable in that case.
   */
  eipNumber?: number;

  /** Document title, e.g. "ERC-20 Token Standard". What makes a label readable. */
  title?: string;

  /**
   * Nearest heading above the chunk, e.g. "Methods".
   *
   * The highest-value field for a reader, and the one that exists nowhere
   * else in the response. In a specification corpus, "Methods" and
   * "Rationale" are different kinds of authority: the first is normative
   * interface text, the second is the authors explaining themselves. A
   * citation without the section hides that difference.
   */
  section?: string;

  /**
   * Path relative to the data directory, e.g. "eip-20.md".
   *
   * The bottom link: the exit from this system back to the original file. Also
   * the natural seed for a URL later, though deriving one needs a repo, branch
   * and anchor convention that does not exist yet.
   */
  sourcePath?: string;

  /**
   * Dense similarity score. Optional, and it must stay optional.
   *
   * `RetrievedChunk.score` is `undefined` — not 0 — for a chunk only the
   * lexical half found, because a 0 reads as "maximally dissimilar" when the
   * truth is "not scored by this retriever". Declaring this `number` would
   * force a 0 to be invented here and reintroduce exactly that bug.
   */
  score?: number;

  /**
   * Position in the retrieved ranking, 1-based.
   *
   * Carried because under fusion it, not `score`, is the authoritative order:
   * RRF blends two rankings whose scores are not on a comparable scale, so
   * `score` need not even be monotonic. Since `citationId` is derived from
   * rank, omitting `rank` would leave a `Source` holding the number you
   * cannot sort on and missing the one you can.
   */
  rank?: number;
}

/**
 * The pipeline's output: an answer and the evidence behind it.
 *
 * This replaces the bare `string` at the *pipeline* boundary, not at the
 * generation boundary. `GenerationService.generate` still returns a string,
 * because prose is genuinely all that generation produces — it never learns
 * that sources exist. The two are joined one layer up, in `answerQuestion`.
 */
export interface RAGResponse {
  answer: string;
  /**
   * The attributed evidence, in retrieval order.
   *
   * Empty when the answer did not rest on the retrieved chunks — see
   * `looksLikeRefusal`. An empty array is a claim in its own right ("nothing
   * here supports this"), which is why it is distinguished from a populated
   * one rather than merged into it.
   */
  sources: Source[];
  /**
   * How many chunks retrieval actually returned.
   *
   * Kept separately from `sources.length` because the two differ exactly when
   * it matters: on a refusal, sources are suppressed but chunks were still
   * retrieved, and "5 chunks retrieved, none cited" is a far more useful
   * thing to print than a silent absence.
   */
  chunksRetrieved: number;
  /**
   * Which model produced the answer, e.g. "gpt-4o-mini".
   *
   * Part of the response rather than of a debug channel because it is
   * provenance too — of the *answer* rather than of the evidence. An answer
   * that cannot be traced to a model is not reproducible, and comparing two
   * runs means knowing whether the model changed under you.
   */
  model: string;
  /**
   * Which rule set ran: "extraction" or "synthesis".
   *
   * The single most useful field for interpreting an unexpected answer, and
   * the reason it is here. The two prompts have opposite failure modes —
   * extraction refuses to emit a function signature, synthesis emits one and
   * pins it to the evidence — so the same question answered under the wrong
   * mode produces a *correct-looking* refusal from the wrong rule set. With
   * `--mode=auto` the choice is a heuristic over the question's wording, so
   * it is a guess the output has to disclose.
   */
  mode: "extraction" | "synthesis";
}
