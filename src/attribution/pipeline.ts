/**
 * The RAG pipeline: question -> RAGResponse.
 *
 * This is the join, and it is the only file that knows both halves exist:
 *
 *                        RetrievedChunk[]
 *                              |
 *              +---------------+---------------+
 *              v                               v
 *      GenerationService                  mapSources
 *      (LLM, async, paid, stochastic)     (pure, sync, free, deterministic)
 *              v                               v
 *           answer                          Source[]
 *              +---------------+---------------+
 *                              v
 *                          RAGResponse
 *
 * A fork, not a chain. The two branches read the same input and share nothing
 * else, and the independence is what buys the guarantees:
 *
 *  - Sources cannot be fabricated, because the branch that produces them has
 *    no generative step. The prompt *asks* the model not to invent an EIP
 *    number; this asks nothing, because the mapper cannot.
 *  - Sources survive a generation failure. They are computed from data
 *    already in hand, so a timeout or a 500 costs the answer and not the
 *    evidence — a reader can be shown what was retrieved instead of nothing.
 *  - Failures stay attributable. A wrong source is a mapper or ingestion bug;
 *    a wrong answer is a prompt or retrieval bug. Fused, every failure is one
 *    undifferentiated "the RAG is bad" — the same argument `generator/types.ts`
 *    makes for separating retrieval from generation, one level down.
 *
 * The join lives here rather than inside `RAGGenerationService` for the
 * second of those reasons. If generation owned the mapper, attribution would
 * inherit generation's failure modes and the guarantee would be gone.
 * `GenerationService.generate` still returns a plain string, which keeps its
 * own contract true: generation composes prose, and never learns that
 * citations exist.
 */
import { looksLikeRefusal, mapSources } from "./sourceMapper.js";
import type { RAGResponse } from "./types.js";
import type { RAGGenerationService } from "../generator/generationService.js";
import type { Retriever } from "../vectorstore/retriever.js";
import type { RetrievedChunk } from "../vectorstore/types.js";

export interface AnswerQuestionOptions {
  retriever: Retriever;
  /**
   * Typed as the concrete service, not the `GenerationService` interface,
   * because `generateDetailed` is deliberately not on that interface: a
   * caller wanting only an answer should not have to know a prompt exists.
   * The pipeline is exactly the caller that does want it — `model` and `mode`
   * belong in the response, and `--show-prompt` needs the strings that were
   * actually sent.
   */
  generator: RAGGenerationService;
  /** How many chunks to retrieve. 5 is the evaluated baseline (Recall@5 88%). */
  k?: number;
  /**
   * Suppress sources when the answer declines for want of evidence.
   *
   * On by default, because the alternative asserts something false: a
   * "SOURCES" heading means "the evidence behind this answer", and a refusal
   * has no evidence behind it. Switchable because it is a *policy*, and a
   * caller that wants to see everything retrieval returned — an eval harness,
   * say — should not have to defeat a heuristic to get it.
   */
  suppressSourcesOnRefusal?: boolean;
}

/**
 * Retrieve, generate and attribute, in one call.
 *
 * Retrieval runs first because both branches need its output; from there the
 * two are independent. They are not run concurrently: generation is the only
 * async half, and mapping is a synchronous `.map()` over five objects, so
 * there is nothing to overlap. `Promise.all` here would be theatre.
 */
export async function answerQuestion(
  question: string,
  options: AnswerQuestionOptions,
): Promise<
  RAGResponse & {
    chunks: RetrievedChunk[];
    systemPrompt: string;
    userPrompt: string;
  }
> {
  const {
    retriever,
    generator,
    k = 5,
    suppressSourcesOnRefusal = true,
  } = options;

  // Stage 1 — retrieval. No LLM. Ends with text plus provenance.
  const chunks = await retriever.retrieve(question, k);

  // Stage 2a — attribution. Pure, free, deterministic. Computed *before*
  // generation, not after, so that a failure in the LLM call cannot take the
  // evidence with it.
  const allSources = mapSources(chunks);

  // Stage 2b — generation. Sees the chunks; never sees the sources.
  const generated = await generator.generateDetailed(question, chunks);

  // The join, and the one policy decision that needs both halves in view.
  const sources =
    suppressSourcesOnRefusal && looksLikeRefusal(generated.answer)
      ? []
      : allSources;

  // `chunks` and the two prompt strings are returned alongside rather than
  // added to `RAGResponse`, because they are debugging needs: putting five
  // full passages and two multi-kilobyte prompts in the response type would
  // burden every consumer that only wanted an answer and its citations.
  // `model` and `mode` *are* in the response, because they are provenance of
  // the answer rather than diagnostics about it.
  return {
    answer: generated.answer,
    sources,
    chunksRetrieved: chunks.length,
    model: generated.model,
    mode: generated.mode,
    chunks,
    systemPrompt: generated.systemPrompt,
    userPrompt: generated.userPrompt,
  };
}
