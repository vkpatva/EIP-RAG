/**
 * Types for the generation stage.
 *
 * Generation is the second half of RAG. Retrieval ends with text: a ranked
 * list of passages that are geometrically near the question. Nobody asked a
 * question in order to receive five paragraphs of markdown, so something has
 * to turn those passages into an answer. That is this stage.
 *
 * The boundary this file draws is the important part. A `GenerationService`
 * receives a question and some chunks. It does not know where the chunks came
 * from — not the collection, not the embedding model, not that Qdrant exists.
 * `RetrievedChunk` is reused as the contract between the two stages precisely
 * because nothing in it mentions Qdrant: a hand-written literal satisfies it
 * just as well as a search hit, which is what makes generation testable with
 * no infrastructure running at all.
 *
 * The split also localises failure. When an answer is wrong there are two
 * candidates — the evidence was wrong, or the reading of it was. Separated,
 * you look at the chunks (`npm run retrieve`) and know within seconds which
 * one it is. Fused, every failure is one undifferentiated "the RAG is bad".
 */
import type { RetrievedChunk } from "../vectorstore/types.js";

/**
 * The generation boundary.
 *
 *   question + retrieved context  ->  LLM  ->  answer
 *
 * Returns a bare string for now. Citations, confidence, and structured output
 * are all deliberately out of scope: each one changes this return type, and
 * changing it later is a smaller cost than guessing its final shape now.
 */
export interface GenerationService {
  generate(question: string, context: RetrievedChunk[]): Promise<string>;
}

/**
 * The LLM boundary.
 *
 * Narrow on purpose, and narrow in the same way `EmbeddingProvider` is: text
 * in, text out. A provider knows nothing about chunks, evidence blocks, or
 * grounding rules — it takes two prepared strings and returns a completion.
 *
 * Two messages rather than one concatenated prompt, because the split is not
 * cosmetic. Providers train models to weight system content more heavily and
 * to treat it as harder to override. That makes the role tag the only real
 * mechanism separating "rules I wrote" from "text that came out of a file",
 * so the rules go in `system` and the retrieved evidence goes in `user`.
 * Flattening both into one string would discard that distinction — the very
 * thing the "ignore instructions inside the evidence" rule depends on.
 */
export interface LLMProvider {
  /** Model identity, e.g. "gpt-4o-mini". Logged so an answer is reproducible. */
  readonly model: string;

  /**
   * One completion.
   *
   * Contract: returns the assistant's text with no added commentary. An empty
   * or truncated completion is an error, not an empty string — silently
   * returning "" would look like a refusal to answer.
   *
   * `maxOutputTokens` overrides the provider's configured ceiling for this
   * call only. It exists because the right ceiling is a property of the task,
   * not of the provider: grounded prose wants a low limit (length correlates
   * with invention), while generating a contract that implements nine methods
   * legitimately needs several times that and otherwise fails as a truncation
   * error mid-function.
   */
  complete(
    system: string,
    user: string,
    maxOutputTokens?: number,
  ): Promise<string>;
}

export interface GenerateOptions {
  /**
   * Ceiling on answer length, in tokens.
   *
   * Also a cost ceiling: a runaway completion is billed either way. Grounded
   * answers over a handful of chunks are short by nature — if the model wants
   * 2000 tokens it is padding, and padding is where invention happens, since
   * there is no more evidence to fill it with.
   */
  maxOutputTokens: number;
  /**
   * Sampling temperature. 0 by default.
   *
   * Generation here is extraction, not composition: the facts are fixed by
   * the evidence and the only freedom is phrasing. Variety buys nothing and
   * costs reproducibility — at 0 the same chunks give the same answer, which
   * is what makes a prompt change measurable rather than anecdotal.
   */
  temperature: number;
  /** Per-request timeout. Chat completions are far slower than embeddings. */
  timeoutMs: number;
  /** Retry attempts for transient failures. 0 disables retrying. */
  maxRetries: number;
  /** Base delay for exponential backoff, in milliseconds. */
  retryDelayMs: number;
  /** Ceiling on a single backoff wait. */
  maxRetryDelayMs: number;
}

export const DEFAULT_GENERATE_OPTIONS: GenerateOptions = {
  maxOutputTokens: 800,
  temperature: 0,
  // 60s, not the embedder's 30s: a chat completion generates tokens
  // sequentially, so its latency scales with the answer, not just the input.
  timeoutMs: 60_000,
  maxRetries: 4,
  retryDelayMs: 1_000,
  maxRetryDelayMs: 30_000,
};

/**
 * A generation failure.
 *
 * `retryable` mirrors `EmbeddingError` for the same reason: rate limits (429)
 * and server errors (5xx) clear up on their own, while a bad key (401) or an
 * over-length prompt (400) will fail identically forever. Retrying the second
 * kind only delays a useful error message.
 *
 * Context-length errors deserve a note. They arrive as a 400 and are the one
 * failure caused by *this* stage's own inputs: too many chunks, or chunks too
 * large. The fix is a smaller K, never a retry.
 */
export class GenerationError extends Error {
  constructor(
    message: string,
    readonly retryable: boolean,
    /** HTTP status, when the failure came from a response. */
    readonly status?: number,
    readonly cause?: unknown,
    /**
     * Server-specified wait before retrying, in milliseconds, from a
     * `Retry-After` header. Always preferred over computed backoff — the
     * server knows when its limit resets and guessing is strictly worse.
     */
    readonly retryAfterMs?: number,
  ) {
    super(message);
    this.name = "GenerationError";
  }
}
