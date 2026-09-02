/**
 * Types for the embedding stage.
 *
 * An embedding is a fixed-length vector of floats that encodes the *meaning*
 * of a piece of text. Two chunks about the same idea produce vectors that
 * point in nearly the same direction, even with no words in common. That is
 * what makes semantic retrieval possible later.
 *
 * The vector is a lossy, one-way projection: you cannot recover the text from
 * it. The original `text` is always carried alongside.
 */
import type { Chunk } from "../chunker/types.js";

/**
 * Whether a text is being stored or searched with.
 *
 * Some models are *asymmetric*: they were trained to encode a question and the
 * passage that answers it differently, and expect to be told which is which.
 * Voyage takes this as `input_type`; other families use a `query: ` /
 * `passage: ` prefix. Getting it wrong measurably degrades retrieval.
 *
 * This is not a second model — it is the same model told what role the text
 * plays, so §4 still holds: queries and documents must share a provider.
 * Symmetric models (OpenAI) ignore it entirely.
 */
export type InputType = "document" | "query";

/**
 * A chunk plus its vector.
 *
 * `model` and `dimensions` are stored deliberately. Every embedding model
 * defines its own coordinate space, so vectors from two different models are
 * not comparable — comparing them yields a plausible-looking number that means
 * nothing. Recording the model's identity next to the vector turns a model
 * swap into a detectable mismatch instead of silently wrong search results.
 */
export interface EmbeddedChunk extends Chunk {
  /** The vector. Length always equals `dimensions`, regardless of text length. */
  embedding: number[];
  /** Model identity, e.g. "text-embedding-3-small". */
  model: string;
  /** Vector length. A property of the model, not of the text. */
  dimensions: number;
}

/**
 * The provider boundary.
 *
 * Deliberately narrow: text in, vectors out. A provider knows nothing about
 * `Chunk`, documents, or batching policy. Two payoffs:
 *
 *  1. Swapping providers touches one file.
 *  2. Embedding a *query* at search time reuses this exact method, which is
 *     how you guarantee queries and documents share a coordinate space.
 */
export interface EmbeddingProvider {
  /** Model identity, copied onto every `EmbeddedChunk`. */
  readonly model: string;
  /** Vector length this provider produces. */
  readonly dimensions: number;
  /** Max inputs accepted in a single call. The orchestrator respects this. */
  readonly maxBatchSize: number;

  /**
   * Embed one batch.
   *
   * Contract: returns exactly `texts.length` vectors, in the same order as
   * `texts`. Implementations are responsible for restoring order if their
   * transport does not guarantee it.
   *
   * `inputType` is optional because symmetric providers have no use for it;
   * they ignore it. Asymmetric providers should default to "document", the
   * far more common case — a corpus is embedded once, queries arrive one at
   * a time through `embedQuery`.
   */
  embedBatch(texts: string[], inputType?: InputType): Promise<number[][]>;
}

export interface EmbedOptions {
  /**
   * Texts per request. Clamped to the provider's `maxBatchSize`.
   *
   * The tradeoff: larger batches amortise per-request overhead (~50-200ms of
   * TCP/TLS/auth that dwarfs the compute for a short chunk) and spend your
   * requests-per-minute budget more slowly. But a failed batch loses all of
   * its work, so larger batches also make retries more expensive.
   */
  batchSize: number;
  /** Retry attempts per batch for transient failures. 0 disables retrying. */
  maxRetries: number;
  /** Base delay for exponential backoff, in milliseconds. */
  retryDelayMs: number;
  /** Ceiling on a single backoff wait. Caps exponential growth. */
  maxRetryDelayMs: number;
  /**
   * Minimum spacing between requests, in milliseconds.
   *
   * Proactive pacing for a rate-limited account. Backoff is reactive — it only
   * slows down *after* a 429, and on a tight limit that means most requests
   * fail once before succeeding. Spacing them from the start is both faster
   * overall and gentler on the provider. 0 disables it.
   */
  requestIntervalMs: number;
  /** Called after each batch. Useful for progress output on a long run. */
  onProgress?: (done: number, total: number) => void;
}

export const DEFAULT_EMBED_OPTIONS: EmbedOptions = {
  // 50, not 100: a 100-chunk response is several MB and can exceed the
  // provider's request timeout while the body is still streaming.
  batchSize: 50,
  maxRetries: 5,
  retryDelayMs: 500,
  maxRetryDelayMs: 60_000,
  requestIntervalMs: 0,
};

/**
 * An embedding failure.
 *
 * `retryable` is the field that matters: rate limits (429) and server errors
 * (5xx) are worth retrying, a bad API key (401) or malformed input (400) is
 * not. Retrying a 401 just burns time and produces a worse error message.
 */
export class EmbeddingError extends Error {
  constructor(
    message: string,
    readonly retryable: boolean,
    /** HTTP status, when the failure came from a response. */
    readonly status?: number,
    readonly cause?: unknown,
    /**
     * Server-specified wait before retrying, in milliseconds, from a
     * `Retry-After` header. Always prefer this over computed backoff — the
     * server knows when the limit resets and guessing is strictly worse.
     */
    readonly retryAfterMs?: number,
  ) {
    super(message);
    this.name = "EmbeddingError";
  }
}
