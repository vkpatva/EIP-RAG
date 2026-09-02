/**
 * OpenAI embedding provider.
 *
 * The only file in the pipeline that knows OpenAI exists. Everything above it
 * talks to `EmbeddingProvider`. Written against the raw HTTP endpoint rather
 * than the SDK: one dependency fewer, and the request/response shape stays
 * visible while learning.
 */
import { EmbeddingError } from "./types.js";
import type { EmbeddingProvider } from "./types.js";

const ENDPOINT = "https://api.openai.com/v1/embeddings";

/** Native vector length per model, used when `dimensions` is not overridden. */
const MODEL_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
};

export interface OpenAIProviderOptions {
  apiKey?: string;
  model?: string;
  /**
   * Truncate vectors to this length.
   *
   * The `text-embedding-3-*` models are trained so a prefix of the vector is
   * still a usable embedding ("Matryoshka"), trading a little accuracy for
   * less storage and faster search. Leave unset to get the model's native
   * length. Whatever you pick becomes part of the model's identity: vectors
   * of different lengths cannot be compared.
   */
  dimensions?: number;
  /** Per-request timeout. Guards against a hung connection stalling the run. */
  timeoutMs?: number;
}

/** The subset of the response we rely on. */
interface EmbeddingsResponse {
  data: Array<{ index: number; embedding: number[] }>;
}

export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly model: string;
  readonly dimensions: number;
  /** OpenAI accepts up to 2048 inputs per request. */
  readonly maxBatchSize = 2048;

  readonly #apiKey: string;
  readonly #timeoutMs: number;
  /** Only sent when explicitly requested, so we get the native length by default. */
  readonly #requestedDimensions?: number;

  constructor(options: OpenAIProviderOptions = {}) {
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "OpenAI API key missing. Set OPENAI_API_KEY or pass { apiKey }.",
      );
    }
    this.#apiKey = apiKey;
    this.model = options.model ?? "text-embedding-3-small";
    this.#requestedDimensions = options.dimensions;
    this.#timeoutMs = options.timeoutMs ?? 30_000;

    const native = MODEL_DIMENSIONS[this.model];
    if (options.dimensions === undefined && native === undefined) {
      throw new Error(
        `Unknown native dimensions for "${this.model}". Pass { dimensions } explicitly.`,
      );
    }
    this.dimensions = options.dimensions ?? native!;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    if (texts.length > this.maxBatchSize) {
      throw new EmbeddingError(
        `Batch of ${texts.length} exceeds maxBatchSize ${this.maxBatchSize}.`,
        false,
      );
    }

    const response = await this.#post({
      model: this.model,
      input: texts,
      ...(this.#requestedDimensions !== undefined && {
        dimensions: this.#requestedDimensions,
      }),
    });

    if (!response.ok) {
      throw await this.#toError(response);
    }

    const body = await this.#readBody(response);
    return this.#toVectors(body, texts.length);
  }

  /** Issue the request, converting network/timeout failures into retryable errors. */
  async #post(payload: unknown): Promise<Response> {
    try {
      return await fetch(ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.#apiKey}`,
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(this.#timeoutMs),
      });
    } catch (cause) {
      throw this.#toTransportError(cause);
    }
  }

  /**
   * Read and parse the response body.
   *
   * Separate from `#post` because `fetch` resolves as soon as headers arrive,
   * while the body is still streaming — and the abort signal stays armed for
   * that whole window. A large batch (1536 floats x 100 chunks is several MB)
   * can therefore time out *here*, not at the request. Parsing outside a
   * try/catch let that DOMException escape as an uncaught crash instead of
   * being retried as the transient failure it is.
   */
  async #readBody(response: Response): Promise<EmbeddingsResponse> {
    try {
      return (await response.json()) as EmbeddingsResponse;
    } catch (cause) {
      throw this.#toTransportError(cause);
    }
  }

  /**
   * Classify a thrown transport failure.
   *
   * DNS failures, connection resets, aborted body reads, and timeouts are all
   * transient, so they are retryable. `AbortSignal.timeout` throws a
   * DOMException rather than a plain Error, which is why this checks `name`
   * rather than the constructor.
   */
  #toTransportError(cause: unknown): EmbeddingError {
    const timedOut = cause instanceof Error && cause.name === "TimeoutError";
    return new EmbeddingError(
      timedOut
        ? `Request timed out after ${this.#timeoutMs}ms — ` +
          `lower batchSize or raise timeoutMs.`
        : `Network error calling OpenAI: ${(cause as Error).message}`,
      true,
      undefined,
      cause,
    );
  }

  /**
   * Classify an error response.
   *
   * The split that matters is retryable vs. not. 429 and 5xx are transient and
   * clear up on their own; 401 and 400 will fail identically forever, so
   * retrying them only delays a useful error message.
   */
  async #toError(response: Response): Promise<EmbeddingError> {
    const detail = await response.text().catch(() => "");
    const retryable = response.status === 429 || response.status >= 500;
    return new EmbeddingError(
      `OpenAI embeddings failed: ${response.status} ${response.statusText}` +
        (detail ? ` — ${detail.slice(0, 300)}` : ""),
      retryable,
      response.status,
      undefined,
      this.#retryAfterMs(response),
    );
  }

  /**
   * Parse a `Retry-After` header into milliseconds.
   *
   * The header is either a delay in seconds or an HTTP date; both forms are
   * legal and providers use both. Returns undefined when absent or malformed,
   * leaving the caller to fall back on computed backoff.
   */
  #retryAfterMs(response: Response): number | undefined {
    const raw = response.headers.get("retry-after");
    if (!raw) return undefined;

    const seconds = Number(raw);
    if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);

    const date = Date.parse(raw);
    return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
  }


  /**
   * Restore input order and validate the shape.
   *
   * The response carries an explicit `index` per item and the API does not
   * promise array order. Sorting by `index` rather than trusting position is
   * cheap insurance: a silent misalignment here would attach every vector to
   * the wrong chunk, and nothing downstream could detect it.
   */
  #toVectors(body: EmbeddingsResponse, expected: number): number[][] {
    if (!Array.isArray(body.data) || body.data.length !== expected) {
      throw new EmbeddingError(
        `Expected ${expected} embeddings, received ${body.data?.length ?? 0}.`,
        false,
      );
    }

    const vectors = new Array<number[]>(expected);
    for (const item of body.data) {
      if (item.index < 0 || item.index >= expected || vectors[item.index]) {
        throw new EmbeddingError(
          `Malformed response: bad or duplicate index ${item.index}.`,
          false,
        );
      }
      if (item.embedding.length !== this.dimensions) {
        throw new EmbeddingError(
          `Expected ${this.dimensions}-dim vector, received ${item.embedding.length}.`,
          false,
        );
      }
      vectors[item.index] = item.embedding;
    }
    return vectors;
  }
}
