/**
 * Voyage AI embedding provider.
 *
 * The second implementation of `EmbeddingProvider`, and the one that shows why
 * the interface is worth having: Voyage is *asymmetric*. It wants to be told
 * whether a text is a stored document or a search query, because it encodes
 * the two differently. OpenAI is symmetric and ignores the distinction.
 *
 * Everything above this file is unchanged by that difference.
 */
import { EmbeddingError } from "./types.js";
import type { EmbeddingProvider, InputType } from "./types.js";

const ENDPOINT = "https://api.voyageai.com/v1/embeddings";

/** Native vector length per model, used when `outputDimension` is not set. */
const MODEL_DIMENSIONS: Record<string, number> = {
  "voyage-3": 1024,
  "voyage-3-lite": 512,
  "voyage-3-large": 1024,
  "voyage-code-3": 1024,
};

export interface VoyageProviderOptions {
  apiKey?: string;
  model?: string;
  /**
   * Truncate vectors to this length, where the model supports it.
   *
   * Same tradeoff as OpenAI's `dimensions`: less storage and faster search for
   * a little accuracy. The chosen length is part of the model's identity —
   * vectors of different lengths are not comparable.
   */
  outputDimension?: number;
  /** Per-request timeout. Guards against a hung connection stalling the run. */
  timeoutMs?: number;
}

/** The subset of the response we rely on. */
interface EmbeddingsResponse {
  data: Array<{ index: number; embedding: number[] }>;
}

export class VoyageEmbeddingProvider implements EmbeddingProvider {
  readonly model: string;
  readonly dimensions: number;
  /**
   * Voyage accepts up to 1000 inputs per request.
   *
   * Note this is not the only limit — there is also a cap on total tokens per
   * request (120k for voyage-3), which a batch of long chunks can hit while
   * well under 1000 items. That surfaces as a retryable 400 rather than
   * something we can predict from item count alone; see `#toError`.
   */
  readonly maxBatchSize = 1000;

  readonly #apiKey: string;
  readonly #timeoutMs: number;
  readonly #outputDimension?: number;

  constructor(options: VoyageProviderOptions = {}) {
    const apiKey = options.apiKey ?? process.env.VOYAGE_API_KEY;
    if (!apiKey) {
      throw new Error(
        "Voyage API key missing. Set VOYAGE_API_KEY or pass { apiKey }.",
      );
    }
    this.#apiKey = apiKey;
    this.model = options.model ?? "voyage-3";
    this.#outputDimension = options.outputDimension;
    this.#timeoutMs = options.timeoutMs ?? 30_000;

    const native = MODEL_DIMENSIONS[this.model];
    if (options.outputDimension === undefined && native === undefined) {
      throw new Error(
        `Unknown native dimensions for "${this.model}". Pass { outputDimension } explicitly.`,
      );
    }
    this.dimensions = options.outputDimension ?? native!;
  }

  /**
   * `inputType` defaults to "document": bulk embedding is the common path,
   * and `embedQuery` passes "query" explicitly at search time.
   */
  async embedBatch(
    texts: string[],
    inputType: InputType = "document",
  ): Promise<number[][]> {
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
      input_type: inputType,
      ...(this.#outputDimension !== undefined && {
        output_dimension: this.#outputDimension,
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
        : `Network error calling Voyage: ${(cause as Error).message}`,
      true,
      undefined,
      cause,
    );
  }

  /**
   * Classify an error response.
   *
   * Same retryable/not split as the OpenAI provider, with one addition: a 400
   * naming the token limit is a batch that was too large in *tokens* rather
   * than in items. That is not retryable as-is — the same payload fails
   * identically — so it is reported as permanent with an actionable message.
   */
  async #toError(response: Response): Promise<EmbeddingError> {
    const detail = await response.text().catch(() => "");
    const retryable = response.status === 429 || response.status >= 500;

    if (response.status === 400 && /token/i.test(detail)) {
      return new EmbeddingError(
        `Voyage rejected the batch on token count — lower batchSize. ` +
          `${detail.slice(0, 300)}`,
        false,
        400,
      );
    }

    return new EmbeddingError(
      `Voyage embeddings failed: ${response.status} ${response.statusText}` +
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
   * As with OpenAI: sort by the response's `index` rather than trusting array
   * position. A silent misalignment would attach every vector to the wrong
   * chunk, and nothing downstream could detect it.
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
