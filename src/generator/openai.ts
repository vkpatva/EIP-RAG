/**
 * OpenAI chat provider.
 *
 * The only file in the pipeline that knows OpenAI's chat API exists.
 * Everything above it talks to `LLMProvider`. Written against the raw HTTP
 * endpoint rather than the SDK, matching `embedder/openai.ts`: one dependency
 * fewer, and the request/response shape stays visible while learning.
 *
 * The whole file is HTTP plumbing — auth, timeouts, error classification,
 * response parsing. None of it mentions chunks, evidence, or grounding. That
 * is the point of the provider boundary: prompt design and transport concerns
 * grow independently, and testing generation needs neither an API key nor a
 * mocked `fetch`, only a three-line fake `LLMProvider`.
 */
import { GenerationError } from "./types.js";
import type { GenerateOptions, LLMProvider } from "./types.js";

const ENDPOINT = "https://api.openai.com/v1/chat/completions";

export interface OpenAIChatProviderOptions {
  apiKey?: string;
  model?: string;
  /** Sampling temperature. 0 for grounded extraction; see `GenerateOptions`. */
  temperature?: number;
  /** Ceiling on completion length, in tokens. Also a cost ceiling. */
  maxOutputTokens?: number;
  /** Per-request timeout. Chat latency scales with the answer length. */
  timeoutMs?: number;
}

/** The subset of the response we rely on. */
interface ChatCompletionResponse {
  choices?: Array<{
    message?: { content?: string | null };
    finish_reason?: string;
  }>;
}

export class OpenAIChatProvider implements LLMProvider {
  readonly model: string;

  readonly #apiKey: string;
  readonly #temperature: number;
  readonly #maxOutputTokens: number;
  readonly #timeoutMs: number;

  constructor(
    options: OpenAIChatProviderOptions = {},
    defaults: Pick<
      GenerateOptions,
      "temperature" | "maxOutputTokens" | "timeoutMs"
    > = { temperature: 0, maxOutputTokens: 800, timeoutMs: 60_000 },
  ) {
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "OpenAI API key missing. Set OPENAI_API_KEY or pass { apiKey }.",
      );
    }
    this.#apiKey = apiKey;
    // Env override mirrors OPENAI_EMBEDDING_MODEL, so both stages are
    // configured the same way.
    this.model =
      options.model ?? process.env.OPENAI_CHAT_MODEL ?? "gpt-4o-mini";
    this.#temperature = options.temperature ?? defaults.temperature;
    this.#maxOutputTokens =
      options.maxOutputTokens ?? defaults.maxOutputTokens;
    this.#timeoutMs = options.timeoutMs ?? defaults.timeoutMs;
  }

  async complete(
    system: string,
    user: string,
    maxOutputTokens?: number,
  ): Promise<string> {
    const tokenLimit = maxOutputTokens ?? this.#maxOutputTokens;
    const response = await this.#post({
      model: this.model,
      // Two roles, not one concatenated string. Providers train models to
      // weight system content more heavily and treat it as harder to
      // override, which makes this the mechanism — not the prompt text —
      // that keeps "rules I wrote" apart from "text out of a file".
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature: this.#temperature,
      max_completion_tokens: tokenLimit,
    });

    if (!response.ok) {
      throw await this.#toError(response);
    }

    const body = await this.#readBody(response);
    return this.#toText(body, tokenLimit);
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
   * Separate from `#post` for the reason `embedder/openai.ts` documents:
   * `fetch` resolves as soon as headers arrive while the body is still
   * streaming, and the abort signal stays armed for that whole window. A slow
   * completion can therefore time out *here*. Parsing outside a try/catch
   * would let that DOMException escape as an uncaught crash instead of being
   * retried as the transient failure it is.
   */
  async #readBody(response: Response): Promise<ChatCompletionResponse> {
    try {
      return (await response.json()) as ChatCompletionResponse;
    } catch (cause) {
      throw this.#toTransportError(cause);
    }
  }

  /**
   * Classify a thrown transport failure.
   *
   * DNS failures, resets, aborted body reads, and timeouts are all transient,
   * so retryable. `AbortSignal.timeout` throws a DOMException rather than a
   * plain Error, hence the `name` check rather than an instanceof.
   */
  #toTransportError(cause: unknown): GenerationError {
    const timedOut = cause instanceof Error && cause.name === "TimeoutError";
    return new GenerationError(
      timedOut
        ? `Request timed out after ${this.#timeoutMs}ms — ` +
          `lower maxOutputTokens or raise timeoutMs.`
        : `Network error calling OpenAI: ${(cause as Error).message}`,
      true,
      undefined,
      cause,
    );
  }

  /**
   * Classify an error response.
   *
   * Same split as the embedder: 429 and 5xx clear up on their own, 401 and
   * 400 will fail identically forever. One case is worth naming in the
   * message, because it is the only failure this stage causes itself: a 400
   * for context length means too many or too large chunks were passed in.
   * The fix is a smaller K, and no number of retries will find it.
   */
  async #toError(response: Response): Promise<GenerationError> {
    const detail = await response.text().catch(() => "");
    const retryable = response.status === 429 || response.status >= 500;
    const overLength =
      response.status === 400 && /context length|context_length|too long/i.test(detail);

    return new GenerationError(
      `OpenAI chat completion failed: ${response.status} ${response.statusText}` +
        (detail ? ` — ${detail.slice(0, 300)}` : "") +
        (overLength
          ? "\nThe prompt exceeded the model's context window. Retrieve fewer " +
            "chunks (lower --k) or shorten them; retrying will not help."
          : ""),
      retryable,
      response.status,
      undefined,
      this.#retryAfterMs(response),
    );
  }

  /**
   * Parse a `Retry-After` header into milliseconds.
   *
   * Either a delay in seconds or an HTTP date; both are legal and providers
   * use both. Undefined when absent or malformed, leaving the caller on
   * computed backoff.
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
   * Extract the answer text and validate it.
   *
   * Three failures are checked rather than papered over:
   *
   *  - a missing choice or empty content, which returned as "" would be
   *    indistinguishable from the model declining to answer;
   *  - `finish_reason: "length"`, meaning the answer was cut mid-sentence.
   *    Silently returning a truncated answer is the worst option: it reads
   *    complete and is not;
   *  - `finish_reason: "content_filter"`, which is a refusal, not an answer.
   *
   * None are retryable — each needs a changed request, not a repeated one.
   */
  #toText(body: ChatCompletionResponse, tokenLimit: number): string {
    const choice = body.choices?.[0];
    if (!choice) {
      throw new GenerationError(
        "Malformed response: no choices returned.",
        false,
      );
    }
    if (choice.finish_reason === "content_filter") {
      throw new GenerationError(
        "Completion stopped by OpenAI's content filter.",
        false,
      );
    }

    const text = choice.message?.content?.trim();
    if (!text) {
      throw new GenerationError(
        `Model returned an empty completion (finish_reason: ` +
          `${choice.finish_reason ?? "unknown"}).`,
        false,
      );
    }

    if (choice.finish_reason === "length") {
      throw new GenerationError(
        `Answer was truncated at the ${tokenLimit}-token limit. ` +
          `Raise maxOutputTokens — returning a half-sentence as if it were ` +
          `a complete answer would be worse than failing.`,
        false,
      );
    }

    return text;
  }
}
