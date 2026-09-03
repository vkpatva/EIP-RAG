/**
 * The generation stage.
 *
 * Orchestration, and nothing else. Four steps:
 *
 *   format the chunks -> build the prompt -> call the provider -> return text
 *
 * What this file deliberately does not contain: any HTTP (that is
 * `openai.ts`), any prompt text (that is `prompt.ts`), and any notion of
 * where the chunks came from. It never imports the Qdrant client, the
 * embedder, or a collection name — only the `RetrievedChunk` *type*, which is
 * erased at compile time. So generation can be exercised against hand-written
 * chunks with nothing running, which is exactly what the retrieval-quality
 * experiment needs: no real retriever would return deliberately irrelevant
 * chunks for a question, so they have to be supplied by hand.
 */
import {
  buildUserPrompt,
  looksLikeSynthesis,
  SYNTHESIS_SYSTEM_PROMPT,
  SYSTEM_PROMPT,
} from "./prompt.js";
import { DEFAULT_GENERATE_OPTIONS, GenerationError } from "./types.js";
import type {
  GenerateOptions,
  GenerationService,
  LLMProvider,
} from "./types.js";
import type { RetrievedChunk } from "../vectorstore/types.js";

/**
 * A generation plus the prompt that produced it.
 *
 * Returned by `generateDetailed` for inspection. Keeping the exact strings
 * that were sent is what makes a bad answer diagnosable: you can read the
 * evidence the model actually saw rather than the evidence you assume it saw.
 * The two differ more often than you would expect — a chunk truncated
 * upstream, a metadata field missing, an ordering surprise.
 */
export interface GenerationResult {
  answer: string;
  systemPrompt: string;
  userPrompt: string;
  model: string;
  chunksUsed: number;
  /** Which rule set ran. Worth logging: it changes what the answer may claim. */
  mode: "extraction" | "synthesis";
}

/**
 * Which standing rules to apply.
 *
 * "extraction" answers questions about the spec and refuses to go beyond it.
 * "synthesis" builds artifacts that conform to it, pinning the interface to
 * the evidence while allowing implementation knowledge. "auto" picks by
 * heuristic per question — convenient for a CLI, and the wrong choice for an
 * application, which should decide from which feature the user invoked.
 */
export type GenerationMode = "extraction" | "synthesis" | "auto";

export interface RAGGenerationServiceOptions
  extends Partial<GenerateOptions> {
  provider: LLMProvider;
  /** Defaults to "extraction": the stricter mode, chosen when unsure. */
  mode?: GenerationMode;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class RAGGenerationService implements GenerationService {
  private readonly provider: LLMProvider;
  private readonly options: GenerateOptions;
  private readonly mode: GenerationMode;

  constructor(options: RAGGenerationServiceOptions) {
    const { provider, mode, ...overrides } = options;
    this.provider = provider;
    this.mode = mode ?? "extraction";
    this.options = { ...DEFAULT_GENERATE_OPTIONS, ...overrides };
  }

  /** Resolve "auto" against the question; the explicit modes pass through. */
  private resolveMode(question: string): "extraction" | "synthesis" {
    if (this.mode !== "auto") return this.mode;
    return looksLikeSynthesis(question) ? "synthesis" : "extraction";
  }

  /** The interface method: question + evidence in, answer out. */
  async generate(
    question: string,
    context: RetrievedChunk[],
  ): Promise<string> {
    return (await this.generateDetailed(question, context)).answer;
  }

  /**
   * Same generation, with the prompt returned alongside.
   *
   * Not part of `GenerationService` on purpose. Callers that only want an
   * answer should not have to know a prompt exists; this is the seam for dev
   * scripts and evaluation, where seeing the input is the whole point.
   */
  async generateDetailed(
    question: string,
    context: RetrievedChunk[],
  ): Promise<GenerationResult> {
    if (question.trim().length === 0) {
      throw new GenerationError("Question is empty.", false);
    }

    // An empty context is passed through rather than short-circuited with a
    // canned "I don't know". Two reasons. The prompt already instructs the
    // model to state that the evidence is insufficient, so a hardcoded string
    // here would be a second, silently divergent copy of that policy. And a
    // real answer proves the grounding rules are working on the case that
    // matters most — a system that only refuses when code forces it to has
    // not been shown to refuse at all.
    const userPrompt = buildUserPrompt(question, context);
    const mode = this.resolveMode(question);
    const systemPrompt =
      mode === "synthesis" ? SYNTHESIS_SYSTEM_PROMPT : SYSTEM_PROMPT;

    // Synthesis needs a higher ceiling than extraction. The 800-token
    // default is calibrated for grounded prose, where length correlates with
    // invention; a contract implementing nine methods is legitimately longer
    // than that and gets truncated mid-function. Only raised when the caller
    // left it at the default, so an explicit maxOutputTokens still wins.
    const maxOutputTokens =
      mode === "synthesis" &&
      this.options.maxOutputTokens === DEFAULT_GENERATE_OPTIONS.maxOutputTokens
        ? 2_400
        : this.options.maxOutputTokens;

    const answer = await this.#completeWithRetry(
      systemPrompt,
      userPrompt,
      maxOutputTokens,
    );

    return {
      answer,
      systemPrompt,
      userPrompt,
      model: this.provider.model,
      chunksUsed: context.length,
      mode,
    };
  }

  /**
   * Call the provider, retrying only transient failures.
   *
   * Same policy as the embedder, for the same reason: a 429 or 5xx clears up
   * on its own, while a 401 or an over-length prompt fails identically
   * forever, so retrying it only delays a useful error message. A server
   * `Retry-After` is honoured verbatim over computed backoff — the server
   * knows when its limit resets.
   */
  async #completeWithRetry(
    system: string,
    user: string,
    maxOutputTokens?: number,
  ): Promise<string> {
    const { maxRetries, retryDelayMs, maxRetryDelayMs } = this.options;
    let lastError: unknown;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.provider.complete(system, user, maxOutputTokens);
      } catch (error) {
        lastError = error;

        const retryable =
          error instanceof GenerationError ? error.retryable : false;
        if (!retryable || attempt === maxRetries) throw error;

        const backoff = Math.min(
          retryDelayMs * 2 ** attempt,
          maxRetryDelayMs,
        );
        const wait =
          error instanceof GenerationError && error.retryAfterMs !== undefined
            ? error.retryAfterMs
            : backoff;
        await sleep(wait);
      }
    }

    throw lastError;
  }
}
