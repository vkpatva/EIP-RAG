/**
 * Batching and retry orchestration.
 *
 * Provider-agnostic on purpose: it talks only to `EmbeddingProvider`. Adding a
 * new provider means supplying limits and a transport, never re-implementing
 * the loop below.
 *
 * Batches run sequentially. Concurrency is a separate lever that would raise
 * throughput, but it makes rate-limit behaviour much harder to reason about —
 * worth adding only once measurement shows it is needed.
 */
import { DEFAULT_EMBED_OPTIONS, EmbeddingError } from "./types.js";
import type {
  EmbedOptions,
  EmbeddedChunk,
  EmbeddingProvider,
  InputType,
} from "./types.js";
import type { Chunk } from "../chunker/types.js";

/** Split into batches of at most `size`. */
function batch<T>(items: T[], size: number): T[][] {
  const batches: T[][] = [];
  for (let i = 0; i < items.length; i += size) {
    batches.push(items.slice(i, i + size));
  }
  return batches;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Embed one batch, retrying transient failures with exponential backoff.
 *
 * Only errors flagged `retryable` are retried — a 401 or a malformed request
 * fails the same way every time, so retrying it wastes the backoff window and
 * buries the real error behind a delay.
 */
async function embedWithRetry(
  provider: EmbeddingProvider,
  texts: string[],
  options: EmbedOptions,
  inputType: InputType,
): Promise<number[][]> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= options.maxRetries; attempt++) {
    try {
      return await provider.embedBatch(texts, inputType);
    } catch (error) {
      lastError = error;

      const retryable = error instanceof EmbeddingError && error.retryable;
      if (!retryable || attempt === options.maxRetries) break;

      // A `Retry-After` header is authoritative: the server knows when its
      // limit resets, so honor it verbatim rather than guessing. Otherwise
      // fall back to exponential backoff with jitter — the jitter matters
      // when several batches are limited together, since without it they
      // all retry at the same instant and trip the limit again.
      const serverDelay = (error as EmbeddingError).retryAfterMs;
      const delay =
        serverDelay ??
        Math.min(
          options.retryDelayMs * 2 ** attempt,
          options.maxRetryDelayMs,
        );
      await sleep(serverDelay !== undefined ? delay : delay + Math.random() * delay * 0.25);
    }
  }

  throw lastError;
}

/**
 * Embed a collection of chunks.
 *
 * Returns the original chunk fields plus `embedding`, `model`, and
 * `dimensions`, in the same order as the input. Rejects on the first batch
 * that exhausts its retries: a partially embedded corpus would produce a
 * quietly incomplete index, which is worse than a loud failure.
 */
export async function embedChunks(
  chunks: Chunk[],
  provider: EmbeddingProvider,
  options: Partial<EmbedOptions> = {},
): Promise<EmbeddedChunk[]> {
  const opts: EmbedOptions = { ...DEFAULT_EMBED_OPTIONS, ...options };

  if (opts.batchSize < 1) throw new Error("batchSize must be at least 1.");
  // A provider's limit is a hard cap; the option is only a request.
  const size = Math.min(opts.batchSize, provider.maxBatchSize);

  const embedded: EmbeddedChunk[] = [];
  let lastRequestAt = 0;

  for (const group of batch(chunks, size)) {
    // Proactive pacing for rate-limited accounts. Waits only for the time
    // still owed since the previous request, so it costs nothing when the
    // work between batches already exceeded the interval.
    if (opts.requestIntervalMs > 0 && lastRequestAt > 0) {
      const owed = opts.requestIntervalMs - (Date.now() - lastRequestAt);
      if (owed > 0) await sleep(owed);
    }
    lastRequestAt = Date.now();

    const vectors = await embedWithRetry(
      provider,
      group.map((c) => c.text),
      opts,
      // Chunks are corpus content, always. Queries go through `embedQuery`.
      "document",
    );

    if (vectors.length !== group.length) {
      throw new EmbeddingError(
        `Provider "${provider.model}" returned ${vectors.length} vectors for ` +
          `${group.length} inputs, violating the EmbeddingProvider contract.`,
        false,
      );
    }

    for (let i = 0; i < group.length; i++) {
      embedded.push({
        ...group[i]!,
        embedding: vectors[i]!,
        model: provider.model,
        dimensions: provider.dimensions,
      });
    }

    opts.onProgress?.(embedded.length, chunks.length);
  }

  return embedded;
}

/**
 * Embed a single query.
 *
 * Routes through the same provider instance as `embedChunks`, so a query and
 * the documents it searches are guaranteed to share a coordinate space.
 * Comparing vectors from two different models returns a perfectly plausible
 * number that means nothing at all.
 *
 * The "query" input type is the one thing this does differently. On an
 * asymmetric provider it selects the query-side encoding; on a symmetric one
 * it is ignored. Passing "document" here instead would not fail — it would
 * just quietly retrieve worse.
 */
export async function embedQuery(
  text: string,
  provider: EmbeddingProvider,
  options: Partial<EmbedOptions> = {},
): Promise<number[]> {
  // Routed through the same retry path as bulk embedding: a query is one
  // request against the same rate limit, and a transient 429 at search time
  // should back off rather than fail the whole search.
  const opts: EmbedOptions = { ...DEFAULT_EMBED_OPTIONS, ...options };
  const [vector] = await embedWithRetry(provider, [text], opts, "query");
  if (!vector) {
    throw new EmbeddingError("Provider returned no vector for query.", false);
  }
  return vector;
}

/**
 * Cosine similarity: the cosine of the angle between two vectors, in [-1, 1].
 *
 * Included here to make the retrieval intuition concrete — the vector store
 * will do this for you later. It compares direction only, ignoring magnitude,
 * so a long chunk and a short one on the same topic can still match closely.
 *
 * Read scores as a ranking, not a percentage. Real embeddings occupy a narrow
 * cone, so unrelated text still scores well above 0 and any absolute threshold
 * has to be measured per model rather than assumed.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(
      `Cannot compare ${a.length}-dim and ${b.length}-dim vectors — ` +
        `they are almost certainly from different models.`,
    );
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}
