/**
 * The retriever: the one place where the embedding provider and the vector
 * store meet.
 *
 * Everything else in this stage is deliberately kept apart — `qdrant.ts` never
 * imports a provider, `embedder/` never imports Qdrant. They must agree on
 * exactly one thing, and it is the thing that fails silently when it is wrong:
 * the query must be embedded by the same model that produced the stored
 * vectors. Concentrating that into a single file means there is one place to
 * get right rather than one per call site.
 *
 * The flow, end to end:
 *
 *   question → embedQuery() → query vector → Qdrant search → Top-K → chunks
 *
 * No LLM is involved. Retrieval ends with text; deciding what to *say* about
 * that text is a separate stage.
 */
import { embedQuery } from "../embedder/embedChunks.js";
import type { EmbeddingProvider } from "../embedder/types.js";
import type { QdrantVectorRepository } from "./qdrant.js";
import type { ChunkPayload, RetrievedChunk } from "./types.js";

export interface Retriever {
  retrieve(query: string, topK: number): Promise<RetrievedChunk[]>;
}

export interface QdrantRetrieverOptions {
  repository: QdrantVectorRepository;
  provider: EmbeddingProvider;
  /**
   * Drop hits scoring below this. Off by default (0).
   *
   * Worth understanding rather than switching on: Top-K is a *ranking*, not a
   * relevance test. Ask "how do I build a React app?" and Qdrant will still
   * return five chunks — the five least-unrelated ones in the corpus. The
   * only signal that nothing relevant exists is that the scores are low. A
   * threshold turns that signal into a filter, but the right cutoff is a
   * property of the model and the corpus, so it has to be measured, not
   * guessed. Leaving it at 0 keeps the raw scores visible while you look.
   */
  minScore?: number;
}

export class QdrantRetriever implements Retriever {
  private readonly repository: QdrantVectorRepository;
  private readonly provider: EmbeddingProvider;
  private readonly minScore: number;

  constructor(options: QdrantRetrieverOptions) {
    const { repository, provider } = options;

    // The guard that matters. Two different models produce vectors in
    // unrelated coordinate spaces; comparing across them yields scores that
    // look entirely ordinary and mean nothing at all. Nothing downstream
    // would ever notice, so refuse to construct the retriever at all.
    if (repository.model !== provider.model) {
      throw new Error(
        `Collection "${repository.collection}" holds ${repository.model} ` +
          `vectors, but the query provider is ${provider.model}. Vectors ` +
          `from different models are not comparable.`,
      );
    }
    if (repository.dimensions !== provider.dimensions) {
      throw new Error(
        `Collection expects ${repository.dimensions}-dimensional vectors; ` +
          `provider produces ${provider.dimensions}.`,
      );
    }

    this.repository = repository;
    this.provider = provider;
    this.minScore = options.minScore ?? 0;
  }

  async retrieve(query: string, topK: number): Promise<RetrievedChunk[]> {
    // Same provider as the corpus, flagged as a query so asymmetric models
    // (Voyage) use their query-side encoding rather than the document one.
    const vector = await embedQuery(query, this.provider, {
      maxRetries: 6,
      retryDelayMs: 2_000,
      maxRetryDelayMs: 90_000,
    });

    const result = await this.repository.raw.query(this.repository.collection, {
      query: vector,
      limit: topK,
      // The text and provenance are the point of the search — without the
      // payload a hit is only an id and a number.
      with_payload: true,
      // The stored vectors are not needed to answer; asking for them would
      // move 1536 floats per hit across the wire for nothing.
      with_vector: false,
      score_threshold: this.minScore > 0 ? this.minScore : undefined,
    });

    return result.points.map((point) => {
      const payload = point.payload as unknown as ChunkPayload;
      return {
        chunkId: payload.chunkId,
        documentId: payload.documentId,
        text: payload.text,
        score: point.score,
        metadata: {
          eipNumber: payload.eipNumber,
          title: payload.title,
          section: payload.section,
          sourcePath: payload.sourcePath,
        },
      };
    });
  }
}
