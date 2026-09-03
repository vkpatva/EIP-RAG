/**
 * Hybrid retrieval: dense search and BM25, fused by rank.
 *
 * Implements the same `Retriever` interface as `QdrantRetriever`, so every
 * script that takes a retriever works unchanged and the two can be compared
 * on the same eval by swapping one constructor.
 *
 * The `score` on returned chunks stays the *dense* cosine score, not the
 * fused one. Fusion reorders; it does not produce a calibrated relevance
 * number, and RRF scores (order 1/60) would be actively misleading in output
 * that a human reads as "how good is this match". A chunk found only by BM25
 * has no dense score, so it reports 0 and is visibly a lexical-only hit.
 */
import type { EmbeddingProvider } from "../embedder/types.js";
import type { QdrantRetriever } from "../vectorstore/retriever.js";
import type { Retriever } from "../vectorstore/retriever.js";
import type { RetrievedChunk } from "../vectorstore/types.js";
import { BM25Index } from "./bm25.js";
import { reciprocalRankFusion } from "./fuse.js";
import type { FusionOptions } from "./fuse.js";

export interface HybridRetrieverOptions {
  /** The dense half. Already validated for model/dimension agreement. */
  dense: QdrantRetriever;
  /** The lexical half, built over the same chunks that were indexed. */
  bm25: BM25Index;
  /** Maps a BM25 document id back to a full chunk. */
  chunkById: Map<string, RetrievedChunk>;
  /**
   * How many candidates to pull from each retriever before fusing.
   *
   * Must exceed the final K: fusion can only promote a chunk that at least
   * one list returned, so a candidate pool equal to K would leave nothing to
   * reorder. The default multiplier of 4 is the usual starting point — deep
   * enough that a chunk ranked 12th by dense search can win on lexical
   * agreement, shallow enough to stay one Qdrant call.
   */
  candidateMultiplier?: number;
  fusion?: Partial<FusionOptions>;
  provider?: EmbeddingProvider;
}

export class HybridRetriever implements Retriever {
  readonly #dense: QdrantRetriever;
  readonly #bm25: BM25Index;
  readonly #chunkById: Map<string, RetrievedChunk>;
  readonly #candidateMultiplier: number;
  readonly #fusion: Partial<FusionOptions>;

  constructor(options: HybridRetrieverOptions) {
    this.#dense = options.dense;
    this.#bm25 = options.bm25;
    this.#chunkById = options.chunkById;
    this.#candidateMultiplier = options.candidateMultiplier ?? 4;
    this.#fusion = options.fusion ?? {};
  }

  async retrieve(query: string, topK: number): Promise<RetrievedChunk[]> {
    const depth = Math.max(topK * this.#candidateMultiplier, topK);

    // Dense is the only async half, and BM25 is pure CPU over in-memory maps,
    // so there is nothing to parallelize: awaiting first costs nothing.
    const denseHits = await this.#dense.retrieve(query, depth);
    const lexicalHits = this.#bm25.search(query, depth);

    const fused = reciprocalRankFusion(
      [
        { name: "dense", ids: denseHits.map((h) => h.chunkId) },
        { name: "bm25", ids: lexicalHits.map((h) => h.id) },
      ],
      this.#fusion,
    );

    // Dense hits carry their payload already; a BM25-only hit has to be
    // looked up in the chunk map. Preferring the live hit keeps the real
    // cosine score rather than the map's placeholder.
    const denseByid = new Map(denseHits.map((h) => [h.chunkId, h]));

    const results: RetrievedChunk[] = [];
    for (const hit of fused) {
      const dense = denseByid.get(hit.id);
      const chunk = dense ?? this.#chunkById.get(hit.id);
      if (!chunk) continue; // Indexed in BM25 but absent from Qdrant: skip.
      results.push({
        ...chunk,
        // Only a dense hit has a comparable score. A lexical-only hit leaves
        // it undefined rather than 0: the chunk was never scored in the
        // embedding space, and a 0 would read as "maximally dissimilar".
        score: dense?.score,
        rank: results.length + 1,
        retrievedBy: hit.ranks,
      });
      if (results.length === topK) break;
    }
    return results;
  }
}
