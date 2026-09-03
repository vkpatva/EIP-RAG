/**
 * Reciprocal Rank Fusion: combine ranked lists that do not share a scale.
 *
 * The problem this solves is that the two retrievers' scores cannot be added.
 * Cosine similarity lands in a narrow band — this corpus produces roughly
 * 0.27..0.73, with unrelated text still scoring near 0.4 — while BM25 is
 * unbounded and depends on query length and term rarity, so one query's 8.0
 * and another's 2.0 say nothing about each other. Normalizing each list to
 * 0..1 is the obvious alternative and is worse: min-max over the top K makes
 * the top hit 1.0 by construction, whether it scored 0.73 or 0.31, which
 * throws away the one thing a low top score does tell you.
 *
 * RRF instead uses only position. A document's contribution from each list is
 * 1/(k + rank), so agreement between the two retrievers is what promotes a
 * chunk, and a first place is worth a fixed amount regardless of the score
 * that earned it. That is the right currency here: the useful signal is
 * "both methods think this is relevant", not "this scored 0.51".
 */

export interface FusionOptions {
  /**
   * RRF's rank-smoothing constant. Conventionally 60; 2 here, measured.
   *
   * It sets how sharply rank 1 beats rank 2, and 60 is wrong for this corpus.
   * That value comes from fusing many lists over web-scale collections, where
   * flattening the curve is what you want. Over 418 chunks and two lists it
   * makes positions 1 and 12 differ by only ~18% (1/61 vs 1/72), so merely
   * *appearing in both lists* outweighs *ranking first in either* — junk that
   * placed mid-list in both floated above the correct answer at dense rank 1.
   * At k=2 the top ranks dominate again and agreement becomes a tie-breaker
   * rather than the deciding term.
   */
  k: number;
  /**
   * Weight per list, keyed by list name. Defaults to 1 for any list not
   * named. A weight scales that list's whole contribution, so it expresses
   * "trust lexical matching less than semantic" without touching ranks.
   */
  weights?: Record<string, number>;
}

export const DEFAULT_FUSION_OPTIONS: FusionOptions = {
  k: 2,
  // BM25 at half weight. It is the more brittle of the two here: short
  // link-list and "Implementation" sections score highly on term density
  // while containing no answer, so lexical evidence is treated as a strong
  // hint rather than a peer of semantic similarity.
  weights: { dense: 1, bm25: 0.5 },
};

/** A named ranked list. Order is the ranking; ids must be unique within it. */
export interface RankedList {
  name: string;
  ids: string[];
}

export interface FusedHit {
  id: string;
  /**
   * Summed RRF contribution. Small (order 1/60 per list) and meaningful only
   * as an ordering — deliberately not passed downstream as a relevance score,
   * because it no longer carries the "is anything here actually relevant?"
   * signal that a raw cosine score does.
   */
  score: number;
  /** Which lists contributed, and at what 1-based rank. For debugging. */
  ranks: Record<string, number>;
}

/**
 * Fuse ranked lists into one ordering.
 *
 * Ties break toward the document found by more lists, then by id, so the
 * output is deterministic — a fused ranking that shuffled between identical
 * runs would make every downstream change unmeasurable.
 */
export function reciprocalRankFusion(
  lists: RankedList[],
  options: Partial<FusionOptions> = {},
): FusedHit[] {
  const { k, weights } = { ...DEFAULT_FUSION_OPTIONS, ...options };
  const fused = new Map<string, FusedHit>();

  for (const list of lists) {
    const weight = weights?.[list.name] ?? 1;
    if (weight === 0) continue;

    for (const [position, id] of list.ids.entries()) {
      const rank = position + 1;
      let hit = fused.get(id);
      if (!hit) {
        hit = { id, score: 0, ranks: {} };
        fused.set(id, hit);
      }
      hit.score += weight / (k + rank);
      hit.ranks[list.name] = rank;
    }
  }

  return [...fused.values()].sort(
    (a, b) =>
      b.score - a.score ||
      Object.keys(b.ranks).length - Object.keys(a.ranks).length ||
      a.id.localeCompare(b.id),
  );
}
