/**
 * BM25: lexical retrieval over the chunk corpus.
 *
 * This exists because dense embeddings and keyword matching fail in opposite
 * directions, and the failure that motivated it is concrete. Asked "what
 * functions do I need in my ERC20 contract?", the dense index ranked ERC-1155
 * and ERC-721 prose above EIP-20's own method list: to an embedding model,
 * "functions required by standard X" and "functions required by standard Y"
 * are nearly the same sentence, and the token that distinguishes them — the
 * literal string "ERC20" — is exactly what gets smoothed away. BM25 cannot
 * see paraphrase at all, but it sees that token sharply. Neither is better;
 * they are wrong about different queries, which is what makes combining them
 * worth more than tuning either.
 *
 * Implemented here rather than pulled in: the whole corpus is a few hundred
 * chunks that already sit in memory, so the index is a couple of maps built
 * at startup. A search server would be a second piece of infrastructure to
 * run, keep in sync with Qdrant, and reason about when results disagree.
 */

/** Standard BM25 parameters. */
export interface BM25Options {
  /**
   * Term-frequency saturation. At 1.2, a term appearing ten times in a chunk
   * counts only ~3x a single occurrence — past a few mentions, repetition
   * stops being evidence of relevance and starts being an artifact of length.
   */
  k1: number;
  /**
   * Length normalization, 0..1. At 0.75 a long chunk is penalized for its
   * length but not fully: the synthesized interface-overview chunks are long
   * *because* they are dense lists of exactly the signatures a query names,
   * and normalizing them to death would bury the chunks that answer best.
   */
  b: number;
}

export const DEFAULT_BM25_OPTIONS: BM25Options = { k1: 1.2, b: 0.75 };

/**
 * Words that carry no discriminating signal in this corpus.
 *
 * Two groups. English function words are the usual list. The rest are terms
 * that appear in nearly every EIP — "ethereum", "standard", "contract",
 * "must" — and so behave like stopwords *here* even though they would be
 * meaningful in a general corpus. Leaving them in makes every query match
 * every document a little, which is precisely the flat-score problem BM25 is
 * being added to fix. IDF already down-weights them; removing them outright
 * also stops them from padding the term count.
 */
const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
  "from", "have", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of",
  "on", "or", "should", "so", "that", "the", "their", "then", "there", "these",
  "this", "to", "use", "used", "using", "was", "what", "when", "which", "who",
  "why", "will", "with", "would", "you", "your", "need", "needs", "want",
  // Corpus-specific: present in nearly every document.
  "ethereum", "eip", "erc", "standard", "specification", "must", "may",
  "should", "shall", "required", "recommended", "optional", "contract",
  "contracts", "implementation", "implementing",
]);

/**
 * Rewrite a *query*'s standard references to also target the owning document.
 *
 * Applied to queries only, never to the corpus. "What functions does ERC-20
 * need?" is asking about EIP-20 itself, so the query should reach for the
 * `doc20` marker that only EIP-20's chunks carry — while keeping `std20` so a
 * genuinely relevant mention elsewhere can still match, just more weakly.
 */
function expandQueryReferences(terms: string[]): string[] {
  const expanded = [...terms];
  for (const term of terms) {
    const m = /^std(\d+)$/.exec(term);
    if (m) expanded.push(`doc${m[1]}`);
  }
  return expanded;
}

/**
 * Split text into match terms.
 *
 * The interesting part is what is deliberately kept. Identifiers in this
 * corpus carry the signal: `balanceOf`, `safeTransferFrom`, `0xd9b67a26`.
 * So camelCase is indexed both whole and split into parts — a user typing
 * "balance of" should reach `balanceOf`, and one typing `balanceOf` should
 * match it exactly. Digits stay attached to letters, because "ERC20" and
 * "EIP-1559" are the most discriminating tokens a query can contain, and a
 * tokenizer that split them into "erc" + "20" would throw away the one thing
 * distinguishing two otherwise identical specification sections.
 */
export function tokenize(text: string): string[] {
  const terms: string[] = [];

  const lowered = text.toLowerCase();

  // Standard references first, and canonically. "ERC-20", "erc 20", "ERC20",
  // "EIP-20" and "eip20" all name the same document, but a plain tokenizer
  // turns the hyphenated forms into a stopword plus a bare "20" — so the
  // standard's identity survives only as a number shared with every other
  // 20 in the corpus. All spellings collapse to one "std20" term, and both
  // prefixes map to it because the documents and their readers disagree
  // about which to use (erc-20.md has `eip: 20` in its frontmatter).
  const referenced = new Set<string>();
  for (const m of lowered.matchAll(/\b(?:erc|eip)[\s\-_]?(\d+)\b/g)) {
    referenced.add(`std${m[1]}`);
  }
  terms.push(...referenced);

  for (const raw of lowered.match(/[a-z0-9_]+/g) ?? []) {
    // Already captured above, canonically. Skipping the raw form keeps a
    // reference from counting two or three times toward term frequency.
    if (/^(?:erc|eip)\d+$/.test(raw)) continue;
    if (STOPWORDS.has(raw)) continue;
    // Single characters are dropped as noise *except* digits. "type-2
    // transaction" and "type 2" are how people say what the specs write as
    // `0x02`, and discarding the 2 left the query as "explain type
    // transactions" — which matched EIP-712's "typed structured data" and
    // EIP-1's "types of EIP" far better than EIP-1559's actual envelope.
    if (raw.length > 1 || /^\d$/.test(raw)) terms.push(raw);
  }

  // "type 2" / "type-2" / "0x02": one canonical term for a transaction type,
  // so all three spellings meet. The specs write the hex form inside RLP
  // expressions (`0x02 || rlp([...])`) while readers ask for "type-2", and
  // nothing else bridges those two vocabularies.
  for (const m of lowered.matchAll(/\btype[\s\-_]?(\d{1,3})\b/g)) {
    terms.push(`txtype${Number(m[1])}`);
  }
  for (const m of lowered.matchAll(/\b0x0?([0-9a-f])\b/g)) {
    terms.push(`txtype${parseInt(m[1]!, 16)}`);
  }

  // camelCase parts, from the original casing which the lowercase pass lost.
  for (const identifier of text.match(/[a-z]+(?:[A-Z][a-z]+)+/g) ?? []) {
    for (const part of identifier.split(/(?=[A-Z])/)) {
      const lower = part.toLowerCase();
      if (!STOPWORDS.has(lower) && lower.length > 2) terms.push(lower);
    }
  }

  return terms;
}

/** One indexed document. `id` is opaque here — the caller maps it back. */
export interface BM25Document {
  id: string;
  text: string;
}

export interface BM25Hit {
  id: string;
  /**
   * Unbounded BM25 score. Comparable within one query's results and
   * meaningless across queries — which is why fusion downstream uses ranks,
   * not these numbers.
   */
  score: number;
}

export class BM25Index {
  /** term -> (document id -> occurrences in that document) */
  readonly #postings = new Map<string, Map<string, number>>();
  /** document id -> total terms, for length normalization */
  readonly #lengths = new Map<string, number>();
  #averageLength = 0;
  readonly #options: BM25Options;

  constructor(documents: BM25Document[], options: Partial<BM25Options> = {}) {
    this.#options = { ...DEFAULT_BM25_OPTIONS, ...options };

    for (const doc of documents) {
      const terms = tokenize(doc.text);
      this.#lengths.set(doc.id, terms.length);

      for (const term of terms) {
        let postings = this.#postings.get(term);
        if (!postings) {
          postings = new Map();
          this.#postings.set(term, postings);
        }
        postings.set(doc.id, (postings.get(doc.id) ?? 0) + 1);
      }
    }

    const total = [...this.#lengths.values()].reduce((a, b) => a + b, 0);
    this.#averageLength = this.#lengths.size > 0 ? total / this.#lengths.size : 0;
  }

  get size(): number {
    return this.#lengths.size;
  }

  /**
   * Score every document containing at least one query term, best first.
   *
   * Only documents that share a term are scored — the postings lists give
   * those directly, so an unrelated corpus costs nothing. Documents matching
   * no term simply do not appear, which is the honest answer for a lexical
   * index and the reason it must be fused with a dense one rather than used
   * alone: a correct paraphrase sharing no vocabulary scores zero here.
   */
  search(query: string, topK: number): BM25Hit[] {
    const scores = new Map<string, number>();
    const documentCount = this.#lengths.size;
    const { k1, b } = this.#options;

    for (const term of new Set(expandQueryReferences(tokenize(query)))) {
      const postings = this.#postings.get(term);
      if (!postings) continue;

      // Standard BM25 IDF. A term in nearly every document approaches zero
      // weight; the +0.5 smoothing keeps it from going negative outright.
      const idf = Math.log(
        1 + (documentCount - postings.size + 0.5) / (postings.size + 0.5),
      );

      for (const [id, frequency] of postings) {
        const length = this.#lengths.get(id) ?? 0;
        const normalized =
          this.#averageLength > 0 ? length / this.#averageLength : 1;
        const denominator = frequency + k1 * (1 - b + b * normalized);
        scores.set(id, (scores.get(id) ?? 0) + idf * (frequency * (k1 + 1)) / denominator);
      }
    }

    return [...scores]
      .map(([id, score]) => ({ id, score }))
      .sort((x, y) => y.score - x.score)
      .slice(0, topK);
  }
}
