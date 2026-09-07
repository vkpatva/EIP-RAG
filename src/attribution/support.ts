/**
 * Support checking: which cited chunks did the answer actually draw on?
 *
 * The problem this addresses. A `SOURCES` list built from `mapSources` is
 * every chunk that was *retrieved*, which is a true statement but a weaker one
 * than a reader assumes: the heading reads as "the evidence behind this
 * answer", and a chunk the model read and ignored is not that. On a real run
 * of "what are functions required in ERC20" at K=10, eight of ten cited
 * chunks contained none of the nine signatures the answer listed — four of
 * them from ERC-1155, four from EIP-20 sections like `History > Copyright`.
 * Citing all ten attributes the answer to evidence that did not support it.
 *
 * Why this is not solved by filtering on what the retriever already knows:
 *
 *  - **Not by document.** It would need the question's subject parsed from its
 *    wording, and would break the ERC-721 vs ERC-1155 comparison, where a
 *    multi-document list is correct. It would also have missed most of the
 *    noise above, which was EIP-20's own off-topic sections.
 *  - **Not by score.** On that run the top ERC-1155 chunk scored 0.5459 while
 *    `Methods (overview)` — the one chunk holding the answer — scored 0.5339,
 *    and the `allowance` chunk that also supported it had no score at all,
 *    being a BM25-only hit. Any threshold keeps the noise and drops the
 *    signal.
 *  - **Not by asking the model.** Its list would be a prediction, which is the
 *    fabrication this module exists to remove.
 *
 * So the check here is neither a score nor a judgement. It asks a question
 * with a yes-or-no answer: does this chunk's text literally contain a
 * distinctive string the answer used? That is `String.includes`, over text
 * both sides already hold — deterministic, free, and impossible to fabricate,
 * which are the same properties that make `mapSources` trustworthy.
 *
 * What it deliberately is *not*: a faithfulness check. It shows that a chunk
 * contains text the answer reproduced; it cannot show that the answer's
 * *claims* follow from the evidence, and it says nothing about prose that
 * paraphrases rather than quotes. A chunk can support an answer without
 * sharing a single distinctive string with it. That is why an unsupported
 * chunk is *demoted and labelled* rather than deleted — see `SupportLevel`.
 */
import type { Source } from "./types.js";
import type { RetrievedChunk } from "../vectorstore/types.js";

/**
 * How strongly a chunk is known to have contributed.
 *
 * Three levels rather than a boolean, because the honest answer has a middle:
 *
 *  - "quoted"    — the chunk contains a distinctive string the answer used.
 *                  Positive evidence of contribution.
 *  - "unknown"   — no shared distinctive string was found. This is *not*
 *                  proof the chunk was unused: a paraphrased answer shares no
 *                  exact strings with its source. Absence of evidence.
 *  - "unchecked" — the answer contained nothing checkable (no signatures, no
 *                  quotable identifiers), so the check could not run at all
 *                  and every chunk keeps its retrieved status.
 *
 * Collapsing "unknown" into "unused" would be the same overclaim in the
 * opposite direction: dropping a chunk that did support a paraphrased answer.
 */
export type SupportLevel = "quoted" | "unknown" | "unchecked";

export interface CheckedSource extends Source {
  support: SupportLevel;
  /** The strings this chunk shares with the answer. Empty unless "quoted". */
  quoted: string[];
}

/**
 * The three patterns below are what counts as a distinctive string.
 *
 * Function signatures and identifiers, because in a specification corpus they
 * are what an answer reproduces verbatim and what a reader most needs to
 * trace: `transferFrom` in an answer either came from a chunk or came from
 * training weights, and which one it was is exactly the question.
 *
 * Deliberately narrow. Matching on ordinary words would find something in
 * every chunk and make the check meaningless — the value comes from a term
 * being rare enough that a shared occurrence is not a coincidence.
 */

/**
 * A called or declared identifier: `transferFrom(`, `keccak256(`.
 *
 * The trailing paren is what makes this a code match rather than a word
 * match, and camelCase or `_`-containing names are required for the same
 * reason: an English sentence does not contain `safeTransferFrom`, so a
 * shared occurrence is not a coincidence.
 */
const IDENTIFIER = /\b([a-z][A-Za-z0-9_]*[A-Z_][A-Za-z0-9_]{2,})\s*\(/g;

/**
 * A camelCase or ALLCAPS identifier appearing as bare text: an event name
 * (`Approval`), a constant (`INTERFACE_ID`), a type (`ERC1155TokenReceiver`).
 *
 * Requires an internal capital or underscore, which is what excludes ordinary
 * prose. An earlier version accepted any capitalised word and matched `Some`
 * and `Specifically` out of an English sentence — enough "terms" to trip the
 * checkable threshold while matching nothing, so a well-grounded EIP-55
 * answer was reported as possibly ungrounded. A term is only usable here if
 * its presence in both answer and chunk would be surprising.
 */
const EVENT_OR_TYPE = /\b([A-Za-z][a-z0-9]*(?:[A-Z][A-Za-z0-9]*)+|[A-Z][A-Z0-9]*_[A-Z0-9_]+)\b/g;

/**
 * A whole function signature with its parameter list, e.g.
 * `allowance(address _owner, address _spender)`. Matched verbatim against
 * chunk text: parameter names are what a model reconstructing from training
 * weights gets wrong, so an exact match is evidence of copying.
 */
const SIGNATURE = /\b[a-z][A-Za-z0-9_]*\s*\([^)]{4,200}\)/g;

/**
 * Terms too common in this corpus for a shared occurrence to mean anything.
 *
 * `transfer` and `approve` are on this list, which is the correction that
 * matters. The first version of this check matched them, and every ERC-1155
 * chunk in a run about ERC-20 came back "supported" — ERC-1155's backwards
 * compatibility text discusses ERC-20's `transfer` at length, so the term is
 * shared without the chunk supporting anything. A term is only evidence of
 * contribution if its appearance in both places would be a coincidence.
 */
const STOPWORDS = new Set([
  // Solidity and prose scaffolding.
  "function",
  "contract",
  "interface",
  "returns",
  "public",
  "external",
  "internal",
  "private",
  "view",
  "pure",
  "payable",
  "address",
  "uint256",
  "uint8",
  "bool",
  "string",
  "bytes",
  "mapping",
  "event",
  "emit",
  "require",
  "MUST",
  "SHOULD",
  "MAY",
  "OPTIONAL",
  "REQUIRED",
  "RECOMMENDED",
  "NOT",
  "SHALL",
  "This",
  "The",
  "ERC",
  "EIP",
  "ERC20",
  "ERC721",
  "ERC1155",
  "Ethereum",
  "Solidity",
  // Section names, which appear in headers across the corpus.
  "Specification",
  "Abstract",
  "Motivation",
  "Rationale",
  "Implementation",
  "Copyright",
  "Standard",
  "Token",
  "Methods",
  "Events",
  "Note",
  "NOTES",
  "Summary",
  "Security",
  "Considerations",
  "Backwards",
  "Compatibility",
  // The cross-standard vocabulary: real identifiers, but shared by every
  // token standard and discussed in each one's compatibility section, so a
  // match on these alone says nothing about which document an answer used.
  "transfer",
  "approve",
  "balanceOf",
  "totalSupply",
  "allowance",
  "transferFrom",
  "safeTransferFrom",
  "supportsInterface",
  "Transfer",
  "Approval",
]);

/**
 * How many distinctive terms a chunk must share before it counts.
 *
 * One term is a coincidence in a corpus where every token standard discusses
 * every other. Two independent rare terms in the same chunk is not.
 */
const MIN_SHARED_TERMS = 2;

/**
 * How much quotable material an answer needs before the check runs at all.
 *
 * Below this the answer is prose, and prose that paraphrases its source
 * shares no exact strings with it — so "no match" would mean "this check does
 * not apply here", not "unsupported". Reporting the difference is the point.
 */
const MIN_CHECKABLE = 3;

/** Extract the distinctive terms an answer used. */
export function distinctiveTerms(text: string): string[] {
  const terms = new Set<string>();

  for (const match of text.matchAll(IDENTIFIER)) {
    const term = match[1]!;
    if (!STOPWORDS.has(term)) terms.add(term);
  }
  for (const match of text.matchAll(EVENT_OR_TYPE)) {
    const term = match[1]!;
    if (!STOPWORDS.has(term)) terms.add(term);
  }

  return [...terms];
}

/**
 * Mark each source with whether its chunk shares a distinctive term with the
 * answer.
 *
 * The comparison is against `chunk.text`, the same string the model was shown,
 * so a match means the term was literally available to be copied. Case is
 * preserved: `transferFrom` and `transferfrom` are different identifiers in
 * Solidity, and lowercasing would let a prose mention count as a signature.
 */
export function checkSupport(
  answer: string,
  sources: Source[],
  chunks: RetrievedChunk[],
): CheckedSource[] {
  const terms = distinctiveTerms(answer);
  const textById = new Map(chunks.map((c) => [c.chunkId, c.text]));

  // Exact signature lines from the answer, e.g.
  // "allowance(address _owner, address _spender)". A whole signature matching
  // verbatim is decisive on its own — parameter names are the part a model
  // reconstructing from memory gets wrong (`_owner` vs `owner`, `_value` vs
  // `amount`), so an exact match is evidence of copying rather than recall.
  const signatures = [
    ...new Set(
      [...answer.matchAll(SIGNATURE)].map((m) =>
        m[0].replace(/\s+/g, " ").trim(),
      ),
    ),
  ];

  // Whether the check can run at all. Signatures count toward this, and must:
  // an answer that is nothing but nine signatures has no camelCase prose
  // identifiers (they are all in STOPWORDS as cross-standard vocabulary), so
  // gating on `terms` alone declared the most checkable answer in the corpus
  // unverifiable.
  //
  // Below the bar the answer is prose, and prose that paraphrases its source
  // shares no exact strings with it — so "no match" would mean "this check
  // does not apply here", not "unsupported". Reporting that difference is the
  // point. The asymmetry is deliberate: a quoted signature proves a chunk was
  // used, while finding none proves nothing, so an answer with too little
  // quotable material is reported as *unchecked* rather than unsupported. The
  // check declines to render a verdict it cannot support, which is the
  // standard the rest of this module is held to.
  //
  // The threshold exists because of a real false alarm. "Why do some Ethereum
  // addresses contain uppercase letters?" is answered from EIP-55 in grounded
  // prose that paraphrases throughout, and an earlier version reported that
  // correct answer as possibly ungrounded.
  if (signatures.length + terms.length < MIN_CHECKABLE) {
    return sources.map((source) => ({
      ...source,
      support: "unchecked" as const,
      quoted: [],
    }));
  }

  return sources.map((source) => {
    const text = textById.get(source.chunkId) ?? "";
    const normalised = text.replace(/\s+/g, " ");

    const matchedSignatures = signatures.filter((sig) =>
      normalised.includes(sig),
    );
    const quoted = terms.filter((term) => text.includes(term));

    const supported =
      matchedSignatures.length > 0 || quoted.length >= MIN_SHARED_TERMS;

    return {
      ...source,
      support: supported ? ("quoted" as const) : ("unknown" as const),
      // Signatures first: they are the more convincing evidence, and the
      // renderer shows only the first few.
      quoted: [...matchedSignatures, ...quoted],
    };
  });
}
