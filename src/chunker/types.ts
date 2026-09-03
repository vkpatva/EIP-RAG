/**
 * Types for the chunking stage.
 *
 * A `Chunk` is a retrievable unit: small enough to embed sharply, and
 * carrying enough provenance to be cited and traced without the loader.
 */
import type { DocumentSource } from "../loader/types.js";

export interface ChunkOptions {
  /** Target maximum chunk length in characters. */
  chunkSize: number;
  /** Characters repeated from the end of the previous chunk. Must be < chunkSize. */
  chunkOverlap: number;
  /**
   * Sections shorter than this are packed together with their siblings.
   *
   * EIPs put each method under its own `####` heading, so splitting on every
   * heading turns ERC-20's interface into nine ~60-character chunks. Nothing
   * then contains "the list of functions", which is what a question like
   * "what functions do I need?" is actually asking for — and a 60-character
   * fragment has too little semantic surface to win a similarity contest
   * against a 200-word prose block. Packing restores both.
   */
  minChunkSize: number;
  /**
   * Emit one synthesized "interface overview" chunk per heading whose
   * children are mostly code signatures.
   *
   * Packing alone cannot answer "what functions do I need?": ERC-20's nine
   * method sections total ~4400 characters, so no size-bounded chunk can hold
   * the list. The list is what the question asks for, so it has to be built
   * rather than found — one chunk per interface holding every signature under
   * it, with the prose left to the detailed chunks.
   */
  synthesizeOverviews: boolean;
}

export const DEFAULT_CHUNK_OPTIONS: ChunkOptions = {
  chunkSize: 1000,
  chunkOverlap: 150,
  minChunkSize: 400,
  synthesizeOverviews: true,
};

export interface Chunk {
  /** Unique: `${documentId}:${index}`. */
  id: string;
  /** The `Document.id` this came from. Ties the chunk back to its file. */
  documentId: string;
  /** The chunk text, a verbatim slice of `Document.content`. */
  text: string;
  /**
   * What actually gets embedded: `text` prefixed with a provenance header
   * ("EIP-20 — A standard interface for tokens · Methods > approve").
   *
   * Separate from `text` because the two have different jobs. `text` is
   * evidence shown to the LLM and to you, and must stay a verbatim slice so
   * `charStart`/`charEnd` keep meaning something. `embedText` is a retrieval
   * key, and it exists because a chunk body frequently omits the very terms
   * a searcher types: ERC-20's `totalSupply` section never says "ERC-20",
   * "token", or "interface" anywhere in its body, so a query naming any of
   * them had nothing to match against.
   */
  embedText: string;
  /** 0-based position of this chunk within its document. */
  index: number;

  /** EIP/ERC number from frontmatter, when present. */
  eip?: number;
  /** Document title from frontmatter, when present. */
  title?: string;
  /** Nearest Markdown heading above this text, e.g. "Specification". */
  section?: string;

  /** Offsets into `Document.content`: content.slice(charStart, charEnd) === text. */
  charStart: number;
  charEnd: number;

  /**
   * True for a synthesized overview chunk rather than a verbatim slice.
   *
   * `charStart`/`charEnd` still bound the region it summarizes, but
   * `content.slice(charStart, charEnd) !== text` for these — the one
   * exception to that invariant, flagged so a citation can say so.
   */
  synthetic?: true;

  /** Full source provenance, copied from the document. */
  source: DocumentSource;
}
