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
}

export const DEFAULT_CHUNK_OPTIONS: ChunkOptions = {
  chunkSize: 1000,
  chunkOverlap: 150,
};

export interface Chunk {
  /** Unique: `${documentId}:${index}`. */
  id: string;
  /** The `Document.id` this came from. Ties the chunk back to its file. */
  documentId: string;
  /** The chunk text, a verbatim slice of `Document.content`. */
  text: string;
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

  /** Full source provenance, copied from the document. */
  source: DocumentSource;
}
