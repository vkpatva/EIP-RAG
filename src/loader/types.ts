/**
 * Core types for the document loading stage.
 *
 * A `Document` is the raw unit of knowledge that enters the pipeline.
 * Nothing here knows about chunking, embeddings, or retrieval.
 */

/**
 * Parsed YAML frontmatter.
 *
 * Values are `unknown` on purpose: EIP frontmatter is not a fixed schema
 * (`requires` may be absent, `category` only exists for Standards Track, etc).
 * Validating those fields is a separate concern from loading them.
 */
export type Frontmatter = Record<string, unknown>;

/** Where a document came from. This is what makes citation possible later. */
export interface DocumentSource {
  /** Absolute path on the current machine. Not stable across checkouts. */
  filePath: string;
  /** Path relative to the data directory, e.g. "eip-1559.md". The identity key. */
  relativePath: string;
  /** Base file name, e.g. "eip-1559.md". */
  fileName: string;
  /** File size in bytes. */
  bytes: number;
  /** Filesystem last-modified time. */
  modifiedAt: Date;
}

export interface Document {
  /** Stable across runs and machines: derived from `source.relativePath`. */
  id: string;
  /** The Markdown body, verbatim, with the frontmatter block removed. */
  content: string;
  /** Parsed frontmatter, or `{}` when the file has none. */
  frontmatter: Frontmatter;
  /** Provenance. */
  source: DocumentSource;
  /** Hash of the raw file contents. Changes when the document is edited. */
  contentHash: string;
}

/** A single file that failed to load. Other files are unaffected. */
export interface LoadError {
  filePath: string;
  message: string;
}

export interface LoadResult {
  documents: Document[];
  errors: LoadError[];
}
