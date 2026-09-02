/**
 * Types for the vector storage stage.
 *
 * The boundary this file draws: a `VectorRepository` knows how to persist
 * vectors and find them again. It does not know what an embedding *is*, which
 * model produced it, or how to turn text into one. That is the embedder's job.
 *
 * Keeping the two apart means swapping Qdrant for pgvector touches one file,
 * and swapping OpenAI for Voyage touches a different one. The only place they
 * must agree is the retriever — and one place to guard is far easier to keep
 * correct than many.
 */
import type { EmbeddedChunk } from "../embedder/types.js";

/**
 * A Qdrant point's payload: the JSON stored alongside the vector.
 *
 * Flat on purpose. Qdrant filters on payload keys, and a flat key like
 * `documentId` is simpler to filter on than a nested path. This shape is the
 * storage format, deliberately decoupled from `Chunk` — the pipeline's
 * internal type can evolve without a re-ingest, as long as this mapping is
 * updated with it.
 *
 * `text` is here because a vector is a one-way projection: you cannot recover
 * the passage from its floats. If the text is not in the payload, a search hit
 * is just an id and a score, and answering the user needs a second lookup
 * against a second store that can drift out of sync.
 */
export interface ChunkPayload {
  /** `Chunk.id`, e.g. "eip-1559.md:3". The human-readable identity. */
  chunkId: string;
  /** `Chunk.documentId`. What `deleteDocument` filters on. */
  documentId: string;
  /** The chunk text, verbatim. */
  text: string;
  /** Position within the document. Useful for ordering neighbouring hits. */
  index: number;

  /** EIP/ERC number. Recall@K in the eval is computed on this field. */
  eipNumber?: number;
  /** Document title, for readable output. */
  title?: string;
  /** Nearest heading above the text. Tells you *why* a chunk matched. */
  section?: string;
  /** Path relative to the data directory. The citation key. */
  sourcePath: string;
}

/**
 * A chunk returned by a search, with its similarity score.
 *
 * Distinct from `ChunkPayload` because the read side has different needs from
 * the write side: it carries a score, and it groups provenance under
 * `metadata` so a consumer can pass the whole object around without caring
 * which fields are storage bookkeeping.
 */
export interface RetrievedChunk {
  chunkId: string;
  documentId: string;
  text: string;
  /** Cosine similarity, roughly 0..1 here. Higher is more similar. */
  score: number;
  metadata: {
    eipNumber?: number;
    title?: string;
    section?: string;
    sourcePath?: string;
  };
}

/**
 * The storage boundary.
 *
 * Four operations, each existing for a distinct reason:
 *
 *  - `createCollection` — a collection's vector config is immutable, so it is
 *    a schema step that must happen once before any write.
 *  - `upsertChunks` — the write path. Upsert, not insert, so that re-running
 *    ingestion replaces points instead of duplicating them.
 *  - `deleteDocument` — chunk ids are positional (`file.md:7`). If an edited
 *    document re-chunks into fewer pieces, upsert overwrites the first N and
 *    leaves the tail behind as orphans: stale text that still ranks. Deleting
 *    by document before reinserting is the only clean update.
 *  - `getCollectionInfo` — the verification handle. Point count is how you
 *    prove idempotency rather than assume it.
 */
export interface VectorRepository {
  createCollection(): Promise<void>;
  upsertChunks(chunks: EmbeddedChunk[]): Promise<void>;
  deleteDocument(documentId: string): Promise<void>;
  getCollectionInfo(): Promise<CollectionInfo>;
}

/** What we actually want to know about a collection, extracted from Qdrant's reply. */
export interface CollectionInfo {
  name: string;
  exists: boolean;
  /** Points stored. The number that must not double on a second ingest. */
  pointsCount: number;
  /** Configured vector length. Must equal the provider's `dimensions`. */
  vectorSize?: number;
  /** Configured metric, expected to be "Cosine". */
  distance?: string;
  /** Qdrant's own status string, e.g. "green". */
  status?: string;
}
