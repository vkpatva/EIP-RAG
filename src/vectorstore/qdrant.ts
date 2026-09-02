/**
 * A Qdrant-backed `VectorRepository`.
 *
 * Small on purpose. This file's entire job is to translate between the
 * pipeline's `EmbeddedChunk` and Qdrant's `{ id, vector, payload }` point, and
 * to own the collection's configuration. It never imports an embedding
 * provider — see the boundary note in `types.ts`.
 */
import { QdrantClient } from "@qdrant/js-client-rest";

import type { EmbeddedChunk } from "../embedder/types.js";
import { chunkPointId } from "./pointId.js";
import type {
  ChunkPayload,
  CollectionInfo,
  VectorRepository,
} from "./types.js";

export interface QdrantRepositoryOptions {
  /** Qdrant REST endpoint. */
  url?: string;
  /** API key, for a hosted cluster. Local Qdrant needs none. */
  apiKey?: string;
  /**
   * Collection name. Defaults to one derived from the model — see
   * `collectionNameFor`, which is where the model-mismatch guard lives.
   */
  collection?: string;
  /**
   * Vector length. Must equal the embedding model's output dimension.
   *
   * Qdrant compares vectors coordinate by coordinate, so a 1024-float query
   * against 1536-float storage has no valid pairing and is rejected outright.
   * That hard failure is the *good* case: it is loud and immediate.
   */
  dimensions: number;
  /** Model identity, recorded so the collection can be traced back to it. */
  model: string;
  /**
   * Points per upsert request.
   *
   * One request per point wastes a round trip each time; one request for all
   * 428 sends several megabytes of floats in a single body and risks a
   * timeout mid-stream. Batching is the middle ground, and a failed batch
   * loses only that batch's work.
   */
  batchSize?: number;
}

/**
 * Derive a collection name from the model.
 *
 * This is the defence against the quiet version of a model swap. A loud
 * mismatch (1536 vs 1024) errors on insert and you find out at once. But two
 * different models that happen to share a dimension will compare cleanly and
 * return scores like 0.43 that look entirely plausible and mean nothing —
 * every model defines its own coordinate space, and comparing across them is
 * noise dressed as a result.
 *
 * Naming the collection after the model makes the wrong pairing structurally
 * impossible rather than merely unlikely: a different model simply addresses
 * a different collection.
 */
export function collectionNameFor(model: string): string {
  return `eip_chunks_${model.replace(/[^a-zA-Z0-9]+/g, "_").toLowerCase()}`;
}

/** Map a chunk to the flat payload Qdrant stores beside the vector. */
function toPayload(chunk: EmbeddedChunk): ChunkPayload {
  return {
    chunkId: chunk.id,
    documentId: chunk.documentId,
    text: chunk.text,
    index: chunk.index,
    eipNumber: chunk.eip,
    title: chunk.title,
    section: chunk.section,
    sourcePath: chunk.source.relativePath,
  };
}

export class QdrantVectorRepository implements VectorRepository {
  readonly collection: string;
  readonly dimensions: number;
  readonly model: string;
  private readonly client: QdrantClient;
  private readonly batchSize: number;

  constructor(options: QdrantRepositoryOptions) {
    this.client = new QdrantClient({
      url: options.url ?? process.env.QDRANT_URL ?? "http://localhost:6333",
      apiKey: options.apiKey ?? process.env.QDRANT_API_KEY,
    });
    this.dimensions = options.dimensions;
    this.model = options.model;
    this.collection = options.collection ?? collectionNameFor(options.model);
    this.batchSize = options.batchSize ?? 64;
  }

  /** Escape hatch for verification scripts that need the raw client. */
  get raw(): QdrantClient {
    return this.client;
  }

  /**
   * Create the collection if it is absent.
   *
   * A collection's vector config is immutable, so this is a schema step: it
   * must run once, before any write, and cannot be corrected afterwards
   * except by dropping and re-ingesting.
   *
   * Cosine is the right metric here because it measures the *angle* between
   * vectors and ignores their length. A 200-character chunk and a
   * 1000-character chunk on the same topic should score comparably; only
   * direction — meaning — should count. It is also what these models are
   * trained and normalised for, so it is the default for a reason.
   *
   * If the collection already exists with a different vector size, that is a
   * real error and not something to paper over: silently writing into a
   * mismatched collection is exactly the failure this design exists to
   * prevent.
   */
  async createCollection(): Promise<void> {
    const existing = await this.getCollectionInfo();

    if (existing.exists) {
      if (existing.vectorSize !== this.dimensions) {
        throw new Error(
          `Collection "${this.collection}" is configured for ` +
            `${existing.vectorSize}-dimensional vectors, but ${this.model} ` +
            `produces ${this.dimensions}. A collection's vector size is ` +
            `immutable — drop it and re-ingest, or use a different name.`,
        );
      }
      return;
    }

    await this.client.createCollection(this.collection, {
      vectors: { size: this.dimensions, distance: "Cosine" },
    });

    // A payload index turns `deleteDocument`'s filter from a scan into a
    // lookup. At 428 points this is invisible; the habit matters at 400k.
    await this.client.createPayloadIndex(this.collection, {
      field_name: "documentId",
      field_schema: "keyword",
    });
  }

  /**
   * Store chunks, replacing any point that already holds the same chunk id.
   *
   * Idempotent by construction: the point id is a pure hash of `chunk.id`
   * (see `pointId.ts`), so the second run of an ingest overwrites the first
   * rather than appending beside it. Run this ten times and the point count
   * stays put.
   */
  async upsertChunks(chunks: EmbeddedChunk[]): Promise<void> {
    if (chunks.length === 0) return;

    // Catch a wrong-shaped vector here rather than letting Qdrant reject the
    // batch with a message that does not name the offending chunk.
    for (const chunk of chunks) {
      if (chunk.embedding.length !== this.dimensions) {
        throw new Error(
          `Chunk ${chunk.id} has a ${chunk.embedding.length}-dimensional ` +
            `vector, but collection "${this.collection}" expects ` +
            `${this.dimensions}.`,
        );
      }
    }

    for (let i = 0; i < chunks.length; i += this.batchSize) {
      const batch = chunks.slice(i, i + this.batchSize);
      await this.client.upsert(this.collection, {
        // Block until the write is searchable. Without it the point count
        // read straight after an ingest can lag, which makes the idempotency
        // experiment in Part 5 read as a flake.
        wait: true,
        points: batch.map((chunk) => ({
          id: chunkPointId(chunk.id),
          vector: chunk.embedding,
          payload: { ...toPayload(chunk) },
        })),
      });
    }
  }

  /**
   * Remove every point belonging to one document.
   *
   * Needed because chunk ids are positional. Edit `eip-1559.md` so it
   * re-chunks into 9 pieces where it once made 11, re-upsert, and points
   * 0..8 are replaced while 9 and 10 survive as orphans — stale text from the
   * old revision, still indexed, still scoring in Top-K. Deleting by document
   * before reinserting is the only clean update.
   */
  async deleteDocument(documentId: string): Promise<void> {
    await this.client.delete(this.collection, {
      wait: true,
      filter: { must: [{ key: "documentId", match: { value: documentId } }] },
    });
  }

  /**
   * Read the collection's state. The verification handle for Parts 4 and 5.
   *
   * A missing collection is a normal answer here, not an error — callers ask
   * this precisely to find out — so a 404 becomes `exists: false`.
   */
  async getCollectionInfo(): Promise<CollectionInfo> {
    try {
      const info = await this.client.getCollection(this.collection);
      const params = info.config?.params?.vectors;
      // Qdrant returns either a single unnamed vector config or a map of
      // named ones. This project uses the unnamed form.
      const single =
        params && typeof params === "object" && "size" in params
          ? (params as { size: number; distance: string })
          : undefined;

      return {
        name: this.collection,
        exists: true,
        pointsCount: info.points_count ?? 0,
        vectorSize: single?.size,
        distance: single?.distance,
        status: info.status,
      };
    } catch (error) {
      if (isNotFound(error)) {
        return { name: this.collection, exists: false, pointsCount: 0 };
      }
      throw error;
    }
  }
}

function isNotFound(error: unknown): boolean {
  const status = (error as { status?: number })?.status;
  if (status === 404) return true;
  return /not found|doesn't exist|does not exist/i.test(String(error));
}
