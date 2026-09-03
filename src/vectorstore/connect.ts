/**
 * Wiring shared by the retrieval scripts.
 *
 * The rule this file enforces: a query must be embedded by the model that
 * produced the stored vectors. It reads that model from the embedded chunks
 * on disk rather than from `.env`, because .env can drift out of step with
 * what was actually indexed — and a mismatch produces plausible-looking
 * scores rather than an error.
 */
import { readFile } from "node:fs/promises";

import {
  OpenAIEmbeddingProvider,
  VoyageEmbeddingProvider,
} from "../embedder/index.js";
import type { EmbeddedChunk, EmbeddingProvider } from "../embedder/index.js";
import { QdrantRetriever } from "./retriever.js";
import type { Retriever } from "./retriever.js";
import { QdrantVectorRepository } from "./qdrant.js";
import { BM25Index, HybridRetriever } from "../retrieval/index.js";
import type { RetrievedChunk } from "./types.js";

/** Rebuild the provider that produced an index, from its recorded identity. */
export function providerFor(
  model: string,
  dimensions: number,
): EmbeddingProvider {
  if (model.startsWith("text-embedding-")) {
    return new OpenAIEmbeddingProvider({ model, dimensions });
  }
  if (model.startsWith("voyage-")) {
    return new VoyageEmbeddingProvider({ model, outputDimension: dimensions });
  }
  throw new Error(`Cannot rebuild a provider for model "${model}".`);
}

export interface Connection {
  repository: QdrantVectorRepository;
  /**
   * Dense-only or hybrid, depending on `options.hybrid`. Typed as the
   * interface rather than the class so callers cannot depend on which.
   */
  retriever: Retriever;
  model: string;
  dimensions: number;
  pointsCount: number;
  /** Which retrieval mode was built, for output that has to say so. */
  mode: "dense" | "hybrid";
}

export interface ConnectOptions {
  minScore?: number;
  /**
   * Add BM25 and fuse by rank. Off by default so the dense baseline stays
   * the thing you measure against.
   */
  hybrid?: boolean;
  /**
   * Chunks file for the lexical index. Must be the same corpus that was
   * embedded — BM25 over a stale file would rank chunk ids that Qdrant no
   * longer holds, and those get skipped, silently shortening results.
   */
  chunksPath?: string;
  /** Per-list RRF weights, e.g. { dense: 1, bm25: 0.7 }. */
  weights?: Record<string, number>;
  /** RRF rank-smoothing constant. See DEFAULT_FUSION_OPTIONS. */
  rrfK?: number;
}

/**
 * Open a retriever against the collection matching an embeddings file.
 *
 * The file is read only for its model identity, not its vectors — those live
 * in Qdrant now. It is the manifest that says which collection to open.
 */
export async function connect(
  embeddingsPath: string,
  options: ConnectOptions | number = {},
): Promise<Connection> {
  // Numeric second argument kept working: it was `minScore` before hybrid
  // retrieval existed, and several scripts still pass it that way.
  const opts: ConnectOptions =
    typeof options === "number" ? { minScore: options } : options;
  const minScore = opts.minScore ?? 0;
  const chunks = JSON.parse(
    await readFile(embeddingsPath, "utf8"),
  ) as EmbeddedChunk[];
  if (chunks.length === 0) throw new Error(`${embeddingsPath} is empty.`);

  const { model, dimensions } = chunks[0]!;
  const repository = new QdrantVectorRepository({ model, dimensions });

  const info = await repository.getCollectionInfo();
  if (!info.exists) {
    throw new Error(
      `Collection "${repository.collection}" does not exist. ` +
        `Run: npm run index -- --in=${embeddingsPath}`,
    );
  }
  if (info.pointsCount === 0) {
    throw new Error(`Collection "${repository.collection}" is empty.`);
  }

  const provider = providerFor(model, dimensions);
  const dense = new QdrantRetriever({ repository, provider, minScore });

  if (!opts.hybrid) {
    return {
      repository,
      retriever: dense,
      model,
      dimensions,
      pointsCount: info.pointsCount,
      mode: "dense",
    };
  }

  // BM25 indexes `embedText`: the same provenance-headed string the dense
  // side embedded. That matters for lexical matching specifically — a query
  // naming "ERC-20" must reach EIP-20's `totalSupply` section, whose body
  // never contains the standard's name. The header puts it there.
  const chunksPath = opts.chunksPath ?? "data/chunks.json";
  const corpus = JSON.parse(await readFile(chunksPath, "utf8")) as Array<{
    id: string;
    documentId: string;
    text: string;
    embedText?: string;
    eip?: number;
    title?: string;
    section?: string;
    source: { relativePath: string };
  }>;

  const bm25 = new BM25Index(
    corpus.map((c) => ({ id: c.id, text: c.embedText ?? c.text })),
  );

  // Fallback payloads for hits BM25 finds but dense search did not return,
  // so a lexical-only match still arrives with its text and provenance.
  const chunkById = new Map<string, RetrievedChunk>(
    corpus.map((c) => [
      c.id,
      {
        chunkId: c.id,
        documentId: c.documentId,
        text: c.text,
        score: 0,
        metadata: {
          eipNumber: c.eip,
          title: c.title,
          section: c.section,
          sourcePath: c.source.relativePath,
        },
      },
    ]),
  );

  return {
    repository,
    retriever: new HybridRetriever({
      dense,
      bm25,
      chunkById,
      fusion: {
        ...(opts.weights ? { weights: opts.weights } : {}),
        ...(opts.rrfK !== undefined ? { k: opts.rrfK } : {}),
      },
    }),
    model,
    dimensions,
    pointsCount: info.pointsCount,
    mode: "hybrid",
  };
}
