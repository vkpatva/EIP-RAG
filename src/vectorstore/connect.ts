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
import { QdrantVectorRepository } from "./qdrant.js";

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
  retriever: QdrantRetriever;
  model: string;
  dimensions: number;
  pointsCount: number;
}

/**
 * Open a retriever against the collection matching an embeddings file.
 *
 * The file is read only for its model identity, not its vectors — those live
 * in Qdrant now. It is the manifest that says which collection to open.
 */
export async function connect(
  embeddingsPath: string,
  minScore = 0,
): Promise<Connection> {
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
  const retriever = new QdrantRetriever({ repository, provider, minScore });

  return {
    repository,
    retriever,
    model,
    dimensions,
    pointsCount: info.pointsCount,
  };
}
