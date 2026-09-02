/**
 * Deterministic point ids.
 *
 * Qdrant accepts only two id forms: an unsigned integer, or a UUID. It will
 * not take an arbitrary string, so `"eip-1559.md:3"` cannot be used directly.
 *
 * The fix is to hash the chunk id into a UUID. Hashing is a pure function, so
 * the same chunk id yields the same UUID on every run, on every machine,
 * forever. That is the whole property we need: `upsert` means *insert or
 * replace by id*, so a stable id makes re-ingestion overwrite rather than
 * append.
 *
 * The alternative — `crypto.randomUUID()` per point — is the subtle disaster.
 * Ingest twice and 428 points become 856: every chunk stored twice, every
 * Top-5 padded with duplicates, retrieval diversity quietly destroyed while
 * every line of code still appears to work.
 */
import { createHash } from "node:crypto";

/**
 * Map a chunk id to a stable UUID.
 *
 * This is UUIDv5's construction (SHA-1 of a namespace plus a name, with the
 * version and variant bits set) written out rather than pulled from a
 * dependency — it is six lines and makes the determinism visible.
 *
 * The namespace makes the mapping collection-specific in spirit: two projects
 * hashing the same chunk id would collide without one. It is a fixed constant,
 * never a random value, or the determinism is lost.
 */
const NAMESPACE = "eip-rag.chunk";

export function chunkPointId(chunkId: string): string {
  const hash = createHash("sha1").update(`${NAMESPACE}:${chunkId}`).digest();

  // Stamp version 5 into the high nibble of byte 6, and the RFC 4122 variant
  // into the top bits of byte 8. Qdrant validates the UUID form, so these are
  // not cosmetic.
  hash[6] = (hash[6]! & 0x0f) | 0x50;
  hash[8] = (hash[8]! & 0x3f) | 0x80;

  const hex = hash.subarray(0, 16).toString("hex");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join("-");
}
