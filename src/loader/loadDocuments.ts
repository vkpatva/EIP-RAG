import { createHash } from "node:crypto";
import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import matter from "gray-matter";

import type {
  Document,
  Frontmatter,
  LoadError,
  LoadResult,
} from "./types.js";

/** Default location of the EIP corpus, relative to the project root. */
export const DEFAULT_DATA_DIR = "data/EIPs";

/**
 * Stable document ID.
 *
 * Derived from the path *relative to the data directory*, so the same file
 * yields the same ID on any machine, and editing the file does not change it.
 * Use `contentHash` to detect edits.
 */
function makeDocumentId(relativePath: string): string {
  return createHash("sha256")
    .update(relativePath)
    .digest("hex")
    .slice(0, 16);
}

function hashContent(raw: string): string {
  return createHash("sha256").update(raw).digest("hex");
}

/** List `.md` files in a directory, sorted for deterministic output. */
async function findMarkdownFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((e) => e.isFile() && e.name.toLowerCase().endsWith(".md"))
    .map((e) => e.name)
    .sort();
}

/** Load and parse a single Markdown file into a `Document`. */
export async function loadDocument(
  filePath: string,
  baseDir: string,
): Promise<Document> {
  const raw = await readFile(filePath, "utf8");
  const stats = await stat(filePath);

  // gray-matter splits the leading `---` YAML block from the body.
  // Files without frontmatter come back with `data: {}` and the full text.
  const parsed = matter(raw);

  const relativePath = path.relative(baseDir, filePath);

  return {
    id: makeDocumentId(relativePath),
    content: parsed.content,
    frontmatter: parsed.data as Frontmatter,
    source: {
      filePath: path.resolve(filePath),
      relativePath,
      fileName: path.basename(filePath),
      bytes: stats.size,
      modifiedAt: stats.mtime,
    },
    contentHash: hashContent(raw),
  };
}

/**
 * Load every `.md` file in `dir` into a `Document`.
 *
 * A file that fails to read or parse is reported in `errors` rather than
 * aborting the whole load — one bad YAML block should not sink the corpus.
 */
export async function loadDocuments(
  dir: string = DEFAULT_DATA_DIR,
): Promise<LoadResult> {
  const baseDir = path.resolve(dir);
  const fileNames = await findMarkdownFiles(baseDir);

  const documents: Document[] = [];
  const errors: LoadError[] = [];

  for (const fileName of fileNames) {
    const filePath = path.join(baseDir, fileName);
    try {
      documents.push(await loadDocument(filePath, baseDir));
    } catch (error) {
      errors.push({
        filePath,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return { documents, errors };
}
