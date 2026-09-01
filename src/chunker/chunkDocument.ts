import type { Document } from "../loader/types.js";
import {
  DEFAULT_CHUNK_OPTIONS,
  type Chunk,
  type ChunkOptions,
} from "./types.js";

/** A run of body text under one heading. */
interface Section {
  /** Heading text, or undefined for content before the first heading. */
  heading?: string;
  /** Offset of this section's text within Document.content. */
  start: number;
  end: number;
}

const HEADING = /^(#{1,6})\s+(.+?)\s*$/;
const FENCE = /^\s*(```|~~~)/;

/**
 * Split content on Markdown headings.
 *
 * Headings are author-drawn semantic boundaries, so splitting here first means
 * a chunk never straddles two unrelated sections. Lines inside code fences are
 * skipped — erc-55.md has `# All caps` inside a code block, which is a comment,
 * not a heading.
 */
function splitIntoSections(content: string): Section[] {
  const sections: Section[] = [];
  let current: Section = { start: 0, end: content.length };
  let inFence = false;
  let offset = 0;

  for (const line of content.split("\n")) {
    const lineLength = line.length + 1; // +1 for the "\n" removed by split

    if (FENCE.test(line)) {
      inFence = !inFence;
    } else if (!inFence) {
      const match = HEADING.exec(line);
      if (match) {
        // Close the previous section and open a new one at this heading.
        current.end = offset;
        if (content.slice(current.start, current.end).trim()) {
          sections.push(current);
        }
        current = {
          heading: match[2],
          start: offset + lineLength, // section body starts after the heading line
          end: content.length,
        };
      }
    }
    offset += lineLength;
  }

  if (content.slice(current.start, current.end).trim()) sections.push(current);
  return sections;
}

/**
 * Split one section into size-bounded windows, stepping back `chunkOverlap`
 * characters between them so a fact on a boundary survives in both neighbours.
 */
function windowsFor(
  section: Section,
  content: string,
  { chunkSize, chunkOverlap }: ChunkOptions,
): Array<{ start: number; end: number }> {
  const windows: Array<{ start: number; end: number }> = [];
  const step = chunkSize - chunkOverlap;

  let start = section.start;
  while (start < section.end) {
    const end = Math.min(start + chunkSize, section.end);
    if (content.slice(start, end).trim()) windows.push({ start, end });
    if (end >= section.end) break;
    start += step;
  }
  return windows;
}

/** Split one document into chunks, carrying its metadata and provenance along. */
export function chunkDocument(
  document: Document,
  options: ChunkOptions = DEFAULT_CHUNK_OPTIONS,
): Chunk[] {
  const { chunkSize, chunkOverlap } = options;
  if (chunkSize <= 0) throw new Error("chunkSize must be greater than 0");
  if (chunkOverlap < 0) throw new Error("chunkOverlap must not be negative");
  if (chunkOverlap >= chunkSize) {
    throw new Error("chunkOverlap must be smaller than chunkSize");
  }

  const fm = document.frontmatter as { eip?: unknown; title?: unknown };
  const eip = typeof fm.eip === "number" ? fm.eip : undefined;
  const title = typeof fm.title === "string" ? fm.title : undefined;

  const chunks: Chunk[] = [];
  for (const section of splitIntoSections(document.content)) {
    for (const { start, end } of windowsFor(section, document.content, options)) {
      const index = chunks.length;
      chunks.push({
        id: `${document.id}:${index}`,
        documentId: document.id,
        text: document.content.slice(start, end),
        index,
        eip,
        title,
        section: section.heading,
        charStart: start,
        charEnd: end,
        source: document.source,
      });
    }
  }
  return chunks;
}

/** Chunk many documents, preserving input order. */
export function chunkDocuments(
  documents: Document[],
  options: ChunkOptions = DEFAULT_CHUNK_OPTIONS,
): Chunk[] {
  return documents.flatMap((doc) => chunkDocument(doc, options));
}
