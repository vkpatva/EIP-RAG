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
  /** Heading level, 1-6. 0 for pre-heading content. Drives sibling packing. */
  depth: number;
  /**
   * Ancestor headings, outermost first: ["Methods", "approve"]. Retrieval
   * needs this because a leaf heading alone ("approve") is ambiguous across
   * documents, while the path says which interface it belongs to.
   */
  path: string[];
  /** Offset of this section's text within Document.content. */
  start: number;
  end: number;
  /** Leaf headings merged into this section by `packSections`, if any. */
  packedLeaves?: string[];
  /**
   * False when this section was large enough to stand alone, which also bars
   * later siblings from packing into it — a substantial section's boundary is
   * meaningful, and absorbing its neighbours would blur two topics.
   */
  packedFrom?: false;
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
  let current: Section = { depth: 0, path: [], start: 0, end: content.length };
  const ancestors: Array<{ depth: number; heading: string }> = [];
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
        const depth = match[1]!.length;
        const heading = match[2]!;
        // Pop headings at or below this level: they are siblings or deeper,
        // not ancestors of what follows.
        while (ancestors.length && ancestors[ancestors.length - 1]!.depth >= depth) {
          ancestors.pop();
        }
        current = {
          heading,
          depth,
          path: [...ancestors.map((a) => a.heading), heading],
          start: offset + lineLength, // section body starts after the heading line
          end: content.length,
        };
        ancestors.push({ depth, heading });
      }
    }
    offset += lineLength;
  }

  if (content.slice(current.start, current.end).trim()) sections.push(current);
  return sections;
}

/**
 * The RFC 2119 keyword paragraph, which every EIP repeats near-verbatim.
 *
 * It is retrieval poison rather than merely useless: it is dense in exactly
 * the words a requirements question uses ("MUST", "REQUIRED", "SHALL"), so it
 * ranks highly for "what functions do I need?" while containing no answer to
 * any question. Five identical copies also crowd a top-K with duplicates.
 * Dropped only when it is the whole of a section's text — the same words in a
 * real passage are left alone.
 */
const RFC2119 = /key words ["'“”]?MUST["'“”]?,? .{0,120}RFC ?2119/is;

function isBoilerplate(text: string): boolean {
  const trimmed = text.trim();
  // Length guard keeps this from eating a Specification section that merely
  // opens with the keyword paragraph before saying something substantive.
  return trimmed.length < 700 && RFC2119.test(trimmed);
}

/** The whole RFC 2119 sentence, wherever it sits in a chunk. */
const RFC2119_SENTENCE =
  /\s*The key words [^.]*?are to be interpreted as described in RFC ?2119\.\s*/is;

/**
 * Drop the RFC 2119 sentence from text destined for embedding.
 *
 * Distinct from `isBoilerplate`, which discards whole chunks that are nothing
 * but the paragraph. This handles the commoner case: a substantive section —
 * ERC-1155's Specification, say — that merely opens with it. The sentence is
 * dense in "MUST", "REQUIRED" and "SHALL", so it pulls the chunk to the top
 * of any requirements-shaped query ("what functions do I need?") on the
 * strength of words that carry no information about the section's subject.
 * Stripping it from `embedText` only: the sentence stays in `text`, because
 * it is genuinely part of the document and the LLM may quote it.
 */
function stripBoilerplate(text: string): string {
  return text.replace(RFC2119_SENTENCE, "\n\n").trim();
}

/**
 * Pack consecutive undersized sections into shared chunks.
 *
 * A section that is already substantial stands alone — its heading is a real
 * boundary and merging would blur two topics. Runs of small siblings, though,
 * are a single topic that the author happened to sub-divide: ERC-20's nine
 * method subsections are one interface, and one chunk holding all nine is
 * what answers a question about the interface.
 *
 * Packing stops at `chunkSize`, and also whenever the heading path's parent
 * changes, so `Methods > approve` never merges with `Events > Transfer`.
 */
function packSections(sections: Section[], content: string, options: ChunkOptions): Section[] {
  const { chunkSize, minChunkSize } = options;
  const packed: Section[] = [];

  for (const section of sections) {
    const text = content.slice(section.start, section.end);
    if (!text.trim() || isBoilerplate(text)) continue;

    const previous = packed[packed.length - 1];
    const canPack =
      previous !== undefined &&
      // Only small sections pack — a section that already stands on its own
      // keeps its boundary. The accumulator, though, grows to `chunkSize`:
      // capping it at `minChunkSize` instead would stop after the first pair
      // and leave ERC-20's interface spread over four chunks, which is the
      // fragmentation this pass exists to remove.
      text.length < minChunkSize &&
      previous.packedFrom !== false &&
      // Contiguous in the source, so the merged slice stays verbatim.
      section.start >= previous.end &&
      section.start - previous.end < 200 &&
      // Same parent: siblings, not cousins.
      parentOf(previous) === parentOf(section) &&
      section.end - previous.start <= chunkSize;

    if (canPack) {
      previous.end = section.end;
      // The merged chunk covers the parent topic, so label it that way and
      // keep the leaves visible: "Methods > totalSupply, balanceOf, transfer".
      previous.packedLeaves = [...(previous.packedLeaves ?? [previous.heading!]), section.heading!];
      continue;
    }
    packed.push({
      ...section,
      // Remember whether this section opened as a small one. Only small
      // openers accumulate siblings.
      packedFrom: text.length < minChunkSize ? undefined : false,
    });
  }
  return packed;
}

/** The heading one level up, or "" at the top level. */
function parentOf(section: Section): string {
  return section.path.length > 1 ? section.path[section.path.length - 2]! : "";
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

/** A Solidity/JS function or event signature line. */
const SIGNATURE = /^\s*(function|event)\s+\w+\s*\(/m;

/** Every signature line inside a section's text, normalized to one line each. */
function signaturesIn(text: string): string[] {
  const found: string[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (SIGNATURE.test(line)) found.push(line.replace(/\s+/g, " "));
  }
  return found;
}

/**
 * Build one overview chunk per parent heading whose children declare
 * signatures.
 *
 * This is the only synthesized text in the pipeline, and it earns the
 * exception. "What functions do I need in an ERC-20 contract?" is answered by
 * a *list*, and the list exists nowhere in the source as a contiguous span —
 * the author spread it across nine `####` sections totalling far more than
 * one chunk. Retrieval can only return spans it has, so the span has to be
 * constructed. Signatures only: the prose stays in the detailed chunks, which
 * keeps this dense and keeps the two kinds of chunk from competing.
 */
function synthesizeOverviews(
  sections: Section[],
  content: string,
  eip: number | undefined,
  title: string | undefined,
): Array<{ parent: string; text: string; start: number; end: number }> {
  const groups = new Map<
    string,
    { signatures: string[]; start: number; end: number }
  >();

  for (const section of sections) {
    const parent = parentOf(section) || section.heading;
    if (!parent) continue;
    const signatures = signaturesIn(content.slice(section.start, section.end));
    if (signatures.length === 0) continue;

    const group = groups.get(parent);
    if (group) {
      group.signatures.push(...signatures);
      group.end = Math.max(group.end, section.end);
    } else {
      groups.set(parent, {
        signatures: [...signatures],
        start: section.start,
        end: section.end,
      });
    }
  }

  const overviews: Array<{ parent: string; text: string; start: number; end: number }> = [];
  for (const [parent, { signatures, start, end }] of groups) {
    // One signature is not a list; the detailed chunk already covers it.
    if (signatures.length < 2) continue;
    const label = [eip !== undefined ? `EIP-${eip}` : null, title]
      .filter(Boolean)
      .join(" — ");
    const unique = [...new Set(signatures)];
    overviews.push({
      parent,
      text:
        `${label} · ${parent} — complete list\n\n` +
        `All ${parent.toLowerCase()} required by this standard ` +
        `(${unique.length} total):\n\n` +
        unique.map((sig) => `- ${sig}`).join("\n"),
      start,
      end,
    });
  }
  return overviews;
}

/**
 * The provenance header prepended to `text` for embedding only.
 *
 * Restores the identifying terms a chunk body usually lacks. ERC-20's
 * `totalSupply` section is 15 words that never name the standard, the word
 * "token", or the word "interface"; with this header it becomes findable by
 * every one of them. Cheap: a handful of tokens per chunk, paid once.
 */
function buildEmbedText(
  text: string,
  eip: number | undefined,
  title: string | undefined,
  path: string[],
  packedLeaves: string[] | undefined,
): string {
  const parts: string[] = [];
  if (eip !== undefined) parts.push(`EIP-${eip}`);
  if (title) parts.push(title);

  const trail = packedLeaves
    ? [...path.slice(0, -1), packedLeaves.join(", ")]
    : path;
  const heading = trail.filter(Boolean).join(" > ");

  const body = stripBoilerplate(text);
  const label = [parts.join(" — "), heading].filter(Boolean).join(" · ");
  if (!label) return body;

  // `doc20` marks "this chunk *is* from EIP-20", as distinct from the `std20`
  // that any mention of ERC-20 produces. The distinction is load-bearing:
  // eight other standards cite ERC-20 in passing, so `std20` occurs in 35
  // chunks of which only 13 are actually EIP-20 — weak IDF, and it matches
  // discussions of the standard rather than the standard. A term only the
  // real document carries is what a query naming ERC-20 should hit.
  const marker = eip !== undefined ? `doc${eip} ` : "";
  return `${marker}${label}\n\n${body}`;
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
  if (options.minChunkSize > chunkSize) {
    throw new Error("minChunkSize must not exceed chunkSize");
  }

  const fm = document.frontmatter as { eip?: unknown; title?: unknown };
  const eip = typeof fm.eip === "number" ? fm.eip : undefined;
  const title = typeof fm.title === "string" ? fm.title : undefined;

  const chunks: Chunk[] = [];
  const sections = packSections(
    splitIntoSections(document.content),
    document.content,
    options,
  );
  for (const section of sections) {
    for (const { start, end } of windowsFor(section, document.content, options)) {
      const index = chunks.length;
      const text = document.content.slice(start, end);
      chunks.push({
        id: `${document.id}:${index}`,
        documentId: document.id,
        text,
        embedText: buildEmbedText(
          text,
          eip,
          title,
          section.path,
          section.packedLeaves,
        ),
        index,
        eip,
        title,
        section: section.packedLeaves
          ? `${parentOf(section) || section.heading} > ${section.packedLeaves.join(", ")}`
          : section.heading,
        charStart: start,
        charEnd: end,
        source: document.source,
      });
    }
  }

  // Overviews go last so they never shift the positional ids of real chunks.
  if (options.synthesizeOverviews) {
    for (const ov of synthesizeOverviews(sections, document.content, eip, title)) {
      const index = chunks.length;
      chunks.push({
        id: `${document.id}:${index}`,
        documentId: document.id,
        text: ov.text,
        // Already carries its own EIP/heading label, so no second header.
        embedText: ov.text,
        index,
        eip,
        title,
        section: `${ov.parent} (overview)`,
        charStart: ov.start,
        charEnd: ov.end,
        synthetic: true,
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
