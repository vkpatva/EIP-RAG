/**
 * Dev script: load the corpus and print what came back.
 *
 *   npm run inspect              list every document
 *   npm run inspect erc-20       full record for one document
 *   npm run inspect erc-20 --body   ...and print its whole body
 */
import { loadDocuments } from "./loader/index.js";

const args = process.argv.slice(2);
const showBody = args.includes("--body");
const query = args.find((a) => !a.startsWith("--"));

const { documents, errors } = await loadDocuments();

for (const err of errors) console.error(`ERROR ${err.filePath}: ${err.message}`);

if (!query) {
  // No argument: one line per document.
  console.log(`Loaded ${documents.length} documents (${errors.length} errors)\n`);
  for (const doc of documents) {
    const fm = doc.frontmatter as { title?: string; status?: string };
    console.log(
      [
        doc.id,
        doc.source.relativePath.padEnd(16),
        String(doc.content.length).padStart(6) + " chars",
        (fm.status ?? "-").padEnd(9),
        fm.title ?? "(no title)",
      ].join("  "),
    );
  }
  console.log("\nPass a name to see one in full, e.g. npm run inspect erc-20");
} else {
  // Match on file name or id, so "erc-20", "erc-20.md" and the id all work.
  const q = query.toLowerCase();
  const matches = documents.filter(
    (d) => d.source.fileName.toLowerCase().includes(q) || d.id.startsWith(q),
  );

  if (matches.length === 0) {
    console.error(`No document matching "${query}".`);
    console.error("Available:", documents.map((d) => d.source.fileName).join(", "));
    process.exit(1);
  }

  for (const doc of matches) {
    console.log(`\n--- ${doc.source.fileName} ---`);
    console.log({
      ...doc,
      content: showBody ? doc.content : doc.content.slice(0, 200) + "...",
    });
  }
}
