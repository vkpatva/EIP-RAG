# EIP-RAG

A RAG pipeline over Ethereum EIP/ERC specifications, in TypeScript.

Stages are separate and independently runnable: the loader reads disk, the chunker
is a pure function, the embedder is the only stage that talks to a network. Loader
(done), chunker (done), embeddings (done), vector store / generation (not built).

## Setup

```bash
npm install     # Node 22+ (uses --env-file-if-exists)
cp .env.example .env
```

Add an API key to `.env` for whichever embedding provider you use. `.env` is
gitignored; `.env.example` is the tracked template and must never hold a real key.

## Usage

```bash
npm run inspect                   # list loaded documents
npm run inspect -- erc-20         # one document in full (--body for entire text)

npm run chunk                     # chunk all documents -> data/chunks.json
npm run chunk -- --size=500 --overlap=50
npm run chunk -- --dry            # print summary, write nothing
npm run chunk -- --out=path.json

npm run embed                     # chunks.json -> data/embeddings.json
npm run embed -- --provider=voyage --out=data/embeddings-voyage.json
npm run embed -- --limit=8        # cheap smoke test before a full run
npm run embed -- --batch=25 --interval=25000    # rate-limited accounts

npm run search                    # run a built-in probe set
npm run search -- "how are base fees burned?" --k=8 --text
npm run search -- --index=data/embeddings-voyage.json --interval=22000

npm run eval                      # score eval/queries.json, print a report
npm run eval -- --out=eval/results-openai.json

npm run build                     # type-check and compile to dist/
```

`--` is required so npm passes flags through to the script.

## Pipeline

```
data/EIPs/*.md -> Document[] -> Chunk[] -> EmbeddedChunk[] -> (vector store, generation)
                   loader      chunker       embedder
```

```
data/EIPs/        source markdown (committed)
data/chunks.json  generated output (gitignored)
src/loader/       types.ts, loadDocuments.ts, index.ts
src/chunker/      types.ts, chunkDocument.ts, index.ts
src/embedder/     types.ts, openai.ts, voyage.ts, embedChunks.ts, index.ts
src/inspect.ts    dev script: view documents
src/chunk.ts      dev script: run chunking
src/embed.ts      dev script: run embedding
src/search.ts     dev script: semantic search over an index
src/eval.ts       dev script: score a labelled query set
eval/queries.json 60 labelled questions (50 positive, 10 negative)
data/embeddings*.json  generated output (gitignored, ~13 MB)
```

## Loader

```ts
import { loadDocuments } from "./src/loader/index.js";
const { documents, errors } = await loadDocuments();   // default: data/EIPs
```

```ts
interface Document {
  id: string;               // sha256(relativePath).slice(0, 16)
  content: string;          // markdown body, verbatim, frontmatter removed
  frontmatter: Frontmatter; // parsed YAML; {} when absent
  source: DocumentSource;   // filePath, relativePath, fileName, bytes, modifiedAt
  contentHash: string;      // sha256 of the raw file
}
```

A file that fails to parse lands in `errors` and the rest still load. A missing
directory throws — a typo'd path should fail loudly, not return an empty result.

## Chunker

```ts
import { chunkDocuments } from "./src/chunker/index.js";
const chunks = chunkDocuments(documents, { chunkSize: 1000, chunkOverlap: 150 });
```

```ts
interface Chunk {
  id: string;          // `${documentId}:${index}`
  documentId: string;
  text: string;        // verbatim slice of Document.content
  index: number;
  eip?: number;        // from frontmatter
  title?: string;      // from frontmatter
  section?: string;    // nearest markdown heading above this text
  charStart: number;   // content.slice(charStart, charEnd) === text
  charEnd: number;
  source: DocumentSource;
}
```

Splits on markdown headings first, then into fixed-size windows within each section,
so a chunk never straddles two sections. Overlap repeats the tail of one chunk at the
head of the next.

Defaults are 1000/150 characters. Smaller chunks match more precisely but can lose the
context needed to interpret them; larger chunks read better but dilute the embedding.
EIPs are dense specs where a sentence often depends on a definition just above it, so
the default sits at the larger end.

## Embedder

```ts
import { embedChunks, OpenAIEmbeddingProvider } from "./src/embedder/index.js";
const provider = new OpenAIEmbeddingProvider();          // reads OPENAI_API_KEY
const embedded = await embedChunks(chunks, provider);    // Chunk[] -> EmbeddedChunk[]
```

```ts
interface EmbeddedChunk extends Chunk {
  embedding: number[];   // length always === dimensions, whatever the text length
  model: string;         // "text-embedding-3-small"
  dimensions: number;    // 1536
}

interface EmbeddingProvider {
  readonly model: string;
  readonly dimensions: number;
  readonly maxBatchSize: number;
  embedBatch(texts: string[], inputType?: InputType): Promise<number[][]>;
}
```

Two providers implement that interface: OpenAI (`text-embedding-3-small`, 1536d,
symmetric) and Voyage (`voyage-3`, 1024d, asymmetric). Pick with
`EMBEDDING_PROVIDER` in `.env` or `--provider=` on the command line.

An embedding is a lossy, one-way projection: the vector cannot be turned back
into text, which is why the original `text` travels with it. Position in the
space encodes meaning, so two chunks about the same idea land near each other
with no words in common — that is what makes retrieval work on paraphrase.

## Retrieval evaluation

`eval/queries.json` holds 60 labelled questions — 50 with an expected document,
10 negative controls that no document in the corpus answers. `npm run eval`
scores hit@k after collapsing several chunks from one document into one result.

| Metric | OpenAI `text-embedding-3-small` (1536d) | Voyage `voyage-3` (1024d) |
|---|---|---|
| hit@1 | 37/50 · 74% | **42/50 · 84%** |
| hit@3 | 44/50 · 88% | **48/50 · 96%** |
| hit@5 | 46/50 · 92% | **48/50 · 96%** |
| all docs found on multi-answer questions | 7/11 · 64% | **9/11 · 82%** |

By question type (hit@1):

| Type | n | OpenAI | Voyage |
|---|--:|--:|--:|
| natural | 6 | 100% | 100% |
| technical | 14 | 93% | **100%** |
| comparison | 11 | 82% | **91%** |
| product | 14 | 50% | **64%** |
| indirect | 5 | 40% | **60%** |

Voyage leads on every type. Head to head on the 50 positive questions it wins 6
that OpenAI misses and loses 1, which is a real difference rather than noise —
plausibly because it is asymmetric and encodes the query side differently from
the corpus side. Its scores are uniformly lower, which means nothing: each model
calibrates its own range, so only hit rates compare across columns.

Questions phrased in spec vocabulary do best; questions phrased as a product
goal ("I want users to...") do worst, because the gap between how a user asks
and how a spec is written is exactly the gap the embedding has to bridge.

Both models miss the same document, `erc-2771` — OpenAI on 4 questions, Voyage
on 2. See Limitations.

### Negative controls

Ten questions no document in the corpus answers ("How does Ethereum mining
work?", "How can I reduce my JavaScript bundle size?"). There is no right
answer; what is measured is whether confidence stays low.

| | OpenAI | Voyage |
|---|---|---|
| mean top score, answerable | 0.529 | 0.519 |
| mean top score, unanswerable | 0.411 | 0.419 |
| highest unanswerable | 0.612 | 0.668 |

Scores separate on average but overlap badly per question. On both models the
worst case is "How do I write a Solidity function that transfers tokens?"
scoring against `erc-20` *above* the median answerable question. A fixed score
threshold is therefore not a usable "I don't know" signal.

Of the two summary numbers the eval reports, `topScore` (the best chunk's
similarity) is the better one: only 1 of 10 negatives exceeds the positive
median, on both models. `spread` (top score minus 5th) is much weaker —
5 of 10 for OpenAI, 2 of 10 for Voyage — so it should not be used as a
confidence signal. Calibration, not accuracy, is this pipeline's weak point.

## Design notes

**`id` is path-derived, `contentHash` is content-derived.** `id` is stable across
machines and unchanged by edits; `contentHash` changes on every edit. That pair makes
incremental re-indexing possible: same `id` + new `contentHash` means re-embed, new
`id` means a new document. A content-derived `id` would make an edit look like a new
document.

**Headings inside code fences are ignored.** `erc-55.md` contains `# All caps` inside a
fenced block. A plain `/^#+\s/` regex treats it as a heading and splits the code block
in half, yielding 8 chunks with three garbage ones instead of 5 clean ones. The splitter
tracks fence state. This affects any corpus with shell, Python, or YAML samples.

**Provenance is copied onto every chunk.** After chunking, nothing in the text says
where it came from, and it is not recoverable later. `documentId` + `source` are what
make citation, per-document deletes from a vector store, and debugging possible.

**The provider interface takes `string[]`, not `Chunk[]`.** A provider's job is
text to vectors; it knows nothing about chunks or documents. That keeps the
OpenAI/Voyage files swappable and lets `embedQuery` reuse the same code path at
search time — which is how queries and documents are guaranteed to share a
coordinate space.

**`model` and `dimensions` are stored on every vector.** Each model defines its
own coordinate space, so vectors from two models are not comparable: comparing
them returns a plausible number that means nothing. Recording the model turns a
mismatch into a detectable error instead of silently wrong results. `search.ts`
rebuilds its provider from the index's stored `model`, not from `.env`, so
searching an old index with a new default still embeds the query correctly.

**`inputType` is optional on the interface.** Voyage is asymmetric — it encodes a
question and the passage answering it differently, and wants to be told which it
has. OpenAI is symmetric and ignores it. `embedChunks` passes `"document"`,
`embedQuery` passes `"query"`. Making it required would force a meaningless
argument on every symmetric provider.

**Responses are reordered by the API's `index` field, not by array position.**
Neither provider promises order. A silent misalignment would attach every vector
to the wrong chunk with no detectable symptom.

**Retries distinguish transient from permanent.** 429 and 5xx back off and retry;
401 and 400 fail immediately. A `Retry-After` header, when present, is honored
verbatim — the server knows when its limit resets and guessing is worse. A batch
that exhausts its retries fails the whole run rather than writing a partial
index, because a quietly incomplete index is worse than a loud failure.

**Body reads are inside the timeout guard.** `fetch` resolves when headers
arrive, but the abort signal stays armed while the body streams, and a
100-chunk response is several MB. Parsing outside the guard let a timeout escape
as an uncaught `DOMException` instead of being retried.

**Offsets round-trip.** `content.slice(charStart, charEnd) === text` holds for all 428
chunks, which is only true because the loader preserves the body byte-for-byte. Worth
re-checking after any splitter change.

## Limitations

- Chunk text is body prose only, so its embedding carries no title or section context.
  Prepending it was tried and did not help (see below); the metadata is on the chunk
  if you want to revisit it.
- Windows cut mid-sentence. The usual upgrade is recursive splitting: prefer `\n\n`,
  then `\n`, then `. `, then hard-cut.
- The loader reads one flat directory. Nested dirs need `{ recursive: true }` in
  `findMarkdownFiles`; `relativePath` already handles nesting, so ids stay stable.
- Chunk size counts characters, not tokens. The two diverge on code-heavy passages.
- **Exact identifiers retrieve poorly.** "What is EIP-712?" returns `eip-1`, because
  0 of the 40 `eip-712` chunks contain the string "EIP-712" — the body never names
  itself, and `eip-1` has a section literally titled "What is an EIP?". Prepending
  the title and EIP number to each chunk did not fix it: the bare string `"EIP-712"`
  scores 0.919 against the query, but wrapped in a sentence it drops to 0.590, below
  `eip-1`'s 0.781. Embeddings encode topic, not tokens. The fix is hybrid search —
  keyword matching for identifiers alongside vectors for concepts — at the vector
  store, so store `eip` as a filterable field.
- **Multi-concept questions dilute.** One vector averaging two topics matches
  neither sharply: a combined "in-game currency AND unique items" query scored 0.347
  where each half alone scored 0.534 and 0.417. Query decomposition belongs in the
  retrieval stage, not the embedder.
- **Small documents under-retrieve.** Chunk count correlates with hit@1 at +0.41.
  `erc-2771` (14 chunks) accounts for every miss on both models — 3 of OpenAI's 4
  and both of Voyage's 2 — losing to `erc-4337` (69 chunks) on every "someone else
  pays the fee" phrasing. But size is not
  destiny: `erc-1271` scores 7/7 on 13 chunks, because its questions use its own
  vocabulary. `erc-55` (5 chunks, 4 of them raw code) has almost no prose to match.
- Batches run sequentially and batch size counts texts, not tokens. Voyage's real
  limit is 120k tokens per request, which a batch of long chunks can hit while well
  under its 1000-item cap.