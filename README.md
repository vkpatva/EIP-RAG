# EIP-RAG

A RAG pipeline over Ethereum EIP/ERC specifications, in TypeScript.

Stages are separate and independently runnable: the loader reads disk, the chunker
is a pure function, the embedder is the only stage that talks to a network. Loader
(done), chunker (done), embeddings (done), vector store (done), generation (not built).

## Setup

```bash
npm install                 # Node 22+ (uses --env-file-if-exists)
cp .env.example .env
docker compose up -d        # Qdrant on :6333 (dashboard at /dashboard)
```

Add an API key to `.env` for whichever embedding provider you use. `.env` is
gitignored; `.env.example` is the tracked template and must never hold a real key.

## Usage

```bash
npm run inspect                   # list loaded documents
npm run inspect -- erc-20         # one document in full (--body for entire text)

npm run chunk                     # chunk all documents -> data/chunks.json
npm run chunk -- --size=500 --overlap=50
npm run chunk -- --min=200        # sibling-packing floor (default 400)
npm run chunk -- --no-overviews   # skip synthesized interface-overview chunks
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

npm run index                     # embeddings.json -> Qdrant
npm run index -- --in=data/embeddings-voyage.json   # separate collection
npm run index -- --recreate       # drop the collection first

npm run verify                    # prove the collection holds what it should

npm run retrieve                  # Top-K for a built-in probe set
npm run retrieve -- "How can a smart contract verify a signature?" --k=5
npm run retrieve -- --chars=800   # more of each chunk's text

npm run eval:retrieval            # Recall@1/3/5 against eval/queries.json
npm run eval:retrieval -- --out=eval/retrieval-openai.json

npm run build                     # type-check and compile to dist/
```

`--` is required so npm passes flags through to the script.

## Pipeline

```
data/EIPs/*.md -> Document[] -> Chunk[] -> EmbeddedChunk[] -> Qdrant -> (generation)
                   loader      chunker       embedder       vectorstore

                                       question -> embedQuery -> search -> Top-K
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
src/vectorstore/  types.ts, pointId.ts, qdrant.ts, retriever.ts, connect.ts, index.ts
src/index-chunks.ts   dev script: load embeddings into Qdrant
src/verify.ts     dev script: inspect the stored collection
src/retrieve.ts   dev script: Top-K retrieval against Qdrant
src/eval-retrieval.ts dev script: Recall@K over the labelled set
docker-compose.yml    local Qdrant, version pinned to the client
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
const chunks = chunkDocuments(documents, {
  chunkSize: 1000,
  chunkOverlap: 150,
  minChunkSize: 400,
  synthesizeOverviews: true,
});
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
  embedText: string;   // what is embedded: text + provenance header
  charStart: number;   // content.slice(charStart, charEnd) === text
  charEnd: number;
  synthetic?: true;    // a built overview chunk, not a verbatim slice
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

### Sibling packing

Splitting on every heading level fragments a spec. EIPs put each method under its own
`####` heading, so ERC-20's nine methods became nine chunks of ~60 characters each:

```
before:  339847731d356207:7   totalSupply    (60ch)
         339847731d356207:8   balanceOf      (95ch)
         339847731d356207:9   transfer      (180ch)      19 chunks for erc-20.md
after:   339847731d356207:2   Methods > name, symbol, decimals, totalSupply  (975ch)
         339847731d356207:3   Methods > balanceOf, transfer                  (559ch)
                                                          13 chunks for erc-20.md
```

Two failures, compounding. Nothing contained *the list* a question like "what
functions do I need?" is asking for; and a 60-character fragment has too little
semantic surface to outrank a 200-word prose block from another standard. Sections
under `minChunkSize` now pack together with their siblings up to `chunkSize`,
stopping when the parent heading changes so `Methods > approve` never merges with
`Events > Transfer`.

### Synthesized overview chunks

Packing alone cannot fix it. ERC-20's full method list is ~4400 characters, so no
size-bounded chunk can hold it — and the list is exactly what the question wants.
So it is built rather than found: one chunk per heading whose children declare
signatures, holding every signature under it and nothing else.

```
EIP-20 — Token Standard · Methods — complete list

All methods required by this standard (9 total):

- function name() public view returns (string)
- function totalSupply() public view returns (uint256)
- function balanceOf(address _owner) public view returns (uint256 balance)
...
```

12 such chunks across the corpus, built by regex over `function`/`event` lines — no
LLM, deterministic, free. They are the one place `content.slice(charStart, charEnd)
!== text`, so they carry `synthetic: true` rather than hiding the exception.

### `embedText` vs `text`

`text` is evidence: a verbatim slice, shown to the LLM and to you, keeping the
offset invariant meaningful. `embedText` is a retrieval key — the same text with a
provenance header, and the RFC 2119 keyword paragraph stripped out:

```
EIP-20 — Token Standard · Token > Methods > balanceOf, transfer

Returns the account balance of another account with address `_owner`. ...
```

Both halves earn their place. ERC-20's `totalSupply` body never contains the string
"ERC-20", "token", or "interface", so a query naming any of them had nothing to
match. And the RFC 2119 sentence ("The key words MUST, REQUIRED, SHALL...") is
repeated near-verbatim in every EIP and is dense in exactly the words a requirements
question uses — it pulled unrelated Specification sections to rank 1 for "what
functions do I need?" on the strength of words carrying no information about the
section's subject. Stripped from the embedding, kept in the evidence.

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

This is the pre-Qdrant measurement: brute-force cosine straight over
`data/embeddings.json`, scored on documents after collapsing. The numbers below are
from the earlier 428-chunk corpus and have not been re-run since the chunker rework.

| Metric | OpenAI `text-embedding-3-small` |
|---|---|
| hit@1 | 37/50 · 74% |
| hit@3 | 44/50 · 88% |
| hit@5 | 46/50 · 92% |

By question type (hit@1 / hit@5):

| Type | n | hit@1 | hit@5 |
|---|--:|--:|--:|
| natural | 6 | 100% | 100% |
| technical | 14 | 93% | 93% |
| comparison | 11 | 82% | 100% |
| product | 14 | 50% | 86% |
| indirect | 5 | 40% | 80% |

Questions phrased in spec vocabulary do best; questions phrased as a product
goal ("I want users to...") do worst, because the gap between how a user asks
and how a spec is written is exactly the gap the embedding has to bridge.

Negative controls: mean top score 0.411 vs 0.529 for answerable questions, with
one at 0.612 ("How do I write a Solidity function that transfers tokens?" →
`erc-20`). Scores separate on average but overlap individually, so an absolute
score threshold is not a reliable "I don't know" signal.

## Vector store

Qdrant, via `@qdrant/js-client-rest`. `docker-compose.yml` pins the server version to
the client's major.minor — the client warns on a mismatch, and a `latest` tag can
change the storage format under you. (It did, once, during this build: a collection
written by 1.12 would not load on 1.19. The data is reproducible from
`data/embeddings.json`, so the fix was `docker compose down -v` and re-index; a real
corpus would need an export.)

**One collection per model.** The name is derived from the model —
`eip_chunks_text_embedding_3_small`, `eip_chunks_voyage_3` — never passed as a flag.
A dimension mismatch (1536 vs 1024) is a loud error Qdrant raises itself. The quiet
failure is two *different* models sharing a dimension: their vectors live in
unrelated coordinate spaces, so the comparison returns plausible numbers that mean
nothing, and nothing errors. Deriving the name from the model makes that pairing
unreachable rather than merely unlikely.

**Deterministic point ids.** `chunkPointId` hashes `chunk.id` into a UUIDv5 (Qdrant
accepts only integers and UUIDs, not `"eip-1559.md:3"`). Upsert means insert-or-
replace-by-id, so a stable id makes re-ingestion idempotent:

```
run 1: before 0    after 418  delta +418
run 2: before 418  after 418  delta +0
run 3: before 418  after 418  delta +0
```

With `randomUUID()` the same three runs give 418 -> 836 -> 1254, and a Top-5 collapses
to two distinct chunks repeated — retrieval diversity destroyed while every line of
code still appears to work. Measured, not assumed.

**`deleteDocument` exists because chunk ids are positional.** Edit a document so it
re-chunks into 9 pieces where it made 11, re-upsert, and points 0..8 are replaced
while 9 and 10 survive as orphans: stale text, still indexed, still ranking. Delete
by `documentId` then reinsert is the only clean update. `documentId` carries a
payload index so that filter is a lookup rather than a scan.

Cosine, because it measures angle and ignores magnitude: a 200-character chunk and a
1000-character one on the same topic should score comparably. These models are also
trained and normalised for it — `npm run verify` reports a stored vector magnitude of
exactly 1.0000.

`npm run verify` checks the things that fail silently: collection config, point count
against the source file, and one point fetched *by its derived id* with
`with_vector: true` — Qdrant omits vectors by default, so that is the only way to
confirm one was actually stored rather than a payload-only point.

## Retrieval

```
question -> embedQuery() -> query vector -> Qdrant search -> Top-K -> RetrievedChunk[]
```

`QdrantRetriever` is the only file that imports both an embedding provider and the
repository — deliberately. `qdrant.ts` never imports a provider; `embedder/` never
imports Qdrant. They must agree on exactly one thing, and it is the thing that fails
silently, so its constructor refuses to build when the collection's model or
dimension does not match the provider's:

```
wrong model, same dims: blocked — Collection "eip_chunks_text_embedding_3_small"
  holds text-embedding-3-small vectors, but the query provider is
  some-other-1536-model. Vectors from different models are not comparable.
wrong dims:             blocked — Collection expects 1536-dimensional vectors;
  provider produces 1024.
```

The model is read from the embedded chunks on disk, not from `.env`, because `.env`
can drift out of step with what was actually indexed.

`minScore` defaults to 0. Top-K is a ranking, not a relevance test: ask "How do I
build a React application?" and Qdrant returns five chunks regardless — the five
least-unrelated in the corpus, at 0.20-0.25. Low scores are the only signal that
nothing relevant exists, and the right cutoff has to be measured (see below), not
guessed.

At 418 points Qdrant brute-forces rather than building an HNSW index, so these
results are exact. Approximation only starts mattering around 10k+ points.

## Vector retrieval evaluation

`npm run eval:retrieval` scores `eval/queries.json` on Recall@K — did the EIP that
should answer the question appear anywhere in the top K? The crudest useful measure:
if Recall@5 is poor, nothing downstream can recover, because the answer was never
retrieved. Recall is scored on the *document*, not the chunk — which of eip-1559's
30 chunks surfaced is not the question.

text-embedding-3-small, 418 chunks, 50 answerable queries:

```
Recall@1  36/50  72.0%
Recall@3  40/50  80.0%
Recall@5  42/50  84.0%

By type:                          By difficulty:
  comparison  n=11  R@5  82%        easy    n=14  R@5  86%
  indirect    n= 5  R@5  80%        medium  n=26  R@5  81%
  natural     n= 6  R@5 100%        hard    n=10  R@5  90%
  product     n=14  R@5  71%
  technical   n=14  R@5  93%
```

The breakdowns are where the signal is; the aggregate hides it.

- **`indirect` is the weak spot at 60%**, not `hard` (90%). Difficulty labels track
  how hard the *concept* is; retrieval instead fails on how far the phrasing sits
  from the document's own vocabulary. `technical` questions score 93% because they
  reuse the spec's words — `supportsInterface`, `isValidSignature`.
- **`natural` scores 100% and `product` 79%** on the same underlying standards. A
  product framing ("I want to make sure a user hasn't typed an invalid address")
  describes a goal; the document describes a mechanism. q05 misses entirely at R@5
  while q04, the same standard asked as a question about capitalisation, lands at
  R@1. This is the query-rewriting-shaped hole.
- **Negatives do not separate.** Answerable queries score 0.30-0.68 at Top-1;
  negatives 0.33-0.61. The ranges overlap by 0.31, so no single score threshold
  distinguishes "found it" from "found nothing". q59 ("How do I write a Solidity
  function that transfers tokens?") scores 0.6123 — higher than 30 of the 50 real
  questions — because it is genuinely about the same topic as ERC-20 while not being
  answerable from it. **Topical similarity is not answerability**, and cosine
  measures only the first. A generation stage cannot use a score cutoff to decide
  whether to answer.

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

**Offsets round-trip.** `content.slice(charStart, charEnd) === text` holds for every
chunk except the 12 synthesized overviews, which are flagged `synthetic: true` for
exactly that reason. It is only true because the loader preserves the body
byte-for-byte. Worth re-checking after any splitter change.

## Limitations

- Chunk text now carries a provenance header for embedding (`embedText`), which
  reverses an earlier finding recorded here. Prepending the title alone did not help;
  prepending EIP number + title + heading path, *and* stripping the RFC 2119
  boilerplate, does — it is what makes the ERC-20 method chunks reachable by a query
  naming ERC-20 at all. The earlier negative result was about a weaker header.
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
  `eip-1`'s 0.781. Embeddings encode topic, not tokens. Still true through Qdrant:
  "What is EIP-712?" returns eip-1 at rank 1 (0.6288). The fix is hybrid search —
  keyword matching for identifiers alongside vectors for concepts. `eipNumber` is now
  stored on every point, so the filterable side of that is already in place.
- **Multi-concept questions dilute.** One vector averaging two topics matches
  neither sharply: a combined "in-game currency AND unique items" query scored 0.347
  where each half alone scored 0.534 and 0.417. Query decomposition belongs in the
  retrieval stage, not the embedder.
- **Small documents under-retrieve.** Chunk count correlates with hit@1 at +0.41.
  `erc-2771` (14 chunks) accounts for 3 of the 4 eval misses, losing to `erc-4337`
  (69 chunks) on every "someone else pays the fee" phrasing. But size is not
  destiny: `erc-1271` scores 7/7 on 13 chunks, because its questions use its own
  vocabulary. `erc-55` (5 chunks, 4 of them raw code) has almost no prose to match.
- Batches run sequentially and batch size counts texts, not tokens. Voyage's real
  limit is 120k tokens per request, which a batch of long chunks can hit while well
  under its 1000-item cap.
- **No relevance threshold is possible yet.** Answerable and unanswerable queries
  produce overlapping Top-1 score ranges (0.30-0.68 vs 0.33-0.61), so retrieval
  cannot currently tell a generation stage "I found nothing." Fixing this needs a
  signal cosine does not provide — a reranker, or an LLM judging the retrieved text.
- **Re-ingestion after a content edit is not wired up.** `deleteDocument` exists and
  works, but `npm run index` only upserts. A document that re-chunks into fewer
  pieces leaves orphan points behind. This is not hypothetical: the chunker rework
  took the corpus from 428 to 418 chunks, and a plain `npm run index` left a
  collection of 432 points — 14 orphans holding stale text that still ranked. Use
  `npm run index -- --recreate` after any chunker change until delete-then-upsert per
  document is wired up, keyed on `Document.contentHash`, which the loader computes.
- **No filtered search.** The payload carries `eipNumber`, `section` and
  `documentId`, and Qdrant can filter on all three, but `retrieve()` exposes no way
  to say "only within EIP-1559" or "only Specification sections".
- **The Voyage collection is indexed but not evaluated.** `npm run index --
  --in=data/embeddings-voyage.json` fills `eip_chunks_voyage_3` (418 points, 1024
  dims). Scoring it needs `--interval` pacing against Voyage's rate limit, and was
  not run here.
