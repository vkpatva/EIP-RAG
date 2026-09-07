/**
 * Dev script: end-to-end RAG. Retrieve, then generate.
 *
 *   npm run generate                            the built-in probe set
 *   npm run generate -- "What is EIP-712?"
 *   npm run generate -- --k=5
 *   npm run generate -- --dense                 dense-only (hybrid is default)
 *   npm run generate -- --show-prompt           print what was sent to the LLM
 *   npm run generate -- --in=data/embeddings-voyage.json
 *   npm run generate -- --interval=25000        pace queries (rate limits)
 *
 * This is the only file that touches both stages, and it is a *script*, not a
 * component — the coupling lives here rather than inside either stage. The
 * two halves are visible in the output on purpose: the retrieved chunks are
 * printed before the answer, because when an answer is wrong the first
 * question is always whether the evidence was wrong or the reading of it was.
 */
import { connect } from "./vectorstore/connect.js";
import { EmbeddingError } from "./embedder/index.js";
import {
  GenerationError,
  OpenAIChatProvider,
  RAGGenerationService,
} from "./generator/index.js";
// Shared with `npm run ask`, deliberately. These two scripts printing the
// same retrieval block from two copies of the code is how they drifted apart
// in the first place — one grew a flag default the other did not, and the
// same question produced answers that looked like a bug in one of them.
import { formatRetrieval } from "./attribution/index.js";

/**
 * A probe set chosen to exercise different generation behaviours: a direct
 * lookup whose retrieval is known to be poor, two paraphrases that share
 * little vocabulary with their target text, a comparison spanning two
 * documents, and three questions the corpus cannot answer at all.
 */
const DEFAULT_QUERIES = [
  "What is EIP-712?",
  "Why do some Ethereum addresses contain uppercase letters?",
  "How can I create unique digital items for my game?",
  "What's the difference between a unique game item and a system where I can have many copies of different items?",
  "How do I build a React application?",
  "What is the current price of ETH?",
  "Who invented Ethereum?",
];

function flag(name: string, fallback: string): string {
  return (
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ??
    fallback
  );
}

const inPath = flag("in", "data/embeddings.json");
const modeFlag = flag("mode", "auto") as "extraction" | "synthesis" | "auto";
// Hybrid by default, matching `npm run ask`. It was opt-in while the dense
// baseline was the thing being measured against; now that hybrid is the
// evaluated configuration (88% R@5), an opt-in flag meant the two scripts
// silently retrieved different chunks for the same question and produced
// answers that looked like a bug in one of them. `--dense` still selects the
// baseline, and `--hybrid` is accepted so existing invocations keep working.
const hybrid = !process.argv.includes("--dense");
const rrfK = Number(flag("rrf-k", "2"));
const bm25Weight = Number(flag("bm25-weight", "0.5"));
const k = Number(flag("k", "5"));
const chars = Number(flag("chars", "220"));
const queryIntervalMs = Number(flag("interval", "0"));
const showPrompt = process.argv.includes("--show-prompt");

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const queries = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const toRun = queries.length > 0 ? queries : DEFAULT_QUERIES;

const { retriever, model, dimensions, pointsCount, repository, mode } =
  await connect(inPath, {
  hybrid,
  weights: { dense: 1, bm25: bm25Weight },
  rrfK,
});

const service = new RAGGenerationService({
  provider: new OpenAIChatProvider(),
  // "auto" here, not in the library default: a CLI is the right place for a
  // heuristic, since the user can see the mode in the output and re-run with
  // an explicit flag if it guessed wrong.
  mode: modeFlag,
});

console.log(
  `${repository.collection} · ${pointsCount} points · ${model} · ` +
    `${dimensions} dims · ${mode} · top ${k}\n`,
);

try {
  let first = true;
  for (const question of toRun) {
    if (!first && queryIntervalMs > 0) await sleep(queryIntervalMs);
    first = false;

    console.log("=".repeat(76));
    console.log(`QUESTION: ${question}\n`);

    // Stage 1 — retrieval. No LLM involved; ends with text.
    const hits = await retriever.retrieve(question, k);

    // The block includes the score spread, which says whether the ranking
    // discriminated or just handed back the corpus in near-arbitrary order —
    // a flat spread on a high top score usually means the query matched a
    // theme, not a passage.
    console.log(formatRetrieval(hits, chars));
    console.log();

    // Stage 2 — generation. Knows only the question and the chunks.
    const result = await service.generateDetailed(question, hits);

    if (showPrompt) {
      console.log(`PROMPT SENT (system, ${result.systemPrompt.length} chars):`);
      console.log(result.systemPrompt);
      console.log(`\nPROMPT SENT (user, ${result.userPrompt.length} chars):`);
      console.log(result.userPrompt);
      console.log();
    }

    console.log(`ANSWER (${result.model} · ${result.mode}):`);
    console.log(result.answer);
    console.log();
  }
} catch (error) {
  if (error instanceof EmbeddingError) {
    console.error(`\nQuery embedding failed: ${error.message}`);
    process.exit(1);
  }
  if (error instanceof GenerationError) {
    console.error(`\nGeneration failed: ${error.message}`);
    process.exit(1);
  }
  throw error;
}
