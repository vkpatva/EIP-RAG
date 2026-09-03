/**
 * Dev script: the attributed RAG pipeline.
 *
 *   npm run ask                                 the built-in probe set
 *   npm run ask -- "What is EIP-712?"
 *   npm run ask -- --k=10                       hybrid is the default
 *   npm run ask -- --show-chunks                print retrieval above the answer
 *   npm run ask -- --show-prompt                print what was sent to the LLM
 *   npm run ask -- --mode=extraction            force strict evidence-only answers
 *   npm run ask -- --json                       RAGResponse as JSON
 *   npm run ask -- --keep-sources               do not suppress on refusal
 *   npm run ask -- --dense                      dense-only retrieval
 *   npm run ask -- --interval=25000             pace queries (rate limits)
 *
 * A superset of `npm run generate`: same retrieval defaults, same mode
 * heuristic, same diagnostics (`--show-chunks`, `--show-prompt`, scores and
 * spread), plus the source list. `generate` is kept unchanged as the Module 5
 * script — it is the smaller thing, useful when the sources are noise.
 *
 * The rendering lives in `attribution/render.ts` rather than here, so that
 * both halves of the output are built by the same code a UI would call. What
 * stays in this file is argument parsing and the order things print in.
 *
 * `--show-chunks` is what makes Part 8's check possible by eye: every
 * citation must name a chunk id that appears in the retrieval block above it.
 */
import { connect } from "./vectorstore/connect.js";
import { EmbeddingError } from "./embedder/index.js";
import {
  GenerationError,
  OpenAIChatProvider,
  RAGGenerationService,
} from "./generator/index.js";
import {
  answerQuestion,
  formatRetrieval,
  renderResponse,
} from "./attribution/index.js";

/**
 * The Module 6 test set: a direct lookup, a paraphrase sharing little
 * vocabulary with its target, a specification enumeration, a comparison
 * spanning two documents, and an out-of-corpus negative control.
 */
const DEFAULT_QUERIES = [
  "What is EIP-712?",
  "Why do some Ethereum addresses contain uppercase letters?",
  "What functions are required by ERC-20?",
  "What's the difference between ERC-721 and ERC-1155?",
  "How do I build a React application?",
];

function flag(name: string, fallback: string): string {
  return (
    process.argv.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ??
    fallback
  );
}

const inPath = flag("in", "data/embeddings.json");
// "auto" matches `npm run generate`, and for the same reason its comment
// gives: a CLI is the right place for a heuristic, since the mode is printed
// in the output and can be overridden with --mode=. Defaulting to
// "extraction" here was a bug: "can you write me an ERC20 interface" is a
// synthesis request, and under extraction rules rule 3 forbids emitting a
// function signature, so the pipeline refused a question it could answer.
const modeFlag = flag("mode", "auto") as "extraction" | "synthesis" | "auto";
const hybrid = !process.argv.includes("--dense");
const rrfK = Number(flag("rrf-k", "2"));
const bm25Weight = Number(flag("bm25-weight", "0.5"));
const k = Number(flag("k", "5"));
const chars = Number(flag("chars", "220"));
const queryIntervalMs = Number(flag("interval", "0"));
const asJson = process.argv.includes("--json");
const showChunks = process.argv.includes("--show-chunks");
const showPrompt = process.argv.includes("--show-prompt");
const keepSources = process.argv.includes("--keep-sources");

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const queries = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const toRun = queries.length > 0 ? queries : DEFAULT_QUERIES;

const { retriever, model, dimensions, pointsCount, repository, mode } =
  await connect(inPath, {
    hybrid,
    weights: { dense: 1, bm25: bm25Weight },
    rrfK,
  });

const generator = new RAGGenerationService({
  provider: new OpenAIChatProvider(),
  mode: modeFlag,
});

if (!asJson) {
  console.log(
    `${repository.collection} · ${pointsCount} points · ${model} · ` +
      `${dimensions} dims · ${mode} · top ${k}\n`,
  );
}

try {
  const collected: unknown[] = [];
  let first = true;

  for (const question of toRun) {
    if (!first && queryIntervalMs > 0) await sleep(queryIntervalMs);
    first = false;

    const response = await answerQuestion(question, {
      retriever,
      generator,
      k,
      suppressSourcesOnRefusal: !keepSources,
    });

    if (asJson) {
      // `chunks` and the prompts are dropped: the JSON shape is the API
      // contract, and five full passages plus two multi-kilobyte prompts is a
      // debugging payload, not a response.
      const {
        chunks: _chunks,
        systemPrompt: _system,
        userPrompt: _user,
        ...rest
      } = response;
      collected.push({ question, ...rest });
      continue;
    }

    // Retrieval printed *before* the answer, so the source list below can be
    // checked against it line by line. Every citation must name a chunk id
    // that appears here; one that does not would mean the model, not the
    // application, produced it.
    if (showChunks) {
      console.log("=".repeat(76));
      console.log(`QUESTION: ${question}\n`);
      console.log(formatRetrieval(response.chunks, chars));
      console.log();
    }

    if (showPrompt) {
      console.log(
        `PROMPT SENT (system, ${response.systemPrompt.length} chars):`,
      );
      console.log(response.systemPrompt);
      console.log(`\nPROMPT SENT (user, ${response.userPrompt.length} chars):`);
      console.log(response.userPrompt);
      console.log();
    }

    console.log(renderResponse(question, response));
    console.log();
  }

  if (asJson) console.log(JSON.stringify(collected, null, 2));
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
