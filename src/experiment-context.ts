/**
 * Experiment: how retrieval quality determines generation quality.
 *
 *   npm run experiment
 *   npm run experiment -- --show-prompt
 *
 * The same question is asked three times with the same prompt, the same model,
 * and temperature 0. The *only* variable is the evidence. Whatever differs in
 * the answers is therefore attributable to retrieval alone.
 *
 * Note what this file does not import: no Qdrant client, no embedder, no
 * collection. The chunks are hand-written literals, because no real retriever
 * would ever return deliberately irrelevant chunks for this question — Case B
 * is only constructible *because* generation was kept independent of the
 * store. If the two stages were fused, this experiment could not be run at
 * all, which is the separation argument made concrete rather than asserted.
 *
 * The texts are copied verbatim from the corpus so the comparison is honest;
 * only their selection is synthetic.
 */
import {
  GenerationError,
  OpenAIChatProvider,
  RAGGenerationService,
} from "./generator/index.js";
import type { RetrievedChunk } from "./vectorstore/types.js";

const QUESTION =
  "What problem does ERC-1155 solve that ERC-20 and ERC-721 do not?";

/** Terse constructor — the experiment needs many literals to stay readable. */
function chunk(
  chunkId: string,
  eipNumber: number,
  section: string,
  text: string,
  score: number,
): RetrievedChunk {
  return {
    chunkId,
    documentId: `eip-${eipNumber}`,
    text,
    score,
    metadata: {
      eipNumber,
      title: `EIP-${eipNumber}`,
      section,
      sourcePath: `data/EIPs/eip-${eipNumber}.md`,
    },
  };
}

/** Case A — five chunks that actually answer the question. */
const RELEVANT: RetrievedChunk[] = [
  chunk(
    "eip-1155:2",
    1155,
    "Motivation",
    "Tokens standards like ERC-20 and ERC-721 require a separate contract to be " +
      "deployed for each token type or collection. This places a lot of redundant " +
      "bytecode on the Ethereum blockchain and limits certain functionality by the " +
      "nature of separating each token contract into its own permissioned address.",
    0.71,
  ),
  chunk(
    "eip-1155:1",
    1155,
    "Abstract",
    "This standard outlines a smart contract interface that can represent any " +
      "number of fungible and non-fungible token types. Existing standards such as " +
      "ERC-20 require deployment of separate contracts per token type.",
    0.68,
  ),
  chunk(
    "eip-1155:3",
    1155,
    "Motivation",
    "New functionality is possible with this design such as transferring multiple " +
      "token types at once, saving on transaction costs. Trading (escrow / atomic " +
      "swaps) of multiple tokens can be built on top of this standard and it " +
      "removes the need to approve individual token contracts separately.",
    0.65,
  ),
  chunk(
    "eip-721:1",
    721,
    "Abstract",
    "The following standard allows for the implementation of a standard API for " +
      "NFTs within smart contracts. This standard provides basic functionality to " +
      "track and transfer NFTs.",
    0.61,
  ),
  chunk(
    "eip-20:2",
    20,
    "Motivation",
    "A standard interface allows any tokens on Ethereum to be re-used by other " +
      "applications: from wallets to decentralized exchanges.",
    0.58,
  ),
];

/**
 * Case B — five chunks from the same corpus, none bearing on the question.
 *
 * Real texts, deliberately wrong selection. This is what a retrieval failure
 * looks like from generation's side: five confident-looking excerpts, all
 * about something else. Scores are plausible mid-range values, because a
 * failed retrieval does not announce itself with zeros.
 */
const IRRELEVANT: RetrievedChunk[] = [
  chunk(
    "eip-1:19",
    1,
    "EIP Header Preamble",
    "Each EIP must begin with an RFC 822 style header preamble, preceded and " +
      "followed by three hyphens. This header is also termed front matter by " +
      "Jekyll. The headers must appear in the following order.",
    0.42,
  ),
  chunk(
    "eip-55:4",
    55,
    "Test Cases",
    "# All caps\n0x52908400098527886E0F7030069857D2E4169EE7\n" +
      "0x8617E340B3D01FA5F11F306F4090FD50E238070D\n# All Lower\n" +
      "0xde709f2102306220921060314715629080e2fb77",
    0.39,
  ),
  chunk(
    "eip-1559:23",
    1559,
    "GASPRICE",
    "Previous to this change, GASPRICE represented both the ETH paid by the " +
      "signer per gas for a transaction as well as the ETH received by the miner " +
      "per gas. As of this change, GASPRICE now only represents the amount of ETH " +
      "paid by the signer per gas.",
    0.37,
  ),
  chunk(
    "eip-712:14",
    712,
    "eth_signTypedData",
    "The sign method calculates an Ethereum specific signature. Note: the " +
      "address to sign with must be unlocked.",
    0.35,
  ),
  chunk(
    "eip-1:30",
    1,
    "Consensus Layer Specifications",
    "Links to specific commits of files within the Ethereum Consensus Layer " +
      "Specifications may be included using normal markdown syntax.",
    0.33,
  ),
];

/** Case C — one relevant chunk. Correct, but only part of the picture. */
const SINGLE: RetrievedChunk[] = [RELEVANT[0]!];

const CASES = [
  { name: "Case A — 5 highly relevant chunks", chunks: RELEVANT },
  { name: "Case B — 5 irrelevant chunks", chunks: IRRELEVANT },
  { name: "Case C — 1 relevant chunk", chunks: SINGLE },
];

const showPrompt = process.argv.includes("--show-prompt");

const service = new RAGGenerationService({
  provider: new OpenAIChatProvider(),
});

console.log(`QUESTION (identical in all three cases):\n${QUESTION}\n`);
console.log("Same prompt, same model, temperature 0.");
console.log("The only variable is the evidence.\n");

try {
  for (const { name, chunks } of CASES) {
    console.log("=".repeat(76));
    console.log(`${name}\n`);

    console.log("EVIDENCE GIVEN:");
    for (const [i, c] of chunks.entries()) {
      console.log(
        `  [${i + 1}] ${c.score?.toFixed(2) ?? " -- "}  ` +
          `EIP-${c.metadata.eipNumber}  ` +
          `${c.metadata.section}`,
      );
    }
    console.log();

    const result = await service.generateDetailed(QUESTION, chunks);

    if (showPrompt) {
      console.log(`USER PROMPT (${result.userPrompt.length} chars):`);
      console.log(result.userPrompt);
      console.log();
    }

    console.log("ANSWER:");
    console.log(result.answer);
    console.log();
  }
} catch (error) {
  if (error instanceof GenerationError) {
    console.error(`\nGeneration failed: ${error.message}`);
    process.exit(1);
  }
  throw error;
}
