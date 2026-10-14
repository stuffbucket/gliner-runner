import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

import { KNOWN_MODELS } from "../../src/index.js";
import { DEMO_CASES, demoCasesFor } from "../../src/classify-cli.js";
import { CLI } from "../helpers.js";

/**
 * Guarded real-model test of `classify` and `demo`. Runs only when GLINER_RUNNER_ONNX_MODELS_DIR points at a models
 * directory holding the prepared artifact (`<dir>/onnx/<owner>--<name>/<revision>/`); each model is skipped when its
 * artifact is absent. The expected labels are the ones the demo documents for decide-340m.
 */
const MODELS_DIR = process.env["GLINER_RUNNER_ONNX_MODELS_DIR"];

function run(args: string[]): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [CLI, ...args, "--models-dir", MODELS_DIR as string], {
      env: { PATH: process.env["PATH"] ?? "" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
    child.once("exit", (code) => resolve({ code, stdout, stderr }));
  });
}

function skipFor(modelId: string): string | false {
  if (MODELS_DIR === undefined) {
    return "set GLINER_RUNNER_ONNX_MODELS_DIR to run";
  }
  const model = KNOWN_MODELS.find((known) => known.modelId === modelId);
  if (model === undefined) {
    return `unknown model ${modelId}`;
  }
  const artifact = join(MODELS_DIR, "onnx", model.repository.replace("/", "--"), model.pinnedRevision);
  return existsSync(join(artifact, "manifest.json")) ? false : `no artifact for ${modelId} in ${MODELS_DIR}`;
}

/** Labels printed on each `Result:` line of the demo, in order. */
function demoResults(stdout: string): string[][] {
  return stdout
    .split("\n")
    .filter((line) => line.startsWith("   Result:  "))
    .map((line) => line.slice("   Result:  ".length).split(", ").map((part) => part.replace(/ \(.*$/, "")));
}

test("real decide-340m: classify selects the expected labels", { skip: skipFor("decide-340m"), timeout: 5 * 60_000 }, async () => {
  const ticket = await run(["classify", "--model", "decide-340m", "--labels", "billing,support,shipping", "I was charged twice for my last order."]);
  assert.equal(ticket.code, 0, ticket.stderr);
  assert.match(ticket.stdout, /^billing \(confidence 0\.\d{3}\)\n$/);

  const json = await run(["classify", "--model", "decide-340m", "--labels", "billing,support,shipping", "--json", "Where is my parcel? It is two weeks late."]);
  assert.equal(json.code, 0, json.stderr);
  const native = JSON.parse(json.stdout) as { category: { value: string; confidence: number; probabilities: Record<string, number> }; _meta: { feasible: boolean } };
  assert.equal(native.category.value, "shipping");
  assert.deepEqual(Object.keys(native.category.probabilities), ["billing", "support", "shipping"]);
  assert.equal(native._meta.feasible, true);

  const multi = await run(["classify", "--labels", "battery,screen,price,delivery", "--multi", "The battery is weak and the delivery was slow."]);
  assert.equal(multi.code, 0, multi.stderr);
  assert.match(multi.stdout, /^battery \(0\.\d{3}\), delivery \(0\.\d{3}\)\n$/);

  const described = await run([
    "classify", "--labels", "refund,exchange,question",
    "--describe", "refund=The customer wants their money back",
    "--describe", "exchange=The customer wants a different item or size",
    "Can I swap the blue jacket I bought for the same one in a larger size?",
  ]);
  assert.equal(described.code, 0, described.stderr);
  assert.match(described.stdout, /^exchange \(confidence/);
});

test("real decide-340m: demo selects the expected labels", { skip: skipFor("decide-340m"), timeout: 5 * 60_000 }, async () => {
  const result = await run(["demo"]);
  assert.equal(result.code, 0, result.stderr);
  const cases = demoCasesFor("decide-340m");
  assert.deepEqual(
    demoResults(result.stdout),
    cases.map((demo) => [...demo.expected]),
  );
});

test("real decide-multilingual: demo classifies the Spanish, French and German examples correctly", { skip: skipFor("decide-multilingual"), timeout: 5 * 60_000 }, async () => {
  const result = await run(["demo", "--model", "decide-multilingual"]);
  assert.equal(result.code, 0, result.stderr);
  const results = demoResults(result.stdout);
  assert.equal(results.length, DEMO_CASES.length);
  // The multi-label example is model dependent; the single-label examples, including the non-English ones, are not.
  DEMO_CASES.forEach((demo, index) => {
    if (!demo.spec.multi) {
      assert.deepEqual(results[index], [...demo.expected], demo.title);
    }
  });
});
