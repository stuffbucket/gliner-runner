import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import {
  InferenceRequestSchema,
  InvalidSchemaError,
  createServer,
  defaultSettings,
  type InferenceRequest,
  type ResolvedModel,
} from "../../src/index.js";
import {
  Gliner2Preprocessor,
  compileRunnerSchema,
  decode,
  type Gliner2Tokenizer,
  type RunnerClassificationSchema,
} from "../../src/gliner2/index.js";
import { OnnxArtifactCatalog } from "../../src/onnx/artifacts.js";
import { OnnxBackend } from "../../src/onnx/backend.js";
import { createComponents } from "../../src/onnx/factory.js";
import { wireRequest } from "../helpers.js";
import { MODEL, positionLogitsGraph, preprocessingConfig, syntheticTokenizer, writeSyntheticArtifact } from "./synthetic.js";

const TOKENIZERS_SPECIFIER = "@huggingface/tokenizers";
const { Tokenizer } = (await import(TOKENIZERS_SPECIFIER)) as {
  Tokenizer: new (json: object, config: object) => Gliner2Tokenizer;
};
const tokenizer = syntheticTokenizer();
const preprocessor = new Gliner2Preprocessor(new Tokenizer(tokenizer.json, tokenizer.config));

const TASKS = {
  intent: { labels: ["billing", "support", "urgent"], max_labels: 2, min_labels: 0, activation: "sigmoid" },
  route: { labels: ["billing", "support"] },
};

function request(text: string, overrides: Record<string, unknown> = {}): InferenceRequest {
  return InferenceRequestSchema.parse(
    wireRequest({
      text,
      schema: { kind: "classification", tasks: TASKS },
      options: { threshold: 0.3, include_confidence: true },
      ...overrides,
    }),
  );
}

/** The synthetic graph returns each label position as its logit, so the expected output is computable. */
function expected(
  req: InferenceRequest,
  taskNames?: readonly string[],
): { output: string; inputTokens: number } {
  const tasks = (req.schema as { tasks: Record<string, unknown> }).tasks;
  const compiled = compileRunnerSchema({
    kind: "classification",
    tasks: (taskNames ?? Object.keys(tasks)).map((name) => [name, tasks[name]]),
  } as RunnerClassificationSchema);
  const row = preprocessor.buildInput(req.text, compiled);
  const result = decode(compiled, Float32Array.from(row.labelPositions), {
    candidateThreshold: req.options.threshold,
    includeConfidence: req.options.include_confidence,
  });
  return { output: result.serialize(), inputTokens: row.inputTokens };
}

async function loadedBackend(options: ConstructorParameters<typeof OnnxBackend>[1] = {}): Promise<OnnxBackend> {
  const { modelsDir } = writeSyntheticArtifact();
  const model = await new OnnxArtifactCatalog(modelsDir).resolve("decide-340m");
  const backend = new OnnxBackend(model, options);
  await backend.load();
  return backend;
}

const TEXTS = [
  "my card was charged twice",
  "urgent billing support",
  "my card was charged twice. my card was charged twice again and again and again",
];

test("inferBatch pads, slices each row to its labels, decodes, and counts tokens", async () => {
  const backend = await loadedBackend();
  try {
    const requests = TEXTS.map((text) => request(text));
    const results = await backend.inferBatch(requests);
    assert.equal(results.length, 3);
    requests.forEach((req, index) => {
      const want = expected(req);
      const got = results[index];
      assert.equal(
        (got?.output as { serialize(): string }).serialize(),
        want.output,
      );
      assert.deepEqual(got?.usage, { inputTokens: want.inputTokens, outputTokens: 0 });
    });
    assert.notEqual(results[0]?.usage.inputTokens, results[2]?.usage.inputTokens);
    assert.deepEqual(await backend.inferBatch([]), []);
  } finally {
    await backend.close();
  }
});

test("chunked execution returns the same results in request order", async () => {
  const whole = await loadedBackend();
  const byRow = await loadedBackend({ maxRowsPerRun: 1 });
  const byTokens = await loadedBackend({ maxPaddedTokensPerRun: 1 });
  try {
    const requests = [...TEXTS, ...TEXTS].map((text) => request(text, { options: { include_confidence: false, threshold: 0.2 } }));
    const render = async (backend: OnnxBackend): Promise<string[]> =>
      (await backend.inferBatch(requests)).map(
        (result) => `${(result.output as { serialize(): string }).serialize()}|${result.usage.inputTokens}`,
      );
    const baseline = await render(whole);
    assert.deepEqual(await render(byRow), baseline);
    assert.deepEqual(await render(byTokens), baseline);
    assert.equal(baseline.length, 6);
  } finally {
    await Promise.all([whole.close(), byRow.close(), byTokens.close()]);
  }
});

test("schema errors carry the PyTorch messages; incompatible batches and unloaded use fail", async () => {
  const backend = await loadedBackend();
  try {
    const invalid = request("hello", {
      schema: { kind: "classification", tasks: { intent: { labels: ["yes", "no"], min_labels: 2, max_labels: 1 } } },
    });
    await assert.rejects(backend.inferBatch([invalid]), (error: unknown) => {
      assert.ok(error instanceof InvalidSchemaError);
      assert.equal(error.message, "task 'intent': min_labels (2) exceeds max_labels (1)");
      return true;
    });
    const reserved = request("hello", {
      schema: { kind: "classification", tasks: { intent: { labels: ["yes", "[P]no"] } } },
    });
    await assert.rejects(backend.inferBatch([reserved]), /label name may not contain '\[P\]'/);
    const unknownLabel = request("hello", {
      schema: {
        kind: "classification",
        tasks: { decision: { labels: ["yes", "no"] } },
        constraints: [{ type: "LabelRef", task: "decision", label: "maybe" }],
      },
    });
    await assert.rejects(backend.inferBatch([unknownLabel]), /'maybe' is not a label of 'decision'/);
    // A malformed constraint dict (unknown field) is a schema problem too, not an internal error.
    const malformed = request("hello", {
      schema: {
        kind: "classification",
        tasks: { decision: { labels: ["yes", "no"] } },
        constraints: [{ type: "LabelRef", task: "decision", label: "yes", bogus: 1 }],
      },
    });
    await assert.rejects(backend.inferBatch([malformed]), InvalidSchemaError);

    await assert.rejects(
      backend.inferBatch([request("a"), request("b", { options: { threshold: 0.9 } })]),
      /incompatible requests/,
    );
  } finally {
    await backend.close();
  }
  await assert.rejects(backend.inferBatch([request("a")]), /must be loaded/);
});

test("close is safe before load, repeated, and after a failed load", async () => {
  const { modelsDir, directory } = writeSyntheticArtifact();
  const model: ResolvedModel = await new OnnxArtifactCatalog(modelsDir).resolve("decide-340m");

  const never = new OnnxBackend(model);
  await never.close();
  await never.close();

  // Corrupt the graph after verification: load must fail cleanly and close must still succeed.
  writeFileSync(join(directory, "model.onnx"), "not an onnx graph");
  const broken = new OnnxBackend(model);
  await assert.rejects(broken.load());
  await broken.close();
  await broken.close();
  await assert.rejects(broken.inferBatch([request("a")]), /must be loaded/);
  await assert.rejects(broken.load());

  const good = writeSyntheticArtifact();
  const reloaded = new OnnxBackend(await new OnnxArtifactCatalog(good.modelsDir).resolve("decide-340m"));
  await reloaded.load();
  await reloaded.load();
  await reloaded.close();
  await reloaded.load();
  assert.equal((await reloaded.inferBatch([request("a b")])).length, 1);
  await reloaded.close();
  assert.equal(MODEL.modelId, model.modelId);
});

test("a tokenizer whose marker ids disagree with gliner2.json is rejected at load", async () => {
  const bad = writeSyntheticArtifact({
    files: (() => {
      const config = preprocessingConfig(MODEL) as { special_tokens: Record<string, number> };
      config.special_tokens["[L]"] = 99;
      return {
        "gliner2.json": JSON.stringify(config),
        "model.onnx": positionLogitsGraph(),
        "tokenizer.json": JSON.stringify(tokenizer.json),
        "tokenizer_config.json": JSON.stringify(tokenizer.config),
      };
    })(),
  });
  const backend = new OnnxBackend(await new OnnxArtifactCatalog(bad.modelsDir).resolve("decide-340m"));
  await assert.rejects(backend.load(), /\[L\].*does not match/);
  await backend.close();
});

test("server end to end: batching, key order of integer-like tasks, usage, and no fallback", async () => {
  const { modelsDir } = writeSyntheticArtifact();
  const settings = { ...defaultSettings(), host: "127.0.0.1", port: 0, modelsDir, batchWindowMs: 25 };
  const server = createServer({ ...(await createComponents(settings)), settings });
  const base = `http://127.0.0.1:${(await server.listen()).port}`;
  const post = async (path: string, body: string): Promise<{ status: number; text: string }> => {
    const response = await fetch(`${base}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
    });
    return { status: response.status, text: await response.text() };
  };
  try {
    const bodies = TEXTS.map((text) => JSON.stringify(wireRequest({ text, schema: { kind: "classification", tasks: TASKS } })));
    const concurrent = await Promise.all(bodies.map((body) => post("/v1/infer", body)));
    for (const [index, result] of concurrent.entries()) {
      assert.equal(result.status, 200, result.text);
      const want = expected(request(TEXTS[index] as string, { options: {} }));
      const parsed = JSON.parse(result.text) as { usage: unknown; precision: string };
      assert.deepEqual(parsed.usage, { inputTokens: want.inputTokens, outputTokens: 0 });
      assert.ok(result.text.includes(`"output":${expected(request(TEXTS[index] as string, { options: { threshold: 0.5, include_confidence: true } })).output}`));
    }
    const metrics = await (await fetch(`${base}/metrics`)).text();
    assert.match(metrics, /metric="batches_total"\} 1\n/);
    assert.match(metrics, /metric="requests_submitted_total"\} 3\n/);

    // Integer-like task names keep their declared order in the prompt, logits and output keys: the raw body is
    // written by hand because a JS object literal would already have reordered "10" and "2".
    const raw =
      '{"model":"decide-340m","backend":"onnx","precision":"fp32","operation":"classify",' +
      '"text":"my card was charged twice","schema":{"kind":"classification","tasks":' +
      '{"10":{"labels":["billing","support"]},"2":{"labels":["urgent","card"]},"name":{"labels":["a","b"]}}}}';
    const reference = InferenceRequestSchema.parse(JSON.parse(raw));
    const declared = expected(reference, ["10", "2", "name"]);
    const reordered = expected(reference, ["2", "10", "name"]);
    assert.notEqual(declared.output, reordered.output);
    assert.equal(declared.output.indexOf('"10"') < declared.output.indexOf('"2"'), true);

    const single = await post("/v1/infer", raw);
    assert.equal(single.status, 200, single.text);
    assert.ok(single.text.includes(`"output":${declared.output},`), single.text);
    assert.deepEqual(JSON.parse(single.text).usage, { inputTokens: declared.inputTokens, outputTokens: 0 });

    const batched = await post("/v1/batch", `{"requests":[${raw},${raw}]}`);
    assert.equal(batched.status, 200, batched.text);
    assert.equal(batched.text.split(`"output":${declared.output},`).length, 3, batched.text);

    // Another backend is never substituted.
    const pytorch = await post("/v1/infer", JSON.stringify(wireRequest({ backend: "pytorch" })));
    assert.equal(pytorch.status, 409);
  } finally {
    await server.close();
  }
});

async function withSyntheticServer<T>(
  options: { batchWindowMs?: number },
  body: (post: (path: string, raw: string) => Promise<{ status: number; text: string }>, base: string) => Promise<T>,
): Promise<T> {
  const { modelsDir } = writeSyntheticArtifact();
  const settings = { ...defaultSettings(), host: "127.0.0.1", port: 0, modelsDir, batchWindowMs: options.batchWindowMs ?? 0 };
  const server = createServer({ ...(await createComponents(settings)), settings });
  const base = `http://127.0.0.1:${(await server.listen()).port}`;
  const post = async (path: string, raw: string): Promise<{ status: number; text: string }> => {
    const response = await fetch(`${base}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: raw,
    });
    return { status: response.status, text: await response.text() };
  };
  try {
    return await body(post, base);
  } finally {
    await server.close();
  }
}

const RAW_PREFIX =
  '{"model":"decide-340m","backend":"onnx","precision":"fp32","operation":"classify",' +
  '"text":"my card was charged twice","schema":{"kind":"classification","tasks":';

test("__proto__ task, label and description names reach GLiNER2 compilation like any other string", async () => {
  const raw =
    `${RAW_PREFIX}{"__proto__":{"labels":["__proto__","support"],"label_descriptions":{"__proto__":"the proto label"}},` +
    '"b":{"labels":["billing","urgent"]}}}}';
  const reference = InferenceRequestSchema.parse(JSON.parse(raw));
  const tasks = (reference.schema as { tasks: object }).tasks;
  assert.deepEqual(Object.keys(tasks), ["__proto__", "b"]);
  const want = expected(reference, ["__proto__", "b"]);
  assert.ok(want.output.includes('"__proto__"'), want.output);

  await withSyntheticServer({}, async (post) => {
    const single = await post("/v1/infer", raw);
    assert.equal(single.status, 200, single.text);
    assert.ok(single.text.includes(`"output":${want.output},`), single.text);
    assert.deepEqual(JSON.parse(single.text).usage, { inputTokens: want.inputTokens, outputTokens: 0 });
    const batched = await post("/v1/batch", `{"requests":[${raw},${raw}]}`);
    assert.equal(batched.status, 200, batched.text);
    assert.equal(batched.text.split(`"output":${want.output},`).length, 3, batched.text);

    // A __proto__ constraint field is an unexpected keyword exactly as in Python (not silently dropped).
    const constrained = await post(
      "/v1/infer",
      `${RAW_PREFIX}{"d":{"labels":["yes","no"]}},"constraints":[{"type":"AnySelected","task":"d","__proto__":1}]}}`,
    );
    assert.equal(constrained.status, 422, constrained.text);
    assert.match(JSON.parse(constrained.text).detail.message, /unexpected keyword argument '__proto__'/);

    // Validation errors name the key as sent.
    const invalid = await post("/v1/infer", `${RAW_PREFIX}{"__proto__":{"labels":[]}}}}`);
    assert.equal(invalid.status, 422, invalid.text);
    assert.deepEqual(JSON.parse(invalid.text).detail[0].loc, [
      "body",
      "schema",
      "classification",
      "tasks",
      "__proto__",
      "labels",
    ]);
  });
});

test("aliases of one resolved model batch together and each response echoes its own model", async () => {
  const aliases = ["decide-340m", MODEL.repository, `decide-340m@${MODEL.pinnedRevision}`];
  await withSyntheticServer({ batchWindowMs: 40 }, async (post, base) => {
    const results = await Promise.all(
      aliases.map((model) => post("/v1/infer", JSON.stringify(wireRequest({ model, schema: { kind: "classification", tasks: TASKS } })))),
    );
    results.forEach((result, index) => {
      assert.equal(result.status, 200, result.text);
      assert.equal(JSON.parse(result.text).model, aliases[index]);
    });
    const metrics = await (await fetch(`${base}/metrics`)).text();
    assert.match(metrics, /metric="batches_total"\} 1\n/);
    assert.match(metrics, /metric="requests_submitted_total"\} 3\n/);
  });
});

test("inferBatch accepts requests naming the resolved model through different aliases", async () => {
  const backend = await loadedBackend();
  try {
    const requests = ["decide-340m", MODEL.repository, `decide-340m@${MODEL.pinnedRevision}`].map((model) =>
      request("urgent billing support", { model }),
    );
    const results = await backend.inferBatch(requests);
    assert.equal(results.length, 3);
    const outputs = new Set(results.map((result) => (result.output as { serialize(): string }).serialize()));
    assert.equal(outputs.size, 1);
  } finally {
    await backend.close();
  }
});
