import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  DEMO_CASES,
  buildRequest,
  demoCasesFor,
  formatClassification,
  parseClassifyArguments,
  readClassification,
  runClassifyCommand,
  runDemoCommand,
  type ClassifyIo,
} from "../src/classify-cli.js";
import { InvalidSchemaError } from "../src/errors.js";
import { OrderedObject } from "../src/gliner2/ordered-json.js";
import { InMemoryArtifactCatalog, type InferenceRequest } from "../src/index.js";
import { CLI, FakeFactory, installedCatalog } from "./helpers.js";

const LABELS = "billing,support,shipping";

/** The native output shape of the real backend for the CLI's single task. */
function nativeOutput(value: string | string[], probabilities: Record<string, number>, confidence: number): OrderedObject {
  return new OrderedObject([
    [
      "category",
      new OrderedObject([
        ["value", value],
        ["confidence", confidence],
        ["probabilities", new OrderedObject(Object.entries(probabilities))],
      ]),
    ],
    [
      "_meta",
      new OrderedObject([
        ["feasible", true],
        ["decoder", "independent"],
        ["exact", true],
        ["objective", 1],
        ["violations", []],
      ]),
    ],
  ]);
}

interface Captured {
  io: ClassifyIo;
  stdout: () => string;
  stderr: () => string;
  requests: InferenceRequest[];
  factory: FakeFactory;
}

/** A synthetic backend that picks the first label of a single-label task and the first two of a multi-label task. */
function capture(options: { catalog?: InMemoryArtifactCatalog; stdin?: string; terminal?: boolean } = {}): Captured {
  const factory = new FakeFactory();
  const requests: InferenceRequest[] = [];
  factory.configure = (backend) => {
    backend.inferBatch = async (batch) => {
      requests.push(...batch);
      return batch.map((request) => {
        const schema = request.schema;
        if (schema.kind !== "classification") {
          throw new Error("classification only");
        }
        const task = Object.values(schema.tasks)[0];
        assert.ok(task);
        const probabilities = Object.fromEntries(task.labels.map((label, index) => [label, index === 0 ? 0.9 : 0.1]));
        const multi = task.max_labels === null;
        return {
          output: nativeOutput(multi ? task.labels.slice(0, 2) : (task.labels[0] as string), probabilities, 0.9),
          usage: { inputTokens: request.text.length, outputTokens: 0 as const },
        };
      });
    };
  };
  let out = "";
  let err = "";
  const catalog = options.catalog ?? installedCatalog();
  const io: ClassifyIo = {
    env: {},
    stdout: (text) => void (out += text),
    stderr: (text) => void (err += text),
    stdinIsTerminal: options.terminal ?? false,
    readStdin: async () => options.stdin ?? "",
    components: async () => ({ catalog, backends: [factory] }),
  };
  return { io, stdout: () => out, stderr: () => err, requests, factory };
}

const emptyCatalog = (): InMemoryArtifactCatalog => new InMemoryArtifactCatalog({ destination: "/models" });

test("parseClassifyArguments reads labels, flags and the text", () => {
  const parsed = parseClassifyArguments([
    "--labels", LABELS, "--labels", "returns", "--model", "decide-1b", "--multi", "--threshold", "0.3",
    "--describe", "billing=Invoices and charges", "--describe", "support=Help with the product = anything",
    "--json", "--models-dir", "/tmp/m", "A text",
  ]);
  assert.deepEqual(parsed.spec, {
    labels: ["billing", "support", "shipping", "returns"],
    multi: true,
    threshold: 0.3,
    descriptions: { billing: "Invoices and charges", support: "Help with the product = anything" },
  });
  assert.equal(parsed.model, "decide-1b");
  assert.equal(parsed.text, "A text");
  assert.equal(parsed.readStdin, false);
  assert.equal(parsed.json, true);
  assert.equal(parsed.modelsDir, "/tmp/m");
});

test("parseClassifyArguments defaults to decide-340m, one label, threshold 0.5, trimmed labels", () => {
  const parsed = parseClassifyArguments(["--labels", " billing , support ", "x"]);
  assert.equal(parsed.model, "decide-340m");
  assert.deepEqual(parsed.spec, { labels: ["billing", "support"], multi: false, threshold: 0.5, descriptions: {} });
});

test("the text comes from stdin when it is - or missing", () => {
  assert.deepEqual(
    [parseClassifyArguments(["--labels", "a,b", "-"]), parseClassifyArguments(["--labels", "a,b"])].map((p) => [p.text, p.readStdin]),
    [[undefined, true], [undefined, true]],
  );
});

test("parseClassifyArguments rejects bad input with a specific message", () => {
  const cases: [string[], RegExp][] = [
    [["x"], /--labels is required/],
    [["--labels", "a,,b", "x"], /empty label/],
    [["--labels", "a,b", "--threshold", "1", "x"], /--threshold must be a number greater than 0 and less than 1/],
    [["--labels", "a,b", "--threshold", "abc", "x"], /--threshold must be a number/],
    [["--labels", "a,b", "--describe", "a", "x"], /--describe expects label=description/],
    [["--labels", "a,b", "--describe", "a=", "x"], /--describe expects label=description/],
    [["--labels", "a,b", "--describe", "c=thing", "x"], /not in --labels: c/],
    [["--labels", "a,b", "--describe", "a=x", "--describe", "a=y", "t"], /given twice for label 'a'/],
    [["--labels", "a,b", "one", "two"], /expected one text argument but got 2/],
    [["--labels", "a,b", "--bogus", "x"], /bogus/],
  ];
  for (const [args, message] of cases) {
    assert.throws(() => parseClassifyArguments(args), message, args.join(" "));
  }
});

test("buildRequest builds a single-label or multi-label classification request", () => {
  const single = buildRequest("decide-340m", "hello", { labels: ["a", "b"], multi: false, threshold: 0.5, descriptions: {} });
  assert.equal(single.backend, "onnx");
  assert.equal(single.operation, "classify");
  assert.deepEqual(single.schema, {
    kind: "classification",
    tasks: {
      category: {
        labels: ["a", "b"], label_descriptions: null, min_labels: 1, max_labels: 1, ordered: false,
        threshold: 0.5, activation: "auto", temperature: 1, default: null, instruction: null,
      },
    },
    constraints: [],
  });
  const multi = buildRequest("decide-340m", "hello", { labels: ["a", "b"], multi: true, threshold: 0.3, descriptions: { a: "the a" } });
  const task = (multi.schema as { tasks: Record<string, Record<string, unknown>> }).tasks["category"];
  assert.deepEqual([task?.["min_labels"], task?.["max_labels"], task?.["threshold"], task?.["label_descriptions"]], [0, null, 0.3, { a: "the a" }]);
  assert.equal(multi.options.threshold, 0.3);
});

test("buildRequest reports invalid schemas and empty text as one usage message", () => {
  const spec = { labels: ["a", "a"], multi: false, threshold: 0.5, descriptions: {} };
  assert.throws(() => buildRequest("decide-340m", "x", spec), /invalid schema: classification labels must be unique/);
  assert.throws(() => buildRequest("decide-340m", "  ", { ...spec, labels: ["a", "b"] }), /text must not be empty/);
});

test("formatClassification prints the label and confidence, or each selected label with its probability", () => {
  const single = readClassification(nativeOutput("billing", { billing: 0.8887, support: 0.1 }, 0.8887));
  assert.ok(single);
  assert.equal(formatClassification(single, { threshold: 0.5 }), "billing (confidence 0.889)");
  const multi = readClassification(nativeOutput(["billing", "support"], { billing: 0.94, support: 0.8, shipping: 0.1 }, 0.7));
  assert.ok(multi);
  assert.equal(formatClassification(multi, { threshold: 0.5 }), "billing (0.940), support (0.800)");
  const none = readClassification(nativeOutput([], { billing: 0.2 }, 0.8));
  assert.ok(none);
  assert.equal(formatClassification(none, { threshold: 0.6 }), "no label reached the 0.6 threshold");
  assert.equal(readClassification({ label: "useful" }), undefined);
});

test("classify prints one readable line", async () => {
  const run = capture();
  assert.equal(await runClassifyCommand(["--labels", LABELS, "I was charged twice"], run.io), 0);
  assert.equal(run.stdout(), "billing (confidence 0.900)\n");
  assert.equal(run.stderr(), "");
  assert.equal(run.requests.length, 1);
  assert.equal(run.requests[0]?.model, "decide-340m");
  assert.equal(run.requests[0]?.text, "I was charged twice");
});

test("classify --multi prints every selected label", async () => {
  const run = capture();
  assert.equal(await runClassifyCommand(["--labels", LABELS, "--multi", "Refund and ship"], run.io), 0);
  assert.equal(run.stdout(), "billing (0.900), support (0.100)\n");
});

test("classify --json prints the native result on one line", async () => {
  const run = capture();
  assert.equal(await runClassifyCommand(["--labels", "billing,support", "--json", "text"], run.io), 0);
  const lines = run.stdout().split("\n");
  assert.equal(lines.length, 2);
  assert.deepEqual(JSON.parse(lines[0] as string), {
    category: { value: "billing", confidence: 0.9, probabilities: { billing: 0.9, support: 0.1 } },
    _meta: { feasible: true, decoder: "independent", exact: true, objective: 1, violations: [] },
  });
});

test("classify reads the text from stdin for - and for no text argument", async () => {
  for (const args of [["--labels", LABELS, "-"], ["--labels", LABELS]]) {
    const run = capture({ stdin: "My parcel is late\n" });
    assert.equal(await runClassifyCommand(args, run.io), 0);
    assert.equal(run.requests[0]?.text, "My parcel is late");
  }
});

test("classify without text on a terminal says so and exits 2", async () => {
  const run = capture({ terminal: true });
  assert.equal(await runClassifyCommand(["--labels", LABELS], run.io), 2);
  assert.match(run.stderr(), /^error: no text given; pass the text as an argument or pipe it on stdin/);
  assert.equal(run.stderr().trimEnd().split("\n").length, 1);
});

test("a missing model names the exact install command and exits 1", async () => {
  const run = capture({ catalog: emptyCatalog() });
  assert.equal(await runClassifyCommand(["--labels", LABELS, "text"], run.io), 1);
  assert.equal(run.stdout(), "");
  assert.equal(
    run.stderr(),
    "error: model decide-340m is not installed. Install it with: gliner-runner models install decide-340m\n",
  );
  const other = capture({ catalog: emptyCatalog() });
  assert.equal(await runClassifyCommand(["--labels", LABELS, "--model", "decide-multilingual", "--models-dir", "/m", "t"], other.io), 1);
  assert.match(other.stderr(), /gliner-runner models install decide-multilingual --models-dir \/m\n$/);
});

test("a model that is not in the catalog is a usage error that lists the known ones", async () => {
  const run = capture();
  assert.equal(await runClassifyCommand(["--labels", LABELS, "--model", "gpt", "text"], run.io), 2);
  assert.match(run.stderr(), /^error: unknown model 'gpt'; known models: decide-340m, decide-multilingual, decide-1b/);
});

test("an invalid schema is a one-line error with exit 2 before any model is touched", async () => {
  const run = capture({ catalog: emptyCatalog() });
  assert.equal(await runClassifyCommand(["--labels", "a,b,a", "text"], run.io), 2);
  assert.match(run.stderr(), /^error: invalid schema: classification labels must be unique/);
  assert.equal(run.factory.created.length, 0);
});

test("semantic schema errors from the backend are reported as invalid schema", async () => {
  const run = capture();
  run.factory.configure = (backend) => {
    backend.inferError = new InvalidSchemaError("label 'x' is reserved");
  };
  assert.equal(await runClassifyCommand(["--labels", "a,b", "text"], run.io), 2);
  assert.match(run.stderr(), /^error: invalid schema: label 'x' is reserved/);
});

test("classify --help prints usage and runs nothing", async () => {
  const run = capture();
  assert.equal(await runClassifyCommand(["--help"], run.io), 0);
  assert.match(run.stdout(), /^Usage: gliner-runner classify --labels/);
  assert.equal(run.factory.created.length, 0);
});

test("demo runs every English example and prints readable results", async () => {
  const run = capture();
  assert.equal(await runDemoCommand([], run.io), 0);
  const english = demoCasesFor("decide-340m");
  assert.equal(english.length, DEMO_CASES.filter((demo) => demo.multilingualOnly !== true).length);
  assert.equal(run.requests.length, english.length);
  assert.match(run.stdout(), /^gliner-runner demo \(model: decide-340m\)\n/);
  assert.equal(run.stdout().split("\n").filter((line) => line.startsWith("   Result:")).length, english.length);
  assert.match(run.stdout(), /Support ticket routing/);
  assert.match(run.stdout(), /\(several allowed\)/);
  assert.match(run.stdout(), /exchange: The customer wants a different item or size/);
  assert.match(run.stdout(), /Sentiment/);
  assert.match(run.stdout(), /Priority/);
  assert.match(run.stdout(), /Classified \d+ texts in /);
  assert.doesNotMatch(run.stdout(), /Spanish/);
});

test("demo with decide-multilingual adds Spanish, French and German examples", async () => {
  const catalog = new InMemoryArtifactCatalog({
    destination: "/models",
    installed: { "decide-multilingual": { e173bca1f0c4217d7a55b3b0c705d5ece8a0218d: "/models/m" } },
  });
  const run = capture({ catalog });
  assert.equal(await runDemoCommand(["--model", "decide-multilingual"], run.io), 0);
  assert.match(run.stdout(), /Spanish ticket/);
  assert.match(run.stdout(), /French ticket/);
  assert.match(run.stdout(), /German ticket/);
  assert.equal(run.requests.length, DEMO_CASES.length);
  assert.ok(run.requests.every((request) => request.model === "decide-multilingual"));
});

test("demo with a missing model prints the install command and exits 1", async () => {
  const run = capture({ catalog: emptyCatalog() });
  assert.equal(await runDemoCommand([], run.io), 1);
  assert.equal(run.stdout(), "");
  assert.equal(
    run.stderr(),
    "error: model decide-340m is not installed. Install it with: gliner-runner models install decide-340m\n",
  );
});

test("demo rejects extra arguments and unknown models", async () => {
  const extra = capture();
  assert.equal(await runDemoCommand(["now"], extra.io), 2);
  assert.match(extra.stderr(), /unexpected argument: now/);
  const unknown = capture();
  assert.equal(await runDemoCommand(["--model", "nope"], unknown.io), 2);
  assert.match(unknown.stderr(), /unknown model 'nope'/);
});

test("every demo case is a valid request whose expected labels are declared", () => {
  for (const demo of DEMO_CASES) {
    assert.ok(demo.expected.length > 0);
    for (const label of demo.expected) {
      assert.ok(demo.spec.labels.includes(label), `${demo.title}: ${label}`);
    }
    buildRequest("decide-340m", demo.text, demo.spec);
  }
});

// --- through the real CLI entry point ------------------------------------------------------------------------------

function cli(args: string[], input?: string): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [CLI, ...args], {
      // HOME is a scratch directory, so no real model can be found.
      env: { PATH: process.env["PATH"] ?? "", HOME: mkdtempSync(join(tmpdir(), "gliner-home-")) },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
    child.once("exit", (code) => resolve({ code, stdout, stderr }));
    child.stdin.end(input ?? "");
  });
}

test("--help and no arguments list every command", async () => {
  for (const args of [["--help"], ["help"], []]) {
    const result = await cli(args);
    assert.equal(result.code, args.length === 0 ? 2 : 0);
    for (const command of [
      "models install", "models list", "models scan", "models prepare", "classify", "demo", "serve", "update check", "--version",
    ]) {
      assert.match(result.stdout, new RegExp(`^  ${command}\\b.+\\S$`, "m"), `${args.join(" ")}: ${command}`);
    }
  }
});

test("classify and demo without an installed model fail with the install command", async () => {
  const classify = await cli(["classify", "--labels", LABELS, "text"]);
  assert.equal(classify.code, 1);
  assert.equal(classify.stdout, "");
  assert.match(classify.stderr, /^error: model decide-340m is not installed\. Install it with: gliner-runner models install decide-340m\n$/);
  const demo = await cli(["demo", "--model", "decide-1b"]);
  assert.equal(demo.code, 1);
  assert.match(demo.stderr, /gliner-runner models install decide-1b\n$/);
});

test("classify via the CLI reads stdin and validates arguments", async () => {
  const noLabels = await cli(["classify", "text"]);
  assert.equal(noLabels.code, 2);
  assert.match(noLabels.stderr, /^error: --labels is required/);
  const empty = await cli(["classify", "--labels", LABELS], "");
  assert.equal(empty.code, 2);
  assert.match(empty.stderr, /text must not be empty/);
  const piped = await cli(["classify", "--labels", LABELS], "text from a pipe\n");
  assert.equal(piped.code, 1);
  assert.match(piped.stderr, /is not installed/);
});

test("an unknown command exits 2 and shows the command list", async () => {
  const result = await cli(["frobnicate"]);
  assert.equal(result.code, 2);
  assert.match(result.stderr, /^error: unknown command: frobnicate\n/);
  assert.match(result.stderr, /classify {14}Classify one text/);
});
