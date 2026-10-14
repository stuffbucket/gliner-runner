/**
 * End-to-end parity of the ONNX runner against the PyTorch golden outputs.
 *
 * Guarded: runs only when GLINER_RUNNER_ONNX_MODELS_DIR points at prepared artifacts. It starts the real server
 * (createComponents + createServer), replays every golden corpus in singleton, repeated, batch and concurrent
 * modes, compares with the PyTorch golden and writes `node-vs-pytorch-<model_id>-<corpus>.{json,md}` reports.
 *
 *   GLINER_RUNNER_ONNX_MODELS_DIR     artifact root (<root>/onnx/<owner>--<name>/<revision>)
 *   GLINER_RUNNER_ONNX_PARITY_DIR     PyTorch golden directory (default <repo>/tests/fixtures/onnx-parity/goldens, checked in)
 *   GLINER_RUNNER_ONNX_PARITY_OUT_DIR report directory (default <repo>/dist/onnx-parity-run, git-ignored; never the goldens)
 *   GLINER_RUNNER_ONNX_CORPUS_DIR     request corpora (default <repo>/tests/fixtures/onnx-parity)
 *   GLINER_RUNNER_ONNX_PARITY_MODELS  optional comma-separated model ids to run (default: every installed one)
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { after, before, describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import { KNOWN_MODELS, createServer, defaultSettings, type RunnerServer, type Settings } from "../../src/index.js";
import { OrderedObject, serializeOrdered, type OrderedJson } from "../../src/gliner2/index.js";
import { createComponents } from "../../src/onnx/factory.js";
import {
  NUMERIC_TOLERANCE,
  compareOutputs,
  emptyStats,
  mergeStats,
  parseOrdered,
  type NumericStats,
} from "./parity-compare.js";

function packageRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    if (existsSync(join(dir, "package.json"))) return dir;
    const parent = dirname(dir);
    if (parent === dir) throw new Error("package.json not found");
    dir = parent;
  }
}
const ROOT = packageRoot();
const REPO = resolve(ROOT, "..", "..");

const MODELS_DIR = process.env["GLINER_RUNNER_ONNX_MODELS_DIR"];
const PARITY_DIR =
  process.env["GLINER_RUNNER_ONNX_PARITY_DIR"] ?? join(REPO, "tests", "fixtures", "onnx-parity", "goldens");
const REPORT_DIR = process.env["GLINER_RUNNER_ONNX_PARITY_OUT_DIR"] ?? join(REPO, "dist", "onnx-parity-run");
const CORPUS_DIR = process.env["GLINER_RUNNER_ONNX_CORPUS_DIR"] ?? join(REPO, "tests", "fixtures", "onnx-parity");
const MODEL_FILTER = (process.env["GLINER_RUNNER_ONNX_PARITY_MODELS"] ?? "")
  .split(",")
  .map((entry) => entry.trim())
  .filter((entry) => entry.length > 0);

interface GoldenCase {
  case_id: string;
  status: "success" | "error";
  http_status: number;
  usage?: { inputTokens: number; outputTokens: number };
  error?: { code: string; message: string; retryable: boolean };
  repeats?: number;
}

interface GoldenBatchResult extends GoldenCase {}

interface Golden {
  path: string;
  model_id: string;
  repository: string;
  revision: string;
  corpus_id: string;
  corpus_sha256: string;
  reference: unknown;
  singleton: GoldenCase[];
  batches: Array<{ batch_group: string; case_ids: string[]; results: GoldenBatchResult[] }>;
  /** Ordered outputs by case id (singleton) and by `group/index` (batches). */
  singletonOutputs: Map<string, OrderedJson>;
  batchOutputs: Map<string, OrderedJson>;
}

interface CorpusCase {
  case_id: string;
  request: Record<string, unknown>;
}

function loadGolden(path: string): Golden | undefined {
  const text = readFileSync(path, "utf8");
  const plain = JSON.parse(text) as Record<string, unknown>;
  if (plain["kind"] !== "gliner-runner-onnx-reference-outputs") return undefined;
  const ordered = parseOrdered(text) as OrderedObject;
  const singletonOutputs = new Map<string, OrderedJson>();
  for (const entry of ordered.get("singleton") as OrderedObject[]) {
    singletonOutputs.set(entry.get("case_id") as string, entry.get("output") as OrderedJson);
  }
  const batchOutputs = new Map<string, OrderedJson>();
  for (const batch of ordered.get("batches") as OrderedObject[]) {
    (batch.get("results") as OrderedObject[]).forEach((entry) => {
      batchOutputs.set(`${batch.get("batch_group") as string}/${entry.get("case_id") as string}`, entry.get("output") as OrderedJson);
    });
  }
  return { ...(plain as unknown as Golden), path, singletonOutputs, batchOutputs };
}

function discover(): Golden[] {
  if (MODELS_DIR === undefined || !existsSync(PARITY_DIR)) return [];
  const goldens: Golden[] = [];
  for (const name of readdirSync(PARITY_DIR).sort()) {
    if (!name.endsWith(".json") || name.startsWith("node-vs-pytorch-")) continue;
    const golden = loadGolden(join(PARITY_DIR, name));
    if (golden === undefined) continue;
    const model = KNOWN_MODELS.find((candidate) => candidate.modelId === golden.model_id);
    if (model === undefined || model.pinnedRevision !== golden.revision) continue;
    if (MODEL_FILTER.length > 0 && !MODEL_FILTER.includes(golden.model_id)) continue;
    const artifact = join(MODELS_DIR, "onnx", model.repository.replace("/", "--"), golden.revision, "manifest.json");
    if (existsSync(artifact)) goldens.push(golden);
  }
  return goldens;
}

type Mode = "singleton" | "repeated" | "batch" | "concurrent-group" | "concurrent-all";

interface CaseRecord {
  mode: Mode;
  case_id: string;
  http_status: number;
  expected_http_status: number;
  exact_semantic_match: boolean;
  mismatches: string[];
  numeric: NumericStats;
  input_tokens: number | null;
  expected_input_tokens: number | null;
}

interface BatchingEvidence {
  mode: Mode;
  requests: number;
  batches: number;
  submitted: number;
}

interface HttpResult {
  status: number;
  text: string;
}

async function post(base: string, path: string, body: unknown): Promise<HttpResult> {
  const response = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: response.status, text: await response.text() };
}

async function metric(base: string, name: string): Promise<number> {
  const text = await (await fetch(`${base}/metrics`)).text();
  const match = new RegExp(`metric="${name}"\\} (\\S+)`).exec(text);
  return match === null ? 0 : Number(match[1]);
}

/** Compares one HTTP answer with one golden entry. */
function check(
  mode: Mode,
  expected: GoldenCase,
  expectedOutput: OrderedJson | undefined,
  actual: { status: number; text?: string; body?: Record<string, unknown>; ordered?: OrderedJson },
): CaseRecord {
  const mismatches: string[] = [];
  let numeric = emptyStats();
  let inputTokens: number | null = null;
  if (actual.status !== expected.http_status) {
    mismatches.push(`http status ${actual.status} != ${expected.http_status}`);
  }
  const body = actual.body ?? (JSON.parse(actual.text ?? "null") as Record<string, unknown>);
  const ordered = actual.ordered ?? parseOrdered(actual.text ?? "null");
  if (expected.status === "success") {
    if (actual.status === 200) {
      const comparison = compareOutputs((ordered as OrderedObject).get("output"), expectedOutput);
      mismatches.push(...comparison.mismatches);
      numeric = comparison.numeric;
      const usage = body["usage"] as { inputTokens: number; outputTokens: number } | undefined;
      inputTokens = usage?.inputTokens ?? null;
      if (JSON.stringify(usage) !== JSON.stringify(expected.usage)) {
        mismatches.push(`usage ${JSON.stringify(usage)} != ${JSON.stringify(expected.usage)}`);
      }
      if (body["error"] !== null) mismatches.push("error must be null on success");
    } else {
      mismatches.push(`expected success, got ${JSON.stringify(body["detail"])}`);
    }
  } else {
    const detail = body["detail"] as Record<string, unknown> | undefined;
    for (const key of ["code", "message", "retryable"] as const) {
      if (detail?.[key] !== expected.error?.[key]) {
        mismatches.push(`error.${key} ${JSON.stringify(detail?.[key])} != ${JSON.stringify(expected.error?.[key])}`);
      }
    }
  }
  return {
    mode,
    case_id: expected.case_id,
    http_status: actual.status,
    expected_http_status: expected.http_status,
    exact_semantic_match: mismatches.length === 0,
    mismatches,
    numeric,
    input_tokens: inputTokens,
    expected_input_tokens: expected.usage?.inputTokens ?? null,
  };
}

function summarize(records: readonly CaseRecord[], mode: Mode) {
  const own = records.filter((record) => record.mode === mode);
  const numeric = emptyStats();
  for (const record of own) mergeStats(numeric, record.numeric);
  return {
    requests: own.length,
    exact_semantic_matches: own.filter((record) => record.exact_semantic_match).length,
    mismatches: own.filter((record) => !record.exact_semantic_match).length,
    numeric: {
      compared_values: numeric.compared,
      exact_zero_diffs: numeric.exactZero,
      synthetic_zero_diffs: numeric.syntheticZero,
      over_tolerance_diffs: numeric.overTolerance,
      max_raw_abs_diff: numeric.maxAbsDiff,
    },
  };
}

const MODES: readonly Mode[] = ["singleton", "repeated", "batch", "concurrent-group", "concurrent-all"];

function versions(): Record<string, string> {
  const read = (name: string): string => {
    try {
      return (JSON.parse(readFileSync(join(ROOT, "node_modules", name, "package.json"), "utf8")) as { version: string }).version;
    } catch {
      return "unknown";
    }
  };
  return {
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    "onnxruntime-node": read("onnxruntime-node"),
    "@huggingface/tokenizers": read("@huggingface/tokenizers"),
  };
}

function markdown(report: ReturnType<typeof buildReport>): string {
  const lines = [
    `# Node ONNX vs PyTorch parity: ${report.model_id} / ${report.corpus_id}`,
    "",
    `- Repository: \`${report.repository}\` at \`${report.revision}\``,
    `- Corpus sha256: \`${report.corpus_sha256}\``,
    `- Runtime: ${Object.entries(report.runtime).map(([k, v]) => `${k} ${v}`).join(", ")}`,
    `- Generated: ${report.generated_at}`,
    `- Numeric tolerance (absolute, gating): ${report.tolerance}; differences below it are counted as synthetic zero (diagnostic only)`,
    `- Overall: **${report.passed ? "PASS" : "FAIL"}**`,
    "",
    "| Mode | Requests | Exact semantic matches | Mismatches | Numeric values | Exact zero | Synthetic zero (<1e-4) | Over tolerance | Max raw abs diff |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ];
  for (const [mode, s] of Object.entries(report.modes)) {
    lines.push(
      `| ${mode} | ${s.requests} | ${s.exact_semantic_matches} | ${s.mismatches} | ${s.numeric.compared_values} | ` +
        `${s.numeric.exact_zero_diffs} | ${s.numeric.synthetic_zero_diffs} | ${s.numeric.over_tolerance_diffs} | ` +
        `${s.numeric.max_raw_abs_diff.toExponential(3)} |`,
    );
  }
  lines.push("", "## Dynamic batching evidence", "", "| Mode | Requests | Backend batches | Submitted requests |", "| --- | ---: | ---: | ---: |");
  for (const b of report.batching) lines.push(`| ${b.mode} | ${b.requests} | ${b.batches} | ${b.submitted} |`);
  lines.push(
    "",
    "## Overall",
    "",
    `- Requests compared: ${report.overall.requests}; exact semantic matches: ${report.overall.exact_semantic_matches}`,
    `- Max raw absolute numeric difference: ${report.overall.max_raw_abs_diff.toExponential(3)}`,
    `- Repeat runs byte-identical: ${report.repeat_identical}`,
  );
  const failures = report.records.filter((record) => !record.exact_semantic_match);
  if (failures.length > 0) {
    lines.push("", "## Mismatches", "");
    for (const record of failures) lines.push(`- ${record.mode} ${record.case_id}: ${record.mismatches.join("; ")}`);
  }
  lines.push("", "## Per-case maximum raw numeric difference", "", "| Mode | Case | Max raw abs diff |", "| --- | --- | ---: |");
  for (const record of report.records.filter((r) => r.mode === "singleton")) {
    lines.push(`| ${record.mode} | ${record.case_id} | ${record.numeric.maxAbsDiff.toExponential(3)} |`);
  }
  return `${lines.join("\n")}\n`;
}

function buildReport(
  golden: Golden,
  records: CaseRecord[],
  batching: BatchingEvidence[],
  repeatIdentical: boolean,
) {
  const modes = Object.fromEntries(MODES.map((mode) => [mode, summarize(records, mode)])) as Record<
    Mode,
    ReturnType<typeof summarize>
  >;
  const overall = emptyStats();
  for (const record of records) mergeStats(overall, record.numeric);
  const passed = records.every((record) => record.exact_semantic_match) && repeatIdentical;
  return {
    format_version: 1,
    kind: "gliner-runner-node-vs-pytorch-parity",
    model_id: golden.model_id,
    repository: golden.repository,
    revision: golden.revision,
    corpus_id: golden.corpus_id,
    corpus_sha256: golden.corpus_sha256,
    golden_file: golden.path,
    reference: golden.reference,
    runtime: versions(),
    generated_at: new Date().toISOString(),
    tolerance: NUMERIC_TOLERANCE,
    passed,
    repeat_identical: repeatIdentical,
    modes,
    overall: {
      requests: records.length,
      exact_semantic_matches: records.filter((record) => record.exact_semantic_match).length,
      max_raw_abs_diff: overall.maxAbsDiff,
      compared_values: overall.compared,
      synthetic_zero_diffs: overall.syntheticZero,
    },
    batching,
    records: records.map((record) => ({
      ...record,
      numeric: {
        compared: record.numeric.compared,
        exactZero: record.numeric.exactZero,
        syntheticZero: record.numeric.syntheticZero,
        overTolerance: record.numeric.overTolerance,
        maxAbsDiff: record.numeric.maxAbsDiff,
      },
    })),
  };
}

const goldens = discover();
const skipReason =
  MODELS_DIR === undefined
    ? "GLINER_RUNNER_ONNX_MODELS_DIR is not set"
    : goldens.length === 0
      ? `no golden outputs with installed artifacts in ${PARITY_DIR}`
      : false;

describe("node ONNX runner parity against PyTorch goldens", { skip: skipReason }, () => {
  const byModel = new Map<string, Golden[]>();
  for (const golden of goldens) byModel.set(golden.model_id, [...(byModel.get(golden.model_id) ?? []), golden]);

  for (const [modelId, modelGoldens] of byModel) {
    describe(modelId, () => {
      let server: RunnerServer;
      let base = "";

      before(async () => {
        const settings: Settings = {
          ...defaultSettings(),
          host: "127.0.0.1",
          port: 0,
          modelsDir: MODELS_DIR as string,
          batchWindowMs: 25,
          maxBatchSize: 16,
        };
        const components = await createComponents(settings);
        server = createServer({ ...components, settings, log: (message) => process.stderr.write(`${message}\n`) });
        base = `http://127.0.0.1:${(await server.listen()).port}`;
      });
      after(async () => {
        await server?.close();
      });

      for (const golden of modelGoldens) {
        test(`${golden.model_id} ${golden.corpus_id}`, { timeout: 30 * 60 * 1000 }, async () => {
          const corpusPath = join(CORPUS_DIR, `corpus-${golden.corpus_id}.json`);
          const corpusBytes = readFileSync(corpusPath);
          assert.equal(createHash("sha256").update(corpusBytes).digest("hex"), golden.corpus_sha256, "corpus digest");
          const corpus = JSON.parse(corpusBytes.toString("utf8")) as { cases: CorpusCase[] };
          const cases = new Map(corpus.cases.map((entry) => [entry.case_id, entry]));
          const expectedByCase = new Map(golden.singleton.map((entry) => [entry.case_id, entry]));
          const request = (caseId: string): Record<string, unknown> => ({
            ...(cases.get(caseId) as CorpusCase).request,
            model: golden.model_id,
            backend: "onnx",
            precision: "fp32",
          });
          const expectedOutput = (caseId: string): OrderedJson | undefined => golden.singletonOutputs.get(caseId);

          const records: CaseRecord[] = [];
          const batching: BatchingEvidence[] = [];
          let repeatIdentical = true;
          const infer = (caseId: string): Promise<HttpResult> => post(base, "/v1/infer", request(caseId));
          const outputText = (text: string): string => {
            const root = parseOrdered(text);
            return root instanceof OrderedObject && root.get("output") !== undefined
              ? serializeOrdered(root.get("output") as OrderedJson)
              : text;
          };
          const record = (mode: Mode, caseId: string, result: HttpResult): void => {
            records.push(
              check(mode, expectedByCase.get(caseId) as GoldenCase, expectedOutput(caseId), result),
            );
          };

          // Singleton: one request at a time, in corpus order.
          const firstOutputs = new Map<string, string>();
          for (const entry of golden.singleton) {
            const result = await infer(entry.case_id);
            record("singleton", entry.case_id, result);
            firstOutputs.set(entry.case_id, outputText(result.text));
          }

          // Repeated: replay each case; every run must equal the golden and be byte-identical to the first run.
          for (const entry of golden.singleton) {
            for (let run = 0; run < Math.max(2, entry.repeats ?? 2); run += 1) {
              const result = await infer(entry.case_id);
              record("repeated", entry.case_id, result);
              if (outputText(result.text) !== firstOutputs.get(entry.case_id)) repeatIdentical = false;
            }
          }

          // Batch: /v1/batch with the golden batch groups.
          for (const group of golden.batches) {
            if (group.results.some((entry) => entry.status !== "success")) continue;
            const before = [await metric(base, "batches_total"), await metric(base, "requests_submitted_total")];
            const result = await post(base, "/v1/batch", { requests: group.case_ids.map(request) });
            assert.equal(result.status, 200, result.text);
            const root = parseOrdered(result.text) as OrderedObject;
            const responses = root.get("responses") as OrderedObject[];
            const plain = (JSON.parse(result.text) as { responses: Array<Record<string, unknown>> }).responses;
            assert.equal(responses.length, group.case_ids.length);
            group.results.forEach((expected, index) => {
              const caseId = group.case_ids[index] as string;
              assert.equal(expected.case_id, caseId);
              const response = responses[index] as OrderedObject;
              records.push(
                check(
                  "batch",
                  expected,
                  golden.batchOutputs.get(`${group.batch_group}/${caseId}`),
                  { status: 200, body: plain[index] as Record<string, unknown>, ordered: response },
                ),
              );
            });
            batching.push({
              mode: "batch",
              requests: group.case_ids.length,
              batches: (await metric(base, "batches_total")) - (before[0] as number),
              submitted: (await metric(base, "requests_submitted_total")) - (before[1] as number),
            });
          }

          // Concurrent /v1/infer per batch group, then every case at once: dynamic batching.
          for (const group of golden.batches) {
            const before = [await metric(base, "batches_total"), await metric(base, "requests_submitted_total")];
            const results = await Promise.all(group.case_ids.map((caseId) => infer(caseId)));
            group.case_ids.forEach((caseId, index) => record("concurrent-group", caseId, results[index] as HttpResult));
            batching.push({
              mode: "concurrent-group",
              requests: group.case_ids.length,
              batches: (await metric(base, "batches_total")) - (before[0] as number),
              submitted: (await metric(base, "requests_submitted_total")) - (before[1] as number),
            });
          }
          {
            const ids = golden.singleton.map((entry) => entry.case_id);
            const before = [await metric(base, "batches_total"), await metric(base, "requests_submitted_total")];
            const results = await Promise.all(ids.map((caseId) => infer(caseId)));
            ids.forEach((caseId, index) => record("concurrent-all", caseId, results[index] as HttpResult));
            batching.push({
              mode: "concurrent-all",
              requests: ids.length,
              batches: (await metric(base, "batches_total")) - (before[0] as number),
              submitted: (await metric(base, "requests_submitted_total")) - (before[1] as number),
            });
          }

          const report = buildReport(golden, records, batching, repeatIdentical);
          mkdirSync(REPORT_DIR, { recursive: true });
          const stem = join(REPORT_DIR, `node-vs-pytorch-${golden.model_id}-${golden.corpus_id}`);
          writeFileSync(`${stem}.json`, `${JSON.stringify(report, null, 2)}\n`);
          writeFileSync(`${stem}.md`, markdown(report));

          const failures = records.filter((entry) => !entry.exact_semantic_match);
          assert.deepEqual(
            failures.map((entry) => `${entry.mode} ${entry.case_id}: ${entry.mismatches.join("; ")}`),
            [],
          );
          assert.ok(repeatIdentical, "repeated runs must be byte-identical");
          const grouped = batching.filter((entry) => entry.mode === "concurrent-group" && entry.requests > 1);
          assert.ok(
            grouped.every((entry) => entry.batches < entry.submitted),
            `concurrent requests of one schema must share a backend batch: ${JSON.stringify(grouped)}`,
          );
        });
      }
    });
  }
});
