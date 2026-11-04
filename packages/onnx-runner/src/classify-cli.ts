import { StringDecoder } from "node:string_decoder";
import { parseArgs } from "node:util";

import { z, ZodError } from "zod";

import { BackendRegistry } from "./backend.js";
import { KNOWN_MODELS } from "./catalog.js";
import type { ArtifactCatalog } from "./catalog.js";
import { loadComponents, type RunnerComponents } from "./components.js";
import { InferenceRequestSchema, type InferenceRequest } from "./contracts.js";
import {
  BackendUnavailableError,
  GlinerRunnerError,
  InvalidSchemaError,
  ModelDownloadRequiredError,
  ModelIntegrityError,
  domainHttpError,
  type ErrorDetail,
} from "./errors.js";
import { Runtime } from "./runtime.js";
import { defaultModelsDir, resolveSettings, type Settings } from "./settings.js";
import { stringifyWire } from "./wire.js";

export const DEFAULT_MODEL = "decide-340m";
/** Name of the single classification task the CLI builds; it is the key in the native JSON output. */
export const TASK_NAME = "category";

export const CLASSIFY_USAGE = `Usage: gliner-runner classify --labels <a,b,c> [options] "<text>"
       gliner-runner classify --jsonl [--labels <a,b,c>] [options] < requests.jsonl

Classifies one text into labels you choose. Runs locally; no server is needed.
The text is read from stdin when it is "-" or omitted.

Options:
  --labels <a,b,c>        Labels to choose from (comma-separated, may be repeated; required
                          unless every --jsonl line gives its own "labels")
  --model <id>            decide-340m (default), decide-multilingual or decide-1b
  --multi                 Allow several labels (default: exactly one label)
  --threshold <0-1>       Confidence needed to select a label (default 0.5, exclusive bounds)
  --describe <label=text> Describe a label to help the model (repeatable)
  --json                  Print the full native JSON result instead of one line
  --jsonl                 Classify many texts with one model load: read one JSON request per
                          line on stdin, write one JSON result per line on stdout (see below)
  --models-dir <path>     Model root (default: GLINER_RUNNER_MODELS_DIR, else the user cache)
  --help, -h              Show this help

Install a model first: gliner-runner models install decide-340m

Examples:
  gliner-runner classify --labels billing,support,shipping "I was charged twice."
  gliner-runner classify --labels billing,support,shipping --multi "Refund me and ship the new one."
  echo "Where is my parcel?" | gliner-runner classify --labels billing,support,shipping

Many texts (--jsonl):
  Loading the model takes a few seconds; --jsonl pays that once for all the lines, so use it
  whenever a script or program classifies more than a handful of texts. Each input line is
  a JSON object; only "text" is required, the other fields default to the options:
    {"id": "t1", "text": "...", "labels": ["a", "b"], "multi_label": true, "threshold": 0.4,
     "label_descriptions": {"a": "what a means"}}
  "labels" replaces --labels for that line (--describe entries for its labels still apply).
  Each result is written as soon as it is ready, in input order, one line per request:
    {"id":"t1","line":1,"output":{...the --json result...}}
    {"id":"t2","line":2,"detail":{"code":"invalid_request","message":"...","retryable":false}}
  "id" is copied from the request when given; "line" is the input line number. Blank lines
  are skipped. A failed line gets a "detail" error (the HTTP API's error envelope) and the
  rest continue. The command ends when stdin ends: exit code 0 when every line succeeded, 1
  when any failed. Problems with the options or the model are reported on stderr before any
  line is read (exit 2 or 1). Ctrl-C (SIGINT) or SIGTERM stops it at once.

  printf '%s\\n' '{"text": "I was charged twice."}' '{"text": "Where is my parcel?"}' |
    gliner-runner classify --jsonl --labels billing,support,shipping
`;

export const DEMO_USAGE = `Usage: gliner-runner demo [--model <id>] [--models-dir <path>]

Runs a few example classifications with an installed model and prints the results.

Options:
  --model <id>            decide-340m (default), decide-multilingual or decide-1b.
                          decide-multilingual also classifies Spanish, French and German examples.
  --models-dir <path>     Model root (default: GLINER_RUNNER_MODELS_DIR, else the user cache)
  --help, -h              Show this help

Install a model first: gliner-runner models install decide-340m
`;

export interface ClassifyIo {
  env: NodeJS.ProcessEnv;
  stdout(text: string): void;
  stderr(text: string): void;
  /** Whether stdin is an interactive terminal (then there is no piped text to read). */
  stdinIsTerminal: boolean;
  readStdin(): Promise<string>;
  /** Reads stdin line by line for `--jsonl`; defaults to splitting `readStdin()` once it has all been read. */
  readLines?: (() => AsyncIterable<string>) | undefined;
  /** Assembles the catalog and backends; defaults to the ONNX implementation (`loadComponents`). */
  components?: ((settings: Settings) => Promise<RunnerComponents>) | undefined;
  signal?: AbortSignal | undefined;
}

class UsageError extends Error {}
/** A request whose classification schema is invalid; `--jsonl` reports it as `invalid_schema`. */
class SchemaUsageError extends UsageError {}
/** A failure the user can act on; printed as one line, exit code 1. */
class RunError extends Error {}

export interface ClassifySpec {
  labels: string[];
  multi: boolean;
  threshold: number;
  descriptions: Record<string, string>;
}

export interface ClassificationView {
  /** One label (single-label task), the selected labels (multi-label task) or `null` for an empty single result. */
  value: string | string[] | null;
  confidence: number | null;
  probabilities: Record<string, number>;
}

// --- argument parsing --------------------------------------------------------------------------------------------

function splitLabels(values: readonly string[]): string[] {
  const labels: string[] = [];
  for (const raw of values) {
    for (const piece of raw.split(",")) {
      const label = piece.trim();
      if (label === "") {
        throw new UsageError(`--labels has an empty label in '${raw}'; use --labels billing,support,shipping`);
      }
      labels.push(label);
    }
  }
  return labels;
}

function parseDescriptions(values: readonly string[]): Record<string, string> {
  const descriptions: Record<string, string> = {};
  for (const raw of values) {
    const separator = raw.indexOf("=");
    const label = separator < 0 ? "" : raw.slice(0, separator).trim();
    const text = separator < 0 ? "" : raw.slice(separator + 1).trim();
    if (label === "" || text === "") {
      throw new UsageError(`--describe expects label=description, got '${raw}'`);
    }
    if (Object.hasOwn(descriptions, label)) {
      throw new UsageError(`--describe given twice for label '${label}'`);
    }
    descriptions[label] = text;
  }
  return descriptions;
}

function parseThreshold(raw: string | undefined): number {
  if (raw === undefined) {
    return 0.5;
  }
  const value = raw.trim() === "" ? Number.NaN : Number(raw);
  if (!(value > 0 && value < 1)) {
    throw new UsageError(`--threshold must be a number greater than 0 and less than 1, got '${raw}'`);
  }
  return value;
}

export interface ParsedClassify {
  help: boolean;
  model: string;
  text: string | undefined;
  readStdin: boolean;
  json: boolean;
  /** Read one JSON request per stdin line and write one JSON result per line. */
  jsonl: boolean;
  modelsDir: string | undefined;
  spec: ClassifySpec;
}

export function parseClassifyArguments(argv: readonly string[]): ParsedClassify {
  let parsed;
  try {
    parsed = parseArgs({
      args: [...argv],
      allowPositionals: true,
      strict: true,
      options: {
        labels: { type: "string", multiple: true },
        model: { type: "string" },
        multi: { type: "boolean" },
        threshold: { type: "string" },
        describe: { type: "string", multiple: true },
        json: { type: "boolean" },
        jsonl: { type: "boolean" },
        "models-dir": { type: "string" },
        help: { type: "boolean", short: "h" },
      },
    });
  } catch (error) {
    throw new UsageError(error instanceof Error ? error.message : String(error));
  }
  const { values, positionals } = parsed;
  const base = {
    help: values.help === true,
    model: values.model ?? DEFAULT_MODEL,
    json: values.json === true,
    jsonl: values.jsonl === true,
    modelsDir: values["models-dir"],
  };
  if (base.help) {
    return { ...base, text: undefined, readStdin: false, spec: { labels: [], multi: false, threshold: 0.5, descriptions: {} } };
  }
  if (base.jsonl) {
    if (positionals.length > 0) {
      throw new UsageError("--jsonl reads the texts from stdin, one JSON request per line; do not pass a text argument");
    }
    if (base.json) {
      throw new UsageError("--jsonl already writes JSON; leave out --json");
    }
  }
  if (positionals.length > 1) {
    throw new UsageError(
      `expected one text argument but got ${positionals.length}; put the text in quotes, or pass it on stdin`,
    );
  }
  if (values.labels === undefined && !base.jsonl) {
    throw new UsageError("--labels is required, for example --labels billing,support,shipping");
  }
  // With --jsonl the labels may come from each line instead.
  const labels = splitLabels(values.labels ?? []);
  const descriptions = parseDescriptions(values.describe ?? []);
  const unknown = values.labels === undefined ? [] : Object.keys(descriptions).filter((label) => !labels.includes(label));
  if (unknown.length > 0) {
    throw new UsageError(`--describe names a label that is not in --labels: ${unknown.join(", ")}`);
  }
  const text = positionals[0];
  return {
    ...base,
    text: text === "-" ? undefined : text,
    readStdin: !base.jsonl && (text === undefined || text === "-"),
    spec: { labels, multi: values.multi === true, threshold: parseThreshold(values.threshold), descriptions },
  };
}

// --- requests and results ----------------------------------------------------------------------------------------

/** The wire request for one classification; validation errors surface as `UsageError` one-liners. */
export function buildRequest(model: string, text: string, spec: ClassifySpec): InferenceRequest {
  const descriptions = Object.keys(spec.descriptions).length === 0 ? undefined : spec.descriptions;
  try {
    return InferenceRequestSchema.parse({
      model,
      backend: "onnx",
      precision: "fp32",
      operation: "classify",
      text,
      schema: {
        kind: "classification",
        tasks: {
          [TASK_NAME]: {
            labels: spec.labels,
            ...(descriptions === undefined ? {} : { label_descriptions: descriptions }),
            // A multi-label task may select none; the default single-label task always selects exactly one.
            ...(spec.multi ? { min_labels: 0, max_labels: null } : {}),
            threshold: spec.threshold,
          },
        },
      },
      options: { threshold: spec.threshold },
    });
  } catch (error) {
    if (error instanceof ZodError) {
      const message = describeIssue(error);
      throw error.issues[0]?.path[0] === "schema" ? new SchemaUsageError(message) : new UsageError(message);
    }
    throw error;
  }
}

function describeIssue(error: ZodError): string {
  const issue = error.issues[0];
  if (issue === undefined) {
    return "invalid request";
  }
  const path = issue.path.map(String);
  if (path[0] === "text") {
    return "the text must not be empty";
  }
  if (path[0] === "schema") {
    return `invalid schema: ${issue.message}`;
  }
  return `invalid request: ${path.join(".")}: ${issue.message}`;
}

/** Reads the CLI's task out of a native `output` value; `undefined` when the shape is not the native one. */
export function readClassification(output: unknown): ClassificationView | undefined {
  const plain: unknown = JSON.parse(stringifyWire(output));
  if (typeof plain !== "object" || plain === null) {
    return undefined;
  }
  const task = (plain as Record<string, unknown>)[TASK_NAME];
  if (typeof task !== "object" || task === null) {
    return undefined;
  }
  const { value, confidence, probabilities } = task as Record<string, unknown>;
  const isLabel = (item: unknown): item is string => typeof item === "string";
  if (!(value === null || isLabel(value) || (Array.isArray(value) && value.every(isLabel)))) {
    return undefined;
  }
  if (typeof probabilities !== "object" || probabilities === null) {
    return undefined;
  }
  return {
    value: value as ClassificationView["value"],
    confidence: typeof confidence === "number" ? confidence : null,
    probabilities: probabilities as Record<string, number>,
  };
}

const score = (value: number): string => value.toFixed(3);

/** One readable line: `billing (confidence 0.998)`, or `billing (0.941), shipping (0.812)` for a multi-label task. */
export function formatClassification(view: ClassificationView, spec: Pick<ClassifySpec, "threshold">): string {
  const { value } = view;
  if (Array.isArray(value)) {
    return value.length === 0
      ? `no label reached the ${spec.threshold} threshold`
      : value.map((label) => `${label} (${score(view.probabilities[label] ?? Number.NaN)})`).join(", ");
  }
  if (value === null) {
    return "no label selected";
  }
  return view.confidence === null ? value : `${value} (confidence ${score(view.confidence)})`;
}

// --- running -----------------------------------------------------------------------------------------------------

interface Session {
  runtime: Runtime;
  catalog: ArtifactCatalog;
  modelsDir: string;
  modelsDirFlag: string | undefined;
}

function installCommand(modelId: string, modelsDirFlag: string | undefined): string {
  return `gliner-runner models install ${modelId}${modelsDirFlag === undefined ? "" : ` --models-dir ${modelsDirFlag}`}`;
}

async function openSession(model: string, modelsDirFlag: string | undefined, io: ClassifyIo): Promise<Session & { modelId: string }> {
  const fromEnv = io.env["GLINER_RUNNER_MODELS_DIR"];
  const modelsDir =
    modelsDirFlag !== undefined && modelsDirFlag !== ""
      ? modelsDirFlag
      : fromEnv !== undefined && fromEnv !== ""
        ? fromEnv
        : defaultModelsDir(io.env);
  const settings = resolveSettings({ modelsDir, batchWindowMs: 0 });
  const components = await (io.components ?? ((value) => loadComponents(value)))(settings);
  const modelId = components.catalog.canonicalModelId(model);
  if (modelId === undefined || !KNOWN_MODELS.some((known) => known.modelId === modelId)) {
    throw new UsageError(`unknown model '${model}'; known models: ${KNOWN_MODELS.map((known) => known.modelId).join(", ")}`);
  }
  const runtime = new Runtime({
    catalog: components.catalog,
    backends: new BackendRegistry(components.backends),
    config: {
      device: settings.device,
      queueCapacity: settings.queueCapacity,
      maxBatchSize: settings.maxBatchSize,
      batchWindowMs: settings.batchWindowMs,
      modelIdleTtlSeconds: 0,
      memoryLimitBytes: settings.memoryLimitBytes,
    },
  });
  return { runtime, catalog: components.catalog, modelsDir, modelsDirFlag, modelId };
}

/** Turns a runner error into the one-line message the user sees. */
function explain(error: unknown, modelId: string, modelsDirFlag: string | undefined): Error {
  if (error instanceof ModelDownloadRequiredError) {
    return new RunError(`model ${modelId} is not installed. Install it with: ${installCommand(modelId, modelsDirFlag)}`);
  }
  if (error instanceof InvalidSchemaError) {
    return new UsageError(`invalid schema: ${error.message}`);
  }
  if (error instanceof ModelIntegrityError) {
    return new RunError(
      `model ${modelId} failed verification (${error.message}). Reinstall it with: ${installCommand(modelId, modelsDirFlag)}`,
    );
  }
  if (error instanceof BackendUnavailableError) {
    return new RunError("this build has no ONNX backend");
  }
  if (error instanceof GlinerRunnerError) {
    return new RunError(error.message);
  }
  return error instanceof Error ? error : new Error(String(error));
}

async function infer(session: Session, request: InferenceRequest, io: ClassifyIo, modelId: string) {
  try {
    return await session.runtime.infer(request, io.signal);
  } catch (error) {
    throw explain(error, modelId, session.modelsDirFlag);
  }
}

function fail(error: unknown, io: ClassifyIo, usage: string): number {
  if (error instanceof UsageError) {
    io.stderr(`error: ${error.message} (see: ${usage})\n`);
    return 2;
  }
  io.stderr(`error: ${error instanceof Error ? error.message : String(error)}\n`);
  return 1;
}

/** Runs `gliner-runner classify <argv>`; returns the exit code (0 ok, 1 failure, 2 usage). */
export async function runClassifyCommand(argv: readonly string[], io: ClassifyIo): Promise<number> {
  const hint = "gliner-runner classify --help";
  let session: (Session & { modelId: string }) | undefined;
  try {
    const parsed = parseClassifyArguments(argv);
    if (parsed.help) {
      io.stdout(CLASSIFY_USAGE);
      return 0;
    }
    if (parsed.jsonl) {
      session = await openSession(parsed.model, parsed.modelsDir, io);
      return await runJsonl(session, parsed.spec, io);
    }
    let text = parsed.text;
    if (parsed.readStdin) {
      if (io.stdinIsTerminal) {
        throw new UsageError("no text given; pass the text as an argument or pipe it on stdin");
      }
      text = (await io.readStdin()).trimEnd();
    }
    // Build the request first so an invalid schema is reported before any model is touched.
    const draft = buildRequest(parsed.model, text ?? "", parsed.spec);
    session = await openSession(parsed.model, parsed.modelsDir, io);
    const request = { ...draft, model: session.modelId };
    const response = await infer(session, request, io, session.modelId);
    if (parsed.json) {
      io.stdout(`${stringifyWire(response.output)}\n`);
      return 0;
    }
    const view = readClassification(response.output);
    if (view === undefined) {
      throw new RunError(`unexpected result from the model: ${stringifyWire(response.output)}`);
    }
    io.stdout(`${formatClassification(view, parsed.spec)}\n`);
    return 0;
  } catch (error) {
    return fail(error, io, hint);
  } finally {
    await session?.runtime.close();
  }
}

// --- many texts: --jsonl -----------------------------------------------------------------------------------------

/** One `--jsonl` input line; every field but `text` defaults to the command-line options. */
const JsonlRequestSchema = z.strictObject({
  id: z.union([z.string(), z.number()], { error: "must be a string or a number" }).optional(),
  text: z.string(),
  labels: z.array(z.string()).min(1).optional(),
  // The names of POST /v1/classify_text and GLiNER2's classify_text.
  multi_label: z.boolean().optional(),
  threshold: z.number().optional(),
  label_descriptions: z.record(z.string(), z.string()).optional(),
});
type JsonlRequest = z.output<typeof JsonlRequestSchema>;

/** The classification a line asks for: its own fields over the command-line defaults. */
function lineSpec(line: JsonlRequest, defaults: ClassifySpec): ClassifySpec {
  const labels = line.labels ?? defaults.labels;
  if (labels.length === 0) {
    throw new UsageError('no labels: give --labels, or a "labels" array on the line');
  }
  const threshold = line.threshold ?? defaults.threshold;
  if (!(threshold > 0 && threshold < 1)) {
    throw new UsageError(`threshold must be greater than 0 and less than 1, got ${threshold}`);
  }
  // --describe entries apply to the line's labels that they name; a line's own label_descriptions must name its labels.
  const descriptions =
    line.label_descriptions ?? Object.fromEntries(Object.entries(defaults.descriptions).filter(([label]) => labels.includes(label)));
  const unknown = Object.keys(descriptions).filter((label) => !labels.includes(label));
  if (unknown.length > 0) {
    throw new UsageError(`label_descriptions name a label that is not in labels: ${unknown.join(", ")}`);
  }
  return { labels, multi: line.multi_label ?? defaults.multi, threshold, descriptions };
}

function describeLineIssue(error: ZodError): string {
  const issue = error.issues[0];
  if (issue === undefined) {
    return "invalid request";
  }
  if (issue.path.length === 0) {
    return issue.code === "unrecognized_keys"
      ? `unknown field${issue.keys.length === 1 ? "" : "s"} ${issue.keys.map((key) => `"${key}"`).join(", ")}; ` +
          "expected id, text, labels, multi_label, threshold, label_descriptions"
      : 'each line must be a JSON object such as {"text": "..."}';
  }
  return `"${issue.path.map(String).join(".")}": ${issue.message}`;
}

/** The error envelope of the HTTP API (`detail` of a failed response) for one failed line. */
function lineErrorDetail(error: unknown): ErrorDetail {
  if (error instanceof SchemaUsageError) {
    return { code: "invalid_schema", message: error.message, retryable: false };
  }
  if (error instanceof UsageError) {
    return { code: "invalid_request", message: error.message, retryable: false };
  }
  const mapped = domainHttpError(error)?.detail;
  if (mapped !== undefined && !Array.isArray(mapped) && typeof mapped === "object") {
    return mapped as ErrorDetail;
  }
  return { code: "internal_error", message: error instanceof Error ? error.message : String(error), retryable: false };
}

function idOf(value: unknown): string | number | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }
  const id = (value as Record<string, unknown>)["id"];
  return typeof id === "string" || typeof id === "number" ? id : undefined;
}

async function* splitLines(read: () => Promise<string>): AsyncGenerator<string> {
  const text = await read();
  const lines = text.split("\n");
  if (lines.at(-1) === "") {
    lines.pop();
  }
  yield* lines;
}

/**
 * Splits a stream into lines on "\n" and nothing else, as JSON Lines does. readline also breaks on a lone "\r" and on
 * U+2028 and U+2029, which may appear unescaped inside a JSON string; a trailing "\r" is left for the JSON parser,
 * which reads it as whitespace. The last line needs no newline.
 */
export async function* jsonlLines(chunks: AsyncIterable<Uint8Array | string> | Iterable<Uint8Array | string>): AsyncGenerator<string> {
  const decoder = new StringDecoder("utf8");
  let pending = "";
  for await (const chunk of chunks) {
    pending += typeof chunk === "string" ? chunk : decoder.write(chunk);
    let start = 0;
    for (let end = pending.indexOf("\n"); end !== -1; end = pending.indexOf("\n", start)) {
      yield pending.slice(start, end);
      start = end + 1;
    }
    pending = pending.slice(start);
  }
  pending += decoder.end();
  if (pending !== "") {
    yield pending;
  }
}

/** Exit code of a `--jsonl` run stopped through `io.signal`, as a shell reports SIGINT. */
const INTERRUPTED = 130;

/** Whether `classify <argv>` runs in `--jsonl` mode (false for help and for arguments that do not parse). */
export function wantsJsonl(argv: readonly string[]): boolean {
  try {
    const parsed = parseClassifyArguments(argv);
    return parsed.jsonl && !parsed.help;
  } catch {
    return false;
  }
}

/** Yields lines until the source ends or `signal` aborts, without waiting for a line that never comes. */
async function* linesUntilAborted(source: AsyncIterable<string>, signal: AbortSignal | undefined): AsyncGenerator<string> {
  const iterator = source[Symbol.asyncIterator]();
  const aborted =
    signal === undefined
      ? undefined
      : new Promise<"aborted">((resolve) => {
          if (signal.aborted) {
            resolve("aborted");
          }
          signal.addEventListener("abort", () => resolve("aborted"), { once: true });
        });
  try {
    for (;;) {
      const next = await (aborted === undefined ? iterator.next() : Promise.race([iterator.next(), aborted]));
      if (next === "aborted" || next.done === true) {
        return;
      }
      yield next.value;
    }
  } finally {
    // Not awaited: a source blocked on a read that never comes must not keep the command from finishing.
    void iterator.return?.().catch(() => undefined);
  }
}

/**
 * `classify --jsonl`: one model load, then one JSON result line per JSON request line, in input order, each written
 * as soon as it is ready. Returns 0 when every line succeeded, 1 when any failed, and 130 when `io.signal` aborted it
 * (it then stops reading and writes no further line).
 */
async function runJsonl(session: Session & { modelId: string }, defaults: ClassifySpec, io: ClassifyIo): Promise<number> {
  const { modelId } = session;
  // A missing or damaged model fails the whole command once, on stderr, rather than every line.
  try {
    await session.catalog.resolve(modelId);
  } catch (error) {
    throw explain(error, modelId, session.modelsDirFlag);
  }
  const source = io.readLines?.() ?? splitLines(io.readStdin);
  let lineNumber = 0;
  let failed = 0;
  for await (const next of linesUntilAborted(source, io.signal)) {
    lineNumber += 1;
    // A byte order mark starts a file saved by some Windows tools; it is not part of the first line's JSON.
    const raw = lineNumber === 1 ? next.replace(/^\uFEFF/, "") : next;
    if (raw.trim() === "") {
      continue;
    }
    let id: string | number | undefined;
    let result: Record<string, unknown>;
    try {
      let value: unknown;
      try {
        value = JSON.parse(raw);
      } catch (error) {
        throw new UsageError(`the line is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
      }
      id = idOf(value);
      const parsed = JsonlRequestSchema.safeParse(value);
      if (!parsed.success) {
        throw new UsageError(describeLineIssue(parsed.error));
      }
      const request = buildRequest(modelId, parsed.data.text, lineSpec(parsed.data, defaults));
      const response = await session.runtime.infer(request, io.signal);
      result = { output: response.output };
    } catch (error) {
      if (io.signal?.aborted) {
        break;
      }
      failed += 1;
      result = { detail: lineErrorDetail(error) };
    }
    if (io.signal?.aborted) {
      break;
    }
    io.stdout(`${stringifyWire({ ...(id === undefined ? {} : { id }), line: lineNumber, ...result })}\n`);
  }
  if (io.signal?.aborted) {
    return INTERRUPTED;
  }
  return failed > 0 ? 1 : 0;
}

// --- demo --------------------------------------------------------------------------------------------------------

export interface DemoCase {
  title: string;
  text: string;
  spec: ClassifySpec;
  /** The label(s) decide-340m selects; the guarded real-model test checks them. */
  expected: readonly string[];
  /** Only shown for models that read the text's language. */
  multilingualOnly?: boolean;
}

const single = (labels: string[], descriptions: Record<string, string> = {}): ClassifySpec => ({
  labels,
  multi: false,
  threshold: 0.5,
  descriptions,
});

const TICKET_LABELS = ["billing", "support", "shipping"];

export const DEMO_CASES: readonly DemoCase[] = [
  {
    title: "Support ticket routing",
    text: "I was charged twice for my last order and I want one of the payments refunded.",
    spec: single(TICKET_LABELS),
    expected: ["billing"],
  },
  {
    title: "Support ticket routing",
    text: "My package has been in transit for two weeks and the tracking page has not changed.",
    spec: single(TICKET_LABELS),
    expected: ["shipping"],
  },
  {
    title: "Several topics in one message (multi-label)",
    text: "The battery is weak and the delivery was slow.",
    spec: { labels: ["battery", "screen", "price", "delivery"], multi: true, threshold: 0.5, descriptions: {} },
    expected: ["battery", "delivery"],
  },
  {
    title: "Labels with descriptions",
    text: "Can I swap the blue jacket I bought for the same one in a larger size?",
    spec: single(["refund", "exchange", "question"], {
      refund: "The customer wants their money back",
      exchange: "The customer wants a different item or size",
      question: "The customer asks for information only",
    }),
    expected: ["exchange"],
  },
  {
    title: "Sentiment",
    text: "The checkout was quick and the staff were friendly and helpful.",
    spec: single(["positive", "neutral", "negative"]),
    expected: ["positive"],
  },
  {
    title: "Priority",
    text: "Our payment system is down and customers cannot check out. We need help right now.",
    spec: single(["urgent", "normal", "low"]),
    expected: ["urgent"],
  },
  {
    title: "Spanish ticket",
    text: "Me cobraron dos veces el mismo pedido y quiero que me devuelvan el dinero.",
    spec: single(TICKET_LABELS),
    expected: ["billing"],
    multilingualOnly: true,
  },
  {
    title: "French ticket",
    text: "Mon colis devait arriver il y a deux semaines et je n'ai toujours rien reçu.",
    spec: single(TICKET_LABELS),
    expected: ["shipping"],
    multilingualOnly: true,
  },
  {
    title: "German ticket",
    text: "Die App stürzt jedes Mal ab, sobald ich sie öffne.",
    spec: single(TICKET_LABELS),
    expected: ["support"],
    multilingualOnly: true,
  },
];

export function demoCasesFor(modelId: string): DemoCase[] {
  return DEMO_CASES.filter((demo) => demo.multilingualOnly !== true || modelId === "decide-multilingual");
}

export function selectedLabels(view: ClassificationView): string[] {
  return Array.isArray(view.value) ? view.value : view.value === null ? [] : [view.value];
}

/** Runs `gliner-runner demo <argv>`; returns the exit code (0 ok, 1 failure, 2 usage). */
export async function runDemoCommand(argv: readonly string[], io: ClassifyIo): Promise<number> {
  const hint = "gliner-runner demo --help";
  let session: (Session & { modelId: string }) | undefined;
  try {
    let parsed;
    try {
      parsed = parseArgs({
        args: [...argv],
        allowPositionals: true,
        strict: true,
        options: { model: { type: "string" }, "models-dir": { type: "string" }, help: { type: "boolean", short: "h" } },
      });
    } catch (error) {
      throw new UsageError(error instanceof Error ? error.message : String(error));
    }
    if (parsed.values.help === true) {
      io.stdout(DEMO_USAGE);
      return 0;
    }
    if (parsed.positionals.length > 0) {
      throw new UsageError(`unexpected argument: ${parsed.positionals[0]}`);
    }
    session = await openSession(parsed.values.model ?? DEFAULT_MODEL, parsed.values["models-dir"], io);
    const { modelId } = session;
    const cases = demoCasesFor(modelId);
    const lines = [`gliner-runner demo (model: ${modelId})`, ""];
    const started = Date.now();
    for (const [index, demo] of cases.entries()) {
      const response = await infer(session, buildRequest(modelId, demo.text, demo.spec), io, modelId);
      const view = readClassification(response.output);
      if (view === undefined) {
        throw new RunError(`unexpected result from the model: ${stringifyWire(response.output)}`);
      }
      lines.push(
        `${index + 1}. ${demo.title}`,
        `   Text:    ${demo.text}`,
        `   Labels:  ${demo.spec.labels.join(", ")}${demo.spec.multi ? "  (several allowed)" : ""}`,
        ...Object.entries(demo.spec.descriptions).map(([label, text]) => `            ${label}: ${text}`),
        `   Result:  ${formatClassification(view, demo.spec)}`,
        "",
      );
    }
    lines.push(`Classified ${cases.length} texts in ${((Date.now() - started) / 1000).toFixed(1)} s.`);
    lines.push('Try your own: gliner-runner classify --labels billing,support,shipping "your text"');
    io.stdout(`${lines.join("\n")}\n`);
    return 0;
  } catch (error) {
    return fail(error, io, hint);
  } finally {
    await session?.runtime.close();
  }
}
