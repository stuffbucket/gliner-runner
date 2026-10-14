import { parseArgs } from "node:util";

import { ZodError } from "zod";

import { BackendRegistry } from "./backend.js";
import { KNOWN_MODELS } from "./catalog.js";
import { loadComponents, type RunnerComponents } from "./components.js";
import { InferenceRequestSchema, type InferenceRequest } from "./contracts.js";
import {
  BackendUnavailableError,
  GlinerRunnerError,
  InvalidSchemaError,
  ModelDownloadRequiredError,
  ModelIntegrityError,
} from "./errors.js";
import { Runtime } from "./runtime.js";
import { defaultModelsDir, resolveSettings, type Settings } from "./settings.js";
import { stringifyWire } from "./wire.js";

export const DEFAULT_MODEL = "decide-340m";
/** Name of the single classification task the CLI builds; it is the key in the native JSON output. */
export const TASK_NAME = "category";

export const CLASSIFY_USAGE = `Usage: gliner-runner classify --labels <a,b,c> [options] "<text>"

Classifies one text into labels you choose. Runs locally; no server is needed.
The text is read from stdin when it is "-" or omitted.

Options:
  --labels <a,b,c>        Labels to choose from (required; comma-separated, may be repeated)
  --model <id>            decide-340m (default), decide-multilingual or decide-1b
  --multi                 Allow several labels (default: exactly one label)
  --threshold <0-1>       Confidence needed to select a label (default 0.5, exclusive bounds)
  --describe <label=text> Describe a label to help the model (repeatable)
  --json                  Print the full native JSON result instead of one line
  --models-dir <path>     Model root (default: GLINER_RUNNER_MODELS_DIR, else the user cache)
  --help, -h              Show this help

Install a model first: gliner-runner models install decide-340m

Examples:
  gliner-runner classify --labels billing,support,shipping "I was charged twice."
  gliner-runner classify --labels billing,support,shipping --multi "Refund me and ship the new one."
  echo "Where is my parcel?" | gliner-runner classify --labels billing,support,shipping
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
  /** Assembles the catalog and backends; defaults to the ONNX implementation (`loadComponents`). */
  components?: ((settings: Settings) => Promise<RunnerComponents>) | undefined;
  signal?: AbortSignal | undefined;
}

class UsageError extends Error {}
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
    modelsDir: values["models-dir"],
  };
  if (base.help) {
    return { ...base, text: undefined, readStdin: false, spec: { labels: [], multi: false, threshold: 0.5, descriptions: {} } };
  }
  if (positionals.length > 1) {
    throw new UsageError(
      `expected one text argument but got ${positionals.length}; put the text in quotes, or pass it on stdin`,
    );
  }
  if (values.labels === undefined) {
    throw new UsageError("--labels is required, for example --labels billing,support,shipping");
  }
  const labels = splitLabels(values.labels);
  const descriptions = parseDescriptions(values.describe ?? []);
  const unknown = Object.keys(descriptions).filter((label) => !labels.includes(label));
  if (unknown.length > 0) {
    throw new UsageError(`--describe names a label that is not in --labels: ${unknown.join(", ")}`);
  }
  const text = positionals[0];
  return {
    ...base,
    text: text === "-" ? undefined : text,
    readStdin: text === undefined || text === "-",
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
      throw new UsageError(describeIssue(error));
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
  return { runtime, modelsDir, modelsDirFlag, modelId };
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
