import { lstat } from "node:fs/promises";
import { parseArgs } from "node:util";

import { KNOWN_MODELS, type KnownModel } from "./catalog.js";
import { GlinerRunnerError } from "./errors.js";
import { OnnxArtifactCatalog, artifactDirectory } from "./onnx/artifacts.js";
import { HuggingFaceClient, huggingFaceOptionsFromEnv } from "./onnx/huggingface.js";
import { OnnxModelProvisioner } from "./onnx/install.js";
import { readProcessingRecord } from "./onnx/processing-record.js";
import { loadTemplate, resolveTemplatesDir } from "./onnx/templates.js";
import { defaultModelsDir } from "./settings.js";

export const MODELS_USAGE = `Usage: gliner-runner models <command> [options]

Commands:
  install <model-id>   Prepare the ONNX artifact. A verified local snapshot of the pinned revision is used
                       when one is found (no network); otherwise the pinned Hugging Face files are
                       downloaded. Progress goes to stderr; stdout gets one JSON line
                       {model_id, path, revision}. Downloads need approval: pass --yes, or answer the
                       y/N prompt when stdin is a terminal. Without a terminal and without --yes the
                       command refuses to download.
  list                 Print installed revisions, template availability and processing status as JSON.
  prepare <model-id>   Build the artifact from a local snapshot (offline): --snapshot <dir>, or the first
                       verified snapshot found in the standard locations. Snapshot files are verified
                       by size and SHA-256 first.
  scan                 List every snapshot candidate found for the curated models (location, origin,
                       verified or why not) and the processing status. --model <id> limits it to one
                       model; --json prints JSON. Nothing is written into the snapshot locations.

Snapshot locations, in precedence order: --snapshot; GLINER_RUNNER_SNAPSHOT_DIRS (path list);
HF_HUB_CACHE; $HF_HOME/hub; $XDG_CACHE_HOME (or ~/.cache)/huggingface/hub; the Python runner's provider
store (<user cache>/gliner-runner/models/providers/huggingface); <models-dir>/providers/huggingface.
Every attempt updates <models-dir>/onnx/<owner>--<name>/<revision>.gliner-runner.json.

Options:
  --models-dir <path>      Model root (default: GLINER_RUNNER_MODELS_DIR, else the user cache)
  --templates-dir <path>   Template directory (default: GLINER_RUNNER_ONNX_TEMPLATES_DIR, else bundled)
  --revision <sha>         Revision (default: the pinned revision; only pinned revisions have templates)
  --snapshot <dir>         prepare: directory holding the snapshot files
  --model <id>             scan: only this model
  --json                   scan: print JSON instead of text
  --yes, -y                install: approve the download without prompting
  --help, -h               Show this help

Environment: HF_ENDPOINT (default https://huggingface.co) and HF_TOKEN (optional bearer token).
`;

export interface ModelsIo {
  env: NodeJS.ProcessEnv;
  /** Whether stdin is an interactive terminal. */
  interactive: boolean;
  stdout(text: string): void;
  stderr(text: string): void;
  /** Asks a yes/no question on the terminal; only called when `interactive`. */
  confirm(question: string): Promise<boolean>;
  signal?: AbortSignal | undefined;
}

class UsageError extends Error {}

const GIB = 1024 ** 3;

/** Runs `gliner-runner-onnx models <argv>`; returns the process exit code (0 ok, 1 failure/denied, 2 usage). */
export async function runModelsCommand(argv: readonly string[], io: ModelsIo): Promise<number> {
  const [command, ...rest] = argv;
  if (command === undefined || command === "--help" || command === "-h" || command === "help") {
    io.stdout(MODELS_USAGE);
    return command === undefined ? 2 : 0;
  }
  try {
    switch (command) {
      case "install":
        return await install(rest, io);
      case "list":
        return await list(rest, io);
      case "prepare":
        return await prepare(rest, io);
      case "scan":
        return await scan(rest, io);
      default:
        throw new UsageError(`unknown models command: ${command}`);
    }
  } catch (error) {
    if (error instanceof UsageError) {
      io.stderr(`error: ${error.message}\n\n${MODELS_USAGE}`);
      return 2;
    }
    io.stderr(`error: ${error instanceof Error ? error.message : String(error)}\n`);
    return 1;
  }
}

interface Parsed {
  positionals: string[];
  values: { [name: string]: string | boolean | undefined };
  modelsDir: string;
  templatesDir: string;
}

function parse(
  argv: readonly string[],
  io: ModelsIo,
  wanted: { snapshot?: boolean; yes?: boolean; scan?: boolean },
): Parsed {
  let result;
  try {
    result = parseArgs({
      args: [...argv],
      allowPositionals: true,
      strict: true,
      options: {
        "models-dir": { type: "string" },
        "templates-dir": { type: "string" },
        revision: { type: "string" },
        ...(wanted.snapshot ? { snapshot: { type: "string" } } : {}),
        ...(wanted.yes ? { yes: { type: "boolean", short: "y" } } : {}),
        ...(wanted.scan ? { model: { type: "string" }, json: { type: "boolean" } } : {}),
        help: { type: "boolean", short: "h" },
      },
    });
  } catch (error) {
    throw new UsageError(error instanceof Error ? error.message : String(error));
  }
  const modelsDir = result.values["models-dir"] ?? io.env["GLINER_RUNNER_MODELS_DIR"];
  return {
    positionals: result.positionals,
    values: result.values,
    modelsDir: modelsDir === undefined || modelsDir === "" ? defaultModelsDir(io.env) : modelsDir,
    templatesDir: resolveTemplatesDir(result.values["templates-dir"], io.env),
  };
}

function provisionerFor(parsed: Parsed, io: ModelsIo): OnnxModelProvisioner {
  return new OnnxModelProvisioner({
    modelsDir: parsed.modelsDir,
    templatesDir: parsed.templatesDir,
    huggingface: new HuggingFaceClient(huggingFaceOptionsFromEnv(io.env)),
    env: io.env,
  });
}

function modelArgument(parsed: Parsed, provisioner: OnnxModelProvisioner): { id: string; revision: string; repository: string } {
  const [reference, ...extra] = parsed.positionals;
  if (reference === undefined) {
    throw new UsageError("a model id is required");
  }
  if (extra.length > 0) {
    throw new UsageError(`unexpected argument: ${extra[0]}`);
  }
  const model = provisioner.model(reference);
  if (model === undefined) {
    throw new UsageError(
      `unknown GLiNER2.5 model '${reference}'; known models: ${KNOWN_MODELS.map((known) => known.modelId).join(", ")}`,
    );
  }
  const revision = parsed.values["revision"];
  return {
    id: model.modelId,
    repository: model.repository,
    revision: typeof revision === "string" ? revision : model.pinnedRevision,
  };
}

const line = (value: unknown): string => `${JSON.stringify(value)}\n`;

async function exists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if ((error as { code?: unknown }).code === "ENOENT") {
      return false;
    }
    throw error;
  }
}

async function install(argv: readonly string[], io: ModelsIo): Promise<number> {
  const parsed = parse(argv, io, { yes: true });
  if (parsed.values["help"] === true) {
    io.stdout(MODELS_USAGE);
    return 0;
  }
  const provisioner = provisionerFor(parsed, io);
  const model = modelArgument(parsed, provisioner);
  if (!(await provisioner.canInstall(model.id, model.revision))) {
    throw new Error(
      `no ONNX template for ${model.repository}@${model.revision} in ${provisioner.templatesDir}; ` +
        "only pinned revisions with a template can be installed",
    );
  }

  // Approval is only for downloading: an existing artifact or a verified local snapshot needs no network.
  const present = await exists(artifactDirectory(parsed.modelsDir, model.repository, model.revision));
  const known = provisioner.model(model.id) as KnownModel;
  const template = await loadTemplate(provisioner.templatesDir, known, model.revision);
  const local = present || template === undefined ? undefined : (await provisioner.discover(template, { signal: io.signal })).selected;
  if (!present && local === undefined && parsed.values["yes"] !== true) {
    const download = template?.sourceFiles.reduce((sum, file) => sum + file.size, 0) ?? 0;
    const onDisk = template?.outputs.reduce((sum, file) => sum + file.size, 0) ?? 0;
    const what =
      `download ${model.id}@${model.revision.slice(0, 12)} (${(download / GIB).toFixed(2)} GiB) from ` +
      `${provisioner.huggingFaceEndpoint} and prepare it (${(onDisk / GIB).toFixed(2)} GiB) in ${parsed.modelsDir}`;
    if (!io.interactive) {
      io.stderr(`error: refusing to ${what} without approval; stdin is not a terminal, pass --yes to approve\n`);
      return 1;
    }
    if (!(await io.confirm(`${what[0]?.toUpperCase()}${what.slice(1)}? [y/N] `))) {
      io.stderr("error: download not approved\n");
      return 1;
    }
  }

  const path = await provisioner.install(model.id, model.revision, {
    signal: io.signal ?? new AbortController().signal,
    log: (message) => io.stderr(`${message}\n`),
  });
  io.stdout(line({ model_id: model.id, path, revision: model.revision }));
  return 0;
}

async function prepare(argv: readonly string[], io: ModelsIo): Promise<number> {
  const parsed = parse(argv, io, { snapshot: true });
  if (parsed.values["help"] === true) {
    io.stdout(MODELS_USAGE);
    return 0;
  }
  const snapshot = parsed.values["snapshot"];
  const provisioner = provisionerFor(parsed, io);
  const model = modelArgument(parsed, provisioner);
  const path = await provisioner.prepare(model.id, {
    snapshot: typeof snapshot === "string" && snapshot !== "" ? snapshot : undefined,
    revision: model.revision,
    signal: io.signal,
    log: (message) => io.stderr(`${message}\n`),
  });
  io.stdout(line({ model_id: model.id, path, revision: model.revision }));
  return 0;
}

async function scan(argv: readonly string[], io: ModelsIo): Promise<number> {
  const parsed = parse(argv, io, { scan: true });
  if (parsed.values["help"] === true) {
    io.stdout(MODELS_USAGE);
    return 0;
  }
  if (parsed.positionals.length > 0) {
    throw new UsageError(`unexpected argument: ${parsed.positionals[0]}`);
  }
  const provisioner = provisionerFor(parsed, io);
  const only = parsed.values["model"];
  const selected = KNOWN_MODELS.filter((known) => typeof only !== "string" || provisioner.model(only)?.modelId === known.modelId);
  if (typeof only === "string" && selected.length === 0) {
    throw new UsageError(
      `unknown GLiNER2.5 model '${only}'; known models: ${KNOWN_MODELS.map((known) => known.modelId).join(", ")}`,
    );
  }
  const reports = [];
  for (const known of selected) {
    reports.push(await provisioner.scan(known.modelId, { signal: io.signal, log: (message) => io.stderr(`${message}\n`) }));
  }
  const view = reports.map((report) => ({
    model_id: report.model.modelId,
    repository: report.model.repository,
    revision: report.revision,
    template_available: report.template !== undefined,
    template_error: report.templateError ?? null,
    installed: report.installed,
    processed: report.record?.processed ?? false,
    state: report.record?.state ?? "not_processed",
    selected: report.selected?.location ?? null,
    candidates: report.candidates.map((candidate) => ({
      location: candidate.location,
      origin: candidate.origin,
      layout: candidate.layout,
      revision: candidate.revision ?? null,
      verified: candidate.verified,
      reason: candidate.reason ?? null,
    })),
    record: report.record ?? null,
  }));
  if (parsed.values["json"] === true) {
    io.stdout(`${JSON.stringify({ models_dir: parsed.modelsDir, templates_dir: parsed.templatesDir, models: view }, null, 2)}\n`);
    return 0;
  }
  const lines: string[] = [];
  for (const entry of view) {
    lines.push(`${entry.model_id} (${entry.repository}@${entry.revision.slice(0, 12)})`);
    lines.push(
      `  status: ${entry.state}${entry.processed ? " (processed)" : ""}; artifact ${entry.installed ? "installed" : "not installed"}; ` +
        `template ${entry.template_available ? "available" : (entry.template_error ?? "missing")}`,
    );
    if (entry.candidates.length === 0) {
      lines.push("  no snapshot candidates found");
    }
    for (const candidate of entry.candidates) {
      lines.push(
        `  ${candidate.verified ? "verified" : "rejected"}  ${candidate.location}  [${candidate.origin}]` +
          `${candidate.reason === null ? "" : `  ${candidate.reason}`}`,
      );
    }
    const artifact = entry.record?.artifact;
    if (artifact !== undefined) {
      lines.push(`  artifact: ${artifact.path} (manifest sha256 ${artifact.manifest_sha256})`);
    }
    if (entry.record?.error !== undefined) {
      lines.push(`  last error: ${entry.record.error}`);
    }
  }
  io.stdout(`${lines.join("\n")}\n`);
  return 0;
}

async function list(argv: readonly string[], io: ModelsIo): Promise<number> {
  const parsed = parse(argv, io, {});
  if (parsed.values["help"] === true) {
    io.stdout(MODELS_USAGE);
    return 0;
  }
  if (parsed.positionals.length > 0) {
    throw new UsageError(`unexpected argument: ${parsed.positionals[0]}`);
  }
  const catalog = new OnnxArtifactCatalog(parsed.modelsDir);
  const inventory = await catalog.inventory();
  const models = [];
  for (const model of inventory.models) {
    let template: { available: boolean; error: string | null; download_bytes: number | null } = {
      available: false,
      error: null,
      download_bytes: null,
    };
    try {
      const loaded = await loadTemplate(parsed.templatesDir, model, model.pinnedRevision);
      if (loaded !== undefined) {
        template = {
          available: true,
          error: null,
          download_bytes: loaded.sourceFiles.reduce((sum, file) => sum + file.size, 0),
        };
      }
    } catch (error) {
      template.error = error instanceof GlinerRunnerError || error instanceof Error ? error.message : String(error);
    }
    const record = await readProcessingRecord(parsed.modelsDir, model, model.pinnedRevision);
    models.push({
      model_id: model.modelId,
      repository: model.repository,
      pinned_revision: model.pinnedRevision,
      installed: model.localRevisions.includes(model.pinnedRevision),
      installed_revisions: model.localRevisions,
      template_available: template.available,
      template_error: template.error,
      download_bytes: template.download_bytes,
      processed: record?.processed ?? false,
      state: record?.state ?? "not_processed",
      record: record ?? null,
    });
  }
  io.stdout(`${JSON.stringify({ models_dir: parsed.modelsDir, templates_dir: parsed.templatesDir, models }, null, 2)}\n`);
  return 0;
}
