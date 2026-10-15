import { lstat } from "node:fs/promises";
import { dirname } from "node:path";
import { parseArgs } from "node:util";

import { KNOWN_MODELS, type KnownModel } from "./catalog.js";
import { GlinerRunnerError } from "./errors.js";
import { OnnxArtifactCatalog, artifactDirectory } from "./onnx/artifacts.js";
import { FREE_SPACE_MARGIN_BYTES, freeBytesOf, outputBytes } from "./onnx/assemble.js";
import { HuggingFaceClient, huggingFaceOptionsFromEnv } from "./onnx/huggingface.js";
import { OnnxModelProvisioner } from "./onnx/install.js";
import { readProcessingRecord } from "./onnx/processing-record.js";
import { loadTemplate, resolveTemplatesDir, type OnnxTemplate } from "./onnx/templates.js";
import { defaultModelsDir } from "./settings.js";

export const MODELS_USAGE = `Usage: gliner-runner models <command> [options]

Commands:
  install <model-id>...  Prepare the ONNX artifact. A verified local snapshot of the pinned revision is used
                       when one is found (no network); otherwise the pinned Hugging Face files are
                       downloaded. Progress goes to stderr; stdout gets one JSON line
                       {model_id, path, revision} per model. Downloads need approval: pass --yes, or
                       answer the y/N prompt when stdin is a terminal. Without a terminal and without
                       --yes the command refuses to download.
                       Several ids, or --all (decide-340m, decide-multilingual, decide-1b), install one
                       after another and skip models that are already installed. Before anything is
                       downloaded a plan is printed to stderr: per model "already installed", "found on
                       this machine at <path>" or "will download X.X GB (needs about Y.Y GB disk)", then
                       a total. The command refuses when the free disk space is not enough.
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
  --all                    install: every curated model
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
  /** Free bytes on the filesystem holding a directory; `undefined` if unknown. Defaults to `statfs`. */
  freeBytes?: ((directory: string) => Promise<number | undefined>) | undefined;
  signal?: AbortSignal | undefined;
}

class UsageError extends Error {}

const GIB = 1024 ** 3;
const GB = 1e9;
const gb = (bytes: number): string => (bytes / GB).toFixed(1);

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
  wanted: { snapshot?: boolean; install?: boolean; scan?: boolean },
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
        ...(wanted.install ? { yes: { type: "boolean", short: "y" }, all: { type: "boolean" } } : {}),
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
    freeBytes: io.freeBytes,
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
    throw unknownModel(reference);
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

const unknownModel = (reference: string): UsageError =>
  new UsageError(`unknown GLiNER2.5 model '${reference}'; known models: ${KNOWN_MODELS.map((known) => known.modelId).join(", ")}`);

interface Plan {
  id: string;
  repository: string;
  revision: string;
  template: OnnxTemplate | undefined;
  /** Existing artifact directory, a verified local snapshot to prepare from, or a download. */
  source: "installed" | "local" | "download";
  /** Where the artifact is (installed) or the snapshot was found (local). */
  location: string;
  download: number;
  onDisk: number;
}

/** Where a model would come from, without writing or downloading anything. */
async function plan(
  provisioner: OnnxModelProvisioner,
  parsed: Parsed,
  io: ModelsIo,
  model: { id: string; repository: string; revision: string },
): Promise<Plan> {
  const directory = artifactDirectory(parsed.modelsDir, model.repository, model.revision);
  const present = await exists(directory);
  const template = await loadTemplate(provisioner.templatesDir, provisioner.model(model.id) as KnownModel, model.revision);
  const local = present || template === undefined ? undefined : (await provisioner.discover(template, { signal: io.signal })).selected;
  return {
    ...model,
    template,
    source: present ? "installed" : local === undefined ? "download" : "local",
    location: present ? directory : (local?.location ?? ""),
    download: template?.sourceFiles.reduce((sum, file) => sum + file.size, 0) ?? 0,
    onDisk: template === undefined ? 0 : outputBytes(template),
  };
}

async function install(argv: readonly string[], io: ModelsIo): Promise<number> {
  const parsed = parse(argv, io, { install: true });
  if (parsed.values["help"] === true) {
    io.stdout(MODELS_USAGE);
    return 0;
  }
  const provisioner = provisionerFor(parsed, io);
  if (parsed.values["all"] === true || parsed.positionals.length > 1) {
    return installMany(parsed, provisioner, io);
  }
  const model = modelArgument(parsed, provisioner);
  if (!(await provisioner.canInstall(model.id, model.revision))) {
    throw new Error(
      `no ONNX template for ${model.repository}@${model.revision} in ${provisioner.templatesDir}; ` +
        "only pinned revisions with a template can be installed",
    );
  }

  // Approval is only for downloading: an existing artifact or a verified local snapshot needs no network.
  const { source, download, onDisk } = await plan(provisioner, parsed, io, model);
  if (source === "download" && parsed.values["yes"] !== true) {
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

/** `install --all` or several ids: prints a plan, checks disk space, asks once, then installs one after another. */
async function installMany(parsed: Parsed, provisioner: OnnxModelProvisioner, io: ModelsIo): Promise<number> {
  const all = parsed.values["all"] === true;
  if (all && parsed.positionals.length > 0) {
    throw new UsageError("--all cannot be combined with model ids");
  }
  if (typeof parsed.values["revision"] === "string") {
    throw new UsageError("--revision applies to a single model");
  }
  const references = all ? KNOWN_MODELS.map((known) => known.modelId) : parsed.positionals;
  const models = new Map<string, { id: string; repository: string; revision: string }>();
  for (const reference of references) {
    const known = provisioner.model(reference);
    if (known === undefined) {
      throw unknownModel(reference);
    }
    models.set(known.modelId, { id: known.modelId, repository: known.repository, revision: known.pinnedRevision });
  }

  const plans: Plan[] = [];
  for (const model of models.values()) {
    if (!(await provisioner.canInstall(model.id, model.revision))) {
      throw new Error(`no ONNX template for ${model.repository}@${model.revision} in ${provisioner.templatesDir}`);
    }
    plans.push(await plan(provisioner, parsed, io, model));
  }

  // Downloads and preparation run one at a time and the download copy is removed after each model, so the peak
  // is every prepared artifact plus the largest source download.
  const pending = plans.filter((entry) => entry.source !== "installed");
  const download = plans.reduce((sum, entry) => sum + (entry.source === "download" ? entry.download : 0), 0);
  const largest = Math.max(0, ...pending.filter((entry) => entry.source === "download").map((entry) => entry.download));
  const needed = pending.reduce((sum, entry) => sum + entry.onDisk, 0) + largest;
  for (const entry of plans) {
    io.stderr(
      entry.source === "installed"
        ? `${entry.id}: already installed\n`
        : entry.source === "local"
          ? `${entry.id}: found on this machine at ${entry.location} \u2014 no download needed\n`
          : `${entry.id}: will download ${gb(entry.download)} GB (needs about ${gb(entry.download + entry.onDisk)} GB disk)\n`,
    );
  }
  io.stderr(
    download > 0
      ? `Total: will download ${gb(download)} GB (needs about ${gb(needed)} GB disk)\n`
      : "Total: nothing to download\n",
  );

  if (needed > 0) {
    let directory = parsed.modelsDir;
    while (!(await exists(directory)) && dirname(directory) !== directory) {
      directory = dirname(directory);
    }
    const free = await (io.freeBytes ?? freeBytesOf)(directory);
    if (free !== undefined && free < needed + FREE_SPACE_MARGIN_BYTES) {
      io.stderr(
        `error: not enough free disk space in ${directory}: need about ${gb(needed + FREE_SPACE_MARGIN_BYTES)} GB, ` +
          `${gb(free)} GB available\n`,
      );
      return 1;
    }
  }
  if (download > 0 && parsed.values["yes"] !== true) {
    if (!io.interactive) {
      io.stderr(`error: refusing to download ${gb(download)} GB without approval; stdin is not a terminal, rerun with --yes\n`);
      return 1;
    }
    if (!(await io.confirm(`Download ${gb(download)} GB now? [y/N] `))) {
      io.stderr("error: download not approved\n");
      return 1;
    }
  }

  for (const entry of plans) {
    const installed = entry.source === "installed";
    const path = await provisioner.install(entry.id, entry.revision, {
      signal: io.signal ?? new AbortController().signal,
      log: (message) => installed || io.stderr(`${message}\n`),
    });
    if (installed) {
      io.stderr(`${entry.id}: already installed (${path})\n`);
    }
    io.stdout(line({ model_id: entry.id, path, revision: entry.revision }));
  }
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
