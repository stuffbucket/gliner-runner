import { randomBytes } from "node:crypto";
import { lstat, mkdir, rm } from "node:fs/promises";
import { join } from "node:path";

import { KNOWN_MODELS, parseModelReference, type KnownModel } from "../catalog.js";
import type { InstallContext, ModelProvisioner } from "../downloads.js";
import { ModelDownloadRequiredError, ModelIntegrityError } from "../errors.js";
import { OnnxArtifactCatalog, artifactDirectory, repositoryDirectoryName } from "./artifacts.js";
import { ArtifactExistsError, assembleArtifact, ensureFreeSpace, outputBytes, type Progress } from "./assemble.js";
import { HuggingFaceClient, type DownloadProgress } from "./huggingface.js";
import {
  assemblyStep,
  buildRecord,
  readProcessingRecord,
  writeProcessingRecord,
  type ProcessingRecord,
  type ProcessingState,
} from "./processing-record.js";
import { discoverSnapshots, type DiscoveryResult, type SnapshotCandidate } from "./snapshots.js";
import { TemplateError, loadTemplate, resolveTemplatesDir, type OnnxTemplate } from "./templates.js";

export interface OnnxProvisionerOptions {
  /** Models root the artifacts are installed into. */
  modelsDir: string;
  /** Template directory; defaults to `GLINER_RUNNER_ONNX_TEMPLATES_DIR`, then the bundled templates. */
  templatesDir?: string | undefined;
  models?: readonly KnownModel[];
  huggingface?: HuggingFaceClient;
  /** Environment for snapshot discovery (`GLINER_RUNNER_SNAPSHOT_DIRS`, `HF_HOME`, ...); defaults to `process.env`. */
  env?: NodeJS.ProcessEnv | undefined;
  /** Test seams for snapshot discovery. */
  platform?: NodeJS.Platform | undefined;
  home?: string | undefined;
  /** Test seam for the free-space check. */
  freeBytes?: ((directory: string) => Promise<number | undefined>) | undefined;
}

export interface ScanReport {
  model: KnownModel;
  revision: string;
  /** `undefined` when no template exists for the revision (nothing can be verified or prepared). */
  template: OnnxTemplate | undefined;
  templateError?: string;
  candidates: SnapshotCandidate[];
  selected: SnapshotCandidate | undefined;
  /** The processing record after this scan. */
  record: ProcessingRecord | undefined;
  /** Whether the artifact directory exists. */
  installed: boolean;
}

const MIB = 1024 * 1024;

/** Logs at most once per `intervalMs` (and never loses the final message of a phase, which callers send last). */
function throttled(log: (message: string) => void, intervalMs: number): (message: string, force?: boolean) => void {
  let last = 0;
  return (message, force = false) => {
    const now = Date.now();
    if (force || now - last >= intervalMs) {
      last = now;
      log(message);
    }
  };
}

async function exists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch {
    return false;
  }
}

const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

function describeRejections(candidates: readonly SnapshotCandidate[]): string {
  return candidates.length === 0
    ? "no snapshot directories were found in the standard locations"
    : candidates.map((candidate) => `${candidate.location}: ${candidate.reason ?? "rejected"}`).join("; ");
}

/**
 * Installs curated ONNX artifacts without Python: finds a verified local snapshot of the pinned revision (or
 * downloads the pinned source files named by the bundled template), assembles the canonical artifact and
 * renames it into `<modelsDir>/onnx/<owner>--<name>/<revision>/`. Every attempt updates a processing record next
 * to the artifact directory.
 */
export class OnnxModelProvisioner implements ModelProvisioner {
  readonly destination: string;
  readonly templatesDir: string;
  readonly #models: readonly KnownModel[];
  readonly #hf: HuggingFaceClient;
  readonly #freeBytes: OnnxProvisionerOptions["freeBytes"];
  readonly #env: NodeJS.ProcessEnv;
  readonly #platform: NodeJS.Platform | undefined;
  readonly #home: string | undefined;

  constructor(options: OnnxProvisionerOptions) {
    this.destination = options.modelsDir;
    this.templatesDir = resolveTemplatesDir(options.templatesDir);
    this.#models = options.models ?? KNOWN_MODELS;
    this.#hf = options.huggingface ?? new HuggingFaceClient();
    this.#freeBytes = options.freeBytes;
    this.#env = options.env ?? process.env;
    this.#platform = options.platform;
    this.#home = options.home;
  }

  get huggingFaceEndpoint(): string {
    return this.#hf.endpoint;
  }

  /** The curated model for an id or repository (any `@revision` suffix is ignored). */
  model(reference: string): KnownModel | undefined {
    const { id } = parseModelReference(reference);
    return this.#models.find((model) => model.modelId === id || model.repository === id);
  }

  async #template(model: KnownModel, revision: string): Promise<OnnxTemplate> {
    if (revision !== model.pinnedRevision) {
      throw new ModelIntegrityError(
        `revision ${revision} of ${model.repository} cannot be prepared: only the pinned revision ` +
          `${model.pinnedRevision} has an ONNX template`,
      );
    }
    const template = await loadTemplate(this.templatesDir, model, revision);
    if (template === undefined) {
      throw new ModelIntegrityError(
        `no ONNX template for ${model.repository}@${revision} was found in ${this.templatesDir}`,
      );
    }
    return template;
  }

  async canInstall(modelId: string, revision: string): Promise<boolean> {
    const model = this.model(modelId);
    if (model === undefined || revision !== model.pinnedRevision) {
      return false;
    }
    try {
      return (await loadTemplate(this.templatesDir, model, revision)) !== undefined;
    } catch (error) {
      // A broken template is reported by the job itself, not hidden behind the generic "not installed" error.
      if (error instanceof TemplateError) {
        return true;
      }
      throw error;
    }
  }

  /** Searches the standard snapshot locations (see `discoverSnapshots`); read only. */
  discover(
    template: OnnxTemplate,
    options: { explicit?: string | undefined; exhaustive?: boolean; signal?: AbortSignal | undefined } = {},
  ): Promise<DiscoveryResult> {
    return discoverSnapshots(template.model, template.revision, template.sourceFiles, {
      explicit: options.explicit,
      env: this.#env,
      platform: this.#platform,
      home: this.#home,
      modelsDir: this.destination,
      exhaustive: options.exhaustive,
      signal: options.signal,
    });
  }

  /** Whether the artifact directory already exists; a valid one is reused, an invalid one is an error. */
  async #alreadyInstalled(model: KnownModel, revision: string): Promise<string | undefined> {
    const catalog = new OnnxArtifactCatalog(this.destination, { models: [model] });
    try {
      return (await catalog.resolve(`${model.modelId}@${revision}`)).path;
    } catch (error) {
      if (error instanceof ModelDownloadRequiredError) {
        return undefined;
      }
      throw new ModelIntegrityError(
        `${artifactDirectory(this.destination, model.repository, revision)} exists but is not a valid artifact ` +
          `(${errorMessage(error)}); it is never overwritten, remove it to reinstall`,
      );
    }
  }

  /** Best-effort record update; a record that cannot be written never fails the operation. */
  async #record(
    log: (message: string) => void,
    template: OnnxTemplate,
    state: ProcessingState,
    steps: readonly string[],
    source: { location: string | null; verified: boolean },
    extra: { error?: string; artifactPath?: string; rejected?: readonly SnapshotCandidate[] } = {},
  ): Promise<ProcessingRecord | undefined> {
    const record = buildRecord({
      template,
      state,
      steps,
      source,
      error: extra.error,
      artifactPath: extra.artifactPath,
      rejected: extra.rejected?.map((candidate) => ({ location: candidate.location, reason: candidate.reason ?? "rejected" })),
    });
    try {
      await writeProcessingRecord(this.destination, record);
      return record;
    } catch (error) {
      log(`could not write the processing record: ${errorMessage(error)}`);
      return undefined;
    }
  }

  /** Looks for local snapshots of every kind and updates the model's record (a prepared record is left as is). */
  async scan(modelId: string, options: { signal?: AbortSignal | undefined; log?: (message: string) => void } = {}): Promise<ScanReport> {
    const model = this.model(modelId);
    if (model === undefined) {
      throw new ModelIntegrityError(`unknown GLiNER2.5 model '${modelId}'`);
    }
    const revision = model.pinnedRevision;
    const log = options.log ?? (() => undefined);
    const installed = await exists(artifactDirectory(this.destination, model.repository, revision));
    let template: OnnxTemplate | undefined;
    let templateError: string | undefined;
    try {
      template = await loadTemplate(this.templatesDir, model, revision);
    } catch (error) {
      templateError = errorMessage(error);
    }
    if (template === undefined) {
      return {
        model,
        revision,
        template,
        ...(templateError === undefined ? {} : { templateError }),
        candidates: [],
        selected: undefined,
        record: await readProcessingRecord(this.destination, model, revision),
        installed,
      };
    }
    const found = await this.discover(template, { exhaustive: true, signal: options.signal });
    const existing = await readProcessingRecord(this.destination, model, revision);
    let record = existing;
    if (!(existing?.state === "prepared" && installed)) {
      const rejected = found.candidates.filter((candidate) => !candidate.verified);
      record =
        found.selected === undefined
          ? await this.#record(log, template, "not_processed", [], { location: null, verified: false }, { rejected })
          : await this.#record(
              log,
              template,
              "verified_source",
              ["discovered_local", "verified_source_sha256"],
              { location: found.selected.location, verified: true },
              { rejected },
            );
    }
    return { model, revision, template, candidates: found.candidates, selected: found.selected, record, installed };
  }

  /**
   * Assembles the artifact from a verified source directory and records the outcome. `steps` are the steps already
   * applied to the source (how it was obtained and verified).
   */
  async #assemble(
    template: OnnxTemplate,
    snapshotDir: string,
    source: { location: string; steps: readonly string[] },
    context: { signal?: AbortSignal | undefined; log: (message: string) => void },
  ): Promise<string> {
    const { model } = template;
    const log = throttled(context.log, 2000);
    await this.#record(context.log, template, "verified_source", source.steps, { location: source.location, verified: true });
    try {
      const path = await assembleArtifact({
        template,
        snapshotDir,
        modelsDir: this.destination,
        verifySources: false, // size and SHA-256 were verified by discovery or while downloading
        signal: context.signal,
        freeBytes: this.#freeBytes,
        onProgress: (progress: Progress) =>
          log(
            `${model.modelId}: ${progress.stage}${progress.file === undefined ? "" : ` ${progress.file}`} ` +
              `${(progress.done / MIB).toFixed(1)}/${(progress.total / MIB).toFixed(1)} MiB`,
            progress.done === progress.total,
          ),
      });
      await this.#record(
        context.log,
        template,
        "prepared",
        [...source.steps, assemblyStep(template), "verified_outputs_sha256", "published"],
        { location: source.location, verified: true },
        { artifactPath: path },
      );
      context.log(`${model.modelId}: installed ${path}`);
      return path;
    } catch (error) {
      if (!(error instanceof ArtifactExistsError)) {
        await this.#record(context.log, template, "failed", source.steps, { location: source.location, verified: true }, { error: errorMessage(error) });
      }
      throw error;
    }
  }

  /**
   * Installs a model: reuses a valid existing artifact, else prepares it from a verified local snapshot when one
   * is found (no network), else downloads the pinned source files. Resolves with the artifact directory.
   */
  async install(modelId: string, revision: string, context: InstallContext): Promise<string> {
    const model = this.model(modelId);
    if (model === undefined) {
      throw new ModelIntegrityError(`unknown GLiNER2.5 model '${modelId}'`);
    }
    const template = await this.#template(model, revision);
    const present = await this.#alreadyInstalled(model, revision);
    if (present !== undefined) {
      context.log(`${model.modelId}@${revision} is already installed at ${present}`);
      return present;
    }

    const found = await this.discover(template, { signal: context.signal });
    if (found.selected !== undefined) {
      context.log(`${model.modelId}: using the verified local snapshot ${found.selected.location} (no download)`);
      return this.#assemble(
        template,
        found.selected.location,
        { location: found.selected.location, steps: ["discovered_local", "verified_source_sha256"] },
        context,
      );
    }
    return this.#download(template, context, found.candidates.filter((candidate) => !candidate.verified));
  }

  async #download(
    template: OnnxTemplate,
    context: InstallContext,
    rejected: readonly SnapshotCandidate[],
  ): Promise<string> {
    const { model, revision } = template;
    const parent = join(this.destination, "onnx", repositoryDirectoryName(model.repository));
    const location = `huggingface:${this.#hf.endpoint}`;
    const log = throttled(context.log, 2000);
    const workDirectory = join(parent, `.download-${revision.slice(0, 12)}-${randomBytes(4).toString("hex")}`);
    try {
      await mkdir(parent, { recursive: true });
      const sourceBytes = template.sourceFiles.reduce((sum, file) => sum + file.size, 0);
      await ensureFreeSpace(
        parent,
        sourceBytes + outputBytes(template),
        `downloading and preparing ${model.modelId}`,
        this.#freeBytes,
      );
      await mkdir(workDirectory);
      context.log(
        `${model.modelId}: downloading ${template.sourceFiles.length} files (${(sourceBytes / MIB).toFixed(1)} MiB) ` +
          `from ${this.#hf.endpoint}`,
      );
      await this.#hf.downloadAll(model.repository, revision, template.sourceFiles, workDirectory, {
        signal: context.signal,
        onProgress: (progress: DownloadProgress) =>
          log(
            `${model.modelId}: downloading ${progress.file} ${(progress.bytes / MIB).toFixed(1)}/` +
              `${(progress.total / MIB).toFixed(1)} MiB`,
            progress.fileBytes === progress.fileTotal,
          ),
      });
    } catch (error) {
      await rm(workDirectory, { recursive: true, force: true });
      await this.#record(context.log, template, "failed", [], { location, verified: false }, { error: errorMessage(error), rejected });
      throw error;
    }
    try {
      return await this.#assemble(
        template,
        workDirectory,
        { location, steps: ["downloaded", "verified_source_sha256"] },
        context,
      );
    } finally {
      await rm(workDirectory, { recursive: true, force: true });
    }
  }

  /**
   * Offline preparation from a local snapshot: the explicit `snapshot` directory (which must verify), or, when
   * omitted, the first verified snapshot in the standard locations.
   */
  async prepare(
    modelId: string,
    options: {
      snapshot?: string | undefined;
      revision?: string | undefined;
      signal?: AbortSignal | undefined;
      log?: ((message: string) => void) | undefined;
    } = {},
  ): Promise<string> {
    const model = this.model(modelId);
    if (model === undefined) {
      throw new ModelIntegrityError(`unknown GLiNER2.5 model '${modelId}'`);
    }
    const revision = options.revision ?? model.pinnedRevision;
    const template = await this.#template(model, revision);
    const log = options.log ?? (() => undefined);
    const destination = artifactDirectory(this.destination, model.repository, revision);
    if (await exists(destination)) {
      throw new ArtifactExistsError(destination); // before anything is recorded: the existing record stays accurate
    }
    const found = await this.discover(template, { explicit: options.snapshot, signal: options.signal });
    if (found.selected === undefined) {
      const rejected = found.candidates.filter((candidate) => !candidate.verified);
      const message =
        options.snapshot === undefined
          ? `no verified local snapshot of ${model.repository}@${revision} was found: ${describeRejections(rejected)}`
          : `snapshot ${options.snapshot} does not match ${model.repository}@${revision}: ${describeRejections(rejected.length === 0 ? [{ location: options.snapshot, reason: "not a directory" } as SnapshotCandidate] : rejected)}`;
      await this.#record(log, template, "failed", [], { location: null, verified: false }, { error: message, rejected });
      throw new ModelIntegrityError(message);
    }
    log(`${model.modelId}: preparing from ${found.selected.location}`);
    return this.#assemble(
      template,
      found.selected.location,
      { location: found.selected.location, steps: ["discovered_local", "verified_source_sha256"] },
      { signal: options.signal, log },
    );
  }
}
