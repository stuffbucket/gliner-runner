import { platform } from "#platform";

import { isCommitSha, parseModelReference, type ArtifactCatalog } from "./catalog.js";
import type { DownloadJob, DownloadRequest } from "./contracts.js";
import {
  DownloadJobNotFoundError,
  InvalidDownloadRequestError,
  ModelDownloadRequiredError,
  RunnerClosedError,
  RequestCancelledError,
} from "./errors.js";

export interface InstallContext {
  /** Aborted when the runner shuts down; implementations must stop and clean up promptly. */
  signal: AbortSignal;
  /** Diagnostic sink; never receives request text or credentials. */
  log: (message: string) => void;
}

/**
 * Whatever can fetch and prepare a curated model into the catalog's destination (see `OnnxModelProvisioner`).
 * Rejects with a descriptive error when it cannot; the message becomes the job's `error`.
 */
export interface ModelProvisioner {
  /** The configured models directory; a download job must name exactly this directory. */
  readonly destination: string;
  /** Whether `revision` of `modelId` has everything it needs to be prepared (for example a bundled template). */
  canInstall(modelId: string, revision: string): Promise<boolean>;
  /** Fetches and prepares the artifact; resolves with the installed artifact directory. */
  install(modelId: string, revision: string, context: InstallContext): Promise<string>;
}

interface JobRecord {
  job: DownloadJob;
  /** Settles (never rejects) with the final job state. */
  done: Promise<DownloadJob>;
}

const ACTIVE = new Set<DownloadJob["state"]>(["queued", "running"]);

/** `~` expansion plus symlink resolution, the way Python's `Path.expanduser().resolve()` normalises a path. */
function resolvedPath(path: string): Promise<string> {
  return platform.resolvePath(path);
}

/**
 * In-memory download jobs with the semantics of the Python runner's `DownloadJobManager`: explicit operator
 * approval, one active job per model revision, and a destination that must equal the configured models
 * directory (a mismatch fails the job rather than the request).
 */
export class DownloadJobManager {
  readonly #catalog: ArtifactCatalog;
  readonly #provisioner: ModelProvisioner;
  readonly #log: (message: string) => void;
  readonly #jobs = new Map<string, JobRecord>();
  readonly #shutdown = new AbortController();

  constructor(options: {
    catalog: ArtifactCatalog;
    provisioner: ModelProvisioner;
    log?: ((message: string) => void) | undefined;
  }) {
    this.#catalog = options.catalog;
    this.#provisioner = options.provisioner;
    this.#log = options.log ?? (() => undefined);
  }

  get destination(): string {
    return this.#provisioner.destination;
  }

  async #pinnedRevision(modelId: string): Promise<string> {
    const inventory = await this.#catalog.inventory();
    const model = inventory.models.find((candidate) => candidate.modelId === modelId);
    if (model === undefined) {
      throw new InvalidDownloadRequestError(`unknown GLiNER2.5 model '${modelId}'`);
    }
    return model.pinnedRevision;
  }

  /** Creates and starts a job (`POST /v1/models/{id}/downloads`). */
  async create(modelId: string, request: DownloadRequest): Promise<DownloadJob> {
    if (!request.approved) {
      throw new InvalidDownloadRequestError("approved must be true; downloads never start implicitly");
    }
    const canonical = this.#catalog.canonicalModelId(modelId);
    if (canonical === undefined) {
      throw new InvalidDownloadRequestError(`unknown GLiNER2.5 model '${modelId}'`);
    }
    const revision = request.revision ?? (await this.#pinnedRevision(canonical));
    const existing = this.#active(canonical, revision);
    if (existing !== undefined) {
      throw new InvalidDownloadRequestError(`a download for ${canonical}@${revision} is already running`);
    }
    return { ...this.#start(canonical, revision, request.destination).job };
  }

  /** `GET /v1/model-downloads/{job_id}`. */
  get(jobId: string): DownloadJob {
    const record = this.#jobs.get(jobId.toLowerCase());
    if (record === undefined) {
      throw new DownloadJobNotFoundError(`unknown download job ${jobId.toLowerCase()}`);
    }
    return { ...record.job };
  }

  #active(modelId: string, revision: string): JobRecord | undefined {
    for (const record of this.#jobs.values()) {
      if (record.job.model_id === modelId && record.job.revision === revision && ACTIVE.has(record.job.state)) {
        return record;
      }
    }
    return undefined;
  }

  #start(modelId: string, revision: string, destination: string): JobRecord {
    if (this.#shutdown.signal.aborted) {
      throw new RunnerClosedError("the runner is shutting down");
    }
    const job: DownloadJob = {
      job_id: globalThis.crypto.randomUUID(),
      model_id: modelId,
      revision,
      destination,
      state: "queued",
      error: null,
    };
    const record: JobRecord = { job, done: Promise.resolve(job) };
    record.done = this.#run(record);
    this.#jobs.set(job.job_id, record);
    return record;
  }

  async #run(record: JobRecord): Promise<DownloadJob> {
    const { job } = record;
    // The request that created the job sees "queued"; the job starts on the next tick, as in the Python runner.
    await Promise.resolve();
    record.job = { ...job, state: "running" };
    try {
      const requested = await resolvedPath(job.destination);
      const configured = await resolvedPath(this.#provisioner.destination);
      if (requested !== configured) {
        throw new Error(`destination must equal the configured provider location ${configured}`);
      }
      await this.#provisioner.install(job.model_id, job.revision, { signal: this.#shutdown.signal, log: this.#log });
      // Make the new artifact visible to capabilities and inventory before the job reports success.
      await this.#catalog.inventory();
      record.job = { ...job, state: "succeeded" };
    } catch (error) {
      record.job = { ...job, state: "failed", error: error instanceof Error ? error.message : String(error) };
    }
    return record.job;
  }

  /**
   * Auto-download support: starts (or joins) the install job for a missing model and waits for it. Returns
   * `false` without doing anything when the reference cannot be installed (unknown model, no template for the
   * revision), and throws `ModelDownloadRequiredError` carrying the job's error when the job fails. A caller that
   * goes away stops waiting; the job itself continues for the others.
   */
  async installOnDemand(reference: string, requestedRevision: string | undefined, signal?: AbortSignal): Promise<boolean> {
    const canonical = this.#catalog.canonicalModelId(reference);
    if (canonical === undefined) {
      return false;
    }
    const revision = requestedRevision ?? parseModelReference(reference).revision ?? (await this.#pinnedRevision(canonical));
    if (!isCommitSha(revision) || !(await this.#provisioner.canInstall(canonical, revision))) {
      return false;
    }
    const record = this.#active(canonical, revision) ?? this.#start(canonical, revision, this.#provisioner.destination);
    const cancelled = new Promise<never>((_, reject) => {
      if (signal?.aborted) {
        reject(new RequestCancelledError("request was cancelled while waiting for the model download"));
      }
      signal?.addEventListener(
        "abort",
        () => reject(new RequestCancelledError("request was cancelled while waiting for the model download")),
        { once: true },
      );
    });
    const final = await Promise.race([record.done, cancelled]);
    if (this.#shutdown.signal.aborted) {
      throw new RunnerClosedError("the runner is shutting down");
    }
    if (final.state === "failed") {
      throw new ModelDownloadRequiredError(reference, revision, this.#provisioner.destination, final.error ?? "unknown error");
    }
    return true;
  }

  /** Cancels every running job; `close()` also waits for them to clean up. */
  abort(): void {
    this.#shutdown.abort();
  }

  async close(): Promise<void> {
    this.abort();
    await Promise.allSettled([...this.#jobs.values()].map((record) => record.done));
  }
}
