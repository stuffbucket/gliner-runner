import type { BackendFactory, BackendRegistry } from "./backend.js";
import type { ArtifactCatalog, ResolvedModel } from "./catalog.js";
import type { BackendCapabilities, InferenceRequest, InferenceResponse, PrecisionProfile } from "./contracts.js";
import { ModelMemoryLimitError, RequestCancelledError, RunnerClosedError } from "./errors.js";
import { Metrics } from "./metrics.js";
import {
  defaultMemoryLimitBytes,
  minimumMemoryBytes,
  physicalMemoryBytes,
  processRssBytes,
} from "./resources.js";
import { ModelWorker } from "./scheduler.js";

/** `setTimeout` delays above 2^31-1 ms fire after 1 ms, so the idle TTL is bounded to what a timer can hold. */
export const MAX_MODEL_IDLE_TTL_SECONDS = Math.floor((2 ** 31 - 1) / 1000);

export interface RuntimeConfig {
  device: string;
  queueCapacity: number;
  maxBatchSize: number;
  batchWindowMs: number;
  /** Seconds of inactivity before the resident model is unloaded; 0 keeps it warm; at most {@link MAX_MODEL_IDLE_TTL_SECONDS}. */
  modelIdleTtlSeconds: number;
  /** Explicit memory ceiling; defaults to 75% of physical memory. */
  memoryLimitBytes?: number | undefined;
}

export interface RuntimeOptions {
  catalog: ArtifactCatalog;
  backends: BackendRegistry;
  config: RuntimeConfig;
  metrics?: Metrics;
  /** Memory attributable to the loaded model; defaults to process RSS. */
  memoryProbe?: (device: string) => number;
  totalMemory?: () => number;
}

export interface LoadedState {
  modelId: string;
  revision: string;
  backend: string;
  profile: PrecisionProfile;
  activeRequests: number;
}

interface Resident {
  worker: ModelWorker;
  key: string;
  state: Omit<LoadedState, "activeRequests">;
}

/**
 * Serves requests from exactly one resident model identity at a time: loads lazily, drains and
 * replaces the model when another identity is requested, and can evict it when idle.
 */
export class Runtime {
  readonly config: RuntimeConfig;
  readonly metrics: Metrics;
  readonly #catalog: ArtifactCatalog;
  readonly #backends: BackendRegistry;
  readonly #memoryProbe: (device: string) => number;
  readonly #totalMemory: () => number;
  #resident: Resident | undefined;
  #active = 0;
  #switching = false;
  #generation = 0;
  #idleTimer: NodeJS.Timeout | undefined;
  #closed = false;
  #waiters: (() => void)[] = [];
  /** The model switch (drain, unload, load) in progress, which `close()` must wait for. */
  #switch: Promise<unknown> | undefined;

  constructor(options: RuntimeOptions) {
    const ttl = options.config.modelIdleTtlSeconds;
    if (!(ttl <= MAX_MODEL_IDLE_TTL_SECONDS)) {
      throw new RangeError(
        `modelIdleTtlSeconds must be a number of at most ${MAX_MODEL_IDLE_TTL_SECONDS} seconds, got ${ttl}`,
      );
    }
    this.config = options.config;
    this.metrics = options.metrics ?? new Metrics();
    this.#catalog = options.catalog;
    this.#backends = options.backends;
    this.#memoryProbe = options.memoryProbe ?? processRssBytes;
    this.#totalMemory = options.totalMemory ?? physicalMemoryBytes;
  }

  get accepting(): boolean {
    return !this.#closed;
  }

  get memoryLimitBytes(): number {
    return this.config.memoryLimitBytes ?? defaultMemoryLimitBytes(this.#totalMemory());
  }

  capabilities(): BackendCapabilities[] {
    return this.#backends.capabilities();
  }

  loadedState(): LoadedState | undefined {
    return this.#resident === undefined
      ? undefined
      : { ...this.#resident.state, activeRequests: this.#active };
  }

  async infer(request: InferenceRequest, signal?: AbortSignal): Promise<InferenceResponse> {
    throwIfCancelled(signal);
    const worker = await this.#acquire(request, signal);
    try {
      return await worker.submit(request, signal);
    } finally {
      this.#release();
    }
  }

  /** Stops accepting requests, waits for in-flight ones, then unloads the model. */
  async close(): Promise<void> {
    this.#closed = true;
    this.#switching = true;
    this.#notifyAll();
    while (this.#active > 0) {
      await this.#changed();
    }
    // A load that started before close() finishes (and unloads itself, since the runtime is closed) first.
    await this.#switch?.catch(() => undefined);
    const resident = this.#detach();
    this.#switching = false;
    this.#notifyAll();
    await resident?.worker.close();
  }

  async #acquire(request: InferenceRequest, signal: AbortSignal | undefined): Promise<ModelWorker> {
    this.#assertOpen();
    this.#backends.require(request.backend);
    const model = await this.#catalog.resolve(request.model);
    const { device } = this.config;
    const factory = this.#backends.assertSupported(request, model.modelId, device);
    const key = JSON.stringify([request.backend, model.modelId, model.revision, request.precision, device]);

    while (this.#switching && !this.#closed) {
      await this.#changed();
    }
    this.#assertOpen();
    // A cancelled request must neither start nor wait for a model switch on behalf of a caller that is gone.
    throwIfCancelled(signal);
    if (this.#resident?.key === key) {
      this.#active += 1;
      this.#cancelIdle();
      return this.#resident.worker;
    }

    const profile = `${device}/${request.precision}`;
    const required = minimumMemoryBytes(model.parameterCount, request.precision);
    const limitBytes = this.memoryLimitBytes;
    if (required > limitBytes) {
      throw new ModelMemoryLimitError({ model: request.model, profile, limitBytes, requiredBytes: required });
    }

    this.#switching = true;
    const switched = this.#switchModel({ request, model, factory, key, profile, required, limitBytes, signal });
    this.#switch = switched;
    try {
      return await switched;
    } finally {
      if (this.#switch === switched) {
        this.#switch = undefined;
      }
    }
  }

  /** Drains and unloads the resident model, loads the requested one and makes it resident with one active use. */
  async #switchModel(plan: {
    request: InferenceRequest;
    model: ResolvedModel;
    factory: BackendFactory;
    key: string;
    profile: string;
    required: number;
    limitBytes: number;
    signal: AbortSignal | undefined;
  }): Promise<ModelWorker> {
    const { request, model, factory, key, profile, required, limitBytes, signal } = plan;
    const { device } = this.config;
    try {
      while (this.#active > 0) {
        await this.#changed();
      }
      // Re-check after the drain: the caller may have gone away while it waited, and the resident stays untouched.
      throwIfCancelled(signal);
      await this.#detach()?.worker.close();
      this.#assertOpen();
    } catch (error) {
      this.#loadFailed();
      throw error;
    }

    let worker: ModelWorker | undefined;
    try {
      const backend = factory.create(model, device, request.precision);
      worker = new ModelWorker(backend, {
        queueCapacity: this.config.queueCapacity,
        maxBatchSize: this.config.maxBatchSize,
        batchWindowMs: this.config.batchWindowMs,
        metrics: this.metrics,
      });
      try {
        await worker.start();
      } catch (error) {
        await backend.close().catch(() => undefined);
        throw error;
      }
      const observed = this.#memoryProbe(device);
      if (observed > limitBytes) {
        throw new ModelMemoryLimitError({
          model: request.model,
          profile,
          limitBytes,
          requiredBytes: required,
          observedBytes: observed,
        });
      }
    } catch (error) {
      await worker?.close().catch(() => undefined);
      this.#loadFailed();
      throw error;
    }

    if (this.#closed) {
      this.#switching = false;
      this.#notifyAll();
      await worker.close();
      throw new RunnerClosedError("runtime is closed");
    }
    this.#resident = {
      worker,
      key,
      state: {
        modelId: model.modelId,
        revision: model.revision,
        backend: request.backend,
        profile: { device, precision: request.precision },
      },
    };
    this.#active = 1;
    this.#generation += 1;
    this.#switching = false;
    this.#notifyAll();
    return worker;
  }

  #assertOpen(): void {
    if (this.#closed) {
      throw new RunnerClosedError("runtime is closed");
    }
  }

  #release(): void {
    this.#active -= 1;
    if (this.#active === 0) {
      this.#notifyAll();
      this.#scheduleIdle();
    }
  }

  #loadFailed(): void {
    this.#switching = false;
    this.#notifyAll();
  }

  #detach(): Resident | undefined {
    this.#cancelIdle();
    const resident = this.#resident;
    this.#resident = undefined;
    this.#generation += 1;
    return resident;
  }

  #cancelIdle(): void {
    clearTimeout(this.#idleTimer);
    this.#idleTimer = undefined;
  }

  #scheduleIdle(): void {
    const ttl = this.config.modelIdleTtlSeconds;
    if (ttl <= 0 || this.#resident === undefined || this.#closed) {
      return;
    }
    this.#cancelIdle();
    const generation = this.#generation;
    this.#idleTimer = setTimeout(() => void this.#evict(generation), ttl * 1000);
    // Node keeps the process alive for a pending timer; a browser has no `unref`.
    (this.#idleTimer as { unref?: () => void }).unref?.();
  }

  async #evict(generation: number): Promise<void> {
    if (
      this.#closed ||
      this.#active > 0 ||
      this.#generation !== generation ||
      this.#resident === undefined
    ) {
      return;
    }
    this.#switching = true;
    const resident = this.#detach();
    try {
      await resident?.worker.close();
    } finally {
      this.#switching = false;
      this.#notifyAll();
    }
  }

  #changed(): Promise<void> {
    return new Promise((resolve) => this.#waiters.push(resolve));
  }

  #notifyAll(): void {
    const waiters = this.#waiters;
    this.#waiters = [];
    for (const resolve of waiters) {
      resolve();
    }
  }
}

function throwIfCancelled(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new RequestCancelledError("request was cancelled before a model was acquired");
  }
}
