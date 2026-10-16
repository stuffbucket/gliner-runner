
import type { InferenceBackend, BackendResult } from "./backend.js";
import type { InferenceRequest, InferenceResponse } from "./contracts.js";
import { QueueFullError, RequestCancelledError, RunnerClosedError } from "./errors.js";
import { Metrics } from "./metrics.js";
import { taskOrderOf } from "./task-order.js";
import { delay } from "./timers.js";

export interface WorkerOptions {
  queueCapacity?: number;
  maxBatchSize?: number;
  batchWindowMs?: number;
  metrics?: Metrics;
  /** Monotonic clock in milliseconds. */
  clock?: () => number;
}

interface Work {
  request: InferenceRequest;
  enqueuedAt: number;
  cancelled: boolean;
  settled: boolean;
  resolve(response: InferenceResponse): void;
  reject(error: unknown): void;
}

/** Owns one loaded backend: a serial batching loop over intake queue plus backlog, bounded together by `queueCapacity`. */
export class ModelWorker {
  readonly #backend: InferenceBackend;
  readonly #queueCapacity: number;
  readonly #maxBatchSize: number;
  readonly #batchWindowMs: number;
  readonly #metrics: Metrics;
  readonly #clock: () => number;
  readonly #queue: Work[] = [];
  readonly #backlog: Work[] = [];
  #wake: (() => void) | undefined;
  #starting: Promise<void> | undefined;
  #task: Promise<void> | undefined;
  #accepting = false;
  #stopping = false;
  #inFlight: Work[] = [];

  constructor(backend: InferenceBackend, options: WorkerOptions = {}) {
    const { queueCapacity = 256, maxBatchSize = 16, batchWindowMs = 4 } = options;
    if (
      !Number.isInteger(queueCapacity) ||
      queueCapacity < 1 ||
      !Number.isInteger(maxBatchSize) ||
      maxBatchSize < 1 ||
      !(batchWindowMs >= 0)
    ) {
      throw new RangeError("invalid scheduler configuration");
    }
    this.#backend = backend;
    this.#queueCapacity = queueCapacity;
    this.#maxBatchSize = maxBatchSize;
    this.#batchWindowMs = batchWindowMs;
    this.#metrics = options.metrics ?? new Metrics();
    this.#clock = options.clock ?? (() => performance.now());
  }

  get accepting(): boolean {
    return this.#accepting;
  }

  /** Loads the backend, then starts accepting work. Idempotent. */
  start(): Promise<void> {
    this.#starting ??= this.#load();
    return this.#starting;
  }

  async #load(): Promise<void> {
    await this.#backend.load();
    this.#accepting = true;
    this.#task = this.#run();
    this.#task.catch(() => undefined);
  }

  /**
   * Queues a request and resolves with its response. Aborting `signal` before the request is
   * dispatched removes it from the queue; after dispatch the result is computed and discarded.
   */
  async submit(request: InferenceRequest, signal?: AbortSignal): Promise<InferenceResponse> {
    if (!this.#accepting) {
      throw new RunnerClosedError("model worker is not accepting requests");
    }
    if (signal?.aborted) {
      throw new RequestCancelledError("request was cancelled before it was queued");
    }
    // Backlog (work set aside because it did not match the batch being built) counts toward the bound: otherwise a
    // stream of incompatible requests drains the intake queue into the backlog and grows memory without limit.
    // The PyTorch runner bounds only the intake queue; this is a deliberate stricter bound.
    if (this.#queue.length + this.#backlog.length >= this.#queueCapacity) {
      this.#metrics.increment("requests_rejected_total");
      throw new QueueFullError("model worker queue is full");
    }
    const { promise, resolve, reject } = Promise.withResolvers<InferenceResponse>();
    const work: Work = {
      request,
      enqueuedAt: this.#clock(),
      cancelled: false,
      settled: false,
      resolve: (response) => {
        work.settled = true;
        resolve(response);
      },
      reject: (error) => {
        work.settled = true;
        reject(error);
      },
    };
    this.#queue.push(work);
    this.#metrics.increment("requests_submitted_total");
    this.#updateDepth();
    this.#notify();

    const onAbort = (): void => {
      if (work.settled) {
        return;
      }
      work.cancelled = true;
      for (const list of [this.#queue, this.#backlog]) {
        const index = list.indexOf(work);
        if (index >= 0) {
          list.splice(index, 1);
        }
      }
      this.#metrics.increment("requests_cancelled_total");
      this.#updateDepth();
      work.reject(new RequestCancelledError("request was cancelled"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    try {
      return await promise;
    } finally {
      signal?.removeEventListener("abort", onAbort);
    }
  }

  /** Drains everything accepted so far, then closes the backend. */
  async close(): Promise<void> {
    this.#accepting = false;
    if (this.#task === undefined) {
      await this.#starting?.catch(() => undefined);
      if (this.#task === undefined) {
        return;
      }
    }
    this.#stopping = true;
    this.#notify();
    try {
      await this.#task;
    } finally {
      this.#task = undefined;
      await this.#backend.close();
    }
  }

  async #run(): Promise<void> {
    try {
      for (;;) {
        const first = await this.#next();
        if (first === undefined) {
          return;
        }
        this.#inFlight = [first];
        await this.#collect(this.#inFlight);
        const active = this.#inFlight.filter((work) => !work.cancelled);
        if (active.length > 0) {
          await this.#dispatch(active);
        }
        this.#inFlight = [];
        this.#updateDepth();
      }
    } finally {
      this.#accepting = false;
      this.#failPending(new RunnerClosedError("model worker closed"));
    }
  }

  async #dispatch(active: Work[]): Promise<void> {
    this.#metrics.increment("batches_total");
    this.#metrics.gauge("last_batch_size", active.length);
    try {
      const started = this.#clock();
      const outputs = await this.#backend.inferBatch(active.map((work) => work.request));
      if (outputs.length !== active.length) {
        throw new Error("backend returned the wrong number of results");
      }
      const finished = this.#clock();
      active.forEach((work, index) => {
        if (!work.settled) {
          work.resolve(response(work, outputs[index] as BackendResult, started, finished));
        }
      });
      this.#metrics.increment("requests_completed_total", active.length);
    } catch (error) {
      this.#metrics.increment("batch_failures_total");
      for (const work of active) {
        if (!work.settled) {
          work.reject(error);
        }
      }
    }
  }

  async #next(): Promise<Work | undefined> {
    for (;;) {
      const fromBacklog = this.#backlog.shift();
      if (fromBacklog !== undefined) {
        return fromBacklog;
      }
      const queued = this.#queue.shift();
      if (queued !== undefined) {
        this.#updateDepth();
        return queued;
      }
      if (this.#stopping) {
        return undefined;
      }
      await new Promise<void>((resolve) => {
        this.#wake = resolve;
      });
    }
  }

  /** Extends `batch` (whose first item defines the key) with compatible waiting work. */
  async #collect(batch: Work[]): Promise<void> {
    const key = batchKey((batch[0] as Work).request);
    if (this.#batchWindowMs > 0) {
      await delay(this.#batchWindowMs);
    }
    const waiting = this.#backlog.splice(0);
    for (const work of waiting) {
      if (batch.length < this.#maxBatchSize && batchKey(work.request) === key) {
        batch.push(work);
      } else {
        this.#backlog.push(work);
      }
    }
    while (batch.length < this.#maxBatchSize) {
      const work = this.#queue.shift();
      if (work === undefined) {
        break;
      }
      if (batchKey(work.request) === key) {
        batch.push(work);
      } else {
        this.#backlog.push(work);
      }
    }
    this.#updateDepth();
  }

  #failPending(error: Error): void {
    const pending = [
      ...this.#inFlight.splice(0),
      ...this.#backlog.splice(0),
      ...this.#queue.splice(0),
    ];
    for (const work of pending) {
      if (!work.settled) {
        work.reject(error);
      }
    }
    this.#updateDepth();
  }

  #notify(): void {
    const wake = this.#wake;
    this.#wake = undefined;
    wake?.();
  }

  #updateDepth(): void {
    this.#metrics.gauge("queue_depth", this.#queue.length + this.#backlog.length);
  }
}

function response(
  work: Work,
  result: BackendResult,
  started: number,
  finished: number,
): InferenceResponse {
  const { request } = work;
  return {
    request_id: request.request_id,
    model: request.model,
    backend: request.backend,
    precision: request.precision,
    output: result.output,
    error: null,
    timing: {
      queue_ms: Math.max(0, started - work.enqueuedAt),
      inference_ms: Math.max(0, finished - started),
    },
    usage: result.usage,
  };
}

/** Requests batch together only with equal operation, schema and options after defaulting. */
export function batchKey(request: InferenceRequest): string {
  const { schema } = request;
  const order = schema.kind === "classification" ? taskOrderOf(schema.tasks) : null;
  return canonicalJson([request.operation, schema, request.options, order]);
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  if (typeof value === "object" && value !== null) {
    const entries = Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key])}`);
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(value);
}
