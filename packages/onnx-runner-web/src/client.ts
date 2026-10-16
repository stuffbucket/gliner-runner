import type { HostEvent, HostInit, HostRequest, HostStatus } from "./protocol.js";

export type { HostEvent, HostInit, HostStatus } from "./protocol.js";
export { INSTALL_PEAK_BYTES } from "./limits.js";

export interface TransportReply {
  status: number;
  body: string;
}

/** One way to reach a gliner-runner: the in-tab worker or a runner on this computer over HTTP. */
export interface Transport {
  readonly kind: "tab" | "local";
  request(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<TransportReply>;
}

/** A runner on this computer (`gliner-runner serve`), which must allow the page's origin (`gliner-runner cors`). */
export class HttpTransport implements Transport {
  readonly kind = "local";

  constructor(readonly base: string) {}

  async request(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<TransportReply> {
    const response = await fetch(`${this.base.replace(/\/+$/, "")}${path}`, {
      method,
      ...(body === undefined ? {} : { headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
      ...(signal === undefined ? {} : { signal }),
    });
    return { status: response.status, body: await response.text() };
  }
}

/** The runner in a dedicated worker started with `serveRunner` (see `host.ts`). */
export class WorkerTransport implements Transport {
  readonly kind = "tab";
  readonly #worker: Worker;
  readonly #pending = new Map<number, { resolve(reply: TransportReply): void; reject(error: unknown): void }>();
  readonly #listeners = new Set<(event: HostEvent) => void>();
  #next = 1;
  #status: HostStatus = { state: "starting" };

  constructor(worker: Worker, init: HostInit = {}) {
    this.#worker = worker;
    worker.onmessage = ({ data }: MessageEvent<HostEvent>) => {
      if (data.type === "status") {
        this.#status = data.status;
      }
      if (data.type === "response" || data.type === "cancelled" || data.type === "deleted") {
        const pending = this.#pending.get(data.id);
        this.#pending.delete(data.id);
        if (data.type === "response") {
          pending?.resolve({ status: data.status, body: data.body });
        } else if (data.type === "deleted") {
          pending?.resolve({ status: 204, body: "" });
        } else {
          pending?.reject(new DOMException("The request was cancelled", "AbortError"));
        }
      }
      for (const listener of this.#listeners) {
        listener(data);
      }
    };
    worker.postMessage({ type: "init", options: init } satisfies HostRequest);
    worker.onerror = (event) => {
      this.#status = { state: "failed", reason: event.message || "the runner worker stopped" };
      for (const listener of this.#listeners) {
        listener({ type: "status", status: this.#status });
      }
    };
  }

  get status(): HostStatus {
    return this.#status;
  }

  /** Every event from the worker (status, install progress, the execution provider, log lines). */
  subscribe(listener: (event: HostEvent) => void): () => void {
    this.#listeners.add(listener);
    listener({ type: "status", status: this.#status });
    return () => this.#listeners.delete(listener);
  }

  #send(message: Extract<HostRequest, { id: number }>): Promise<TransportReply> {
    return new Promise((resolve, reject) => {
      this.#pending.set(message.id, { resolve, reject });
      this.#worker.postMessage(message);
    });
  }

  request(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<TransportReply> {
    const id = this.#next++;
    signal?.addEventListener("abort", () => this.#worker.postMessage({ type: "abort", id } satisfies HostRequest), { once: true });
    return this.#send({ type: "request", id, method, path, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  }

  /** Deletes the downloaded model from this site's storage and restarts the runner. */
  async deleteModel(): Promise<void> {
    await this.#send({ type: "delete-model", id: this.#next++ });
  }

  terminate(): void {
    this.#worker.terminate();
    for (const pending of this.#pending.values()) {
      pending.reject(new DOMException("The runner was stopped", "AbortError"));
    }
    this.#pending.clear();
  }
}
