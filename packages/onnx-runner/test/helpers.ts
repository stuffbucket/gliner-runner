import { createServer as createNetServer } from "node:net";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

import {
  type BackendCapabilities,
  type BackendFactory,
  type BackendResult,
  type InferenceBackend,
  type InferenceRequest,
  InMemoryArtifactCatalog,
  InferenceRequestSchema,
  type ResolvedModel,
  type RunnerServer,
  type ServerOptions,
  type Settings,
  createServer,
} from "../src/index.js";

export const PINNED_340M = "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90";
export const PINNED_1B = "67af949e17bebd20f05fe020a971079889e7a462";

export const CAPABILITIES: BackendCapabilities = {
  backend: "onnx",
  operations: ["classify"],
  precisions: ["fp32", "fp16"],
  devices: ["cpu"],
  dynamic_batching: true,
  precision_profiles: [
    { device: "cpu", precision: "fp32" },
    { device: "cpu", precision: "fp16" },
  ],
  validated_models: [],
};

export function makeRequest(overrides: Record<string, unknown> = {}): InferenceRequest {
  return InferenceRequestSchema.parse(wireRequest(overrides));
}

export function wireRequest(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model: "decide-340m",
    backend: "onnx",
    precision: "fp32",
    operation: "classify",
    text: "An urgent invoice",
    schema: {
      kind: "classification",
      tasks: { priority: { labels: ["urgent", "routine"] } },
    },
    ...overrides,
  };
}

export function schemaWith(labels: string[]): Record<string, unknown> {
  return { kind: "classification", tasks: { priority: { labels } } };
}

export class Gate {
  readonly promise: Promise<void>;
  open!: () => void;

  constructor() {
    this.promise = new Promise<void>((resolve) => {
      this.open = resolve;
    });
  }
}

export class FakeBackend implements InferenceBackend {
  readonly batches: InferenceRequest[][] = [];
  readonly events: string[] = [];
  loadError: Error | undefined;
  inferError: Error | undefined;
  gate: Gate | undefined;
  loadGate: Gate | undefined;
  closeGate: Gate | undefined;
  resultCount: ((requests: readonly InferenceRequest[]) => number) | undefined;
  loaded = false;
  closed = false;

  constructor(
    readonly capabilities: BackendCapabilities = CAPABILITIES,
    readonly label = "backend",
  ) {}

  async load(): Promise<void> {
    this.events.push("load:start");
    await this.loadGate?.promise;
    if (this.loadError !== undefined) {
      throw this.loadError;
    }
    this.loaded = true;
    this.events.push("load:end");
  }

  async inferBatch(requests: readonly InferenceRequest[]): Promise<BackendResult[]> {
    this.batches.push([...requests]);
    this.events.push(`infer:${requests.length}`);
    await this.gate?.promise;
    if (this.inferError !== undefined) {
      throw this.inferError;
    }
    const results = requests.map((request) => ({
      output: { label: "useful" },
      usage: { inputTokens: request.text.length, outputTokens: 0 as const },
    }));
    const count = this.resultCount?.(requests);
    return count === undefined ? results : results.slice(0, count);
  }

  async close(): Promise<void> {
    this.events.push("close:start");
    await this.closeGate?.promise;
    this.closed = true;
    this.events.push("close:end");
  }

  get batchSizes(): number[] {
    return this.batches.map((batch) => batch.length);
  }
}

export class FakeFactory implements BackendFactory {
  readonly name = "onnx" as const;
  readonly created: { model: ResolvedModel; device: string; precision: string; backend: FakeBackend }[] =
    [];
  createError: Error | undefined;
  configure: ((backend: FakeBackend, model: ResolvedModel) => void) | undefined;

  constructor(readonly advertised: BackendCapabilities = CAPABILITIES) {}

  capabilities(): BackendCapabilities {
    return this.advertised;
  }

  create(model: ResolvedModel, device: string, precision: string): FakeBackend {
    if (this.createError !== undefined) {
      throw this.createError;
    }
    const backend = new FakeBackend(this.advertised, `${model.modelId}@${model.revision}`);
    this.configure?.(backend, model);
    this.created.push({ model, device, precision, backend });
    return backend;
  }

  get backends(): FakeBackend[] {
    return this.created.map((entry) => entry.backend);
  }
}

export function installedCatalog(): InMemoryArtifactCatalog {
  return new InMemoryArtifactCatalog({
    destination: "/models",
    installed: {
      "decide-340m": { [PINNED_340M]: "/models/decide-340m" },
      "decide-1b": { [PINNED_1B]: "/models/decide-1b" },
    },
  });
}

export interface TestServer {
  server: RunnerServer;
  factory: FakeFactory;
  catalog: InMemoryArtifactCatalog;
  base: string;
  post(path: string, body: unknown): Promise<Response>;
  get(path: string): Promise<Response>;
}

export async function startServer(
  options: {
    settings?: Partial<Settings>;
    factory?: FakeFactory;
    catalog?: InMemoryArtifactCatalog;
    log?: ServerOptions["log"];
  } = {},
): Promise<TestServer> {
  const factory = options.factory ?? new FakeFactory();
  const catalog = options.catalog ?? installedCatalog();
  const server = createServer({
    catalog,
    backends: [factory],
    settings: { host: "127.0.0.1", port: 0, batchWindowMs: 5, ...options.settings },
    ...(options.log === undefined ? {} : { log: options.log }),
  });
  const address = await server.listen();
  const base = `http://127.0.0.1:${address.port}`;
  return {
    server,
    factory,
    catalog,
    base,
    post: (path, body) =>
      fetch(`${base}${path}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: typeof body === "string" ? body : JSON.stringify(body),
      }),
    get: (path) => fetch(`${base}${path}`),
  };
}

export async function until(
  predicate: () => boolean,
  description = "condition",
  timeoutMs = 3000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) {
      throw new Error(`timed out waiting for ${description}`);
    }
    await sleep(2);
  }
}

export const CLI = fileURLToPath(new URL("../src/cli.js", import.meta.url));

export async function freePort(): Promise<number> {
  const server = createNetServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as { port: number };
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return port;
}
