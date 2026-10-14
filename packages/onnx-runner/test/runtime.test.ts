import assert from "node:assert/strict";
import { setTimeout as sleep } from "node:timers/promises";
import test from "node:test";

import {
  BackendRegistry,
  BackendUnavailableError,
  ModelDownloadRequiredError,
  ModelIntegrityError,
  ModelMemoryLimitError,
  RequestCancelledError,
  Runtime,
  RunnerClosedError,
  UnsupportedCapabilityError,
  MAX_MODEL_IDLE_TTL_SECONDS,
  minimumMemoryBytes,
  type RuntimeConfig,
} from "../src/index.js";
import {
  FakeBackend,
  FakeFactory,
  Gate,
  PINNED_1B,
  PINNED_340M,
  installedCatalog,
  makeRequest,
  until,
} from "./helpers.js";

const baseConfig: RuntimeConfig = {
  device: "cpu",
  queueCapacity: 256,
  maxBatchSize: 16,
  batchWindowMs: 2,
  modelIdleTtlSeconds: 0,
  memoryLimitBytes: 64 * 1024 ** 3,
};

function makeRuntime(
  options: {
    config?: Partial<RuntimeConfig>;
    factory?: FakeFactory;
    memoryProbe?: (device: string) => number;
    totalMemory?: () => number;
  } = {},
) {
  const factory = options.factory ?? new FakeFactory();
  const runtime = new Runtime({
    catalog: installedCatalog(),
    backends: new BackendRegistry([factory]),
    config: { ...baseConfig, ...options.config },
    memoryProbe: options.memoryProbe ?? (() => 0),
    ...(options.totalMemory === undefined ? {} : { totalMemory: options.totalMemory }),
  });
  return { runtime, factory };
}

const small = makeRequest({ model: "decide-1b" });

test("one backend and one batch serve concurrent requests for the same identity", async () => {
  const { runtime, factory } = makeRuntime();
  const responses = await Promise.all(
    [1, 2, 3].map((n) => runtime.infer(makeRequest({ text: "x".repeat(n) }))),
  );
  assert.equal(responses.length, 3);
  assert.equal(factory.created.length, 1);
  assert.deepEqual(factory.backends[0]?.batchSizes, [3]);
  assert.deepEqual(runtime.loadedState(), {
    modelId: "decide-340m",
    revision: PINNED_340M,
    backend: "onnx",
    profile: { device: "cpu", precision: "fp32" },
    activeRequests: 0,
  });
  await runtime.close();
  assert.equal(runtime.accepting, false);
  assert.equal(runtime.loadedState(), undefined);
  assert.equal(factory.backends[0]?.closed, true);
});

test("hands the resolved artifact directory and identity to the backend factory", async () => {
  const { runtime, factory } = makeRuntime();
  await runtime.infer(makeRequest({ precision: "fp16" }));
  const [created] = factory.created;
  assert.equal(created?.model.path, "/models/decide-340m");
  assert.equal(created?.model.revision, PINNED_340M);
  assert.equal(created?.model.parameterCount, 486_444_053);
  assert.equal(created?.device, "cpu");
  assert.equal(created?.precision, "fp16");
  await runtime.close();
});

test("aliases of one catalog model share the resident worker", async () => {
  const { runtime, factory } = makeRuntime();
  await runtime.infer(makeRequest({ model: "decide-340m" }));
  await runtime.infer(makeRequest({ model: "fastino/GLiNER2.5-Decide" }));
  await runtime.infer(makeRequest({ model: `decide-340m@${PINNED_340M}` }));
  assert.equal(factory.created.length, 1);
  await runtime.close();
});

test("replacing the resident model closes the old backend before loading the new one", async () => {
  const factory = new FakeFactory();
  const observations: string[] = [];
  factory.configure = (backend, model) => {
    const load = backend.load.bind(backend);
    backend.load = async () => {
      observations.push(
        `${model.modelId} loading; closed=[${factory.backends
          .filter((other) => other !== backend && other.closed)
          .map((other) => other.label)
          .join()}]`,
      );
      await load();
    };
  };
  const { runtime } = makeRuntime({ factory });
  await runtime.infer(makeRequest());
  await runtime.infer(small);
  assert.deepEqual(observations, [
    "decide-340m loading; closed=[]",
    `decide-1b loading; closed=[decide-340m@${PINNED_340M}]`,
  ]);
  assert.equal(runtime.loadedState()?.modelId, "decide-1b");
  await runtime.close();
});

test("evicts the model after the idle TTL and reloads on demand", async () => {
  const { runtime, factory } = makeRuntime({ config: { modelIdleTtlSeconds: 0.03 } });
  await runtime.infer(makeRequest());
  assert.notEqual(runtime.loadedState(), undefined);
  await until(() => runtime.loadedState() === undefined, "idle eviction");
  assert.equal(factory.backends[0]?.closed, true);
  await runtime.infer(makeRequest());
  assert.equal(factory.created.length, 2);
  await runtime.close();
});

test("a request during the idle window cancels eviction", async () => {
  const { runtime, factory } = makeRuntime({ config: { modelIdleTtlSeconds: 0.08 } });
  await runtime.infer(makeRequest());
  await sleep(40);
  await runtime.infer(makeRequest());
  await sleep(50);
  assert.equal(factory.backends[0]?.closed, false);
  await until(() => runtime.loadedState() === undefined, "idle eviction");
  await runtime.close();
});

test("a load waits for an in-progress idle eviction to finish closing", async () => {
  const factory = new FakeFactory();
  const closeGate = new Gate();
  factory.configure = (backend) => {
    backend.closeGate = closeGate;
  };
  const { runtime } = makeRuntime({ factory, config: { modelIdleTtlSeconds: 0.02 } });
  await runtime.infer(makeRequest());
  await until(() => factory.backends[0]?.events.includes("close:start") === true, "eviction start");
  const next = runtime.infer(makeRequest());
  await sleep(30);
  assert.equal(factory.created.length, 1);
  closeGate.open();
  await next;
  assert.equal(factory.created.length, 2);
  await runtime.close();
});

test("rejects a model over the memory floor before creating a backend", async () => {
  const { runtime, factory } = makeRuntime({ config: { memoryLimitBytes: 3_000_000_000 } });
  await assert.rejects(runtime.infer(makeRequest()), (error: unknown) => {
    assert.ok(error instanceof ModelMemoryLimitError);
    assert.equal(error.model, "decide-340m");
    assert.equal(error.profile, "cpu/fp32");
    assert.equal(error.limitBytes, 3_000_000_000);
    assert.equal(error.requiredBytes, 3_650_112_851);
    assert.equal(error.observedBytes, null);
    assert.match(error.message, /estimated minimum is 3650112851 bytes/);
    assert.match(error.message, /no fallback was attempted/);
    return true;
  });
  assert.equal(factory.created.length, 0);
  assert.equal(runtime.loadedState(), undefined);
  await runtime.close();
});

test("a failed memory pre-check leaves the resident model untouched", async () => {
  const { runtime, factory } = makeRuntime({ config: { memoryLimitBytes: 4_000_000_000 } });
  await runtime.infer(makeRequest());
  await assert.rejects(runtime.infer(small), ModelMemoryLimitError);
  assert.equal(runtime.loadedState()?.modelId, "decide-340m");
  assert.equal(factory.backends[0]?.closed, false);
  await runtime.infer(makeRequest());
  assert.equal(factory.created.length, 1);
  await runtime.close();
});

test("post-load memory pressure unloads the model and reports the observation", async () => {
  const { runtime, factory } = makeRuntime({
    config: { memoryLimitBytes: 4_000_000_000 },
    memoryProbe: () => 5_000_000_000,
  });
  await assert.rejects(runtime.infer(makeRequest()), (error: unknown) => {
    assert.ok(error instanceof ModelMemoryLimitError);
    assert.equal(error.observedBytes, 5_000_000_000);
    assert.equal(error.requiredBytes, 3_650_112_851);
    assert.match(error.message, /observed 5000000000 bytes after load/);
    return true;
  });
  assert.equal(factory.backends[0]?.closed, true);
  assert.equal(runtime.loadedState(), undefined);
  await runtime.close();
});

test("memory limits are inclusive at the boundary", async () => {
  const required = minimumMemoryBytes(486_444_053, "fp32");
  const { runtime } = makeRuntime({
    config: { memoryLimitBytes: required },
    memoryProbe: () => required,
  });
  await runtime.infer(makeRequest());
  await runtime.close();
});

test("the memory estimate uses the requested precision", async () => {
  assert.equal(minimumMemoryBytes(486_444_053, "fp32"), 3_650_112_851);
  assert.equal(minimumMemoryBytes(486_444_053, "fp16"), 2_093_491_881);
  assert.equal(minimumMemoryBytes(287_355_159, "fp32"), 2_375_943_929);
  assert.equal(minimumMemoryBytes(1_188_796_693, "fp32"), 8_145_169_747);
  assert.equal(minimumMemoryBytes(1_188_796_693, "fp16"), 4_341_020_329);
  const { runtime } = makeRuntime({ config: { memoryLimitBytes: 3_000_000_000 } });
  await runtime.infer(makeRequest({ precision: "fp16" }));
  await assert.rejects(runtime.infer(makeRequest({ precision: "fp32" })), ModelMemoryLimitError);
  await runtime.close();
});

test("the memory limit defaults to 75% of physical memory", () => {
  const { runtime } = makeRuntime({
    config: { memoryLimitBytes: undefined },
    totalMemory: () => 1000,
  });
  assert.equal(runtime.memoryLimitBytes, 750);
  assert.equal(makeRuntime({ config: { memoryLimitBytes: 123 } }).runtime.memoryLimitBytes, 123);
});

test("recovers after a backend load failure", async () => {
  const factory = new FakeFactory();
  factory.configure = (backend) => {
    if (factory.created.length === 0) {
      backend.loadError = new Error("corrupt graph");
    }
  };
  const { runtime } = makeRuntime({ factory });
  await assert.rejects(runtime.infer(makeRequest()), /corrupt graph/);
  assert.equal(runtime.loadedState(), undefined);
  assert.equal(factory.backends[0]?.closed, true);
  await runtime.infer(makeRequest());
  assert.equal(runtime.loadedState()?.modelId, "decide-340m");
  await runtime.close();
});

test("recovers after the backend factory throws", async () => {
  const { runtime, factory } = makeRuntime();
  factory.createError = new Error("factory exploded");
  await assert.rejects(runtime.infer(makeRequest()), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.equal(error.message, "factory exploded");
    assert.equal(error instanceof UnsupportedCapabilityError, false);
    return true;
  });
  factory.createError = undefined;
  await runtime.infer(makeRequest());
  await runtime.close();
});

test("recovers when closing the previous model fails", async () => {
  const factory = new FakeFactory();
  const { runtime } = makeRuntime({ factory });
  await runtime.infer(makeRequest());
  const first = factory.backends[0] as FakeBackend;
  first.close = async () => {
    throw new Error("close failed");
  };
  await assert.rejects(runtime.infer(small), /close failed/);
  await runtime.infer(small);
  assert.equal(runtime.loadedState()?.modelId, "decide-1b");
  await runtime.close();
});

test("serializes overlapping loads of different models", async () => {
  const factory = new FakeFactory();
  const loadGate = new Gate();
  let concurrent = 0;
  let maximum = 0;
  factory.configure = (backend) => {
    if (factory.created.length === 0) {
      backend.loadGate = loadGate;
    }
    const load = backend.load.bind(backend);
    backend.load = async () => {
      concurrent += 1;
      maximum = Math.max(maximum, concurrent);
      try {
        await load();
      } finally {
        concurrent -= 1;
      }
    };
  };
  const { runtime } = makeRuntime({ factory });
  const first = runtime.infer(makeRequest());
  await until(() => factory.created.length === 1, "first load start");
  const second = runtime.infer(small);
  await sleep(20);
  assert.equal(factory.created.length, 1);
  loadGate.open();
  await Promise.all([first, second]);
  assert.equal(maximum, 1);
  assert.equal(factory.backends[0]?.closed, true);
  assert.equal(runtime.loadedState()?.modelId, "decide-1b");
  await runtime.close();
});

test("alternating models never overlap and every request completes", async () => {
  const { runtime, factory } = makeRuntime();
  const requests = Array.from({ length: 6 }, (_, index) => (index % 2 === 0 ? makeRequest() : small));
  const responses = await Promise.all(requests.map((request) => runtime.infer(request)));
  assert.equal(responses.length, 6);
  const open = factory.backends.filter((backend) => !backend.closed);
  assert.equal(open.length, 1);
  await runtime.close();
  assert.ok(factory.backends.every((backend) => backend.closed));
});

test("close waits for active requests, then rejects new ones", async () => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend) => {
    backend.gate = gate;
  };
  const { runtime } = makeRuntime({ factory });
  const inflight = runtime.infer(makeRequest());
  await until(() => factory.backends[0]?.batches.length === 1, "dispatch");
  let closed = false;
  const closing = runtime.close().then(() => {
    closed = true;
  });
  await sleep(20);
  assert.equal(closed, false);
  assert.equal(runtime.accepting, false);
  await assert.rejects(runtime.infer(makeRequest()), (error: unknown) => {
    assert.ok(error instanceof RunnerClosedError);
    assert.equal(error.message, "runtime is closed");
    return true;
  });
  gate.open();
  await inflight;
  await closing;
  assert.equal(factory.backends[0]?.closed, true);
});

test("passes scheduler settings through to the worker", async () => {
  const { runtime, factory } = makeRuntime({ config: { maxBatchSize: 3 } });
  await Promise.all(Array.from({ length: 5 }, () => runtime.infer(makeRequest())));
  assert.deepEqual(factory.backends[0]?.batchSizes, [3, 2]);
  await runtime.close();
});

test("metrics survive model switches", async () => {
  const { runtime } = makeRuntime();
  await runtime.infer(makeRequest());
  await runtime.infer(small);
  assert.equal(runtime.metrics.snapshot()["requests_completed_total"], 2);
  await runtime.close();
});

test("rejects unknown, uninstalled and malformed model references", async () => {
  const { runtime, factory } = makeRuntime();
  await assert.rejects(runtime.infer(makeRequest({ model: "nope" })), (error: unknown) => {
    assert.ok(error instanceof ModelIntegrityError);
    assert.equal(error instanceof ModelDownloadRequiredError, false);
    return true;
  });
  await assert.rejects(
    runtime.infer(makeRequest({ model: "decide-multilingual" })),
    (error: unknown) =>
      error instanceof ModelDownloadRequiredError &&
      error.destination === "/models" &&
      error.revision === "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d",
  );
  await assert.rejects(
    runtime.infer(makeRequest({ model: "decide-340m@../../etc" })),
    (error: unknown) =>
      error instanceof ModelIntegrityError && !(error instanceof ModelDownloadRequiredError),
  );
  await assert.rejects(
    runtime.infer(makeRequest({ model: `decide-340m@${PINNED_1B}` })),
    ModelDownloadRequiredError,
  );
  assert.equal(factory.created.length, 0);
  await runtime.close();
});

test("backend selection precedes model resolution", async () => {
  const { runtime } = makeRuntime();
  await assert.rejects(
    runtime.infer(makeRequest({ backend: "pytorch", model: "decide-multilingual" })),
    BackendUnavailableError,
  );
  await runtime.close();
});

test("checks capabilities on every request, even for the resident identity", async () => {
  const { runtime, factory } = makeRuntime();
  await runtime.infer(makeRequest());
  const entities = makeRequest({
    operation: "extract_entities",
    schema: { kind: "entities", labels: ["person"] },
  });
  await assert.rejects(runtime.infer(entities), UnsupportedCapabilityError);
  await assert.rejects(
    runtime.infer(makeRequest({ precision: "bf16" })),
    UnsupportedCapabilityError,
  );
  assert.equal(factory.created.length, 1);
  assert.equal(factory.backends[0]?.batches.length, 1);
  await runtime.close();
});

test("explicit backend only: mlx and pytorch are never aliased to onnx", async () => {
  const { runtime, factory } = makeRuntime();
  for (const backend of ["pytorch", "mlx"]) {
    await assert.rejects(runtime.infer(makeRequest({ backend })), BackendUnavailableError);
  }
  assert.equal(factory.created.length, 0);
  await runtime.close();
});

test("close waits for a model load that is still in progress and unloads its result", async () => {
  const factory = new FakeFactory();
  const loadGate = new Gate();
  factory.configure = (backend) => {
    backend.loadGate = loadGate;
  };
  const { runtime } = makeRuntime({ factory });
  const pending = runtime.infer(makeRequest());
  const rejected = assert.rejects(pending, RunnerClosedError);
  await until(() => factory.backends[0]?.events.includes("load:start") === true, "load start");

  let closed = false;
  const closing = runtime.close().then(() => {
    closed = true;
  });
  await sleep(30);
  assert.equal(closed, false, "close() must not resolve while the load is in progress");
  assert.equal(factory.backends[0]?.closed, false);

  loadGate.open();
  await closing;
  await rejected;
  assert.equal(factory.backends[0]?.closed, true, "the freshly loaded backend is closed before close() resolves");
  assert.equal(runtime.loadedState(), undefined);
});

test("close also waits for a load that is waiting for the previous model to drain", async () => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend, model) => {
    if (model.modelId === "decide-340m") {
      backend.gate = gate;
    }
  };
  const { runtime } = makeRuntime({ factory });
  const first = runtime.infer(makeRequest());
  await until(() => factory.backends[0]?.batches.length === 1, "first dispatch");
  const switching = runtime.infer(small);
  const switchingRejected = assert.rejects(switching, RunnerClosedError);
  await sleep(10);
  const closing = runtime.close();
  gate.open();
  await first;
  await closing;
  await switchingRejected;
  assert.equal(factory.created.length, 1);
  assert.equal(factory.backends[0]?.closed, true);
});

test("a cancelled request never starts, or waits to perform, a model switch", async () => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend, model) => {
    if (model.modelId === "decide-340m") {
      backend.gate = gate;
    }
  };
  const { runtime } = makeRuntime({ factory });

  const preAborted = AbortSignal.abort();
  await assert.rejects(runtime.infer(makeRequest(), preAborted), RequestCancelledError);
  assert.equal(factory.created.length, 0);

  const resident = runtime.infer(makeRequest());
  await until(() => factory.backends[0]?.batches.length === 1, "resident dispatch");

  // Aborted while the switch to decide-1b waits for the resident model to drain.
  const controller = new AbortController();
  const waiting = runtime.infer(small, controller.signal);
  const waitingRejected = assert.rejects(waiting, RequestCancelledError);
  await sleep(20);
  controller.abort();
  gate.open();
  await resident;
  await waitingRejected;
  assert.equal(factory.created.length, 1, "no backend was created for the cancelled request");
  assert.equal(factory.backends[0]?.closed, false);
  assert.equal(runtime.loadedState()?.modelId, "decide-340m");

  await runtime.infer(makeRequest());
  await runtime.close();
});

test("the idle TTL is bounded by what a timer can hold", () => {
  const options = (modelIdleTtlSeconds: number) => makeRuntime({ config: { modelIdleTtlSeconds } });
  assert.equal(MAX_MODEL_IDLE_TTL_SECONDS, 2_147_483);
  options(MAX_MODEL_IDLE_TTL_SECONDS);
  for (const bad of [MAX_MODEL_IDLE_TTL_SECONDS + 1, 1e9, Number.POSITIVE_INFINITY, Number.NaN]) {
    assert.throws(() => options(bad), RangeError);
  }
});
