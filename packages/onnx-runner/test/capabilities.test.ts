import assert from "node:assert/strict";
import test from "node:test";

import {
  BackendRegistry,
  BackendUnavailableError,
  UnsupportedCapabilityError,
  supports,
  type BackendCapabilities,
} from "../src/index.js";
import { CAPABILITIES, FakeFactory, makeRequest } from "./helpers.js";

const mpsCapabilities: BackendCapabilities = {
  backend: "onnx",
  operations: ["classify"],
  precisions: ["fp32", "fp16", "bf16"],
  devices: ["cpu", "mps"],
  dynamic_batching: true,
  precision_profiles: [
    { device: "cpu", precision: "fp32" },
    { device: "mps", precision: "fp32" },
    { device: "mps", precision: "fp16" },
  ],
  validated_models: [],
};

test("serves only the explicitly requested backend", () => {
  const registry = new BackendRegistry([new FakeFactory()]);
  assert.deepEqual(registry.capabilities(), [CAPABILITIES]);
  assert.equal(registry.require("onnx").name, "onnx");
});

test("a missing backend is unavailable and never falls back", () => {
  const registry = new BackendRegistry([new FakeFactory()]);
  for (const backend of ["pytorch", "mlx"] as const) {
    assert.throws(
      () => registry.require(backend),
      (error: unknown) =>
        error instanceof BackendUnavailableError &&
        error.message === `backend ${backend} is not installed; no fallback was attempted`,
    );
  }
  assert.throws(() => new BackendRegistry().require("onnx"), BackendUnavailableError);
});

test("rejects duplicate backend registration", () => {
  assert.throws(
    () => new BackendRegistry([new FakeFactory(), new FakeFactory()]),
    /backend already registered: onnx/,
  );
});

test("rejects unadvertised precision, operation and device", () => {
  const registry = new BackendRegistry([new FakeFactory()]);
  const request = makeRequest();
  assert.throws(
    () => registry.assertSupported({ ...request, precision: "bf16" }, "decide-340m", "cpu"),
    (error: unknown) =>
      error instanceof UnsupportedCapabilityError && /no fallback was attempted/.test(error.message),
  );
  assert.throws(
    () => registry.assertSupported(request, "decide-340m", "cuda"),
    UnsupportedCapabilityError,
  );
  assert.throws(
    () =>
      registry.assertSupported(
        { ...request, operation: "extract_entities" },
        "decide-340m",
        "cpu",
      ),
    UnsupportedCapabilityError,
  );
});

test("validated_models restricts the canonical model id, not the raw alias", () => {
  const registry = new BackendRegistry([
    new FakeFactory({ ...CAPABILITIES, validated_models: ["decide-340m"] }),
  ]);
  const alias = makeRequest({ model: `fastino/GLiNER2.5-Decide@${"a".repeat(40)}` });
  registry.assertSupported(alias, "decide-340m", "cpu");
  assert.throws(
    () => registry.assertSupported(makeRequest({ model: "decide-1b" }), "decide-1b", "cpu"),
    UnsupportedCapabilityError,
  );
});

test("negotiates device and precision profiles together", () => {
  const request = (precision: "fp32" | "fp16" | "bf16") => ({
    operation: "classify" as const,
    model: "decide-340m",
    precision,
  });
  assert.equal(supports(mpsCapabilities, request("fp16"), "mps"), true);
  assert.equal(supports(mpsCapabilities, request("fp32"), "mps"), true);
  assert.equal(supports(mpsCapabilities, request("bf16"), "mps"), false);
  assert.equal(supports(mpsCapabilities, request("fp16"), "cpu"), false);
  assert.equal(supports(mpsCapabilities, request("fp32"), "cuda"), false);
  const noProfiles = { ...mpsCapabilities, precision_profiles: [] };
  assert.equal(supports(noProfiles, request("fp16"), "cpu"), true);
});

test("device matching is exact, so cuda:0 does not match cuda", () => {
  const cuda = { ...CAPABILITIES, devices: ["cuda"], precision_profiles: [] };
  const request = { operation: "classify" as const, model: "m", precision: "fp32" as const };
  assert.equal(supports(cuda, request, "cuda"), true);
  assert.equal(supports(cuda, request, "cuda:0"), false);
});
