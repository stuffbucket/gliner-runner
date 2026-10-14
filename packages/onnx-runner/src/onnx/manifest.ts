import { posix } from "node:path";

import type { KnownModel } from "../catalog.js";
import { ModelIntegrityError } from "../errors.js";

export const MANIFEST_NAME = "manifest.json";
export const MANIFEST_KIND = "gliner-runner-onnx-classifier";
export const MANIFEST_FORMAT_VERSION = 1;
export const PREPROCESSING_NAME = "gliner2.json";
export const GRAPH_NAME = "model.onnx";
export const REQUIRED_FILES = [GRAPH_NAME, PREPROCESSING_NAME, "tokenizer.json", "tokenizer_config.json"] as const;

export interface TensorSpec {
  name: string;
  type: string;
  shape: readonly string[];
}

/** The graph contract the runtime feeds and reads; the manifest must declare exactly this. */
export const GRAPH_INPUTS: readonly TensorSpec[] = [
  { name: "input_ids", type: "tensor(int64)", shape: ["batch", "sequence"] },
  { name: "attention_mask", type: "tensor(int64)", shape: ["batch", "sequence"] },
  { name: "label_positions", type: "tensor(int64)", shape: ["batch", "labels"] },
  { name: "label_mask", type: "tensor(bool)", shape: ["batch", "labels"] },
];
export const GRAPH_OUTPUTS: readonly TensorSpec[] = [
  { name: "logits", type: "tensor(float)", shape: ["batch", "labels"] },
];

export interface ArtifactFile {
  path: string;
  size: number;
  sha256: string;
}

/** Parity evidence recorded by the preparation tool; every field is a raw measurement. */
export interface ParityEvidence {
  absoluteTolerance: number;
  /** Largest ONNX versus reference raw logit difference recorded at preparation time. */
  maxAbsLogitDifference: number;
  /** ONNX versus the official GLiNER2 path, when recorded. */
  onnxOfficialMaxAbsLogitDifference: number | null;
  rows: number | null;
  fixtureSuite: string | null;
}

export interface OnnxManifest {
  modelId: string;
  repository: string;
  revision: string;
  parameterCount: number;
  graphPath: string;
  externalData: readonly string[];
  files: readonly ArtifactFile[];
  parity: ParityEvidence;
}

type Json = Record<string, unknown>;

function isRecord(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function same(actual: unknown, expected: unknown): boolean {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

const SHA256 = /^[a-f0-9]{64}$/;

/**
 * Strictly validates a parsed `manifest.json` against the curated model pin. Unknown extra fields are
 * tolerated (provenance such as exporter versions); every field the runtime relies on is checked.
 * Throws `ModelIntegrityError` on the first violation.
 */
export function parseManifest(raw: unknown, model: KnownModel, revision: string): OnnxManifest {
  const fail = (detail: string): never => {
    throw new ModelIntegrityError(`ONNX artifact for '${model.modelId}' at ${revision} is invalid: ${detail}`);
  };
  if (!isRecord(raw)) {
    return fail("manifest.json must be a JSON object");
  }
  const expect = (key: string, expected: unknown): void => {
    if (!same(raw[key], expected)) {
      fail(`manifest ${key} must be ${JSON.stringify(expected)}, found ${JSON.stringify(raw[key])}`);
    }
  };
  expect("format_version", MANIFEST_FORMAT_VERSION);
  expect("kind", MANIFEST_KIND);
  expect("model_id", model.modelId);
  expect("repository", model.repository);
  expect("revision", model.pinnedRevision);
  if (revision !== model.pinnedRevision) {
    fail(`revision ${revision} is not the pinned revision ${model.pinnedRevision}`);
  }
  expect("parameter_count", model.parameterCount);
  expect("operations", ["classify"]);
  expect("precision", "fp32");
  expect("quantization", "none");
  expect("provider", "CPUExecutionProvider");
  expect("opset", 20);
  expect("opset_imports", [{ domain: "ai.onnx", version: 20 }]);
  expect("preprocessing", PREPROCESSING_NAME);

  const graph = raw["graph"];
  if (!isRecord(graph)) {
    return fail("manifest graph must be an object");
  }
  if (graph["path"] !== GRAPH_NAME) {
    fail(`manifest graph.path must be ${JSON.stringify(GRAPH_NAME)}`);
  }
  if (!same(graph["inputs"], GRAPH_INPUTS)) {
    fail("manifest graph.inputs does not match the classification input contract");
  }
  if (!same(graph["outputs"], GRAPH_OUTPUTS)) {
    fail("manifest graph.outputs does not match the classification output contract");
  }
  const external = graph["external_data"];
  if (!Array.isArray(external) || external.some((entry) => typeof entry !== "string")) {
    return fail("manifest graph.external_data must be a list of file names");
  }

  const listed = raw["files"];
  if (!Array.isArray(listed) || listed.length === 0) {
    return fail("manifest files must be a non-empty list");
  }
  const files: ArtifactFile[] = [];
  const names = new Set<string>();
  for (const entry of listed) {
    if (!isRecord(entry)) {
      return fail("manifest files entries must be objects");
    }
    const { path, size, sha256 } = entry;
    if (
      typeof path !== "string" ||
      path.length === 0 ||
      path === MANIFEST_NAME ||
      posix.basename(path) !== path ||
      path === "." ||
      path === ".." ||
      path.includes("\\") ||
      path.includes("\0")
    ) {
      return fail(`manifest lists an unacceptable file path ${JSON.stringify(path)}`);
    }
    if (names.has(path)) {
      return fail(`manifest lists ${path} more than once`);
    }
    if (typeof size !== "number" || !Number.isSafeInteger(size) || size < 0) {
      return fail(`manifest size of ${path} must be a non-negative integer`);
    }
    if (typeof sha256 !== "string" || !SHA256.test(sha256)) {
      return fail(`manifest sha256 of ${path} must be 64 lowercase hex characters`);
    }
    if (Object.keys(entry).some((key) => key !== "path" && key !== "size" && key !== "sha256")) {
      return fail(`manifest file entry for ${path} has unexpected fields`);
    }
    names.add(path);
    files.push({ path, size, sha256 });
  }
  for (const required of REQUIRED_FILES) {
    if (!names.has(required)) {
      fail(`manifest does not list required file ${required}`);
    }
  }
  for (const sidecar of external as string[]) {
    if (!names.has(sidecar)) {
      fail(`manifest external data ${sidecar} is not a listed file`);
    }
  }
  files.sort((left, right) => (left.path < right.path ? -1 : left.path > right.path ? 1 : 0));

  return {
    modelId: model.modelId,
    repository: model.repository,
    revision,
    parameterCount: model.parameterCount,
    graphPath: GRAPH_NAME,
    externalData: [...(external as string[])],
    files,
    parity: parseParity(raw["parity"], fail),
  };
}

function parseParity(value: unknown, fail: (detail: string) => never): ParityEvidence {
  if (!isRecord(value)) {
    return fail("manifest parity evidence is missing");
  }
  const number = (key: string, required: boolean): number | null => {
    const entry = value[key];
    if (entry === undefined && !required) {
      return null;
    }
    if (typeof entry !== "number" || !Number.isFinite(entry) || entry < 0) {
      return fail(`manifest parity.${key} must be a non-negative number`);
    }
    return entry;
  };
  const absoluteTolerance = number("absolute_tolerance", true) as number;
  const maxAbsLogitDifference = number("max_abs_logit_difference", true) as number;
  const official = number("onnx_official_max_abs_logit_difference", false);
  for (const drift of [maxAbsLogitDifference, official]) {
    if (drift !== null && drift > absoluteTolerance) {
      fail(`manifest parity drift ${drift} exceeds its absolute tolerance ${absoluteTolerance}`);
    }
  }
  const suite = value["fixture_suite"];
  return {
    absoluteTolerance,
    maxAbsLogitDifference,
    onnxOfficialMaxAbsLogitDifference: official,
    rows: number("rows", false),
    fixtureSuite: typeof suite === "string" ? suite : null,
  };
}
