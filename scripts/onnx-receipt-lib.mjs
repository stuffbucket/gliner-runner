// Shared logic for the ONNX evaluation receipt: runner identity digest, receipt/ledger/build-info formats and
// the model-free consistency checks that CI runs. Used by scripts/onnx-eval-receipt.mjs (record, check) and
// scripts/verify-onnx-runtime.mjs.
//
// The receipt is produced on a developer machine that has real model artifacts. CI never runs models; it only
// asserts that the checked-in files are coherent and that the (runner identity, receipt) pair is not a replay.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

export const RECEIPT_KIND = "gliner-runner-onnx-eval-receipt";
export const RECEIPT_FORMAT_VERSION = 1;
export const TOLERANCE = 1e-4;
export const RECEIPT_PATH = "release/onnx-eval-receipt.json";
export const LEDGER_PATH = "release/onnx-eval-receipts.jsonl";
export const PACKAGE_DIR = "packages/onnx-runner";
export const BUILD_INFO_PATH = `${PACKAGE_DIR}/src/build-info.ts`;
export const CATALOG_PATH = `${PACKAGE_DIR}/src/catalog.ts`;
export const FIXTURE_DIR = "tests/fixtures/onnx-parity";
export const GOLDEN_DIR = `${FIXTURE_DIR}/goldens`;
export const CORPUS_IDS = ["english-v1", "multilingual-v1"];
export const RECORD_COMMAND = "mise run eval:onnx-receipt";
export const IDENTITY_DOMAIN = "gliner-runner-onnx-identity-v1";

const TEXT_EXTENSIONS = new Set([".ts", ".js", ".mjs", ".json", ".jsonl", ".md", ".yaml", ".yml", ".txt"]);
const HEX64 = /^[a-f0-9]{64}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

// ---------------------------------------------------------------------------------------------------------------
// Hashing and canonical JSON

export function sha256Hex(data) {
  return createHash("sha256").update(data).digest("hex");
}

/** Bytes of a file with CRLF line endings normalized to LF when the file is text. */
export function normalizedBytes(path, bytes) {
  const dot = path.lastIndexOf(".");
  const extension = dot < 0 ? "" : path.slice(dot).toLowerCase();
  if (!TEXT_EXTENSIONS.has(extension)) {
    return bytes;
  }
  return Buffer.from(bytes.toString("utf8").replace(/\r\n/g, "\n"), "utf8");
}

function sortKeys(value) {
  if (Array.isArray(value)) {
    return value.map(sortKeys);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, sortKeys(value[key])]),
    );
  }
  return value;
}

/** Canonical JSON: sorted keys, 2-space indent, trailing newline. */
export function canonicalJson(value) {
  return `${JSON.stringify(sortKeys(value), null, 2)}\n`;
}

// ---------------------------------------------------------------------------------------------------------------
// Files, git and the pnpm lockfile

function git(root, args, options = {}) {
  return execFileSync("git", ["-C", root, ...args], {
    encoding: options.buffer === true ? "buffer" : "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    maxBuffer: 1 << 30,
  });
}

/** Files under the given pathspecs that exist in the working tree, tracked or untracked-but-not-ignored. */
export function listFiles(root, pathspecs) {
  const list = (extra) =>
    git(root, ["ls-files", "-z", ...extra, "--", ...pathspecs])
      .split("\0")
      .filter((entry) => entry !== "");
  const tracked = new Set(list(["--cached"]));
  const untracked = list(["--others", "--exclude-standard"]);
  const files = [...new Set([...tracked, ...untracked])].filter((path) => existsSync(join(root, path))).sort();
  return { files, untracked: untracked.filter((path) => files.includes(path)).sort() };
}

function unquote(value) {
  return value.trim().replace(/^['"]|['"]$/g, "");
}

function section(lines, header) {
  const start = lines.indexOf(header) + 1;
  if (start === 0) {
    throw new Error(`pnpm-lock.yaml has no "${header}" section`);
  }
  let end = start;
  while (end < lines.length && (lines[end] === "" || lines[end].startsWith(" "))) {
    end += 1;
  }
  return lines.slice(start, end);
}

function splitKey(key) {
  const bare = key.split("(", 1)[0];
  const at = bare.lastIndexOf("@");
  return [bare.slice(0, at), bare.slice(at + 1)];
}

/**
 * Exact production dependency closure of a pnpm importer, as sorted `{id: name@version, resolution, line}` entries
 * where `line` is `name@version@resolution` and resolution is the lockfile's integrity (or tarball) descriptor. Ports lockfile_production_closure() from
 * scripts/check_licensing.py; optional dependencies are not followed, exactly as there.
 */
export function lockfileProductionClosure(lockfile, importer) {
  const lines = lockfile.split(/\r?\n/);
  const direct = new Map();
  let inImporter = false;
  let inDependencies = false;
  let current = "";
  for (const line of section(lines, "importers:")) {
    if (/^ {2}\S/.test(line)) {
      inImporter = unquote(line.trim().replace(/:$/, "")) === importer;
      inDependencies = false;
    } else if (inImporter && /^ {4}\S/.test(line)) {
      inDependencies = line.trim() === "dependencies:";
    } else if (inImporter && inDependencies) {
      if (/^ {6}\S/.test(line)) {
        current = unquote(line.trim().replace(/:$/, ""));
      } else if (line.trim().startsWith("version:")) {
        direct.set(current, unquote(line.slice(line.indexOf(":") + 1)).split("(", 1)[0]);
      }
    }
  }

  const resolutions = new Map();
  let owner = "";
  for (const line of section(lines, "packages:")) {
    if (/^ {2}\S/.test(line)) {
      const [name, version] = splitKey(unquote(line.trim().replace(/:$/, "")));
      owner = `${name}@${version}`;
    } else if (/^ {4}resolution:/.test(line) && owner !== "") {
      resolutions.set(owner, line.slice(line.indexOf(":") + 1).trim());
    }
  }

  const edges = new Map();
  let edgeOwner = "";
  let inEdges = false;
  for (const line of section(lines, "snapshots:")) {
    if (/^ {2}\S/.test(line)) {
      const key = unquote(line.trim().replace(/:$/, "").replace(/ \{\}$/, "").replace(/:$/, ""));
      const [name, version] = splitKey(key);
      edgeOwner = `${name}@${version}`;
      if (!edges.has(edgeOwner)) {
        edges.set(edgeOwner, []);
      }
      inEdges = false;
    } else if (/^ {4}\S/.test(line)) {
      inEdges = line.trim() === "dependencies:";
    } else if (inEdges && edgeOwner !== "" && /^ {6}\S/.test(line)) {
      const text = line.trim();
      const colon = text.indexOf(":");
      edges.get(edgeOwner).push(`${unquote(text.slice(0, colon))}@${unquote(text.slice(colon + 1)).split("(", 1)[0]}`);
    }
  }

  const closure = new Set();
  const pending = [...direct].map(([name, version]) => `${name}@${version}`);
  while (pending.length > 0) {
    const item = pending.pop();
    if (closure.has(item)) {
      continue;
    }
    closure.add(item);
    pending.push(...(edges.get(item) ?? []));
  }
  if (closure.size === 0) {
    throw new Error(`pnpm-lock.yaml has no production dependencies for importer ${importer}`);
  }
  return [...closure].sort().map((item) => {
    const resolution = resolutions.get(item);
    if (resolution === undefined) {
      throw new Error(`pnpm-lock.yaml has no resolution for ${item}`);
    }
    return { id: item, resolution, line: `${item}@${resolution}` };
  });
}

/** Curated catalog pins, parsed from the catalog source so identity does not depend on a build. */
export function readCatalogPins(root) {
  const source = readFileSync(join(root, CATALOG_PATH), "utf8");
  const pattern = /modelId:\s*"([^"]+)",\s*repository:\s*"([^"]+)",\s*pinnedRevision:\s*"([a-f0-9]{40})"/g;
  const pins = [...source.matchAll(pattern)].map((match) => ({
    model_id: match[1],
    repository: match[2],
    revision: match[3],
  }));
  if (pins.length === 0) {
    throw new Error(`${CATALOG_PATH}: no curated model pins found`);
  }
  return pins;
}

export function templateDir(pin) {
  return `${PACKAGE_DIR}/templates/${pin.repository.replace("/", "--")}/${pin.revision}`;
}

export function goldenPath(modelId, corpusId) {
  return `${GOLDEN_DIR}/${modelId}-${corpusId}.json`;
}

export function corpusPath(corpusId) {
  return `${FIXTURE_DIR}/corpus-${corpusId}.json`;
}

function readJson(root, path) {
  return JSON.parse(readFileSync(join(root, path), "utf8"));
}

/** sha256 of a tree file, with line endings normalized for text files. */
export function fileSha256(root, path) {
  return sha256Hex(normalizedBytes(path, readFileSync(join(root, path))));
}

// ---------------------------------------------------------------------------------------------------------------
// Runner identity

/**
 * Behavior-relevant inputs of the runner as sorted `{path, sha256}` entries:
 * the TypeScript sources (minus the generated build-info), package.json (without its release-managed `version`),
 * tsconfig.json, templates, the production dependency closure from pnpm-lock.yaml, the parity corpora and goldens,
 * and the curated catalog pins.
 */
export function identityEntries(root) {
  const { files, untracked } = listFiles(root, [
    `${PACKAGE_DIR}/src`,
    `${PACKAGE_DIR}/package.json`,
    `${PACKAGE_DIR}/tsconfig.json`,
    `${PACKAGE_DIR}/templates`,
    FIXTURE_DIR,
  ]);
  const entries = [];
  for (const path of files) {
    if (path === BUILD_INFO_PATH) {
      continue;
    }
    if (path === `${PACKAGE_DIR}/package.json`) {
      const { version: _version, ...rest } = readJson(root, path);
      entries.push({ path, sha256: sha256Hex(canonicalJson(rest)) });
    } else {
      entries.push({ path, sha256: fileSha256(root, path) });
    }
  }
  for (const dependency of lockfileProductionClosure(readFileSync(join(root, "pnpm-lock.yaml"), "utf8"), PACKAGE_DIR)) {
    entries.push({ path: `pnpm-lock:${dependency.id}`, sha256: sha256Hex(dependency.line) });
  }
  for (const pin of readCatalogPins(root)) {
    entries.push({ path: `catalog:${pin.model_id}`, sha256: sha256Hex(`${pin.repository}@${pin.revision}`) });
  }
  entries.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : a.sha256 < b.sha256 ? -1 : 1));
  return { entries, untracked: untracked.filter((path) => path !== BUILD_INFO_PATH) };
}

export function identityDigest(entries) {
  const body = entries.map((entry) => `${entry.path}\t${entry.sha256}\n`).join("");
  return sha256Hex(`${IDENTITY_DOMAIN}\n${body}`);
}

export function computeIdentity(root) {
  const { entries, untracked } = identityEntries(root);
  return { sha256: identityDigest(entries), entries, untracked };
}

export function packageVersion(root) {
  return readJson(root, `${PACKAGE_DIR}/package.json`).version;
}

// ---------------------------------------------------------------------------------------------------------------
// Build info, ledger and receipt writing

export function renderBuildInfo(info) {
  return [
    "// GENERATED by `mise run eval:onnx-receipt` (scripts/onnx-eval-receipt.mjs). Do not edit by hand.",
    "// Excluded from the runner identity digest; see docs/onnx-runtime.md.",
    "export interface BuildInfo {",
    "  /** Package version when the receipt was recorded (the CLI reports the live package version). */",
    "  readonly version: string;",
    "  /** Runner identity digest the evaluation receipt was recorded for. */",
    "  readonly identitySha256: string;",
    "  /** sha256 of release/onnx-eval-receipt.json. */",
    "  readonly evalReceiptSha256: string;",
    "  /** run_id of the evaluation receipt. */",
    "  readonly evalRunId: string;",
    "}",
    "",
    "export const BUILD_INFO: BuildInfo = {",
    `  version: ${JSON.stringify(info.version)},`,
    `  identitySha256: ${JSON.stringify(info.identitySha256)},`,
    `  evalReceiptSha256: ${JSON.stringify(info.evalReceiptSha256)},`,
    `  evalRunId: ${JSON.stringify(info.evalRunId)},`,
    "};",
    "",
  ].join("\n");
}

export function parseBuildInfo(text) {
  const pick = (key) => new RegExp(`^  ${key}: "([^"]*)",$`, "m").exec(text)?.[1];
  const info = {
    version: pick("version"),
    identitySha256: pick("identitySha256"),
    evalReceiptSha256: pick("evalReceiptSha256"),
    evalRunId: pick("evalRunId"),
  };
  return Object.values(info).every((value) => value !== undefined) ? info : undefined;
}

export function ledgerLine(entry) {
  return JSON.stringify({
    receipt_sha256: entry.receipt_sha256,
    identity_sha256: entry.identity_sha256,
    run_id: entry.run_id,
    created_at: entry.created_at,
    previous_receipt_sha256: entry.previous_receipt_sha256,
  });
}

/**
 * Writes the receipt, the generated build-info and one ledger line. Everything is computed before the first write.
 * `receipt` is the full receipt object; returns its sha256.
 */
export function writeReceiptArtifacts(root, receipt) {
  const text = canonicalJson(receipt);
  const sha = sha256Hex(text);
  const buildInfo = renderBuildInfo({
    version: receipt.runner.version,
    identitySha256: receipt.runner.identity_sha256,
    evalReceiptSha256: sha,
    evalRunId: receipt.run_id,
  });
  const line = `${ledgerLine({
    receipt_sha256: sha,
    identity_sha256: receipt.runner.identity_sha256,
    run_id: receipt.run_id,
    created_at: receipt.created_at,
    previous_receipt_sha256: receipt.previous_receipt_sha256,
  })}\n`;
  const ledgerPath = join(root, LEDGER_PATH);
  const existing = existsSync(ledgerPath) ? readFileSync(ledgerPath, "utf8") : "";
  mkdirSync(dirname(ledgerPath), { recursive: true });
  writeFileSync(join(root, RECEIPT_PATH), text);
  writeFileSync(join(root, BUILD_INFO_PATH), buildInfo);
  if (existing !== "" && !existing.endsWith("\n")) {
    appendFileSync(ledgerPath, "\n");
  }
  appendFileSync(ledgerPath, line);
  return sha;
}

// ---------------------------------------------------------------------------------------------------------------
// Receipt schema

const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const isString = (value) => typeof value === "string" && value !== "";
const isHex = (value) => typeof value === "string" && HEX64.test(value);
const isCount = (value) => Number.isInteger(value) && value >= 0;

/** Schema errors of a parsed receipt (empty when valid). */
export function validateReceiptShape(receipt) {
  const errors = [];
  const need = (condition, message) => {
    if (!condition) {
      errors.push(message);
    }
  };
  if (!isObject(receipt)) {
    return ["receipt is not a JSON object"];
  }
  need(receipt.format_version === RECEIPT_FORMAT_VERSION, `format_version must be ${RECEIPT_FORMAT_VERSION}`);
  need(receipt.kind === RECEIPT_KIND, `kind must be "${RECEIPT_KIND}"`);
  need(typeof receipt.run_id === "string" && UUID.test(receipt.run_id), "run_id must be a UUID");
  need(isString(receipt.created_at) && !Number.isNaN(Date.parse(receipt.created_at)), "created_at must be an ISO timestamp");
  need(receipt.previous_receipt_sha256 === null || isHex(receipt.previous_receipt_sha256), "previous_receipt_sha256 must be null or a sha256");
  need(isObject(receipt.runner) && isString(receipt.runner.version) && isHex(receipt.runner.identity_sha256), "runner must have version and identity_sha256");
  need(Array.isArray(receipt.models), "models must be an array");
  for (const model of Array.isArray(receipt.models) ? receipt.models : []) {
    need(
      isObject(model) &&
        isString(model.model_id) &&
        isString(model.repository) &&
        /^[a-f0-9]{40}$/.test(model.revision ?? "") &&
        isHex(model.template_sha256) &&
        isHex(model.manifest_sha256),
      `models entry ${JSON.stringify(model?.model_id)} is malformed`,
    );
  }
  need(Array.isArray(receipt.corpora), "corpora must be an array");
  for (const corpus of Array.isArray(receipt.corpora) ? receipt.corpora : []) {
    need(isObject(corpus) && isString(corpus.id) && isString(corpus.path) && isHex(corpus.sha256), `corpora entry ${JSON.stringify(corpus?.id)} is malformed`);
  }
  need(Array.isArray(receipt.goldens), "goldens must be an array");
  for (const golden of Array.isArray(receipt.goldens) ? receipt.goldens : []) {
    need(isObject(golden) && isString(golden.path) && isHex(golden.sha256), `goldens entry ${JSON.stringify(golden?.path)} is malformed`);
  }
  need(Array.isArray(receipt.results), "results must be an array");
  for (const result of Array.isArray(receipt.results) ? receipt.results : []) {
    need(
      isObject(result) &&
        isString(result.model_id) &&
        isString(result.corpus) &&
        isCount(result.requests) &&
        isCount(result.exact_semantic_matches) &&
        typeof result.max_raw_abs_diff === "number" &&
        Number.isFinite(result.max_raw_abs_diff) &&
        isCount(result.synthetic_zero_diffs) &&
        isCount(result.compared_values) &&
        typeof result.passed === "boolean",
      `results entry ${JSON.stringify(result?.model_id)}/${JSON.stringify(result?.corpus)} is malformed`,
    );
  }
  need(typeof receipt.tolerance === "number", "tolerance must be a number");
  need(isObject(receipt.host), "host must be an object");
  need(receipt.all_passed === true, "all_passed must be true");
  return errors;
}

// ---------------------------------------------------------------------------------------------------------------
// Ledger

/** Parses ledger text into entries; `errors` lists malformed lines. */
export function parseLedger(text) {
  const errors = [];
  const entries = [];
  if (text === "") {
    return { entries, lines: [], errors };
  }
  if (!text.endsWith("\n")) {
    errors.push("ledger must end with a newline");
  }
  const lines = text.replace(/\n$/, "").split("\n");
  lines.forEach((line, index) => {
    const label = `ledger line ${index + 1}`;
    let entry;
    try {
      entry = JSON.parse(line);
    } catch {
      errors.push(`${label} is not valid JSON`);
      return;
    }
    if (
      !isObject(entry) ||
      !isHex(entry.receipt_sha256) ||
      !isHex(entry.identity_sha256) ||
      !(typeof entry.run_id === "string" && UUID.test(entry.run_id)) ||
      !isString(entry.created_at) ||
      !(entry.previous_receipt_sha256 === null || isHex(entry.previous_receipt_sha256))
    ) {
      errors.push(`${label} does not have receipt_sha256, identity_sha256, run_id, created_at and previous_receipt_sha256`);
      return;
    }
    entries.push({ ...entry, line: index + 1 });
  });
  return { entries, lines, errors };
}

// ---------------------------------------------------------------------------------------------------------------
// Checks

function baseFile(root, baseRef, path) {
  try {
    return git(root, ["show", `${baseRef}:${path}`], { buffer: true });
  } catch {
    return null;
  }
}

export function resolveBaseRef(root, baseRef) {
  if (baseRef === undefined || baseRef === "" || /^0+$/.test(baseRef)) {
    return { ref: undefined, note: "no base revision (GLINER_RUNNER_RECEIPT_BASE_REF unset); skipping base comparisons" };
  }
  try {
    git(root, ["rev-parse", "--verify", "--quiet", `${baseRef}^{commit}`]);
  } catch {
    return {
      ref: undefined,
      error: `base revision ${baseRef} does not exist in this checkout; fetch it (actions/checkout fetch-depth: 0) or unset GLINER_RUNNER_RECEIPT_BASE_REF`,
    };
  }
  return { ref: baseRef };
}

/**
 * Model-free consistency checks of the checked-in receipt. Returns `{ errors, notes, identity }`.
 *
 * `probeCli()` must return the parsed `--version --json` output of the built CLI (or throw); it is injected so
 * tests need no build.
 */
export function checkReceipt({ root, baseRef, probeCli }) {
  const errors = [];
  const notes = [];
  const fail = (message) => errors.push(message);

  // (a) files exist and parse; schema; results complete and passing
  const receiptFile = join(root, RECEIPT_PATH);
  const ledgerFile = join(root, LEDGER_PATH);
  if (!existsSync(receiptFile)) {
    fail(`${RECEIPT_PATH} is missing; run \`${RECORD_COMMAND}\` on a machine with the model artifacts and commit the result`);
    return { errors, notes };
  }
  const receiptBytes = readFileSync(receiptFile);
  const receiptSha = sha256Hex(receiptBytes);
  let receipt;
  try {
    receipt = JSON.parse(receiptBytes.toString("utf8"));
  } catch (error) {
    fail(`${RECEIPT_PATH} is not valid JSON: ${error.message}`);
    return { errors, notes };
  }
  const shapeErrors = validateReceiptShape(receipt);
  if (shapeErrors.length > 0) {
    for (const message of shapeErrors) {
      fail(`${RECEIPT_PATH}: ${message}`);
    }
    return { errors, notes };
  }
  if (receiptBytes.toString("utf8") !== canonicalJson(receipt)) {
    fail(`${RECEIPT_PATH} is not in canonical form (sorted keys, 2-space indent, trailing newline); do not edit it by hand`);
  }

  let pins;
  try {
    pins = readCatalogPins(root);
  } catch (error) {
    fail(error.message);
    return { errors, notes };
  }
  if (receipt.tolerance !== TOLERANCE) {
    fail(`receipt tolerance ${receipt.tolerance} is not the required ${TOLERANCE}`);
  }
  for (const pin of pins) {
    const model = receipt.models.find((entry) => entry.model_id === pin.model_id);
    if (model === undefined) {
      fail(`receipt does not cover curated model ${pin.model_id}; run \`${RECORD_COMMAND}\``);
    } else if (model.repository !== pin.repository || model.revision !== pin.revision) {
      fail(`receipt pins ${model.model_id} to ${model.repository}@${model.revision}, catalog pins ${pin.repository}@${pin.revision}; run \`${RECORD_COMMAND}\``);
    }
    for (const corpusId of CORPUS_IDS) {
      const result = receipt.results.find((entry) => entry.model_id === pin.model_id && entry.corpus === corpusId);
      const label = `${pin.model_id} x ${corpusId}`;
      if (result === undefined) {
        fail(`receipt has no result for ${label}`);
      } else if (!result.passed || result.requests === 0 || result.exact_semantic_matches !== result.requests) {
        fail(`receipt result for ${label} did not pass (${result.exact_semantic_matches}/${result.requests} exact semantic matches)`);
      } else if (!(result.max_raw_abs_diff <= TOLERANCE)) {
        fail(`receipt result for ${label} has max_raw_abs_diff ${result.max_raw_abs_diff} > ${TOLERANCE}`);
      }
    }
  }
  for (const result of receipt.results) {
    if (!pins.some((pin) => pin.model_id === result.model_id) || !CORPUS_IDS.includes(result.corpus)) {
      fail(`receipt has an unexpected result for ${result.model_id} x ${result.corpus}`);
    }
  }
  if (!existsSync(ledgerFile)) {
    fail(`${LEDGER_PATH} is missing; the ledger is append-only and must be checked in with the receipt`);
  }

  // (b) identity
  let identity;
  try {
    identity = computeIdentity(root);
  } catch (error) {
    fail(`cannot compute runner identity: ${error.message}`);
    return { errors, notes };
  }
  if (identity.untracked.length > 0) {
    notes.push(`warning: identity inputs not tracked by git (CI will not see them): ${identity.untracked.slice(0, 5).join(", ")}${identity.untracked.length > 5 ? ", ..." : ""}`);
  }
  if (identity.sha256 !== receipt.runner.identity_sha256) {
    fail(
      `stale receipt: runner identity is ${identity.sha256} but the receipt was recorded for ${receipt.runner.identity_sha256}. ` +
        `Code or data used by the ONNX runner changed since the last evaluation; run \`${RECORD_COMMAND}\` locally and commit ` +
        `${RECEIPT_PATH}, ${LEDGER_PATH} and ${BUILD_INFO_PATH}.`,
    );
  }
  const buildInfoFile = join(root, BUILD_INFO_PATH);
  let buildInfo;
  if (!existsSync(buildInfoFile)) {
    fail(`${BUILD_INFO_PATH} is missing; run \`${RECORD_COMMAND}\``);
  } else {
    const text = readFileSync(buildInfoFile, "utf8");
    buildInfo = parseBuildInfo(text);
    if (buildInfo === undefined) {
      fail(`${BUILD_INFO_PATH} cannot be parsed; it is generated, run \`${RECORD_COMMAND}\``);
    } else {
      if (text !== renderBuildInfo(buildInfo)) {
        fail(`${BUILD_INFO_PATH} was edited by hand; it is generated, run \`${RECORD_COMMAND}\``);
      }
      if (buildInfo.identitySha256 !== identity.sha256) {
        fail(`${BUILD_INFO_PATH} identitySha256 ${buildInfo.identitySha256} != recomputed runner identity ${identity.sha256}; run \`${RECORD_COMMAND}\``);
      }
      if (buildInfo.identitySha256 !== receipt.runner.identity_sha256) {
        fail(`${BUILD_INFO_PATH} identitySha256 does not equal the receipt's runner.identity_sha256`);
      }
      // (c) receipt sha embedded in the runner
      if (buildInfo.evalReceiptSha256 !== receiptSha) {
        fail(`${BUILD_INFO_PATH} evalReceiptSha256 ${buildInfo.evalReceiptSha256} != sha256 of ${RECEIPT_PATH} ${receiptSha}; run \`${RECORD_COMMAND}\``);
      }
      if (buildInfo.evalRunId !== receipt.run_id) {
        fail(`${BUILD_INFO_PATH} evalRunId does not equal the receipt run_id`);
      }
      if (buildInfo.version !== receipt.runner.version) {
        fail(`${BUILD_INFO_PATH} version ${buildInfo.version} != receipt runner.version ${receipt.runner.version}`);
      }
    }
  }

  // (c) the built CLI reports the same values
  if (probeCli !== undefined) {
    try {
      const reported = probeCli();
      const expected = {
        identity_sha256: receipt.runner.identity_sha256,
        eval_receipt_sha256: receiptSha,
        eval_run_id: receipt.run_id,
      };
      for (const [key, value] of Object.entries(expected)) {
        if (reported[key] !== value) {
          fail(`built CLI --version --json reports ${key}=${reported[key]}, expected ${value}`);
        }
      }
      if (reported.version !== packageVersion(root)) {
        fail(`built CLI --version reports ${reported.version}, package.json says ${packageVersion(root)}`);
      }
    } catch (error) {
      fail(`cannot run the built CLI --version --json: ${error.message}`);
    }
  }

  // (d) hashes recorded in the receipt equal the files in the tree
  const hashChecks = [];
  for (const pin of pins) {
    const model = receipt.models.find((entry) => entry.model_id === pin.model_id);
    if (model !== undefined) {
      hashChecks.push([`${pin.model_id} template.json`, `${templateDir(pin)}/template.json`, model.template_sha256]);
      hashChecks.push([`${pin.model_id} manifest.json`, `${templateDir(pin)}/manifest.json`, model.manifest_sha256]);
    }
  }
  for (const corpusId of CORPUS_IDS) {
    const corpus = receipt.corpora.find((entry) => entry.id === corpusId);
    if (corpus === undefined) {
      fail(`receipt has no corpus ${corpusId}`);
    } else {
      if (corpus.path !== corpusPath(corpusId)) {
        fail(`receipt corpus ${corpusId} path ${corpus.path} != ${corpusPath(corpusId)}`);
      }
      hashChecks.push([`corpus ${corpusId}`, corpusPath(corpusId), corpus.sha256]);
    }
  }
  for (const pin of pins) {
    for (const corpusId of CORPUS_IDS) {
      const path = goldenPath(pin.model_id, corpusId);
      const golden = receipt.goldens.find((entry) => entry.path === path);
      if (golden === undefined) {
        fail(`receipt has no golden ${path}`);
      } else {
        hashChecks.push([`golden ${path}`, path, golden.sha256]);
      }
    }
  }
  for (const [label, path, expected] of hashChecks) {
    if (!existsSync(join(root, path))) {
      fail(`${label}: ${path} is not checked in`);
    } else if (fileSha256(root, path) !== expected) {
      fail(`${label}: ${path} no longer matches the sha256 in the receipt; run \`${RECORD_COMMAND}\``);
    }
  }

  // (e) anti-replay
  if (!existsSync(ledgerFile)) {
    return { errors, notes, identity: identity.sha256 };
  }
  const ledger = parseLedger(readFileSync(ledgerFile, "utf8"));
  for (const message of ledger.errors) {
    fail(`${LEDGER_PATH}: ${message}`);
  }
  if (ledger.errors.length > 0) {
    return { errors, notes, identity: identity.sha256 };
  }
  const entries = ledger.entries;
  const last = entries.at(-1);
  if (last === undefined) {
    fail(`${LEDGER_PATH} is empty; run \`${RECORD_COMMAND}\``);
    return { errors, notes, identity: identity.sha256 };
  }
  if (last.receipt_sha256 !== receiptSha || last.identity_sha256 !== receipt.runner.identity_sha256 || last.run_id !== receipt.run_id) {
    fail(
      `the last ${LEDGER_PATH} entry (sha ${last.receipt_sha256}, identity ${last.identity_sha256}, run ${last.run_id}) does not ` +
        `match the current receipt (sha ${receiptSha}, identity ${receipt.runner.identity_sha256}, run ${receipt.run_id}); ` +
        "the receipt was replaced without recording it in the ledger",
    );
  }
  // Same receipt bytes earlier in the ledger but paired with a different runner identity: a replay.
  for (const earlier of entries.slice(0, -1)) {
    if (earlier.receipt_sha256 === receiptSha && earlier.identity_sha256 !== last.identity_sha256) {
      fail(
        `replay: receipt ${receiptSha} was already recorded (ledger line ${earlier.line}) for runner identity ${earlier.identity_sha256} ` +
          `but is now paired with ${last.identity_sha256}; a receipt cannot be carried over to changed code. Run \`${RECORD_COMMAND}\`.`,
      );
    }
  }
  const seenSha = new Map();
  const seenRun = new Map();
  for (const entry of entries) {
    if (seenSha.has(entry.receipt_sha256)) {
      fail(`duplicate receipt_sha256 ${entry.receipt_sha256} on ledger lines ${seenSha.get(entry.receipt_sha256)} and ${entry.line}; every evaluation must produce a new receipt`);
    }
    if (seenRun.has(entry.run_id)) {
      fail(`duplicate run_id ${entry.run_id} on ledger lines ${seenRun.get(entry.run_id)} and ${entry.line}`);
    }
    seenSha.set(entry.receipt_sha256, entry.line);
    seenRun.set(entry.run_id, entry.line);
  }
  entries.forEach((entry, index) => {
    const expectedPrevious = index === 0 ? null : entries[index - 1].receipt_sha256;
    if (entry.previous_receipt_sha256 !== expectedPrevious) {
      fail(`ledger line ${entry.line}: previous_receipt_sha256 ${entry.previous_receipt_sha256} breaks the hash chain (expected ${expectedPrevious})`);
    }
  });
  if (receipt.previous_receipt_sha256 !== last.previous_receipt_sha256) {
    fail(`receipt previous_receipt_sha256 ${receipt.previous_receipt_sha256} differs from the last ledger entry's ${last.previous_receipt_sha256}`);
  }

  const resolved = resolveBaseRef(root, baseRef);
  if (resolved.error !== undefined) {
    fail(resolved.error);
  } else if (resolved.note !== undefined) {
    notes.push(resolved.note);
  } else {
    checkAgainstBase({ root, baseRef: resolved.ref, receipt, receiptSha, identity: identity.sha256, ledger, notes, fail });
  }
  return { errors, notes, identity: identity.sha256 };
}

function checkAgainstBase({ root, baseRef, receipt, receiptSha, identity, ledger, notes, fail }) {
  const baseLedgerBytes = baseFile(root, baseRef, LEDGER_PATH);
  const baseLines = baseLedgerBytes === null ? [] : parseLedger(baseLedgerBytes.toString("utf8")).lines;
  for (let index = 0; index < baseLines.length; index += 1) {
    if (ledger.lines[index] !== baseLines[index]) {
      fail(
        `${LEDGER_PATH} is append-only: line ${index + 1} differs from ${baseRef} ` +
          `(${ledger.lines[index] === undefined ? "removed" : "modified"}); only append new lines`,
      );
      break;
    }
  }
  const baseReceiptBytes = baseFile(root, baseRef, RECEIPT_PATH);
  if (baseReceiptBytes === null) {
    notes.push(`${baseRef} has no ${RECEIPT_PATH}; treating this as the first receipt`);
    return;
  }
  const baseSha = sha256Hex(baseReceiptBytes);
  let baseReceipt;
  try {
    baseReceipt = JSON.parse(baseReceiptBytes.toString("utf8"));
  } catch {
    fail(`${RECEIPT_PATH} at ${baseRef} is not valid JSON`);
    return;
  }
  const baseIdentity = baseReceipt?.runner?.identity_sha256;
  if (receiptSha === baseSha) {
    if (identity !== baseIdentity) {
      fail(
        `replay: ${RECEIPT_PATH} is unchanged from ${baseRef} (sha ${baseSha}) but the runner identity changed ` +
          `(${baseIdentity} -> ${identity}). The old evaluation does not cover the new code; run \`${RECORD_COMMAND}\`.`,
      );
    }
    return;
  }
  if (receipt.run_id === baseReceipt.run_id) {
    fail(`receipt run_id ${receipt.run_id} was reused from ${baseRef}; every evaluation needs a fresh run_id`);
  }
  const baseIndex = ledger.entries.findIndex((entry) => entry.receipt_sha256 === baseSha);
  if (baseIndex < 0) {
    fail(`the base receipt ${baseSha} from ${baseRef} is not in ${LEDGER_PATH}`);
  } else if (baseIndex !== baseLines.length - 1) {
    fail(`the base receipt ${baseSha} is not the last entry of ${LEDGER_PATH} at ${baseRef}`);
  }
  // Chain: previous_receipt_sha256 must be the ledger predecessor, which is the base receipt for the first new run.
  // (A branch may hold several local runs; each links to the one before it, and the first links to the base.)
  const firstNew = ledger.entries[baseLines.length];
  if (firstNew !== undefined && firstNew.previous_receipt_sha256 !== baseSha) {
    fail(
      `hash chain broken: the first new evaluation after ${baseRef} has previous_receipt_sha256 ${firstNew.previous_receipt_sha256}, ` +
        `expected the base receipt ${baseSha}`,
    );
  }
}
