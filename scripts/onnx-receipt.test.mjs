// Tests for scripts/onnx-receipt-lib.mjs. They build tiny synthetic git repositories; no models, no network.
//   node --test scripts/onnx-receipt.test.mjs
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { after, describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  BUILD_INFO_PATH,
  CORPUS_IDS,
  LEDGER_PATH,
  RECEIPT_PATH,
  TOLERANCE,
  canonicalJson,
  checkReceipt,
  computeIdentity,
  corpusPath,
  fileSha256,
  goldenPath,
  lockfileProductionClosure,
  packageVersion,
  parseBuildInfo,
  parseLedger,
  readCatalogPins,
  sha256Hex,
  templateDir,
  writeReceiptArtifacts,
} from "./onnx-receipt-lib.mjs";

const REAL_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const scratch = mkdtempSync(join(tmpdir(), "onnx-receipt-test-"));
after(() => rmSync(scratch, { recursive: true, force: true }));

const PINS = [
  ["model-a", "acme/Model-A", "a".repeat(40)],
  ["model-b", "acme/Model-B", "b".repeat(40)],
  ["model-c", "acme/Model-C", "c".repeat(40)],
];

const LOCKFILE = `lockfileVersion: '9.0'

importers:

  .: {}

  packages/onnx-runner:
    dependencies:
      '@scope/tok':
        specifier: 0.2.0
        version: 0.2.0
      zod:
        specifier: 4.6.5
        version: 4.6.5
    devDependencies:
      typescript:
        specifier: 7.0.2
        version: 7.0.2

packages:

  '@scope/tok@0.2.0':
    resolution: {integrity: sha512-tok}

  helper@1.0.0:
    resolution: {integrity: sha512-helper}

  typescript@7.0.2:
    resolution: {integrity: sha512-ts}

  zod@4.6.5:
    resolution: {integrity: sha512-zod}

snapshots:

  '@scope/tok@0.2.0':
    dependencies:
      helper: 1.0.0

  helper@1.0.0: {}

  typescript@7.0.2: {}

  zod@4.6.5: {}
`;

function git(root, ...args) {
  return execFileSync("git", ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "-C", root, ...args], {
    encoding: "utf8",
  }).trim();
}

function put(root, path, content) {
  mkdirSync(dirname(join(root, path)), { recursive: true });
  writeFileSync(join(root, path), content);
}

let counter = 0;

/** A synthetic repository with every identity input, committed once. */
function makeRepo() {
  const root = join(scratch, `repo-${(counter += 1)}`);
  mkdirSync(root, { recursive: true });
  git(root, "init", "-q", "-b", "main");
  put(root, ".gitignore", "dist/\nignored/\n");
  put(root, "pnpm-lock.yaml", LOCKFILE);
  put(root, "packages/onnx-runner/package.json", `${JSON.stringify({ name: "@gliner-runner/onnx-runner", version: "1.0.0", dependencies: { zod: "4.6.5" } }, null, 2)}\n`);
  put(root, "packages/onnx-runner/tsconfig.json", "{}\n");
  put(root, "packages/onnx-runner/src/other.ts", "export const other = 1;\n");
  put(
    root,
    "packages/onnx-runner/src/catalog.ts",
    `export const KNOWN_MODELS = [\n${PINS.map(
      ([modelId, repository, revision]) => `  {\n    modelId: "${modelId}",\n    repository: "${repository}",\n    pinnedRevision: "${revision}",\n    parameterCount: 1,\n  },\n`,
    ).join("")}];\n`,
  );
  for (const pin of readCatalogPins(root)) {
    put(root, `${templateDir(pin)}/template.json`, `{"model":"${pin.model_id}"}\n`);
    put(root, `${templateDir(pin)}/manifest.json`, `{"manifest":"${pin.model_id}"}\n`);
    put(root, `${templateDir(pin)}/graph.onnx`, Buffer.from([0, 1, 2, 13, 10, 3]));
    for (const corpus of CORPUS_IDS) {
      put(root, goldenPath(pin.model_id, corpus), `{"golden":"${pin.model_id}/${corpus}"}\n`);
    }
  }
  for (const corpus of CORPUS_IDS) {
    put(root, corpusPath(corpus), `{"corpus":"${corpus}"}\n`);
  }
  mkdirSync(join(root, "release"), { recursive: true });
  put(root, BUILD_INFO_PATH, "// placeholder\n");
  return root;
}

/** Mirrors `record`: writes a fresh receipt, build-info and ledger line for the current tree. */
function record(root, overrides = {}) {
  const pins = readCatalogPins(root);
  const receiptFile = join(root, RECEIPT_PATH);
  let previous = null;
  try {
    previous = sha256Hex(readFileSync(receiptFile));
  } catch {
    // first receipt
  }
  const receipt = {
    format_version: 1,
    kind: "gliner-runner-onnx-eval-receipt",
    run_id: randomUUID(),
    created_at: new Date().toISOString(),
    previous_receipt_sha256: previous,
    runner: { version: packageVersion(root), identity_sha256: computeIdentity(root).sha256 },
    models: pins.map((pin) => ({
      ...pin,
      template_sha256: fileSha256(root, `${templateDir(pin)}/template.json`),
      manifest_sha256: fileSha256(root, `${templateDir(pin)}/manifest.json`),
    })),
    corpora: CORPUS_IDS.map((id) => ({ id, path: corpusPath(id), sha256: fileSha256(root, corpusPath(id)) })),
    goldens: pins.flatMap((pin) => CORPUS_IDS.map((id) => ({ path: goldenPath(pin.model_id, id), sha256: fileSha256(root, goldenPath(pin.model_id, id)) }))),
    results: pins.flatMap((pin) =>
      CORPUS_IDS.map((corpus) => ({
        model_id: pin.model_id,
        corpus,
        requests: 10,
        exact_semantic_matches: 10,
        max_raw_abs_diff: 2e-6,
        synthetic_zero_diffs: 5,
        compared_values: 20,
        passed: true,
      })),
    ),
    tolerance: TOLERANCE,
    host: { platform: "test", arch: "test", cpu_model: "test", cpu_cores: 1, node: "v24", onnxruntime_node: "1" },
    all_passed: true,
    ...overrides,
  };
  return writeReceiptArtifacts(root, receipt);
}

/** What the built CLI would print for the checked-in files. */
function probeOf(root) {
  return () => {
    const info = parseBuildInfo(readFileSync(join(root, BUILD_INFO_PATH), "utf8"));
    return { version: packageVersion(root), identity_sha256: info.identitySha256, eval_receipt_sha256: info.evalReceiptSha256, eval_run_id: info.evalRunId };
  };
}

function check(root, baseRef, probeCli = probeOf(root)) {
  return checkReceipt({ root, baseRef, probeCli });
}

function commitBase(root) {
  git(root, "add", "-A");
  git(root, "commit", "-q", "-m", "base");
  git(root, "tag", "-f", "base");
}

function errorsOf(result) {
  return result.errors.join("\n");
}

function readReceipt(root) {
  return JSON.parse(readFileSync(join(root, RECEIPT_PATH), "utf8"));
}

describe("runner identity", () => {
  test("is deterministic and independent of line endings, ignored files, build-info and package version", () => {
    const root = makeRepo();
    const first = computeIdentity(root).sha256;
    assert.match(first, /^[a-f0-9]{64}$/);
    assert.equal(computeIdentity(root).sha256, first);

    put(root, "packages/onnx-runner/src/other.ts", "export const other = 1;\r\n".replace(/\r\n/g, "\n"));
    put(root, "packages/onnx-runner/src/crlf.ts", "a\nb\n");
    const withFile = computeIdentity(root).sha256;
    assert.notEqual(withFile, first);
    put(root, "packages/onnx-runner/src/crlf.ts", "a\r\nb\r\n");
    assert.equal(computeIdentity(root).sha256, withFile, "CRLF is normalized");

    put(root, "packages/onnx-runner/src/ignored/x.ts", "ignored\n");
    put(root, "dist/x.ts", "ignored\n");
    put(root, BUILD_INFO_PATH, "// something else\n");
    assert.equal(computeIdentity(root).sha256, withFile, "ignored files and build-info do not count");

    const manifest = JSON.parse(readFileSync(join(root, "packages/onnx-runner/package.json"), "utf8"));
    put(root, "packages/onnx-runner/package.json", `${JSON.stringify({ ...manifest, version: "9.9.9" }, null, 2)}\n`);
    assert.equal(computeIdentity(root).sha256, withFile, "release-managed version does not count");
    put(root, "packages/onnx-runner/package.json", `${JSON.stringify({ ...manifest, version: "9.9.9", dependencies: { zod: "4.7.0" } }, null, 2)}\n`);
    assert.notEqual(computeIdentity(root).sha256, withFile, "dependencies do count");
  });

  test("is the same for a tracked checkout and an untracked working tree, and for a fresh clone", () => {
    const root = makeRepo();
    const before = computeIdentity(root).sha256;
    commitBase(root);
    assert.equal(computeIdentity(root).sha256, before);
    const clone = join(scratch, `clone-${(counter += 1)}`);
    execFileSync("git", ["clone", "-q", root, clone]);
    assert.equal(computeIdentity(clone).sha256, before);
  });

  test("changes with sources, templates, fixtures, goldens, catalog pins and locked integrity", () => {
    const root = makeRepo();
    const base = computeIdentity(root).sha256;
    const mutations = [
      () => put(root, "packages/onnx-runner/src/other.ts", "export const other = 2;\n"),
      () => put(root, "packages/onnx-runner/tsconfig.json", '{"a":1}\n'),
      () => put(root, `${templateDir(readCatalogPins(root)[0])}/graph.onnx`, Buffer.from([9])),
      () => put(root, corpusPath("english-v1"), '{"corpus":"changed"}\n'),
      () => put(root, goldenPath("model-b", "multilingual-v1"), '{"golden":"changed"}\n'),
      () => put(root, "pnpm-lock.yaml", LOCKFILE.replace("sha512-helper", "sha512-evil")),
    ];
    const seen = new Set([base]);
    for (const mutate of mutations) {
      mutate();
      const identity = computeIdentity(root).sha256;
      assert.ok(!seen.has(identity), "every mutation yields a new identity");
      seen.add(identity);
    }
  });

  test("ignores dev-only lockfile entries", () => {
    const root = makeRepo();
    const base = computeIdentity(root).sha256;
    put(root, "pnpm-lock.yaml", LOCKFILE.replace("sha512-ts", "sha512-other"));
    assert.equal(computeIdentity(root).sha256, base);
    const closure = lockfileProductionClosure(LOCKFILE.replace("sha512-ts", "sha512-other"), "packages/onnx-runner");
    assert.deepEqual(closure.map((entry) => entry.line), [
      "@scope/tok@0.2.0@{integrity: sha512-tok}",
      "helper@1.0.0@{integrity: sha512-helper}",
      "zod@4.6.5@{integrity: sha512-zod}",
    ]);
  });

  test("production closure of the real lockfile matches the onnx-runner dependencies", () => {
    const closure = lockfileProductionClosure(readFileSync(join(REAL_ROOT, "pnpm-lock.yaml"), "utf8"), "packages/onnx-runner").map((entry) => entry.id);
    assert.ok(closure.includes("zod@4.6.5"));
    assert.ok(closure.some((id) => id.startsWith("onnxruntime-node@")));
    assert.ok(closure.some((id) => id.startsWith("@huggingface/tokenizers@")));
    assert.ok(!closure.some((id) => id.startsWith("typescript@")), "dev dependencies are excluded");
  });
});

describe("receipt check without a base revision", () => {
  test("accepts a freshly recorded receipt", () => {
    const root = makeRepo();
    record(root);
    const result = check(root, undefined);
    assert.deepEqual(result.errors, []);
    assert.match(result.notes.join("\n"), /skipping base comparisons/);
  });

  test("reports a missing receipt", () => {
    const root = makeRepo();
    assert.match(errorsOf(check(root, undefined)), /release\/onnx-eval-receipt\.json is missing.*mise run eval:onnx-receipt/);
  });

  test("detects a stale receipt after a code change and tells the developer how to fix it", () => {
    const root = makeRepo();
    record(root);
    appendFileSync(join(root, "packages/onnx-runner/src/other.ts"), "export const changed = true;\n");
    const message = errorsOf(check(root, undefined));
    assert.match(message, /stale receipt/);
    assert.match(message, /mise run eval:onnx-receipt/);
  });

  test("detects stale data: corpus, golden and template changes", () => {
    for (const path of [corpusPath("english-v1"), goldenPath("model-c", "english-v1"), `${templateDir(readCatalogPins(makeRepo())[1])}/template.json`]) {
      const root = makeRepo();
      record(root);
      put(root, path, "{\"changed\":true}\n");
      const message = errorsOf(check(root, undefined));
      assert.match(message, /stale receipt/, path);
      assert.match(message, /no longer matches the sha256 in the receipt/, path);
    }
  });

  test("detects build-info that does not match the receipt or the tree", () => {
    const root = makeRepo();
    record(root);
    const original = readFileSync(join(root, BUILD_INFO_PATH), "utf8");
    const info = parseBuildInfo(original);

    put(root, BUILD_INFO_PATH, original.replace(info.evalReceiptSha256, "0".repeat(64)));
    assert.match(errorsOf(check(root, undefined)), /evalReceiptSha256 0{64} != sha256 of release\/onnx-eval-receipt\.json/);

    put(root, BUILD_INFO_PATH, original.replace(info.identitySha256, "1".repeat(64)));
    assert.match(errorsOf(check(root, undefined)), /identitySha256 1{64} != recomputed runner identity/);

    put(root, BUILD_INFO_PATH, original.replace(info.evalRunId, randomUUID()));
    assert.match(errorsOf(check(root, undefined)), /evalRunId does not equal/);

    put(root, BUILD_INFO_PATH, `${original}// edited\n`);
    assert.match(errorsOf(check(root, undefined)), /edited by hand/);

    put(root, BUILD_INFO_PATH, "// garbage\n");
    assert.match(errorsOf(check(root, undefined)), /cannot be parsed/);
  });

  test("detects a receipt file edited after it was embedded", () => {
    const root = makeRepo();
    record(root);
    const receipt = readReceipt(root);
    receipt.host.cpu_model = "edited";
    put(root, RECEIPT_PATH, canonicalJson(receipt));
    assert.match(errorsOf(check(root, undefined)), /evalReceiptSha256 .* != sha256 of release\/onnx-eval-receipt\.json/);
  });

  test("detects a built CLI whose --version disagrees with the receipt", () => {
    const root = makeRepo();
    record(root);
    const probe = () => ({ ...probeOf(root)(), eval_receipt_sha256: "f".repeat(64), eval_run_id: randomUUID() });
    const message = errorsOf(check(root, undefined, probe));
    assert.match(message, /built CLI --version --json reports eval_receipt_sha256=f{64}/);
    assert.match(message, /eval_run_id/);
    const throwing = () => {
      throw new Error("boom");
    };
    assert.match(errorsOf(check(root, undefined, throwing)), /cannot run the built CLI.*boom/);
  });

  test("requires every curated model and corpus to pass within tolerance", () => {
    const root = makeRepo();
    record(root, { results: undefined });
    assert.match(errorsOf(check(root, undefined)), /results must be an array/);

    const failing = makeRepo();
    record(failing);
    const receipt = readReceipt(failing);
    receipt.results[1].max_raw_abs_diff = 2e-4;
    receipt.results[2].passed = false;
    receipt.results.pop();
    receipt.all_passed = true;
    put(failing, RECEIPT_PATH, canonicalJson(receipt));
    const message = errorsOf(check(failing, undefined));
    assert.match(message, /max_raw_abs_diff 0\.0002 > 0\.0001/);
    assert.match(message, /did not pass/);
    assert.match(message, /receipt has no result for model-c x multilingual-v1/);

    const notAllPassed = makeRepo();
    record(notAllPassed, { all_passed: false });
    assert.match(errorsOf(check(notAllPassed, undefined)), /all_passed must be true/);
  });

  test("requires the receipt to cover the catalog pins", () => {
    const root = makeRepo();
    record(root);
    put(root, "packages/onnx-runner/src/catalog.ts", readFileSync(join(root, "packages/onnx-runner/src/catalog.ts"), "utf8").replace("a".repeat(40), "d".repeat(40)));
    assert.match(errorsOf(check(root, undefined)), /(receipt pins model-a|stale receipt)/);
  });

  test("rejects a ledger whose last entry is not the current receipt", () => {
    const root = makeRepo();
    record(root);
    const ledger = readFileSync(join(root, LEDGER_PATH), "utf8");
    writeFileSync(join(root, LEDGER_PATH), "");
    assert.match(errorsOf(check(root, undefined)), /onnx-eval-receipts\.jsonl is empty/);
    const entry = JSON.parse(ledger);
    put(root, LEDGER_PATH, `${JSON.stringify({ ...entry, run_id: randomUUID() })}\n`);
    assert.match(errorsOf(check(root, undefined)), /does not match the current receipt/);
    put(root, LEDGER_PATH, "not json\n");
    assert.match(errorsOf(check(root, undefined)), /not valid JSON/);
  });

  test("rejects duplicate receipt sha and run_id entries in the ledger", () => {
    const root = makeRepo();
    record(root);
    const line = readFileSync(join(root, LEDGER_PATH), "utf8");
    put(root, LEDGER_PATH, line + line);
    assert.match(errorsOf(check(root, undefined)), /duplicate receipt_sha256/);
    assert.match(errorsOf(check(root, undefined)), /duplicate run_id/);

    // A second record that reuses an earlier run id (same run_id, new receipt bytes).
    const second = makeRepo();
    record(second);
    const first = readReceipt(second);
    appendFileSync(join(second, "packages/onnx-runner/src/other.ts"), "export const more = 1;\n");
    record(second, { run_id: first.run_id });
    const message = errorsOf(check(second, undefined));
    assert.match(message, /duplicate run_id/);
  });

  test("rejects a replayed receipt paired with a different identity in the ledger", () => {
    const root = makeRepo();
    record(root);
    const oldReceipt = readFileSync(join(root, RECEIPT_PATH), "utf8");
    const oldLine = JSON.parse(readFileSync(join(root, LEDGER_PATH), "utf8"));
    // Code changes; the developer re-records, then restores the old receipt file and ledger pairing.
    appendFileSync(join(root, "packages/onnx-runner/src/other.ts"), "export const changed = 1;\n");
    record(root);
    put(root, RECEIPT_PATH, oldReceipt);
    const newIdentity = computeIdentity(root).sha256;
    appendFileSync(join(root, LEDGER_PATH), `${JSON.stringify({ ...oldLine, identity_sha256: newIdentity })}\n`);
    const message = errorsOf(check(root, undefined));
    assert.match(message, /replay: receipt .* was already recorded \(ledger line 1\)/);
  });
});

describe("receipt check against a base revision", () => {
  test("accepts the first receipt (base has none) and a normal re-evaluation after a code change", () => {
    const root = makeRepo();
    commitBase(root);
    record(root);
    assert.deepEqual(check(root, "base").errors, []);
    assert.match(check(root, "base").notes.join("\n"), /first receipt/);

    const next = makeRepo();
    record(next);
    commitBase(next);
    appendFileSync(join(next, "packages/onnx-runner/src/other.ts"), "export const changed = 1;\n");
    record(next);
    assert.deepEqual(check(next, "base").errors, []);
    // Several local runs on one branch each chain to the previous one.
    appendFileSync(join(next, "packages/onnx-runner/src/other.ts"), "export const changed2 = 1;\n");
    record(next);
    assert.deepEqual(check(next, "base").errors, []);
  });

  test("accepts an unchanged receipt on unchanged code", () => {
    const root = makeRepo();
    record(root);
    commitBase(root);
    assert.deepEqual(check(root, "base").errors, []);
  });

  test("rejects an unchanged receipt when the runner identity changed (replayed evaluation)", () => {
    const root = makeRepo();
    record(root);
    commitBase(root);
    appendFileSync(join(root, "packages/onnx-runner/src/other.ts"), "export const changed = 1;\n");
    const message = errorsOf(check(root, "base"));
    assert.match(message, /replay: release\/onnx-eval-receipt\.json is unchanged from base .* runner identity changed/);
    assert.match(message, /stale receipt/);
  });

  test("rejects a ledger that is not append-only relative to the base", () => {
    const root = makeRepo();
    record(root);
    commitBase(root);
    appendFileSync(join(root, "packages/onnx-runner/src/other.ts"), "export const changed = 1;\n");
    record(root);
    const lines = readFileSync(join(root, LEDGER_PATH), "utf8").trimEnd().split("\n");

    const rewritten = JSON.parse(lines[0]);
    rewritten.created_at = "2000-01-01T00:00:00.000Z";
    put(root, LEDGER_PATH, `${JSON.stringify(rewritten)}\n${lines[1]}\n`);
    assert.match(errorsOf(check(root, "base")), /append-only: line 1 differs from base \(modified\)/);

    put(root, LEDGER_PATH, `${lines[1]}\n`);
    assert.match(errorsOf(check(root, "base")), /append-only: line 1 differs from base/);
  });

  test("rejects a hash-chain break", () => {
    const root = makeRepo();
    record(root);
    commitBase(root);
    appendFileSync(join(root, "packages/onnx-runner/src/other.ts"), "export const changed = 1;\n");
    record(root, { previous_receipt_sha256: "9".repeat(64) });
    const message = errorsOf(check(root, "base"));
    assert.match(message, /previous_receipt_sha256 9{64}/);
    assert.match(message, /hash chain/);
  });

  test("rejects a new receipt that reuses the base run_id", () => {
    const root = makeRepo();
    record(root);
    commitBase(root);
    const baseRun = readReceipt(root).run_id;
    appendFileSync(join(root, "packages/onnx-runner/src/other.ts"), "export const changed = 1;\n");
    record(root, { run_id: baseRun });
    assert.match(errorsOf(check(root, "base")), /run_id .* was reused/);
  });

  test("rejects an unknown base revision but tolerates the all-zero SHA of a new branch", () => {
    const root = makeRepo();
    record(root);
    commitBase(root);
    assert.match(errorsOf(check(root, "origin/nope")), /does not exist in this checkout/);
    assert.deepEqual(check(root, "0".repeat(40)).errors, []);
  });
});

describe("formats", () => {
  test("canonical JSON sorts keys, indents by two and ends with a newline", () => {
    assert.equal(canonicalJson({ b: 1, a: { d: [1, { z: 1, y: 2 }], c: null } }), `{\n  "a": {\n    "c": null,\n    "d": [\n      1,\n      {\n        "y": 2,\n        "z": 1\n      }\n    ]\n  },\n  "b": 1\n}\n`);
  });

  test("ledger parsing rejects malformed lines", () => {
    assert.deepEqual(parseLedger("").errors, []);
    assert.match(parseLedger('{"x":1}\n').errors[0], /does not have/);
    assert.match(parseLedger('{"x":1}').errors.join(), /newline/);
  });
});
