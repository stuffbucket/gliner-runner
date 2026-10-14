import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";
import { OrderedObject } from "../../src/gliner2/index.js";
import type { Gliner2Tokenizer, OrderedJson } from "../../src/gliner2/index.js";

// `@huggingface/tokenizers@0.2.0` ships type declarations with extensionless relative imports that NodeNext rejects
// (TS2834) unless `skipLibCheck` is on, so the module is loaded through a non-literal specifier and typed locally.
const TOKENIZERS_SPECIFIER = "@huggingface/tokenizers";
const { Tokenizer } = (await import(TOKENIZERS_SPECIFIER)) as {
  Tokenizer: new (tokenizerJson: object, tokenizerConfig: object) => Gliner2Tokenizer;
};

/** Package root (works from both `test/gliner2` and the compiled `dist/test/gliner2`). */
function packageRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    if (existsSync(join(dir, "package.json"))) return dir;
    const parent = dirname(dir);
    if (parent === dir) throw new Error("package.json not found");
    dir = parent;
  }
}
const ROOT = packageRoot();

export function loadFixture<T>(name: string): T {
  return JSON.parse(readFileSync(join(ROOT, "test", "fixtures", "gliner2", name), "utf8")) as T;
}

/** Local model snapshots: `GLINER_RUNNER_TEST_SNAPSHOTS_DIR` defaults to the repo's dist/candidate-models.* dir. */
const DEFAULT_SNAPSHOTS = resolve(ROOT, "..", "..", "dist", "candidate-models.wD8DrF");
export const SNAPSHOTS_DIR = process.env["GLINER_RUNNER_TEST_SNAPSHOTS_DIR"] ?? DEFAULT_SNAPSHOTS;
const SMALL_SNAPSHOT = join(
  homedir(),
  ".cache/huggingface/hub/models--fastino--gliner2.5-small-v1/snapshots/df5910e44bc4ffdb0d95399a83b0ca4516349aa5",
);

/** Tokenizer families keyed like the goldens: small / decide / multi / 1b. */
export const SNAPSHOT_DIRS: Readonly<Record<string, string>> = {
  small: process.env["GLINER_RUNNER_TEST_SMALL_SNAPSHOT"] ?? SMALL_SNAPSHOT,
  decide: join(SNAPSHOTS_DIR, "fastino--GLiNER2.5-Decide", "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90"),
  multi: join(SNAPSHOTS_DIR, "fastino--GLiNER2.5-multi-Decide", "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d"),
  "1b": join(SNAPSHOTS_DIR, "fastino--GLiNER2.5-Decide-1B", "67af949e17bebd20f05fe020a971079889e7a462"),
};

/** Loads the snapshot's tokenizer, or `null` when the snapshot is not present (the caller skips). */
export function loadTokenizer(model: string): Gliner2Tokenizer | null {
  const dir = SNAPSHOT_DIRS[model];
  if (dir === undefined) return null;
  const tj = join(dir, "tokenizer.json");
  const tc = join(dir, "tokenizer_config.json");
  if (!existsSync(tj) || !existsSync(tc)) return null;
  return new Tokenizer(JSON.parse(readFileSync(tj, "utf8")) as object, JSON.parse(readFileSync(tc, "utf8")) as object);
}

/** Relative float tolerance 1e-12 (ulp differences between V8 and libm); everything else must be exactly equal. */
export function compareOrdered(actual: OrderedJson, expected: unknown, path: string, diffs: string[]): void {
  if (typeof actual === "number" && typeof expected === "number") {
    if (actual === expected) return;
    if (Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(actual), Math.abs(expected))) return;
    diffs.push(`${path}: ${actual} vs ${expected}`);
    return;
  }
  if (Array.isArray(actual)) {
    if (!Array.isArray(expected) || expected.length !== actual.length) {
      diffs.push(`${path}: array mismatch`);
      return;
    }
    (actual as readonly OrderedJson[]).forEach((x, i) => compareOrdered(x, expected[i], `${path}[${i}]`, diffs));
    return;
  }
  if (actual instanceof OrderedObject) {
    if (expected === null || typeof expected !== "object" || Array.isArray(expected)) {
      diffs.push(`${path}: expected non-object`);
      return;
    }
    const ek = Object.keys(expected);
    if (JSON.stringify(actual.keys()) !== JSON.stringify(ek)) {
      diffs.push(`${path}: keys ${JSON.stringify(actual.keys())} vs ${JSON.stringify(ek)}`);
      return;
    }
    for (const [k, v] of actual.entries) compareOrdered(v, (expected as Record<string, unknown>)[k], `${path}.${k}`, diffs);
    return;
  }
  if (actual !== expected) diffs.push(`${path}: ${JSON.stringify(actual)} vs ${JSON.stringify(expected)}`);
}
