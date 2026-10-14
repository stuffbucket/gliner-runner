import assert from "node:assert/strict";
import { appendFileSync, lstatSync, mkdirSync, readdirSync, rmSync, symlinkSync, utimesSync, writeFileSync, statSync, readFileSync } from "node:fs";
import { delimiter, join } from "node:path";
import test from "node:test";

import { KNOWN_MODELS, type KnownModel } from "../../src/index.js";
import {
  clearSnapshotDigestCache,
  discoverSnapshots,
  pythonUserCacheDir,
  snapshotRoots,
  verifySnapshotFiles,
} from "../../src/onnx/snapshots.js";
import { loadTemplate } from "../../src/onnx/templates.js";
import { buildTemplateFixture, writeHfCache, writeSnapshotDir, type TemplateFixture } from "./template-fixture.js";

const MODEL = KNOWN_MODELS[0] as KnownModel;
const PIN = MODEL.pinnedRevision;
const SLUG = "fastino--GLiNER2.5-Decide";

async function sources(fixture: TemplateFixture) {
  const template = await loadTemplate(fixture.templatesDir, MODEL, PIN);
  assert.ok(template);
  return template.sourceFiles;
}

/** Recursive listing with file sizes, symlink targets and mtimes: any write would change it. */
function tree(directory: string): string[] {
  const entries: string[] = [];
  const walk = (path: string): void => {
    for (const name of readdirSync(path).sort()) {
      const full = join(path, name);
      const status = lstatSync(full);
      entries.push(`${full}|${status.isSymbolicLink() ? "link" : status.isDirectory() ? "dir" : status.size}|${status.mtimeMs}`);
      if (status.isDirectory()) {
        walk(full);
      }
    }
  };
  walk(directory);
  return entries;
}

interface Layout {
  fixture: TemplateFixture;
  home: string;
  modelsDir: string;
  /** Candidate locations by origin, all holding verified copies. */
  places: Record<string, string>;
  env: NodeJS.ProcessEnv;
}

/** Every standard location populated with a verified snapshot (Linux conventions). */
function populate(): Layout {
  const fixture = buildTemplateFixture();
  const { root, sources: files } = fixture;
  const home = join(root, "home");
  const xdg = join(root, "xdg");
  const places: Record<string, string> = {
    explicit: writeSnapshotDir(join(root, "explicit"), files),
    snapshotDirsA: writeSnapshotDir(join(root, "dirs-a", SLUG, PIN), files),
    snapshotDirsB: writeHfCache(join(root, "dirs-b"), MODEL.repository, PIN, files),
    hfHubCache: writeHfCache(join(root, "hub-cache"), MODEL.repository, PIN, files),
    hfHome: writeHfCache(join(root, "hf-home", "hub"), MODEL.repository, PIN, files),
    xdgHf: writeHfCache(join(xdg, "huggingface", "hub"), MODEL.repository, PIN, files),
    pythonProvider: writeSnapshotDir(join(xdg, "gliner-runner", "models", "providers", "huggingface", SLUG, PIN), files),
    modelsProvider: writeSnapshotDir(join(root, "models", "providers", "huggingface", SLUG, PIN), files),
  };
  return {
    fixture,
    home,
    modelsDir: join(root, "models"),
    places,
    env: {
      GLINER_RUNNER_SNAPSHOT_DIRS: [join(root, "dirs-a"), join(root, "dirs-b")].join(delimiter),
      HF_HUB_CACHE: join(root, "hub-cache"),
      HF_HOME: join(root, "hf-home"),
      XDG_CACHE_HOME: xdg,
    },
  };
}

test("precedence: explicit, GLINER_RUNNER_SNAPSHOT_DIRS, HF_HUB_CACHE, HF_HOME, XDG, Python store, models dir", async () => {
  const layout = populate();
  const files = await sources(layout.fixture);
  const options = { env: layout.env, platform: "linux" as const, home: layout.home, modelsDir: layout.modelsDir };

  const all = await discoverSnapshots(MODEL, PIN, files, { ...options, exhaustive: true });
  assert.deepEqual(
    all.candidates.map((candidate) => [candidate.origin, candidate.location]),
    [
      ["GLINER_RUNNER_SNAPSHOT_DIRS", layout.places["snapshotDirsA"]],
      ["GLINER_RUNNER_SNAPSHOT_DIRS", layout.places["snapshotDirsB"]],
      ["HF_HUB_CACHE", layout.places["hfHubCache"]],
      ["HF_HOME", layout.places["hfHome"]],
      ["xdg-huggingface-cache", layout.places["xdgHf"]],
      ["gliner-runner-provider-store", layout.places["pythonProvider"]],
      ["models-dir-provider", layout.places["modelsProvider"]],
    ],
  );
  assert.ok(all.candidates.every((candidate) => candidate.verified && candidate.files.every((file) => file.verified)));
  assert.equal(all.selected?.location, layout.places["snapshotDirsA"]);

  // Without exhaustive checking, the first verified candidate wins and the rest are only listed.
  const first = await discoverSnapshots(MODEL, PIN, files, options);
  assert.equal(first.selected?.location, layout.places["snapshotDirsA"]);
  assert.match(first.candidates[1]?.reason ?? "", /not checked: an earlier candidate was verified/);

  // An explicit directory is the only thing considered, even when others would verify.
  const explicit = await discoverSnapshots(MODEL, PIN, files, { ...options, explicit: layout.places["explicit"] as string });
  assert.deepEqual(explicit.candidates.map((candidate) => [candidate.origin, candidate.verified]), [["explicit", true]]);
  const bad = await discoverSnapshots(MODEL, PIN, files, { ...options, explicit: join(layout.fixture.root, "absent") });
  assert.deepEqual([bad.candidates, bad.selected], [[], undefined]);

  // Remove the winners one by one: the next location in precedence order takes over.
  const order = ["snapshotDirsA", "snapshotDirsB", "hfHubCache", "hfHome", "xdgHf", "pythonProvider", "modelsProvider"];
  const origins = ["GLINER_RUNNER_SNAPSHOT_DIRS", "GLINER_RUNNER_SNAPSHOT_DIRS", "HF_HUB_CACHE", "HF_HOME", "xdg-huggingface-cache", "gliner-runner-provider-store", "models-dir-provider"];
  for (const [index, key] of order.entries()) {
    const found = await discoverSnapshots(MODEL, PIN, files, options);
    assert.equal(found.selected?.location, layout.places[key], key);
    assert.equal(found.selected?.origin, origins[index]);
    rmSync(layout.places[key] as string, { recursive: true });
  }
  assert.equal((await discoverSnapshots(MODEL, PIN, files, options)).selected, undefined);
});

test("HF_HUB_CACHE, HF_HOME and the XDG default are separate locations; HUGGINGFACE_HUB_CACHE is accepted", () => {
  const roots = snapshotRoots({
    env: { HF_HUB_CACHE: "/a", HF_HOME: "/b", XDG_CACHE_HOME: "/c" },
    platform: "linux",
    home: "/h",
    modelsDir: "/m",
  });
  assert.deepEqual(
    roots.map((root) => [root.origin, root.path]),
    [
      ["HF_HUB_CACHE", "/a"],
      ["HF_HOME", "/b/hub"],
      ["xdg-huggingface-cache", "/c/huggingface/hub"],
      ["gliner-runner-provider-store", "/c/gliner-runner/models/providers/huggingface"],
      ["models-dir-provider", "/m/providers/huggingface"],
      ["models-dir-provider", "/m"],
    ],
  );
  assert.equal(snapshotRoots({ env: { HUGGINGFACE_HUB_CACHE: "/legacy" }, platform: "linux", home: "/h" })[0]?.path, "/legacy");
  assert.equal(snapshotRoots({ env: {}, platform: "linux", home: "/h" })[0]?.path, "/h/.cache/huggingface/hub");
  assert.equal(snapshotRoots({ env: {}, platform: "linux", home: "/h" })[1]?.path, "/h/.cache/gliner-runner/models/providers/huggingface");
});

test("the Python provider store follows the platformdirs cache location of each platform", () => {
  assert.equal(pythonUserCacheDir({}, "darwin", "/Users/u"), "/Users/u/Library/Caches/gliner-runner");
  assert.equal(pythonUserCacheDir({ XDG_CACHE_HOME: "/x" }, "darwin", "/Users/u"), "/Users/u/Library/Caches/gliner-runner");
  assert.equal(pythonUserCacheDir({}, "linux", "/home/u"), "/home/u/.cache/gliner-runner");
  assert.equal(pythonUserCacheDir({ XDG_CACHE_HOME: "/x" }, "linux", "/home/u"), "/x/gliner-runner");
  assert.equal(pythonUserCacheDir({ LOCALAPPDATA: "C:\\L" }, "win32", "C:\\u"), join("C:\\L", "gliner-runner", "Cache"));
  const darwin = snapshotRoots({ env: {}, platform: "darwin", home: "/Users/u" });
  assert.equal(darwin.find((root) => root.origin === "gliner-runner-provider-store")?.path, "/Users/u/Library/Caches/gliner-runner/models/providers/huggingface");
});

test("GLINER_RUNNER_SNAPSHOT_DIRS entries may be snapshot directories, provider roots or HF caches", async () => {
  const fixture = buildTemplateFixture();
  const files = await sources(fixture);
  const direct = writeSnapshotDir(join(fixture.root, "direct"), fixture.sources);
  const found = await discoverSnapshots(MODEL, PIN, files, {
    env: { GLINER_RUNNER_SNAPSHOT_DIRS: direct },
    platform: "linux",
    home: join(fixture.root, "empty-home"),
  });
  assert.equal(found.selected?.location, direct);
  assert.equal(found.selected?.layout, "direct");
});

test("rejections say why: missing file, size, hash, wrong revision, dangling link", async () => {
  const fixture = buildTemplateFixture();
  const files = await sources(fixture);
  const { root } = fixture;
  const home = join(root, "home");
  const hub = join(root, "hub");
  const options = { env: { HF_HUB_CACHE: hub }, platform: "linux" as const, home, exhaustive: true };

  const missing = writeHfCache(hub, MODEL.repository, PIN, { "model.safetensors": fixture.sources["model.safetensors"] as Buffer });
  let result = await discoverSnapshots(MODEL, PIN, files, options);
  assert.equal(result.selected, undefined);
  assert.equal(result.candidates[0]?.reason, "missing tokenizer.json");
  assert.deepEqual(result.candidates[0]?.files.map((file) => file.verified), [false, false, false]);

  rmSync(missing, { recursive: true });
  const sized = writeHfCache(hub, MODEL.repository, PIN, { ...fixture.sources, "tokenizer.json": Buffer.from("short") });
  result = await discoverSnapshots(MODEL, PIN, files, options);
  assert.equal(result.candidates[0]?.reason, `tokenizer.json has 5 bytes, expected ${(fixture.sources["tokenizer.json"] as Buffer).length}`);

  rmSync(sized, { recursive: true });
  clearSnapshotDigestCache();
  const same = Buffer.from(fixture.sources["tokenizer.json"] as Buffer);
  same[2] = (same[2] as number) ^ 1;
  writeHfCache(hub, MODEL.repository, PIN, { ...fixture.sources, "tokenizer.json": same });
  result = await discoverSnapshots(MODEL, PIN, files, options);
  assert.equal(result.candidates[0]?.reason, "sha256 mismatch for tokenizer.json");
  assert.deepEqual(result.candidates[0]?.files.map((file) => [file.path, file.verified]), [
    ["model.safetensors", true],
    ["tokenizer.json", false],
    ["tokenizer_config.json", true],
  ]);

  // A different revision of the same repository is reported, not used.
  const other = "5a7adf72a23b4d311abae6ce050d7f0012bb3416";
  writeHfCache(hub, MODEL.repository, other, fixture.sources);
  result = await discoverSnapshots(MODEL, PIN, files, options);
  const wrong = result.candidates.find((candidate) => candidate.revision === other);
  assert.equal(wrong?.verified, false);
  assert.equal(wrong?.reason, `revision ${other} is not the pinned revision ${PIN}`);
  assert.equal(result.selected, undefined);

  // A dangling symlink (blob garbage-collected) counts as missing.
  const dangling = join(hub, `models--${SLUG}`, "snapshots", PIN, "tokenizer.json");
  rmSync(dangling);
  symlinkSync("../../blobs/gone", dangling);
  result = await discoverSnapshots(MODEL, PIN, files, options);
  assert.equal(result.candidates.find((candidate) => candidate.revision === PIN)?.reason, "missing tokenizer.json");
});

test("a modified file is re-hashed once its modification time changes", async () => {
  const fixture = buildTemplateFixture();
  const files = await sources(fixture);
  const directory = writeSnapshotDir(join(fixture.root, "snap"), fixture.sources);
  assert.equal((await verifySnapshotFiles(directory, files)).verified, true);
  const target = join(directory, "tokenizer_config.json");
  const before = statSync(target);
  const content = readFileSync(target);
  const changed = Buffer.from(content);
  changed[3] = (changed[3] as number) ^ 1;
  writeFileSync(target, changed);
  utimesSync(target, before.atime, new Date(before.mtimeMs + 5000));
  const result = await verifySnapshotFiles(directory, files);
  assert.equal(result.verified, false);
  assert.equal(result.reason, "sha256 mismatch for tokenizer_config.json");
  appendFileSync(target, " ");
  assert.match((await verifySnapshotFiles(directory, files)).reason ?? "", /has \d+ bytes, expected \d+/);
});

test("discovery never writes into the locations it reads", async () => {
  const layout = populate();
  const files = await sources(layout.fixture);
  const roots = ["explicit", "dirs-a", "dirs-b", "hub-cache", "hf-home", "xdg", "models"].map((name) => join(layout.fixture.root, name));
  const before = roots.map(tree);
  await discoverSnapshots(MODEL, PIN, files, {
    env: layout.env,
    platform: "linux",
    home: layout.home,
    modelsDir: layout.modelsDir,
    exhaustive: true,
  });
  assert.deepEqual(roots.map(tree), before);
  assert.deepEqual(readdirSync(layout.places["explicit"] as string).sort(), Object.keys(layout.fixture.sources).sort());
  mkdirSync(join(layout.fixture.root, "unused"), { recursive: true });
});
