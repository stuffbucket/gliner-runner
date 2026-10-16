#!/usr/bin/env node
// Assembles the self-contained, Python-free ONNX runner distribution for the current platform.
//
//   node scripts/package-onnx-runtime.mjs [--no-build] [--output-dir DIR] [--allow-node-mismatch]
//
// Layout of dist/release/onnx-runtime/gliner-runner-onnx-<version>-<platform>-<arch>/:
//   bin/gliner-runner[.cmd]        launcher (resolves its own directory, execs the bundled Node)
//   bin/gliner-runner-onnx[.cmd]   the same launcher under the original command name
//   runtime/node/                  Node.js runtime copied from the running Node executable
//   app/                           compiled app (dist/src only), production node_modules, and templates/
//                                  (the model artifact templates `gliner-runner-onnx models install` needs)
//   licenses/                      license texts that are not part of an npm package
//   LICENSE NOTICE THIRD_PARTY_NOTICES.md release-manifest.json
//
// Model weights and artifacts are never bundled; `gliner-runner-onnx models install <model-id>` downloads the
// pinned snapshot and prepares the artifact from the bundled templates (no Python).
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  chmodSync,
  copyFileSync,
  createReadStream,
  existsSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PACKAGE_DIR = join(ROOT, "packages", "onnx-runner");
const PACKAGE_NAME = "@gliner-runner/onnx-runner";
const LICENSE_FILE = /^(licen[sc]e|notice|copying)(\.[a-z]+)?$/i;
const WIN = process.platform === "win32";
/** Launcher names in bin/: the user-facing command first, then the original name, which keeps working. */
const COMMAND_NAMES = ["gliner-runner", "gliner-runner-onnx"];

function fail(message) {
  process.stderr.write(`package-onnx-runtime: ${message}\n`);
  process.exit(1);
}

function run(command, args, options = {}) {
  execFileSync(command, args, { stdio: "inherit", shell: WIN, ...options });
}

export function sha256File(path) {
  return new Promise((resolveHash, reject) => {
    const hash = createHash("sha256");
    createReadStream(path)
      .on("data", (chunk) => hash.update(chunk))
      .on("error", reject)
      .on("end", () => resolveHash(hash.digest("hex")));
  });
}

const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

/** Every real (non-symlink) package directory in a pnpm virtual store, sorted by name and version. */
export function inventoryPackages(nodeModules) {
  const store = join(nodeModules, ".pnpm");
  const packages = [];
  for (const entry of readdirSync(store, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name === "node_modules") {
      continue;
    }
    const modules = join(store, entry.name, "node_modules");
    for (const name of readdirSync(modules)) {
      const candidates = name.startsWith("@")
        ? readdirSync(join(modules, name)).map((child) => `${name}/${child}`)
        : [name];
      for (const candidate of candidates) {
        const directory = join(modules, candidate);
        if (lstatSync(directory).isSymbolicLink()) {
          continue;
        }
        const manifest = readJson(join(directory, "package.json"));
        const license =
          typeof manifest.license === "string"
            ? manifest.license
            : (manifest.license?.type ?? "UNKNOWN");
        packages.push({
          name: manifest.name,
          version: manifest.version,
          license,
          directory,
          licenseFiles: readdirSync(directory).filter((file) => LICENSE_FILE.test(file)),
        });
      }
    }
  }
  return packages.sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version));
}

function assertLocked(packages) {
  const lockfile = readFileSync(join(ROOT, "pnpm-lock.yaml"), "utf8");
  for (const { name, version } of packages) {
    const entry = new RegExp(`^  '?${escapeRegExp(`${name}@${version}`)}'?:`, "m");
    if (!entry.test(lockfile)) {
      fail(`${name}@${version} is bundled but absent from pnpm-lock.yaml`);
    }
  }
}

function walk(directory, base = directory, files = []) {
  for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) =>
    a.name.localeCompare(b.name),
  )) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      walk(path, base, files);
    } else {
      files.push(relative(base, path).split(sep).join("/"));
    }
  }
  return files;
}

export async function describeTree(root, exclude = new Set()) {
  const entries = [];
  for (const path of walk(root)) {
    if (exclude.has(path)) {
      continue;
    }
    const absolute = join(root, ...path.split("/"));
    const stat = lstatSync(absolute);
    if (stat.isSymbolicLink()) {
      entries.push({ path, type: "symlink", target: readlinkSync(absolute) });
    } else {
      entries.push({
        path,
        type: "file",
        size: stat.size,
        executable: (stat.mode & 0o111) !== 0,
        sha256: await sha256File(absolute),
      });
    }
  }
  return entries;
}

function launcherPosix() {
  return `#!/bin/sh
# GLiNER Runner ONNX launcher. Runs the bundled Node.js runtime; no Python, PyTorch, Transformers,
# or GLiNER2 installation is used or required.
set -eu

self=$0
while [ -h "$self" ]; do
  case $self in */*) dir=\${self%/*} ;; *) dir=. ;; esac
  link=$(readlink "$self")
  case $link in /*) self=$link ;; *) self=$dir/$link ;; esac
done
case $self in */*) bin_dir=\${self%/*} ;; *) bin_dir=. ;; esac
root=$(CDPATH= cd -P -- "$bin_dir/.." && pwd)

# Keep ambient Node configuration from changing which code is loaded.
unset NODE_OPTIONS NODE_PATH
exec "$root/runtime/node/bin/node" "$root/app/dist/src/cli.js" "$@"
`;
}

function launcherCmd() {
  return `@echo off\r
rem GLiNER Runner ONNX launcher. Runs the bundled Node.js runtime.\r
setlocal\r
set "NODE_OPTIONS="\r
set "NODE_PATH="\r
"%~dp0..\\runtime\\node\\node.exe" "%~dp0..\\app\\dist\\src\\cli.js" %*\r
exit /b %ERRORLEVEL%\r
`;
}

function git(...args) {
  try {
    return execFileSync("git", args, { cwd: ROOT, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return undefined;
  }
}

function pinnedNodeVersion() {
  const match = /^node\s*=\s*"([^"]+)"/m.exec(readFileSync(join(ROOT, "mise.toml"), "utf8"));
  return match?.[1];
}

function nodeLicensePath() {
  const candidates = [
    join(dirname(process.execPath), "..", "LICENSE"),
    join(dirname(process.execPath), "LICENSE"),
  ];
  const found = candidates.find((candidate) => existsSync(candidate));
  if (found === undefined || !readFileSync(found, "utf8").startsWith("Node.js is licensed for use")) {
    fail(
      `cannot find the Node.js LICENSE next to ${process.execPath}; ` +
        "package with an official Node.js distribution",
    );
  }
  return found;
}

function renderNotices({ version, node, onnxruntime, packages, nodeLicenseFile }) {
  const lines = [
    "# Third-party notices: GLiNER Runner ONNX runtime distribution",
    "",
    `Generated for gliner-runner-onnx ${version} (${process.platform}-${process.arch}) by`,
    "`scripts/package-onnx-runtime.mjs`. It lists exactly what is bundled in this directory. It is",
    "informational and does not replace the license files retained with each component.",
    "",
    "This distribution contains no Python interpreter, PyTorch, Transformers, GLiNER2, or model",
    "weights. Model artifacts are installed separately and keep their own license metadata; the",
    "curated models' declared license is Apache-2.0 (see the repository's THIRD_PARTY_NOTICES.md).",
    "",
    "## Bundled components",
    "",
    "| Component | Version | License | Notes |",
    "| --- | --- | --- | --- |",
    `| GLiNER Runner ONNX runner | \`${version}\` | Apache-2.0 | \`LICENSE\`, \`NOTICE\` |`,
    `| Node.js runtime | \`${node.version}\` | MIT and bundled third-party licenses | \`licenses/node-LICENSE\`; binary sha256 \`${node.sha256}\` |`,
    `| ONNX Runtime native library | \`${onnxruntime.version}\` | MIT | shipped inside \`onnxruntime-node\`; ${onnxruntime.libraries.join(", ")} |`,
    "",
    "## Bundled npm packages",
    "",
    "| Package | Version | License | License files |",
    "| --- | --- | --- | --- |",
    ...packages.map(
      (pkg) =>
        `| \`${pkg.name}\` | \`${pkg.version}\` | ${pkg.license} | ${
          pkg.licenseFiles.length > 0 ? pkg.licenseFiles.map((file) => `\`${file}\``).join(", ") : "none in package; see notice below"
        } |`,
    ),
    "",
    "## Node.js",
    "",
    "The Node.js runtime is copied unmodified from an official Node.js distribution. Its complete",
    `license, including the licenses of the components bundled into Node.js, is \`${nodeLicenseFile}\`.`,
    "",
    "## ONNX Runtime and onnxruntime-node",
    "",
    "`onnxruntime-node`, `onnxruntime-common`, and the ONNX Runtime native library are published by",
    "Microsoft under the MIT License. The npm packages ship no separate license file. ONNX Runtime's",
    "own third-party notices are published at",
    `<https://github.com/microsoft/onnxruntime/blob/v${onnxruntime.version}/ThirdPartyNotices.txt>.`,
    "",
    "> MIT License",
    ">",
    "> Copyright (c) Microsoft Corporation",
    ">",
    "> Permission is hereby granted, free of charge, to any person obtaining a copy",
    '> of this software and associated documentation files (the "Software"), to deal',
    "> in the Software without restriction, including without limitation the rights",
    "> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell",
    "> copies of the Software, and to permit persons to whom the Software is",
    "> furnished to do so, subject to the following conditions:",
    ">",
    "> The above copyright notice and this permission notice shall be included in",
    "> all copies or substantial portions of the Software.",
    ">",
    '> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR',
    "> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,",
    "> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE",
    "> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER",
    "> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,",
    "> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE",
    "> SOFTWARE.",
    "",
    "## npm package license texts",
    "",
    "The text below is copied from each package's own license file, grouped by identical text. The",
    "same files remain in `app/node_modules`.",
    "",
  ];
  const byText = new Map();
  for (const pkg of packages) {
    for (const file of pkg.licenseFiles) {
      const text = readFileSync(join(pkg.directory, file), "utf8").trim();
      byText.set(text, [...(byText.get(text) ?? []), `${pkg.name}@${pkg.version}`]);
    }
  }
  for (const [text, owners] of [...byText].sort((a, b) => a[1][0].localeCompare(b[1][0]))) {
    lines.push(`### ${owners.join(", ")}`, "", ...text.split("\n").map((line) => (line === "" ? ">" : `> ${line.trimEnd()}`)), "");
  }
  return `${lines.join("\n")}\n`;
}

async function main() {
  const { values } = parseArgs({
    options: {
      "no-build": { type: "boolean", default: false },
      "output-dir": { type: "string", default: join(ROOT, "dist", "release", "onnx-runtime") },
      "allow-node-mismatch": { type: "boolean", default: false },
    },
  });
  const outputDir = resolve(values["output-dir"]);
  const platform = process.platform;
  const arch = process.arch;

  const nodeVersion = process.version.slice(1);
  if (!nodeVersion.startsWith("24.")) {
    fail(`Node 24 is required to package; running ${process.version}`);
  }
  const pinned = pinnedNodeVersion();
  if (pinned !== undefined && pinned !== nodeVersion && !values["allow-node-mismatch"]) {
    fail(`mise.toml pins Node ${pinned} but this is ${nodeVersion}; run through \`mise exec\``);
  }

  const appPackage = readJson(join(PACKAGE_DIR, "package.json"));
  const version = appPackage.version;
  const releaseName = `gliner-runner-onnx-${version}-${platform}-${arch}`;
  const releaseDir = join(outputDir, releaseName);
  const archive = `${releaseDir}.tar.gz`;
  mkdirSync(outputDir, { recursive: true });
  rmSync(releaseDir, { recursive: true, force: true });
  rmSync(archive, { force: true });
  rmSync(`${archive}.sha256`, { force: true });

  if (!values["no-build"]) {
    run("pnpm", ["--filter", `${PACKAGE_NAME}...`, "run", "build"], { cwd: ROOT });
  }
  const compiled = join(PACKAGE_DIR, "dist", "src");
  if (!existsSync(join(compiled, "cli.js"))) {
    fail(`${compiled}/cli.js is missing; build the package first`);
  }
  // Refuse compiled output whose source no longer exists (stale files would ship untested code).
  for (const file of walk(compiled).filter((path) => path.endsWith(".js"))) {
    if (!existsSync(join(PACKAGE_DIR, "src", file.replace(/\.js$/, ".ts")))) {
      fail(`stale compiled file dist/src/${file} has no matching source; rebuild from a clean dist`);
    }
  }

  // Production dependencies come from pnpm-lock.yaml through `pnpm deploy`; the on-install download
  // of optional native components is disabled so packaging is offline and deterministic.
  const appDir = join(releaseDir, "app");
  mkdirSync(releaseDir, { recursive: true });
  const deployEnv = { ...process.env, ONNXRUNTIME_NODE_INSTALL: "skip", CI: "true" };
  if (WIN) {
    deployEnv.npm_config_node_linker = "hoisted";
  }
  run("pnpm", ["--filter", PACKAGE_NAME, "deploy", "--prod", appDir], { cwd: ROOT, env: deployEnv });
  for (const stray of ["src", "test", "dist", "tsconfig.json", "node_modules/.modules.yaml"]) {
    rmSync(join(appDir, stray), { recursive: true, force: true });
  }
  writeFileSync(
    join(appDir, "package.json"),
    `${JSON.stringify(
      {
        name: appPackage.name,
        version,
        private: true,
        type: "module",
        license: appPackage.license,
        // The runner reaches its host through `#platform`; a release is always the Node.js build.
        imports: { "#platform": appPackage.imports["#platform"].default },
        dependencies: appPackage.dependencies,
      },
      null,
      2,
    )}\n`,
  );
  // Artifact templates (template.json, graph.onnx, gliner2.json, manifest.json per curated model): the default
  // template directory is resolved relative to the compiled module, i.e. app/templates.
  const templatesSource = join(PACKAGE_DIR, "templates");
  const templatesTarget = join(appDir, "templates");
  rmSync(templatesTarget, { recursive: true, force: true });
  const templateFiles = existsSync(templatesSource) ? walk(templatesSource) : [];
  if (!templateFiles.some((file) => file.endsWith("/template.json"))) {
    fail(`${templatesSource} has no template.json; the release could not install any model`);
  }
  for (const file of templateFiles) {
    const destination = join(templatesTarget, ...file.split("/"));
    mkdirSync(dirname(destination), { recursive: true });
    copyFileSync(join(templatesSource, ...file.split("/")), destination);
  }
  for (const file of walk(compiled)) {
    if (/\.d\.ts(\.map)?$|\.js\.map$/.test(file)) {
      continue;
    }
    const destination = join(appDir, "dist", "src", ...file.split("/"));
    mkdirSync(dirname(destination), { recursive: true });
    copyFileSync(join(compiled, ...file.split("/")), destination);
  }

  // Keep only the native ONNX Runtime binary for this platform and architecture.
  const ortNative = join(appDir, "node_modules", "onnxruntime-node", "bin", "napi-v6");
  const keep = join(ortNative, platform, arch);
  if (!existsSync(keep)) {
    fail(`onnxruntime-node ships no native binary for ${platform}/${arch}`);
  }
  for (const platformDir of readdirSync(ortNative)) {
    for (const archDir of readdirSync(join(ortNative, platformDir))) {
      if (platformDir !== platform || archDir !== arch) {
        rmSync(join(ortNative, platformDir, archDir), { recursive: true, force: true });
      }
    }
    if (platformDir !== platform) {
      rmSync(join(ortNative, platformDir), { recursive: true, force: true });
    }
  }
  const nativeLibraries = readdirSync(keep).sort();
  if (!nativeLibraries.includes("onnxruntime_binding.node")) {
    fail("onnxruntime_binding.node is missing after pruning");
  }

  const packages = inventoryPackages(join(appDir, "node_modules"));
  assertLocked(packages);
  const bundledNames = new Set(packages.map((pkg) => pkg.name));
  for (const required of Object.keys(appPackage.dependencies)) {
    if (!bundledNames.has(required)) {
      fail(`production dependency ${required} was not installed`);
    }
  }
  const forbidden = packages.filter((pkg) => /^(torch|transformers|gliner2?|onnx|tensorflow)\b/i.test(pkg.name));
  if (forbidden.length > 0) {
    fail(`unexpected inference framework packages bundled: ${forbidden.map((p) => p.name).join(", ")}`);
  }
  const ort = packages.find((pkg) => pkg.name === "onnxruntime-node");

  // Pinned Node.js runtime, copied from the executable that runs this script.
  const nodeTarget = WIN
    ? join(releaseDir, "runtime", "node", "node.exe")
    : join(releaseDir, "runtime", "node", "bin", "node");
  mkdirSync(dirname(nodeTarget), { recursive: true });
  copyFileSync(process.execPath, nodeTarget);
  chmodSync(nodeTarget, 0o755);
  const nodeSha = await sha256File(nodeTarget);
  if (nodeSha !== (await sha256File(process.execPath))) {
    fail("copied Node.js runtime does not match the source executable");
  }
  mkdirSync(join(releaseDir, "licenses"), { recursive: true });
  copyFileSync(nodeLicensePath(), join(releaseDir, "licenses", "node-LICENSE"));

  mkdirSync(join(releaseDir, "bin"), { recursive: true });
  for (const command of COMMAND_NAMES) {
    writeFileSync(join(releaseDir, "bin", command), launcherPosix());
    chmodSync(join(releaseDir, "bin", command), 0o755);
    writeFileSync(join(releaseDir, "bin", `${command}.cmd`), launcherCmd());
  }

  for (const name of ["LICENSE", "NOTICE"]) {
    copyFileSync(join(ROOT, name), join(releaseDir, name));
  }
  const node = { version: nodeVersion, sha256: nodeSha };
  writeFileSync(
    join(releaseDir, "THIRD_PARTY_NOTICES.md"),
    renderNotices({
      version,
      node,
      onnxruntime: { version: ort.version, libraries: nativeLibraries.map((name) => `\`${name}\``) },
      packages,
      nodeLicenseFile: "licenses/node-LICENSE",
    }),
  );

  const manifestPath = "release-manifest.json";
  const files = await describeTree(releaseDir, new Set([manifestPath]));
  const nodeEntry = files.find((entry) => entry.path === relative(releaseDir, nodeTarget).split(sep).join("/"));
  const manifest = {
    formatVersion: 1,
    name: "gliner-runner-onnx",
    version,
    platform,
    arch,
    source: { commit: git("rev-parse", "HEAD") ?? null, dirty: (git("status", "--porcelain") ?? "") !== "" },
    executable: platform === "win32" ? "bin/gliner-runner-onnx.cmd" : "bin/gliner-runner-onnx",
    executables: COMMAND_NAMES.map((command) => `bin/${command}${platform === "win32" ? ".cmd" : ""}`),
    node: {
      version: nodeVersion,
      path: nodeEntry?.path,
      size: nodeEntry?.size,
      sha256: nodeSha,
      license: "licenses/node-LICENSE",
    },
    onnxruntime: {
      npmPackage: `onnxruntime-node@${ort.version}`,
      nativeLibraries: nativeLibraries.map((name) => `app/node_modules/onnxruntime-node/bin/napi-v6/${platform}/${arch}/${name}`),
    },
    packages: packages.map(({ name, version: v, license }) => ({ name, version: v, license })),
    modelsIncluded: false,
    pythonRequired: false,
    modelArtifacts:
      "Weights are not bundled. Install with `gliner-runner-onnx models install <model-id>` into <models-dir>/onnx/.",
    templates: templateFiles
      .filter((file) => file.endsWith("/template.json"))
      .map((file) => `app/templates/${file}`),
    files,
  };
  writeFileSync(join(releaseDir, manifestPath), `${JSON.stringify(manifest, null, 2)}\n`);

  run("tar", ["-czf", archive, "-C", outputDir, releaseName], {
    env: { ...process.env, COPYFILE_DISABLE: "1" },
  });
  const archiveSha = await sha256File(archive);
  writeFileSync(`${archive}.sha256`, `${archiveSha}  ${basename(archive)}\n`);

  const totalBytes = files.reduce((sum, entry) => sum + (entry.size ?? 0), 0);
  process.stdout.write(
    `${JSON.stringify(
      {
        releaseDir,
        archive,
        archiveSha256: archiveSha,
        archiveBytes: lstatSync(archive).size,
        unpackedBytes: totalBytes,
        files: files.length,
        node: node,
        packages: packages.length,
      },
      null,
      2,
    )}\n`,
  );
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
