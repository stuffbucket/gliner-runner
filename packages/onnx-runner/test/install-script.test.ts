import assert from "node:assert/strict";
import { execFile, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, readlinkSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

// Runs the repository's install.sh against a local HTTP server that mimics a GitHub release. No external network.
const INSTALL_SH = new URL("../../../../install.sh", import.meta.url).pathname;

interface Fixture {
  root: string;
  requests: string[];
  base: string;
  env: Record<string, string>;
  home: string;
  binDir: string;
  close(): Promise<void>;
  install(extra?: Record<string, string>, args?: string[]): Promise<{ status: number; stdout: string; stderr: string }>;
}

function makeArchive(root: string, version: string, platform: string): Buffer {
  const name = `gliner-runner-onnx-${version}-${platform}`;
  const dir = join(root, "build", name);
  mkdirSync(join(dir, "bin"), { recursive: true });
  mkdirSync(join(dir, "runtime"), { recursive: true });
  for (const command of ["gliner-runner", "gliner-runner-onnx"]) {
    const launcher = join(dir, "bin", command);
    writeFileSync(launcher, `#!/bin/sh\necho "fake gliner-runner-onnx ${version}"\n[ $# -eq 0 ] || echo "args: $*"\n`);
    chmodSync(launcher, 0o755);
  }
  writeFileSync(join(dir, "runtime", "node"), "not really node\n");
  const archive = join(root, `${name}.tar.gz`);
  const result = spawnSync("tar", ["-czf", archive, "-C", join(root, "build"), name]);
  assert.equal(result.status, 0, String(result.stderr));
  return readFileSync(archive);
}

function shim(root: string, os: string, machine: string): string {
  const dir = join(root, "shim");
  mkdirSync(dir, { recursive: true });
  const script = join(dir, "uname");
  writeFileSync(script, `#!/bin/sh\ncase "$1" in -s) echo ${os} ;; -m) echo ${machine} ;; *) echo ${os} ;; esac\n`);
  chmodSync(script, 0o755);
  return dir;
}

interface Release {
  version: string;
  /** Overrides the digest published in the .sha256 file. */
  sha?: string;
  /** Overrides the digest published in latest.json. */
  latestSha?: string;
}

async function fixture(
  releases: Release[],
  latest: string | undefined,
  system: { os: string; machine: string; platform: string } = { os: "Linux", machine: "x86_64", platform: "linux-x64" },
): Promise<Fixture> {
  const root = mkdtempSync(join(tmpdir(), "gliner-install-test-"));
  const files = new Map<string, Buffer | string>();
  for (const release of releases) {
    const name = `gliner-runner-onnx-${release.version}-${system.platform}.tar.gz`;
    const archive = makeArchive(root, release.version, system.platform);
    const digest = createHash("sha256").update(archive).digest("hex");
    files.set(`/releases/download/v${release.version}/${name}`, archive);
    files.set(`/releases/download/v${release.version}/${name}.sha256`, `${release.sha ?? digest}  ${name}\n`);
    if (release.version === latest) {
      files.set(
        "/releases/latest/download/latest.json",
        JSON.stringify(
          {
            version: release.version,
            channel: "stable",
            assets: { [system.platform]: { url: `http://example.invalid/${name}`, sha256: release.latestSha ?? digest } },
            notes_url: "https://example.test/notes",
          },
          null,
          2,
        ),
      );
    }
  }
  const requests: string[] = [];
  const server: Server = createServer((request, response) => {
    requests.push(request.url ?? "");
    const body = files.get(request.url ?? "");
    if (body === undefined) {
      response.statusCode = 404;
      response.end("not found");
      return;
    }
    response.end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/releases`;
  const home = join(root, "home");
  const binDir = join(root, "bin");
  const env = {
    PATH: `${shim(root, system.os, system.machine)}:${process.env["PATH"] ?? ""}`,
    HOME: join(root, "userhome"),
    GLINER_RUNNER_HOME: home,
    GLINER_RUNNER_BIN_DIR: binDir,
    GLINER_RUNNER_DOWNLOAD_BASE: base,
    TMPDIR: tmpdir(),
  };
  return {
    root,
    requests,
    base,
    env,
    home,
    binDir,
    async close() {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
      rmSync(root, { recursive: true, force: true });
    },
    install: (extra = {}, args = []) =>
      new Promise((resolve) => {
        execFile("/bin/sh", [INSTALL_SH, ...args], { env: { ...env, ...extra }, encoding: "utf8" }, (error, stdout, stderr) => {
          const code = (error as { code?: unknown } | null)?.code;
          resolve({ status: error === null ? 0 : typeof code === "number" ? code : -1, stdout, stderr });
        });
      }),
  };
}

/** Output of the linked launcher; both command names must run the same launcher. */
function runLinked(f: Fixture): string {
  const [user, original] = ["gliner-runner", "gliner-runner-onnx"].map((name) =>
    spawnSync(join(f.binDir, name), { encoding: "utf8" }).stdout.trim(),
  );
  assert.equal(user, original);
  return user as string;
}

test("installs the latest stable release, links the launcher and is idempotent", async () => {
  const f = await fixture([{ version: "1.2.3" }, { version: "1.3.0-rc.1" }], "1.2.3");
  try {
    const first = await f.install();
    assert.equal(first.status, 0, first.stderr);
    for (const command of ["gliner-runner", "gliner-runner-onnx"]) {
      assert.equal(readlinkSync(join(f.binDir, command)), join(f.home, "1.2.3", "bin", command));
    }
    assert.equal(runLinked(f), "fake gliner-runner-onnx 1.2.3");
    assert.ok(existsSync(join(f.home, "1.2.3", "runtime", "node")));
    assert.match(first.stdout, /^ {7}gliner-runner models install decide-340m$/m);
    assert.match(first.stdout, /^ {7}gliner-runner demo$/m);
    assert.match(first.stdout, /^ {7}gliner-runner serve$/m);
    assert.match(first.stdout, /Add .* to your PATH/);
    assert.ok(f.requests.every((url) => !url.includes("1.3.0-rc.1")));

    const downloads = f.requests.filter((url) => url.endsWith(".tar.gz")).length;
    const second = await f.install({ PATH: `${f.env["PATH"]}:${f.binDir}` });
    assert.equal(second.status, 0, second.stderr);
    assert.match(second.stdout, /already installed/);
    assert.doesNotMatch(second.stdout, /Add .* to your PATH/);
    assert.equal(f.requests.filter((url) => url.endsWith(".tar.gz")).length, downloads);
    assert.equal(runLinked(f), "fake gliner-runner-onnx 1.2.3");
  } finally {
    await f.close();
  }
});

test("--playground installs, then hands over to gliner-runner playground with the remaining options", async () => {
  const f = await fixture([{ version: "1.2.3" }], "1.2.3");
  try {
    const scratch = join(f.root, "tmp");
    mkdirSync(scratch);
    const result = await f.install({ TMPDIR: scratch }, ["--playground", "--yes", "--no-open"]);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /Installed gliner-runner 1\.2\.3/);
    assert.match(result.stdout, /fake gliner-runner-onnx 1\.2\.3\nargs: playground --yes --no-open$/m);
    assert.match(result.stdout, /add .* to your PATH/);
    assert.doesNotMatch(result.stdout, /Next steps/);
    assert.deepEqual(readdirSync(scratch), [], "the temporary directory is removed before handing over");
  } finally {
    await f.close();
  }
});

test("--playground reuses a gliner-runner on PATH that has the command, without downloading", async () => {
  const f = await fixture([{ version: "1.2.3" }], "1.2.3");
  try {
    const existing = join(f.root, "existing");
    mkdirSync(existing);
    writeFileSync(
      join(existing, "gliner-runner"),
      `#!/bin/sh
case "$1" in
  --version) echo "gliner-runner-onnx 9.9.9" ;;
  playground) [ "$2" = --help ] && exit 0; echo "existing playground $*" ;;
esac
`,
    );
    chmodSync(join(existing, "gliner-runner"), 0o755);
    const result = await f.install({ PATH: `${existing}:${f.env["PATH"]}` }, ["--playground", "--yes"]);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /Using the installed gliner-runner/);
    assert.match(result.stdout, /existing playground playground --yes/);
    assert.deepEqual(f.requests, []);

    // An older runner without the command is not reused: the latest release is installed instead.
    writeFileSync(join(existing, "gliner-runner"), `#!/bin/sh\necho "gliner-runner-onnx 0.4.0"\n[ "$1" = playground ] && exit 2\nexit 0\n`);
    const upgraded = await f.install({ PATH: `${existing}:${f.env["PATH"]}` }, ["--playground"]);
    assert.equal(upgraded.status, 0, upgraded.stderr);
    assert.match(upgraded.stdout, /args: playground$/m);
  } finally {
    await f.close();
  }
});

test("unknown options are refused before anything is downloaded", async () => {
  const f = await fixture([{ version: "1.2.3" }], "1.2.3");
  try {
    const result = await f.install({}, ["--bogus"]);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /unknown option: --bogus/);
    assert.deepEqual(f.requests, []);
  } finally {
    await f.close();
  }
});

test("GLINER_RUNNER_VERSION pins a release candidate without reading latest.json and repoints the link", async () => {
  const f = await fixture([{ version: "1.2.3" }, { version: "1.3.0-rc.1" }], "1.2.3");
  try {
    assert.equal((await f.install()).status, 0);
    const pinned = await f.install({ GLINER_RUNNER_VERSION: "v1.3.0-rc.1" });
    assert.equal(pinned.status, 0, pinned.stderr);
    assert.equal(runLinked(f), "fake gliner-runner-onnx 1.3.0-rc.1");
    assert.ok(existsSync(join(f.home, "1.2.3", "bin", "gliner-runner-onnx")), "older version is kept");
    assert.equal(f.requests.filter((url) => url.endsWith("latest.json")).length, 1);
  } finally {
    await f.close();
  }
});

test("reinstalls when an existing install lacks the gliner-runner launcher", async () => {
  const f = await fixture([{ version: "1.2.3" }], "1.2.3");
  try {
    // An install from before the alias existed has only bin/gliner-runner-onnx.
    const result = await f.install({ GLINER_RUNNER_VERSION: "1.2.3" });
    assert.equal(result.status, 0, result.stderr);
    rmSync(join(f.home, "1.2.3", "bin", "gliner-runner"));
    const repaired = await f.install();
    assert.equal(repaired.status, 0, repaired.stderr);
    assert.ok(existsSync(join(f.home, "1.2.3", "bin", "gliner-runner")), "an install missing the launcher is replaced");
  } finally {
    await f.close();
  }
});

test("replaces a broken install of the same version", async () => {
  const f = await fixture([{ version: "1.2.3" }], "1.2.3");
  try {
    mkdirSync(join(f.home, "1.2.3"), { recursive: true });
    writeFileSync(join(f.home, "1.2.3", "junk"), "x");
    const result = await f.install();
    assert.equal(result.status, 0, result.stderr);
    assert.equal(runLinked(f), "fake gliner-runner-onnx 1.2.3");
    assert.ok(!existsSync(join(f.home, "1.2.3", "junk")));
  } finally {
    await f.close();
  }
});

test("refuses a checksum mismatch and installs nothing", async () => {
  const f = await fixture([{ version: "1.2.3", sha: "0".repeat(64), latestSha: "0".repeat(64) }], "1.2.3");
  try {
    const result = await f.install();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /checksum mismatch/);
    const pinned = await f.install({ GLINER_RUNNER_VERSION: "1.2.3" });
    assert.notEqual(pinned.status, 0);
    assert.match(pinned.stderr, /checksum mismatch/);
    assert.ok(!existsSync(join(f.home, "1.2.3")));
    assert.ok(!existsSync(join(f.binDir, "gliner-runner")));
    assert.ok(!existsSync(join(f.binDir, "gliner-runner-onnx")));
  } finally {
    await f.close();
  }
});

test("refuses a checksum file that disagrees with latest.json", async () => {
  const f = await fixture([{ version: "1.2.3", latestSha: "f".repeat(64) }], "1.2.3");
  try {
    const result = await f.install();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /does not match latest\.json/);
    assert.ok(!existsSync(join(f.home, "1.2.3")));
  } finally {
    await f.close();
  }
});

test("fails clearly for a missing release and invalid versions", async () => {
  const f = await fixture([{ version: "1.2.3" }], "1.2.3");
  try {
    const missing = await f.install({ GLINER_RUNNER_VERSION: "9.9.9" });
    assert.notEqual(missing.status, 0);
    assert.match(missing.stderr, /could not download/);
    for (const bad of ["latest", "1.2", "../../etc", "1.2.3-beta.1"]) {
      const result = await f.install({ GLINER_RUNNER_VERSION: bad });
      assert.notEqual(result.status, 0, bad);
      assert.match(result.stderr, /GLINER_RUNNER_VERSION must look like/);
    }
  } finally {
    await f.close();
  }
});

test("supports Linux arm64 and macOS arm64 archives", async () => {
  for (const system of [
    { os: "Linux", machine: "aarch64", platform: "linux-arm64" },
    { os: "Darwin", machine: "arm64", platform: "darwin-arm64" },
  ]) {
    const f = await fixture([{ version: "1.2.3" }], "1.2.3", system);
    try {
      const result = await f.install();
      assert.equal(result.status, 0, result.stderr);
      assert.equal(runLinked(f), "fake gliner-runner-onnx 1.2.3");
    } finally {
      await f.close();
    }
  }
});

test("rejects Intel Macs and unsupported systems before downloading", async () => {
  for (const [system, pattern] of [
    [{ os: "Darwin", machine: "x86_64", platform: "darwin-x64" }, /Intel Macs are not supported/],
    [{ os: "Linux", machine: "armv7l", platform: "linux-arm" }, /unsupported CPU architecture/],
    [{ os: "FreeBSD", machine: "amd64", platform: "freebsd-x64" }, /unsupported OS/],
  ] as const) {
    const f = await fixture([{ version: "1.2.3" }], "1.2.3", system);
    try {
      const result = await f.install();
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, pattern);
      assert.deepEqual(f.requests, []);
    } finally {
      await f.close();
    }
  }
});
