import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { readFileSync } from "node:fs";
import test from "node:test";

import { CLI } from "./helpers.js";
import { BREW_UPDATE_COMMAND, INSTALL_COMMAND, compareSemver, installCommandFor, runUpdateCommand } from "../src/update.js";

interface Run {
  status: number;
  stdout: string;
  stderr: string;
}

function cli(args: string[], env: Record<string, string>): Promise<Run> {
  return new Promise((resolve) => {
    execFile(
      process.execPath,
      [CLI, ...args],
      { env: { PATH: process.env["PATH"] ?? "", ...env }, encoding: "utf8" },
      (error, stdout, stderr) => {
        const code = (error as { code?: unknown } | null)?.code;
        resolve({ status: error === null ? 0 : typeof code === "number" ? code : -1, stdout, stderr });
      },
    );
  });
}

async function withServer<T>(
  handler: Parameters<typeof createServer>[1],
  body: (url: string) => Promise<T>,
): Promise<T> {
  const server: Server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    return await body(`http://127.0.0.1:${(server.address() as AddressInfo).port}/latest.json`);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

function currentVersion(): string {
  return (JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8")) as { version: string })
    .version;
}

const serveJson =
  (payload: unknown): Parameters<typeof createServer>[1] =>
  (_request, response) => {
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify(payload));
  };


test("compareSemver follows semver precedence", () => {
  const order = [
    "0.2.1-rc.1",
    "0.2.1-rc.2",
    "0.2.1-rc.10",
    "0.2.1",
    "0.2.2",
    "0.10.0",
    "1.0.0-alpha",
    "1.0.0-alpha.1",
    "1.0.0-alpha.beta",
    "1.0.0-beta",
    "1.0.0",
  ];
  for (let i = 0; i < order.length; i += 1) {
    for (let j = 0; j < order.length; j += 1) {
      assert.equal(compareSemver(order[i]!, order[j]!), Math.sign(i - j), `${order[i]} vs ${order[j]}`);
    }
  }
  assert.equal(compareSemver("v1.2.3", "1.2.3+build"), 0);
  assert.equal(compareSemver("1.2", "1.2.3"), undefined);
  assert.equal(compareSemver("latest", "1.2.3"), undefined);
});

test("update check reports an available update with the install command", async () => {
  await withServer(
    serveJson({ version: "999.0.0", channel: "stable", assets: {}, notes_url: "https://example.test/notes" }),
    async (url) => {
      const text = await cli(["update", "check"], { GLINER_RUNNER_UPDATE_URL: url });
      assert.equal(text.status, 0, text.stderr);
      assert.match(text.stdout, /Update available: .* -> 999\.0\.0/);
      assert.match(text.stdout, /https:\/\/example\.test\/notes/);
      assert.ok(text.stdout.includes(INSTALL_COMMAND));

      const json = await cli(["update", "check", "--json"], { GLINER_RUNNER_UPDATE_URL: url });
      assert.equal(json.status, 0, json.stderr);
      assert.deepEqual(JSON.parse(json.stdout), {
        current: currentVersion(),
        latest: "999.0.0",
        channel: "stable",
        update_available: true,
        notes_url: "https://example.test/notes",
        install_command: INSTALL_COMMAND,
      });
    },
  );
});

test("update check says up to date when the latest stable equals or trails this build", async () => {
  for (const latest of [currentVersion(), "0.0.1"]) {
    await withServer(serveJson({ version: latest, channel: "stable" }), async (url) => {
      const result = await cli(["update", "check", "--json"], { GLINER_RUNNER_UPDATE_URL: url });
      assert.equal(result.status, 0, result.stderr);
      assert.equal((JSON.parse(result.stdout) as { update_available: boolean }).update_available, false);
      const text = await cli(["update", "check"], { GLINER_RUNNER_UPDATE_URL: url });
      assert.match(text.stdout, /is up to date/);
    });
  }
});

test("a release candidate descriptor is never offered", async () => {
  await withServer(serveJson({ version: "999.0.0-rc.1", channel: "stable" }), async (url) => {
    const result = await cli(["update", "check", "--json"], { GLINER_RUNNER_UPDATE_URL: url });
    assert.equal(result.status, 0, result.stderr);
    assert.equal((JSON.parse(result.stdout) as { update_available: boolean }).update_available, false);
  });
  await withServer(serveJson({ version: "999.0.0", channel: "beta" }), async (url) => {
    const result = await cli(["update", "check", "--json"], { GLINER_RUNNER_UPDATE_URL: url });
    assert.equal((JSON.parse(result.stdout) as { update_available: boolean }).update_available, false);
  });
});

test("update check exits 1 on network, HTTP, timeout and malformed-response failures", async () => {
  const unreachable = await cli(["update", "check"], { GLINER_RUNNER_UPDATE_URL: "http://127.0.0.1:1/latest.json" });
  assert.equal(unreachable.status, 1);
  assert.match(unreachable.stderr, /update check failed/);

  await withServer(
    (_request, response) => {
      response.statusCode = 404;
      response.end("nope");
    },
    async (url) => {
      const result = await cli(["update", "check"], { GLINER_RUNNER_UPDATE_URL: url });
      assert.equal(result.status, 1);
      assert.match(result.stderr, /HTTP 404/);
    },
  );
  await withServer(
    (_request, response) => {
      response.end("<html>not json</html>");
    },
    async (url) => {
      const result = await cli(["update", "check"], { GLINER_RUNNER_UPDATE_URL: url });
      assert.equal(result.status, 1);
      assert.match(result.stderr, /valid JSON/);
    },
  );
  await withServer(serveJson({ channel: "stable" }), async (url) => {
    const result = await cli(["update", "check"], { GLINER_RUNNER_UPDATE_URL: url });
    assert.equal(result.status, 1);
  });
  await withServer(serveJson({ version: "not-a-version" }), async (url) => {
    const result = await cli(["update", "check"], { GLINER_RUNNER_UPDATE_URL: url });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /invalid version/);
  });
  await withServer(
    () => {
      // never respond
    },
    async (url) => {
      const started = Date.now();
      const result = await cli(["update", "check"], {
        GLINER_RUNNER_UPDATE_URL: url,
        GLINER_RUNNER_UPDATE_TIMEOUT_MS: "200",
      });
      assert.equal(result.status, 1);
      assert.match(result.stderr, /timed out/);
      assert.ok(Date.now() - started < 5000);
    },
  );
  const badUrl = await cli(["update", "check"], { GLINER_RUNNER_UPDATE_URL: "file:///etc/passwd" });
  assert.equal(badUrl.status, 1);
  assert.match(badUrl.stderr, /http\(s\)/);
});

test("update usage errors exit 2 and help exits 0", async () => {
  assert.equal((await cli(["update"], {})).status, 2);
  assert.equal((await cli(["update", "install"], {})).status, 2);
  assert.equal((await cli(["update", "check", "--bogus"], {})).status, 2);
  const help = await cli(["update", "--help"], {});
  assert.equal(help.status, 0);
  assert.match(help.stdout, /Usage: gliner-runner update check/);
});

test("a Homebrew install is told to run brew upgrade, any other install the curl command", async () => {
  assert.equal(installCommandFor("/opt/homebrew/Cellar/gliner-runner/0.3.0/libexec/app/dist/src/cli.js"), BREW_UPDATE_COMMAND);
  assert.equal(installCommandFor("/home/linuxbrew/.linuxbrew/Cellar/gliner-runner/0.3.0/libexec/app/dist/src/cli.js"), BREW_UPDATE_COMMAND);
  assert.equal(installCommandFor("/Users/me/.local/share/gliner-runner/0.3.0/app/dist/src/cli.js"), INSTALL_COMMAND);
  assert.equal(installCommandFor(undefined), INSTALL_COMMAND);
  assert.equal(BREW_UPDATE_COMMAND, "brew update && brew upgrade gliner-runner");

  await withServer(serveJson({ version: "99.0.0", channel: "stable" }), async (url) => {
    let out = "";
    const code = await runUpdateCommand(
      ["check"],
      {
        env: { GLINER_RUNNER_UPDATE_URL: url },
        appPath: "/opt/homebrew/Cellar/gliner-runner/0.3.0/libexec/app/dist/src/cli.js",
        stdout: (text) => void (out += text),
        stderr: () => undefined,
      },
      "0.3.0",
    );
    assert.equal(code, 0);
    assert.match(out, /Update available: 0\.3\.0 -> 99\.0\.0/);
    assert.ok(out.includes("  brew update && brew upgrade gliner-runner"));
    assert.ok(!out.includes("curl"));
  });
});
