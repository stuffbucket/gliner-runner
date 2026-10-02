import assert from "node:assert/strict";
import { createServer } from "node:net";
import test from "node:test";

import { GlinerSidecar, GlinerSidecarError } from "./dist/sidecar.js";

const fakeRunner = `
const http = await import("node:http");
const portIndex = process.argv.indexOf("--port");
const port = Number(process.argv[portIndex + 1]);
const server = http.createServer((request, response) => {
  if (request.url === "/healthz") {
    response.writeHead(200, {"content-type": "application/json"});
    response.end(JSON.stringify({status: "ok", accepting_requests: true}));
    return;
  }
  response.writeHead(404);
  response.end();
});
server.listen(port, "127.0.0.1");
process.on("SIGTERM", () => server.close(() => process.exit(0)));
`;

async function freePort() {
  const server = createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.equal(typeof address, "object");
  const port = address.port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

test("starts, probes, and stops an Electron sidecar", async () => {
  const sidecar = new GlinerSidecar({
    command: process.execPath,
    commandArguments: ["--input-type=module", "-e", fakeRunner],
    port: await freePort(),
    startupTimeoutMs: 5_000,
  });

  await sidecar.start();
  assert.equal(sidecar.running, true);
  const response = await fetch(`${sidecar.baseUrl}/healthz`);
  assert.equal(response.status, 200);
  await sidecar.stop();
  assert.equal(sidecar.running, false);
});

test("reports an early sidecar exit with bounded diagnostics", async () => {
  const sidecar = new GlinerSidecar({
    command: process.execPath,
    commandArguments: [
      "--input-type=module",
      "-e",
      'process.stderr.write("discard-startup failed"); process.exit(7)',
    ],
    port: await freePort(),
    startupTimeoutMs: 2_000,
    diagnosticLimitBytes: 14,
  });

  await assert.rejects(sidecar.start(), (error) => {
    assert.ok(error instanceof GlinerSidecarError);
    assert.match(error.message, /code=7[\s\S]*startup failed/);
    assert.equal(error.diagnostics, "startup failed");
    return true;
  });
});

test("terminates a sidecar that never becomes ready", async () => {
  const sidecar = new GlinerSidecar({
    command: process.execPath,
    commandArguments: [
      "--input-type=module",
      "-e",
      'process.on("SIGTERM", () => process.exit(0)); setInterval(() => {}, 1000)',
    ],
    port: await freePort(),
    startupTimeoutMs: 100,
    shutdownTimeoutMs: 1_000,
  });

  await assert.rejects(sidecar.start(), /did not become ready within 100 ms/);
  assert.equal(sidecar.running, false);
});
