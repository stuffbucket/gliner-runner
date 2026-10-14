import assert from "node:assert/strict";
import test from "node:test";

import { SettingsError, defaultModelsDir, defaultSettings, parseServeArguments } from "../src/settings.js";

test("defaults match the Python runner", () => {
  const settings = parseServeArguments([], {});
  assert.deepEqual(settings, defaultSettings());
  assert.equal(settings.host, "127.0.0.1");
  assert.equal(settings.port, 8090);
  assert.equal(settings.device, "cpu");
  assert.equal(settings.queueCapacity, 256);
  assert.equal(settings.maxBatchSize, 16);
  assert.equal(settings.batchWindowMs, 4);
  assert.equal(settings.modelIdleTtlSeconds, 0);
  assert.equal(settings.memoryLimitBytes, undefined);
});

test("environment overrides defaults and flags override the environment", () => {
  const env = {
    GLINER_RUNNER_HOST: "0.0.0.0",
    GLINER_RUNNER_PORT: "8080",
    GLINER_RUNNER_DEVICE: "env-device",
    GLINER_RUNNER_MAX_BATCH_SIZE: "8",
    GLINER_RUNNER_MODEL_IDLE_TTL_SECONDS: "2.5",
    GLINER_RUNNER_MEMORY_LIMIT_BYTES: String(1024 ** 3),
    GLINER_RUNNER_MODELS_DIR: "/env/models",
  };
  const fromEnv = parseServeArguments([], env);
  assert.deepEqual(
    [fromEnv.host, fromEnv.port, fromEnv.device, fromEnv.maxBatchSize, fromEnv.modelIdleTtlSeconds],
    ["0.0.0.0", 8080, "env-device", 8, 2.5],
  );
  assert.equal(fromEnv.memoryLimitBytes, 1024 ** 3);
  assert.equal(fromEnv.modelsDir, "/env/models");

  const flagged = parseServeArguments(
    ["--port", "9000", "--device", "flag-device", "--max-batch-size=4", "--models-dir", "/flag"],
    env,
  );
  assert.deepEqual(
    [flagged.host, flagged.port, flagged.device, flagged.maxBatchSize, flagged.modelsDir],
    ["0.0.0.0", 9000, "flag-device", 4, "/flag"],
  );
});

test("accepts every documented flag", () => {
  const settings = parseServeArguments(
    [
      "--host", "::1",
      "--port", "1",
      "--device", "coreml",
      "--models-dir", "/m",
      "--queue-capacity", "3",
      "--max-batch-size", "2",
      "--batch-window-ms", "0",
      "--model-idle-ttl-seconds", "30",
      "--memory-limit-bytes", String(512 * 1024 ** 2),
      "--shutdown-timeout-ms", "100",
      "--max-body-bytes", "4096",
    ],
    {},
  );
  assert.deepEqual(settings, {
    host: "::1",
    port: 1,
    device: "coreml",
    modelsDir: "/m",
    queueCapacity: 3,
    maxBatchSize: 2,
    batchWindowMs: 0,
    modelIdleTtlSeconds: 30,
    memoryLimitBytes: 512 * 1024 ** 2,
    shutdownTimeoutMs: 100,
    maxBodyBytes: 4096,
    autoDownload: false,
    templatesDir: undefined,
  });
});

test("auto-download and the templates directory are opt-in settings", () => {
  assert.equal(parseServeArguments([], {}).autoDownload, false);
  assert.equal(parseServeArguments(["--auto-download"], {}).autoDownload, true);
  assert.equal(parseServeArguments([], { GLINER_RUNNER_AUTO_DOWNLOAD: "1" }).autoDownload, true);
  assert.equal(parseServeArguments([], { GLINER_RUNNER_AUTO_DOWNLOAD: "true" }).autoDownload, true);
  assert.equal(parseServeArguments([], { GLINER_RUNNER_AUTO_DOWNLOAD: "0" }).autoDownload, false);
  assert.equal(parseServeArguments([], { GLINER_RUNNER_AUTO_DOWNLOAD: "" }).autoDownload, false);
  assert.throws(() => parseServeArguments([], { GLINER_RUNNER_AUTO_DOWNLOAD: "maybe" }), /GLINER_RUNNER_AUTO_DOWNLOAD/);
  assert.equal(parseServeArguments([], { GLINER_RUNNER_ONNX_TEMPLATES_DIR: "/t" }).templatesDir, "/t");
  assert.equal(parseServeArguments(["--templates-dir", "/flag"], { GLINER_RUNNER_ONNX_TEMPLATES_DIR: "/t" }).templatesDir, "/flag");
});

test("rejects out-of-range, malformed and unknown arguments", () => {
  const bad: [string[], NodeJS.ProcessEnv, RegExp][] = [
    [["--port", "0"], {}, /--port must be an integer between 1 and 65535/],
    [["--port", "65536"], {}, /--port/],
    [["--port", "80x"], {}, /--port/],
    [["--queue-capacity", "0"], {}, /--queue-capacity must be an integer at least 1/],
    [["--max-batch-size=-1"], {}, /--max-batch-size/],
    [["--batch-window-ms=-0.5"], {}, /--batch-window-ms must be a number of at least 0/],
    [["--model-idle-ttl-seconds", "abc"], {}, /--model-idle-ttl-seconds/],
    [["--model-idle-ttl-seconds", "2147484"], {}, /--model-idle-ttl-seconds must be a number between 0 and 2147483/],
    [[], { GLINER_RUNNER_MODEL_IDLE_TTL_SECONDS: "1e9" }, /GLINER_RUNNER_MODEL_IDLE_TTL_SECONDS must be a number between/],
    [["--shutdown-timeout-ms", "2147483648"], {}, /--shutdown-timeout-ms/],
    [["--max-body-bytes", "0"], {}, /--max-body-bytes must be an integer at least 1/],
    [[], { GLINER_RUNNER_MAX_BODY_BYTES: "lots" }, /GLINER_RUNNER_MAX_BODY_BYTES must be an integer/],
    [["--memory-limit-bytes", "1024"], {}, /--memory-limit-bytes must be an integer at least 536870912/],
    [["--device", ""], {}, /--device must not be empty/],
    [[], { GLINER_RUNNER_PORT: "nope" }, /GLINER_RUNNER_PORT must be an integer/],
    [[], { GLINER_RUNNER_MEMORY_LIMIT_BYTES: "5" }, /GLINER_RUNNER_MEMORY_LIMIT_BYTES/],
    [["--bogus"], {}, /Unknown option/],
    [["stray"], {}, /unexpected argument: stray/],
  ];
  for (const [argv, env, message] of bad) {
    assert.throws(() => parseServeArguments(argv, env), (error: unknown) => {
      assert.ok(error instanceof SettingsError, `${argv.join(" ")} should raise SettingsError`);
      assert.match(error.message, message);
      return true;
    });
  }
});

test("empty environment values are ignored", () => {
  assert.equal(parseServeArguments([], { GLINER_RUNNER_PORT: "" }).port, 8090);
});

test("default models directory follows platform cache conventions", () => {
  assert.match(defaultModelsDir({}, "darwin"), /Library\/Caches\/gliner-runner\/models$/);
  assert.equal(
    defaultModelsDir({ XDG_CACHE_HOME: "/x" }, "linux"),
    "/x/gliner-runner/models",
  );
  assert.match(defaultModelsDir({}, "linux"), /\.cache\/gliner-runner\/models$/);
});

test("idle TTL accepts its maximum and the request body limit is configurable", () => {
  assert.equal(parseServeArguments(["--model-idle-ttl-seconds", "2147483"], {}).modelIdleTtlSeconds, 2147483);
  assert.equal(parseServeArguments([], {}).maxBodyBytes, 16 * 1024 * 1024);
  assert.equal(parseServeArguments([], { GLINER_RUNNER_MAX_BODY_BYTES: "1000" }).maxBodyBytes, 1000);
  assert.equal(
    parseServeArguments(["--max-body-bytes", "2000"], { GLINER_RUNNER_MAX_BODY_BYTES: "1000" }).maxBodyBytes,
    2000,
  );
});
