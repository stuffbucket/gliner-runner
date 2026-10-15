import { homedir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { MIN_MEMORY_LIMIT_BYTES } from "./resources.js";
import { MAX_MODEL_IDLE_TTL_SECONDS } from "./runtime.js";

export interface Settings {
  host: string;
  port: number;
  device: string;
  queueCapacity: number;
  maxBatchSize: number;
  batchWindowMs: number;
  /** Root directory of installed model artifacts. */
  modelsDir: string;
  modelIdleTtlSeconds: number;
  /** Hard memory ceiling in bytes; `undefined` means 75% of physical memory. */
  memoryLimitBytes: number | undefined;
  /** Budget for a graceful shutdown before the CLI forces exit. */
  shutdownTimeoutMs: number;
  /** Largest accepted request body in bytes; larger bodies are rejected with `413 request_too_large`. */
  maxBodyBytes: number;
  /**
   * When true, an inference request for a missing curated model with a bundled template downloads and prepares
   * it (one shared job) and waits. Off by default: inference never downloads implicitly.
   */
  autoDownload: boolean;
  /** Directory of bundled artifact templates; `undefined` means the package's own `templates/`. */
  templatesDir: string | undefined;
  /**
   * Browser origins allowed to call the server (CORS), compared exactly with the request's `Origin` header.
   * Empty (the default) sends no CORS headers.
   */
  corsOrigins: readonly string[];
}

export const DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024;

export class SettingsError extends Error {
  override readonly name = "SettingsError";
}

export function defaultModelsDir(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): string {
  const home = homedir();
  const cache =
    platform === "darwin"
      ? join(home, "Library", "Caches")
      : platform === "win32"
        ? (env["LOCALAPPDATA"] ?? join(home, "AppData", "Local"))
        : (env["XDG_CACHE_HOME"] ?? join(home, ".cache"));
  return join(cache, "gliner-runner", "models");
}

export function defaultSettings(): Settings {
  return {
    host: "127.0.0.1",
    port: 8090,
    device: "cpu",
    queueCapacity: 256,
    maxBatchSize: 16,
    batchWindowMs: 4,
    modelsDir: defaultModelsDir(),
    modelIdleTtlSeconds: 0,
    memoryLimitBytes: undefined,
    shutdownTimeoutMs: 4500,
    maxBodyBytes: DEFAULT_MAX_BODY_BYTES,
    autoDownload: false,
    templatesDir: undefined,
    corsOrigins: [],
  };
}

export function resolveSettings(overrides: Partial<Settings> = {}): Settings {
  return { ...defaultSettings(), ...overrides };
}

type Parse<T> = (raw: string, label: string) => T;

const text: Parse<string> = (raw, label) => {
  if (raw.length === 0) {
    throw new SettingsError(`${label} must not be empty`);
  }
  return raw;
};

function integer(minimum: number, maximum = Number.MAX_SAFE_INTEGER): Parse<number> {
  return (raw, label) => {
    const value = /^[+-]?\d+$/.test(raw.trim()) ? Number(raw) : Number.NaN;
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
      const range =
        maximum === Number.MAX_SAFE_INTEGER
          ? `at least ${minimum}`
          : `between ${minimum} and ${maximum}`;
      throw new SettingsError(`${label} must be an integer ${range}`);
    }
    return value;
  };
}

function decimal(minimum: number, maximum = Number.POSITIVE_INFINITY): Parse<number> {
  return (raw, label) => {
    const value = raw.trim().length > 0 ? Number(raw) : Number.NaN;
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      throw new SettingsError(
        maximum === Number.POSITIVE_INFINITY
          ? `${label} must be a number of at least ${minimum}`
          : `${label} must be a number between ${minimum} and ${maximum}`,
      );
    }
    return value;
  };
}

interface OptionSpec<K extends keyof Settings> {
  flag: string;
  env: string;
  key: K;
  parse: Parse<NonNullable<Settings[K]>>;
}

function option<K extends keyof Settings>(
  key: K,
  flag: string,
  env: string,
  parse: Parse<NonNullable<Settings[K]>>,
): OptionSpec<K> {
  return { key, flag, env, parse };
}

const OPTIONS = [
  option("host", "host", "GLINER_RUNNER_HOST", text),
  option("port", "port", "GLINER_RUNNER_PORT", integer(1, 65535)),
  option("device", "device", "GLINER_RUNNER_DEVICE", text),
  option("modelsDir", "models-dir", "GLINER_RUNNER_MODELS_DIR", text),
  option("queueCapacity", "queue-capacity", "GLINER_RUNNER_QUEUE_CAPACITY", integer(1)),
  option("maxBatchSize", "max-batch-size", "GLINER_RUNNER_MAX_BATCH_SIZE", integer(1)),
  option("batchWindowMs", "batch-window-ms", "GLINER_RUNNER_BATCH_WINDOW_MS", decimal(0)),
  option(
    "modelIdleTtlSeconds",
    "model-idle-ttl-seconds",
    "GLINER_RUNNER_MODEL_IDLE_TTL_SECONDS",
    decimal(0, MAX_MODEL_IDLE_TTL_SECONDS),
  ),
  option(
    "memoryLimitBytes",
    "memory-limit-bytes",
    "GLINER_RUNNER_MEMORY_LIMIT_BYTES",
    integer(MIN_MEMORY_LIMIT_BYTES),
  ),
  option(
    "shutdownTimeoutMs",
    "shutdown-timeout-ms",
    "GLINER_RUNNER_SHUTDOWN_TIMEOUT_MS",
    integer(1, 2 ** 31 - 1),
  ),
  option("maxBodyBytes", "max-body-bytes", "GLINER_RUNNER_MAX_BODY_BYTES", integer(1)),
  option("templatesDir", "templates-dir", "GLINER_RUNNER_ONNX_TEMPLATES_DIR", text),
] as const;

export const AUTO_DOWNLOAD_ENV = "GLINER_RUNNER_AUTO_DOWNLOAD";
export const CORS_ORIGINS_ENV = "GLINER_RUNNER_CORS_ORIGINS";

/** Validates origins such as `http://localhost:5173`; `*`, paths and trailing slashes are rejected. */
export function parseCorsOrigins(raw: readonly string[], label: string): string[] {
  const origins: string[] = [];
  for (const item of raw.map((value) => value.trim()).filter((value) => value.length > 0)) {
    let origin: string | undefined;
    try {
      const url = new URL(item);
      origin = url.origin === item && url.origin !== "null" ? item : undefined;
    } catch {
      origin = undefined;
    }
    if (origin === undefined) {
      throw new SettingsError(`${label} must list origins like http://localhost:5173 (scheme, host, optional port), got '${item}'`);
    }
    if (!origins.includes(origin)) {
      origins.push(origin);
    }
  }
  return origins;
}

function flag(raw: string, label: string): boolean {
  const normalized = raw.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  throw new SettingsError(`${label} must be one of 1, 0, true, false, yes, no, on, off`);
}

export const SERVE_USAGE = `Usage: gliner-runner serve [options]

  --host <host>                      Bind host (default 127.0.0.1)
  --port <port>                      Bind port (default 8090)
  --device <device>                  ONNX Runtime device (default cpu)
  --models-dir <path>                Model artifact root; artifacts are read from
                                     <path>/onnx/<owner>--<name>/<revision>/ (default
                                     <user cache>/gliner-runner/models, e.g.
                                     ~/Library/Caches/gliner-runner/models on macOS,
                                     $XDG_CACHE_HOME or ~/.cache on Linux)
  --queue-capacity <n>               Pending request slots per model (default 256)
  --max-batch-size <n>               Largest micro-batch (default 16)
  --batch-window-ms <ms>             Batch collection window (default 4)
  --model-idle-ttl-seconds <s>       Idle eviction delay; 0 keeps the model warm (at most 2147483)
  --memory-limit-bytes <n>           Memory ceiling (default 75% of RAM, minimum 512 MiB)
  --shutdown-timeout-ms <ms>         Graceful shutdown budget (default 4500)
  --max-body-bytes <n>               Largest request body (default 16777216 = 16 MiB); larger
                                     requests get 413 request_too_large
  --auto-download                    Let an inference request for a missing curated model download and
                                     prepare it (needs a bundled template; default off, so inference
                                     never downloads implicitly). Env GLINER_RUNNER_AUTO_DOWNLOAD=1
  --templates-dir <path>             Artifact template directory (default: the bundled templates; env
                                     GLINER_RUNNER_ONNX_TEMPLATES_DIR)
  --cors-origin <origin>             Let a web page from this origin (for example http://localhost:5173)
                                     call the server from a browser. Repeatable; env
                                     GLINER_RUNNER_CORS_ORIGINS, comma-separated. Only an exactly matching
                                     Origin gets CORS headers and preflight (OPTIONS) answers. Either one
                                     replaces, for this run, the origins saved with 'gliner-runner cors'
                                     (default: common localhost development ports; see: cors --help)

Each option may also be set as GLINER_RUNNER_<NAME>; command-line flags win.

Models are installed with: gliner-runner models install <model-id> (see: models --help);
gliner-runner --version [--json] prints the version, runner identity and evaluation receipt SHAs.
HF_ENDPOINT and HF_TOKEN select the Hugging Face endpoint and an optional bearer token.
`;

/**
 * Parses `serve` arguments with precedence CLI > GLINER_RUNNER_* environment > default. CORS origins fall back to
 * `savedCorsOrigins` (the settings.json list, see `gliner-runner cors`) when neither flag nor environment sets them.
 */
export function parseServeArguments(
  argv: readonly string[],
  env: NodeJS.ProcessEnv,
  savedCorsOrigins: readonly string[] = [],
): Settings {
  let values: Record<string, string | boolean | (string | boolean)[] | undefined>;
  let positionals: string[];
  try {
    ({ values, positionals } = parseArgs({
      args: [...argv],
      allowPositionals: true,
      strict: true,
      options: {
        ...Object.fromEntries(OPTIONS.map(({ flag }) => [flag, { type: "string" as const }])),
        help: { type: "boolean" as const },
        "cors-origin": { type: "string" as const, multiple: true },
        "auto-download": { type: "boolean" as const },
      },
    }));
  } catch (error) {
    throw new SettingsError(error instanceof Error ? error.message : String(error));
  }
  if (positionals.length > 0) {
    throw new SettingsError(`unexpected argument: ${positionals[0]}`);
  }
  const settings = defaultSettings();
  const assign = <K extends keyof Settings>(spec: OptionSpec<K>, raw: string, label: string) => {
    settings[spec.key] = spec.parse(raw, label);
  };
  for (const spec of OPTIONS) {
    const flagValue = values[spec.flag];
    const envValue = env[spec.env];
    if (typeof flagValue === "string") {
      assign(spec, flagValue, `--${spec.flag}`);
    } else if (envValue !== undefined && envValue !== "") {
      assign(spec, envValue, spec.env);
    }
  }
  const originFlags = values["cors-origin"];
  const originEnv = env[CORS_ORIGINS_ENV];
  if (Array.isArray(originFlags)) {
    settings.corsOrigins = parseCorsOrigins(originFlags as string[], "--cors-origin");
  } else if (originEnv !== undefined && originEnv !== "") {
    settings.corsOrigins = parseCorsOrigins(originEnv.split(","), CORS_ORIGINS_ENV);
  } else {
    settings.corsOrigins = [...savedCorsOrigins];
  }
  const autoEnv = env[AUTO_DOWNLOAD_ENV];
  if (values["auto-download"] === true) {
    settings.autoDownload = true;
  } else if (autoEnv !== undefined && autoEnv !== "") {
    settings.autoDownload = flag(autoEnv, AUTO_DOWNLOAD_ENV);
  }
  return settings;
}

export function wantsHelp(argv: readonly string[]): boolean {
  return argv.includes("--help") || argv.includes("-h");
}
