import { spawn } from "node:child_process";
import { createServer as createNetServer } from "node:net";
import { parseArgs } from "node:util";

import { loadComponents } from "./components.js";
import { createServer, type RunnerServer } from "./server.js";
import { SettingsError, parseServeArguments } from "./settings.js";
import { readUserSettings, userSettingsPath, writeCorsOrigins } from "./user-settings.js";

export const PLAYGROUND_URL = "https://stuffbucket.github.io/gliner-runner/playground/";
const DEFAULT_MODEL = "decide-340m";
/** How many ports after `--port` are tried when it is taken by something that cannot be reused. */
const PORT_SEARCH = 20;

export const PLAYGROUND_USAGE = `Usage: gliner-runner playground [options]

Opens the browser playground connected to the runner on this computer, setting up whatever is missing:
  1. allows the playground's address in the saved CORS list (see: gliner-runner cors list);
  2. reuses a runner already serving on --port that allows it, or starts one (on the next free port when
     --port is taken);
  3. downloads and prepares the model if it is not installed yet (asks first; --yes approves);
  4. opens the playground in the default browser, already pointed at this runner.
A runner started here keeps serving until Ctrl+C. The playground address is also printed on stdout.

Options:
  --port <port>     Runner port to reuse or start on (default 8090)
  --model <id>      Model the playground needs (default ${DEFAULT_MODEL})
  --yes, -y         Approve the model download without prompting
  --no-open         Print the playground address instead of opening a browser
  --site <url>      Playground page (default ${PLAYGROUND_URL})
  --help, -h        Show this help

The runner started here reads the same settings and GLINER_RUNNER_* variables as 'gliner-runner serve'.
`;

export interface PlaygroundIo {
  env: NodeJS.ProcessEnv;
  /** Whether stdin is an interactive terminal (a prompt can be answered). */
  interactive: boolean;
  stdout(text: string): void;
  stderr(text: string): void;
  confirm(question: string): Promise<boolean>;
  /** Opens a URL in the default browser; resolves false when no browser could be started. */
  openUrl(url: string): Promise<boolean>;
  /** Resolves when the user asks a runner started here to stop (Ctrl+C, SIGTERM). */
  waitForStop(): Promise<void>;
}

class UsageError extends Error {}

interface Probe {
  /** A gliner-runner answered on the port. */
  runner: boolean;
  /** It sends CORS headers for the playground's origin. */
  allowsOrigin: boolean;
}

interface ModelEntry {
  model_id: string;
  available_locally: boolean;
  download_destination: string;
}

/** Runs `gliner-runner playground <argv>`; returns the exit code (0 ok, 1 failure or declined, 2 usage). */
export async function runPlaygroundCommand(argv: readonly string[], io: PlaygroundIo): Promise<number> {
  let options;
  try {
    options = parse(argv);
  } catch (error) {
    io.stderr(`error: ${(error as Error).message}\n\n${PLAYGROUND_USAGE}`);
    return 2;
  }
  if (options.help) {
    io.stdout(PLAYGROUND_USAGE);
    return 0;
  }
  const say = (text: string): void => io.stderr(`${text}\n`);
  let started: RunnerServer | undefined;
  try {
    const origin = new URL(options.site).origin;
    await allowOrigin(origin, io.env, say);

    let port = options.port;
    let base = `http://127.0.0.1:${port}`;
    const probe = await probeRunner(base, origin);
    if (probe.runner && probe.allowsOrigin) {
      say(`Using the runner already serving on ${base}.`);
    } else {
      if (probe.runner) {
        say(`The runner on ${base} was started before ${origin} was allowed; starting another one.`);
      } else if (!(await portIsFree(port))) {
        say(`Port ${port} is in use by another program; starting the runner on another port.`);
      }
      port = await firstFreePort(port);
      ({ server: started, base } = await startRunner(port, origin, io));
      say(`Started a runner on ${base}.`);
    }

    if (!(await ensureModel(base, options.model, options.yes, io, say))) {
      await started?.close();
      return 1;
    }

    const url = `${options.site}?${new URLSearchParams({ mode: "local", runner: base }).toString()}`;
    io.stdout(`${url}\n`);
    if (options.open && !(await io.openUrl(url))) {
      say("Could not open a browser; open the address above.");
    } else if (options.open) {
      say("Opened the playground in your browser.");
    }
    if (started === undefined) {
      return 0;
    }
    say("The runner keeps serving for the playground; press Ctrl+C to stop it.");
    await io.waitForStop();
    await started.close();
    return 0;
  } catch (error) {
    await started?.close().catch(() => undefined);
    if (error instanceof UsageError || error instanceof SettingsError) {
      io.stderr(`error: ${error.message}\n\n${PLAYGROUND_USAGE}`);
      return 2;
    }
    io.stderr(`error: ${error instanceof Error ? error.message : String(error)}\n`);
    return 1;
  }
}

function parse(argv: readonly string[]): { port: number; model: string; yes: boolean; open: boolean; site: string; help: boolean } {
  let values;
  try {
    ({ values } = parseArgs({
      args: [...argv],
      allowPositionals: false,
      strict: true,
      options: {
        port: { type: "string" },
        model: { type: "string" },
        yes: { type: "boolean", short: "y" },
        "no-open": { type: "boolean" },
        site: { type: "string" },
        help: { type: "boolean", short: "h" },
      },
    }));
  } catch (error) {
    throw new UsageError(error instanceof Error ? error.message : String(error));
  }
  const port = values.port === undefined ? 8090 : Number(values.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new UsageError("--port must be an integer between 1 and 65535");
  }
  const site = values.site ?? PLAYGROUND_URL;
  try {
    const parsed = new URL(site);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
      throw new Error();
    }
  } catch {
    throw new UsageError(`--site must be an http(s) URL, got '${site}'`);
  }
  return {
    port,
    model: values.model ?? DEFAULT_MODEL,
    yes: values.yes === true,
    open: values["no-open"] !== true,
    site,
    help: values.help === true,
  };
}

/** Adds the playground's origin to the saved CORS list (writing the defaults plus it on first use). */
async function allowOrigin(origin: string, env: NodeJS.ProcessEnv, say: (text: string) => void): Promise<void> {
  const settings = await readUserSettings(userSettingsPath(env));
  if (settings.corsOrigins.includes(origin)) {
    return;
  }
  await writeCorsOrigins(settings, [...settings.corsOrigins, origin]);
  say(`Allowed ${origin} to call the runner (saved in ${settings.path}; see: gliner-runner cors list).`);
}

async function probeRunner(base: string, origin: string): Promise<Probe> {
  try {
    const response = await fetch(`${base}/healthz`, { headers: { origin }, signal: AbortSignal.timeout(3000) });
    const body = (await response.json().catch(() => undefined)) as { accepting_requests?: unknown } | undefined;
    const runner = response.ok && typeof body?.accepting_requests === "boolean";
    return { runner, allowsOrigin: runner && response.headers.get("access-control-allow-origin") === origin };
  } catch {
    return { runner: false, allowsOrigin: false };
  }
}

function portIsFree(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const probe = createNetServer();
    probe.once("error", () => resolve(false));
    probe.listen({ host: "127.0.0.1", port }, () => probe.close(() => resolve(true)));
  });
}

async function firstFreePort(from: number): Promise<number> {
  for (let port = from; port < from + PORT_SEARCH && port <= 65535; port += 1) {
    if (await portIsFree(port)) {
      return port;
    }
  }
  throw new Error(`no free port between ${from} and ${from + PORT_SEARCH - 1}; pass --port`);
}

async function startRunner(port: number, origin: string, io: PlaygroundIo): Promise<{ server: RunnerServer; base: string }> {
  const saved = await readUserSettings(userSettingsPath(io.env));
  const settings = parseServeArguments(["--port", String(port), "--host", "127.0.0.1"], io.env, saved.corsOrigins);
  if (!settings.corsOrigins.includes(origin)) {
    // GLINER_RUNNER_CORS_ORIGINS replaces the saved list; the playground still has to be let in.
    settings.corsOrigins = [...settings.corsOrigins, origin];
  }
  const log = (message: string): void => io.stderr(`[gliner-runner-onnx] ${message}\n`);
  const server = createServer({ ...(await loadComponents(settings, { log })), settings, log });
  const address = await server.listen();
  return { server, base: `http://127.0.0.1:${address.port}` };
}

/** Makes sure `model` is installed in the runner at `base`, using its download API (and its models directory). */
async function ensureModel(
  base: string,
  model: string,
  approved: boolean,
  io: PlaygroundIo,
  say: (text: string) => void,
): Promise<boolean> {
  const listing = (await (await fetch(`${base}/v1/models`)).json()) as { models?: ModelEntry[] };
  const entry = listing.models?.find((candidate) => candidate.model_id === model);
  if (entry === undefined) {
    throw new UsageError(`unknown model '${model}' (see: gliner-runner models list)`);
  }
  if (entry.available_locally) {
    return true;
  }
  const question = `${model} is not installed. Download and prepare it now (a 1.95 GB download for decide-340m)? [y/N] `;
  if (!approved) {
    if (!io.interactive) {
      say(`${model} is not installed; pass --yes to download it (stdin is not a terminal, so there is no prompt).`);
      return false;
    }
    if (!(await io.confirm(question))) {
      say("Not downloading; the playground needs the model.");
      return false;
    }
  }
  const created = await fetch(`${base}/v1/models/${encodeURIComponent(model)}/downloads`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ approved: true, destination: entry.download_destination }),
  });
  const job = (await created.json()) as { job_id?: string; detail?: unknown };
  if (!created.ok || job.job_id === undefined) {
    throw new Error(`the runner refused to download ${model}: ${JSON.stringify(job.detail ?? job)}`);
  }
  say(`Downloading and preparing ${model}; this takes a few minutes.`);
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const state = (await (await fetch(`${base}/v1/model-downloads/${job.job_id}`)).json()) as { state: string; error: string | null };
    if (state.state === "succeeded") {
      say(`${model} is installed.`);
      return true;
    }
    if (state.state === "failed") {
      throw new Error(`installing ${model} failed: ${state.error ?? "unknown error"}`);
    }
  }
}

/** Opens `url` with the platform's default handler (`open`, `xdg-open`, or `start` on Windows). */
export function openInBrowser(url: string): Promise<boolean> {
  const [command, args] =
    process.platform === "darwin"
      ? ["open", [url]]
      : process.platform === "win32"
        ? ["cmd", ["/c", "start", "", url]]
        : ["xdg-open", [url]];
  return new Promise((resolve) => {
    try {
      const child = spawn(command, args as string[], { stdio: "ignore", detached: true });
      child.once("error", () => resolve(false));
      child.once("spawn", () => {
        child.unref();
        resolve(true);
      });
    } catch {
      resolve(false);
    }
  });
}
