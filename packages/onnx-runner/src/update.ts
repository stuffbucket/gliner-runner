import { z } from "zod";

export const UPDATE_URL = "https://github.com/stuffbucket/gliner-runner/releases/latest/download/latest.json";
export const INSTALL_COMMAND =
  "curl -fsSL https://github.com/stuffbucket/gliner-runner/releases/latest/download/install.sh | sh";
/** Update command for builds that Homebrew installed (the app lives in a Cellar), so users are not told to run curl. */
export const BREW_UPDATE_COMMAND = "brew update && brew upgrade gliner-runner";

export function installCommandFor(appPath: string | undefined): string {
  return appPath !== undefined && /(^|[\\/])Cellar[\\/]/.test(appPath) ? BREW_UPDATE_COMMAND : INSTALL_COMMAND;
}

const DEFAULT_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 1024 * 1024;

export const UPDATE_USAGE = `Usage: gliner-runner update check [--json]

Commands:
  check                Compare this build with the latest stable release. Prints whether an update is
                       available and the command that installs it. Never installs anything.
                       Exit 0 whether or not an update exists; exit 1 if the check could not be made.

Options:
  --json               Print one JSON object instead of text
  --help, -h           Show this help

Environment: GLINER_RUNNER_UPDATE_URL (default ${UPDATE_URL}),
GLINER_RUNNER_UPDATE_TIMEOUT_MS (default ${DEFAULT_TIMEOUT_MS}).
`;

interface Semver {
  core: [number, number, number];
  pre: string[];
}

const SEMVER = /^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/;

export function parseSemver(text: string): Semver | undefined {
  const match = SEMVER.exec(text.trim());
  if (match === null) {
    return undefined;
  }
  return {
    core: [Number(match[1]), Number(match[2]), Number(match[3])],
    pre: match[4] === undefined ? [] : match[4].split("."),
  };
}

/** Semver precedence: negative when a < b. A prerelease sorts before its release. Undefined if either is not semver. */
export function compareSemver(a: string, b: string): number | undefined {
  const left = parseSemver(a);
  const right = parseSemver(b);
  if (left === undefined || right === undefined) {
    return undefined;
  }
  for (let i = 0; i < 3; i += 1) {
    const delta = left.core[i]! - right.core[i]!;
    if (delta !== 0) {
      return Math.sign(delta);
    }
  }
  if (left.pre.length === 0 || right.pre.length === 0) {
    return Math.sign(right.pre.length - left.pre.length);
  }
  for (let i = 0; i < Math.max(left.pre.length, right.pre.length); i += 1) {
    const x = left.pre[i];
    const y = right.pre[i];
    if (x === undefined) {
      return -1;
    }
    if (y === undefined) {
      return 1;
    }
    const xNumeric = /^\d+$/.test(x);
    const yNumeric = /^\d+$/.test(y);
    if (xNumeric && yNumeric) {
      const delta = Number(x) - Number(y);
      if (delta !== 0) {
        return Math.sign(delta);
      }
    } else if (xNumeric !== yNumeric) {
      return xNumeric ? -1 : 1;
    } else if (x !== y) {
      return x < y ? -1 : 1;
    }
  }
  return 0;
}

const LatestSchema = z.object({
  version: z.string(),
  channel: z.string().optional(),
  notes_url: z.string().optional(),
});

export interface UpdateIo {
  env: NodeJS.ProcessEnv;
  /** Where the running app is installed; used to tell Homebrew installs from the curl installer. */
  appPath?: string | undefined;
  stdout(text: string): void;
  stderr(text: string): void;
}

export interface UpdateResult {
  current: string;
  latest: string;
  channel: "stable";
  update_available: boolean;
  notes_url: string | null;
  install_command: string;
}

class CheckError extends Error {}

async function fetchLatest(url: string, timeoutMs: number): Promise<z.infer<typeof LatestSchema>> {
  let response: Response;
  try {
    response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs), headers: { accept: "application/json" } });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    const cause = (error as { cause?: { message?: string } }).cause?.message ?? (error as Error).message;
    throw new CheckError(timedOut ? `timed out after ${timeoutMs} ms fetching ${url}` : `cannot fetch ${url}: ${cause}`);
  }
  if (!response.ok) {
    throw new CheckError(`${url} returned HTTP ${response.status}`);
  }
  let parsed: unknown;
  try {
    const text = await response.text();
    if (text.length > MAX_BODY_BYTES) {
      throw new Error("response too large");
    }
    parsed = JSON.parse(text);
  } catch (error) {
    throw new CheckError(`${url} did not return valid JSON: ${(error as Error).message}`);
  }
  const result = LatestSchema.safeParse(parsed);
  if (!result.success) {
    throw new CheckError(`${url} is not a release descriptor (missing "version")`);
  }
  return result.data;
}

/** Runs `gliner-runner-onnx update <argv>`; returns the exit code (0 ok, 1 check failed, 2 usage). */
export async function runUpdateCommand(
  argv: readonly string[],
  io: UpdateIo,
  currentVersion: string,
): Promise<number> {
  const [command, ...rest] = argv;
  if (command === undefined || command === "--help" || command === "-h" || command === "help") {
    io.stdout(UPDATE_USAGE);
    return command === undefined ? 2 : 0;
  }
  if (command !== "check") {
    io.stderr(`error: unknown update command: ${command}\n\n${UPDATE_USAGE}`);
    return 2;
  }
  let json = false;
  for (const argument of rest) {
    if (argument === "--json") {
      json = true;
    } else if (argument === "--help" || argument === "-h") {
      io.stdout(UPDATE_USAGE);
      return 0;
    } else {
      io.stderr(`error: unknown option: ${argument}\n\n${UPDATE_USAGE}`);
      return 2;
    }
  }

  const url = io.env["GLINER_RUNNER_UPDATE_URL"] ?? UPDATE_URL;
  const timeout = Number(io.env["GLINER_RUNNER_UPDATE_TIMEOUT_MS"] ?? DEFAULT_TIMEOUT_MS);
  try {
    if (!/^https?:\/\//i.test(url)) {
      throw new CheckError(`GLINER_RUNNER_UPDATE_URL must be an http(s) URL (got ${url})`);
    }
    const latest = await fetchLatest(url, Number.isFinite(timeout) && timeout > 0 ? timeout : DEFAULT_TIMEOUT_MS);
    const parsedLatest = parseSemver(latest.version);
    if (parsedLatest === undefined) {
      throw new CheckError(`${url} has an invalid version: ${latest.version}`);
    }
    // Only stable releases are ever offered: a prerelease or non-stable descriptor is never an update.
    const stable = parsedLatest.pre.length === 0 && (latest.channel === undefined || latest.channel === "stable");
    const order = compareSemver(currentVersion, latest.version);
    const result: UpdateResult = {
      current: currentVersion,
      latest: latest.version.replace(/^v/, ""),
      channel: "stable",
      update_available: stable && order !== undefined && order < 0,
      notes_url: latest.notes_url ?? null,
      install_command: installCommandFor(io.appPath),
    };
    if (json) {
      io.stdout(`${JSON.stringify(result)}\n`);
    } else if (result.update_available) {
      io.stdout(
        [
          `Update available: ${result.current} -> ${result.latest}`,
          ...(result.notes_url === null ? [] : [`Release notes: ${result.notes_url}`]),
          "To update, run:",
          `  ${result.install_command}`,
          "",
        ].join("\n"),
      );
    } else {
      io.stdout(`gliner-runner-onnx ${result.current} is up to date (latest stable: ${result.latest}).\n`);
    }
    return 0;
  } catch (error) {
    if (error instanceof CheckError) {
      io.stderr(`error: update check failed: ${error.message}\n`);
      return 1;
    }
    throw error;
  }
}
