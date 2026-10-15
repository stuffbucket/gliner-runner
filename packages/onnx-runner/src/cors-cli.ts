import { parseArgs } from "node:util";

import { SettingsError, parseCorsOrigins } from "./settings.js";
import {
  DEFAULT_CORS_ORIGINS,
  readUserSettings,
  userSettingsPath,
  writeCorsOrigins,
  type UserSettings,
} from "./user-settings.js";

export const CORS_USAGE = `Usage: gliner-runner cors <command> [options]

Browser origins (scheme://host[:port]) whose web pages may call 'gliner-runner serve'. The list is saved in
$XDG_CONFIG_HOME/gliner-runner/settings.json (default ~/.config/gliner-runner/settings.json; %APPDATA% on
Windows). Origins are numbered from 1, like 'ufw status numbered'; numbers shift after a delete, so run
'list' again before the next change.

Commands:
  list [--json]          Show the numbered origins and the settings file
  add <origin>...        Allow origins, for example https://stuffbucket.github.io
  show <n>               Print origin number n
  set <n> <origin>       Replace origin number n
  delete <n|origin>...   Remove origins by number (or by exact origin)
  reset                  Go back to the defaults: ${DEFAULT_CORS_ORIGINS.join(", ")}

Until the first change the defaults apply and no file is written. Changes take effect the next time
'gliner-runner serve' starts. 'serve --cors-origin' and GLINER_RUNNER_CORS_ORIGINS replace the saved list for
one run.
`;

export interface CorsIo {
  env: NodeJS.ProcessEnv;
  stdout(text: string): void;
  stderr(text: string): void;
}

class UsageError extends Error {}

const RESTART = "Restart 'gliner-runner serve' to apply.\n";

/** Runs `gliner-runner cors <argv>`; returns the process exit code (0 ok, 1 failure, 2 usage). */
export async function runCorsCommand(argv: readonly string[], io: CorsIo): Promise<number> {
  const [command, ...rest] = argv;
  if (command === undefined || command === "--help" || command === "-h" || command === "help") {
    io.stdout(CORS_USAGE);
    return command === undefined ? 2 : 0;
  }
  try {
    const { positionals, json } = parse(rest, command === "list");
    const settings = await readUserSettings(userSettingsPath(io.env));
    switch (command) {
      case "list":
        arity(positionals, 0, 0, "list");
        list(settings, json, io);
        return 0;
      case "add":
        arity(positionals, 1, Infinity, "add <origin>...");
        return await add(settings, positionals, io);
      case "show":
        arity(positionals, 1, 1, "show <n>");
        io.stdout(`${settings.corsOrigins[numberOf(settings, positionals[0]!) - 1]}\n`);
        return 0;
      case "set":
        arity(positionals, 2, 2, "set <n> <origin>");
        return await set(settings, positionals[0]!, positionals[1]!, io);
      case "delete":
        arity(positionals, 1, Infinity, "delete <n|origin>...");
        return await remove(settings, positionals, io);
      case "reset":
        arity(positionals, 0, 0, "reset");
        await writeCorsOrigins(settings, DEFAULT_CORS_ORIGINS);
        io.stdout(`Restored the default origins in ${settings.path}.\n${RESTART}`);
        return 0;
      default:
        throw new UsageError(`unknown cors command: ${command}`);
    }
  } catch (error) {
    if (error instanceof UsageError) {
      io.stderr(`error: ${error.message}\n\n${CORS_USAGE}`);
      return 2;
    }
    io.stderr(`error: ${error instanceof Error ? error.message : String(error)}\n`);
    return 1;
  }
}

function parse(argv: readonly string[], allowJson: boolean): { positionals: string[]; json: boolean } {
  try {
    const { positionals, values } = parseArgs({
      args: [...argv],
      allowPositionals: true,
      strict: true,
      options: allowJson ? { json: { type: "boolean" } } : {},
    });
    return { positionals, json: values.json === true };
  } catch (error) {
    throw new UsageError(error instanceof Error ? error.message : String(error));
  }
}

function arity(positionals: readonly string[], minimum: number, maximum: number, shape: string): void {
  if (positionals.length < minimum || positionals.length > maximum) {
    throw new UsageError(`expected: cors ${shape}`);
  }
}

/** One validated origin, with errors phrased for this command. */
function origin(raw: string): string {
  try {
    return parseCorsOrigins([raw], "origin")[0] ?? raw;
  } catch (error) {
    throw new UsageError(error instanceof SettingsError ? error.message : String(error));
  }
}

/** The 1-based number `raw` names in the list, or a usage error. */
function numberOf(settings: UserSettings, raw: string): number {
  const number = /^\d+$/.test(raw) ? Number(raw) : Number.NaN;
  const count = settings.corsOrigins.length;
  if (!Number.isSafeInteger(number) || number < 1 || number > count) {
    throw new UsageError(
      count === 0
        ? `no origin number ${raw}: the list is empty`
        : `no origin number ${raw}: use 1 to ${count} (see: gliner-runner cors list)`,
    );
  }
  return number;
}

const label = (number: number, count: number): string =>
  `[${String(number).padStart(Math.max(2, String(count).length))}]`;

function list(settings: UserSettings, json: boolean, io: CorsIo): void {
  const origins = settings.corsOrigins;
  if (json) {
    const entries = origins.map((value, index) => ({ id: index + 1, origin: value }));
    io.stdout(`${JSON.stringify({ path: settings.path, saved: settings.saved, origins: entries })}\n`);
    return;
  }
  io.stdout(
    settings.saved
      ? `Saved in ${settings.path}\n`
      : `Defaults (nothing saved yet; the first change creates ${settings.path})\n`,
  );
  if (origins.length === 0) {
    io.stdout("No origins: web pages cannot call the runner.\n");
    return;
  }
  io.stdout("\n");
  origins.forEach((value, index) => io.stdout(`${label(index + 1, origins.length)} ${value}\n`));
}

async function add(settings: UserSettings, raw: readonly string[], io: CorsIo): Promise<number> {
  const next = [...settings.corsOrigins];
  const added: string[] = [];
  for (const value of raw.map(origin)) {
    const existing = next.indexOf(value);
    if (existing >= 0) {
      io.stdout(`Already allowed: ${label(existing + 1, next.length)} ${value}\n`);
      continue;
    }
    next.push(value);
    added.push(value);
  }
  if (added.length === 0) {
    return 0;
  }
  await writeCorsOrigins(settings, next);
  for (const value of added) {
    io.stdout(`Added ${label(next.indexOf(value) + 1, next.length)} ${value}\n`);
  }
  io.stdout(RESTART);
  return 0;
}

async function set(settings: UserSettings, rawNumber: string, rawOrigin: string, io: CorsIo): Promise<number> {
  const number = numberOf(settings, rawNumber);
  const value = origin(rawOrigin);
  const next = [...settings.corsOrigins];
  const existing = next.indexOf(value);
  if (existing === number - 1) {
    io.stdout(`Unchanged: ${label(number, next.length)} ${value}\n`);
    return 0;
  }
  if (existing >= 0) {
    throw new UsageError(`${value} is already allowed as ${label(existing + 1, next.length)}`);
  }
  const previous = next[number - 1];
  next[number - 1] = value;
  await writeCorsOrigins(settings, next);
  io.stdout(`Replaced ${label(number, next.length)} ${previous} with ${value}\n${RESTART}`);
  return 0;
}

async function remove(settings: UserSettings, raw: readonly string[], io: CorsIo): Promise<number> {
  const current = settings.corsOrigins;
  const doomed = new Set<number>();
  for (const item of raw) {
    if (/^\d+$/.test(item)) {
      doomed.add(numberOf(settings, item) - 1);
      continue;
    }
    const index = current.indexOf(origin(item));
    if (index < 0) {
      throw new UsageError(`${item} is not in the list (see: gliner-runner cors list)`);
    }
    doomed.add(index);
  }
  await writeCorsOrigins(
    settings,
    current.filter((_, index) => !doomed.has(index)),
  );
  for (const index of [...doomed].sort((a, b) => a - b)) {
    io.stdout(`Deleted ${label(index + 1, current.length)} ${current[index]}\n`);
  }
  io.stdout(RESTART);
  return 0;
}
