import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import { SettingsError, parseCorsOrigins } from "./settings.js";

/** Key in settings.json that holds the saved browser origins. */
export const CORS_ORIGINS_KEY = "cors_origins";

/**
 * Origins of common local development servers. They apply while settings.json has no `cors_origins` key and are
 * written to the file on the first `gliner-runner cors` change; an explicitly empty list stays empty.
 */
export const DEFAULT_CORS_ORIGINS: readonly string[] = [
  "http://localhost:3000", // Next.js, Create React App, Remix
  "http://localhost:4173", // Vite preview
  "http://localhost:4200", // Angular
  "http://localhost:4321", // Astro
  "http://localhost:5173", // Vite dev server
  "http://localhost:8080", // webpack-dev-server, Vue CLI
];

/** `$XDG_CONFIG_HOME/gliner-runner/settings.json` (default `~/.config`), or `%APPDATA%\gliner-runner` on Windows. */
export function userSettingsPath(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): string {
  const config =
    platform === "win32"
      ? (env["APPDATA"] ?? join(homedir(), "AppData", "Roaming"))
      : env["XDG_CONFIG_HOME"] || join(homedir(), ".config");
  return join(config, "gliner-runner", "settings.json");
}

export interface UserSettings {
  path: string;
  /** Whether settings.json holds a `cors_origins` key; when false `corsOrigins` are the defaults. */
  saved: boolean;
  corsOrigins: string[];
  /** The whole parsed file, so a write keeps keys this version does not know. */
  document: Record<string, unknown>;
}

/** Reads settings.json; a missing file means the defaults. Malformed content throws `SettingsError`. */
export async function readUserSettings(path: string): Promise<UserSettings> {
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return { path, saved: false, corsOrigins: [...DEFAULT_CORS_ORIGINS], document: {} };
    }
    throw new SettingsError(`cannot read ${path}: ${(error as Error).message}`);
  }
  let document: unknown;
  try {
    document = JSON.parse(raw);
  } catch (error) {
    throw new SettingsError(`${path} is not valid JSON: ${(error as Error).message}`);
  }
  if (typeof document !== "object" || document === null || Array.isArray(document)) {
    throw new SettingsError(`${path} must hold a JSON object`);
  }
  const record = document as Record<string, unknown>;
  const stored = record[CORS_ORIGINS_KEY];
  if (stored === undefined) {
    return { path, saved: false, corsOrigins: [...DEFAULT_CORS_ORIGINS], document: record };
  }
  if (!Array.isArray(stored) || stored.some((item) => typeof item !== "string")) {
    throw new SettingsError(`${path}: ${CORS_ORIGINS_KEY} must be a list of origin strings`);
  }
  return {
    path,
    saved: true,
    corsOrigins: parseCorsOrigins(stored as string[], `${path}: ${CORS_ORIGINS_KEY}`),
    document: record,
  };
}

/** Saves the origins into settings.json (creating it and its directory), replacing the file atomically. */
export async function writeCorsOrigins(settings: UserSettings, origins: readonly string[]): Promise<UserSettings> {
  const document = { ...settings.document, [CORS_ORIGINS_KEY]: [...origins] };
  await mkdir(dirname(settings.path), { recursive: true });
  const temporary = `${settings.path}.${process.pid}.tmp`;
  await writeFile(temporary, `${JSON.stringify(document, null, 2)}\n`, "utf8");
  await rename(temporary, settings.path);
  return { path: settings.path, saved: true, corsOrigins: [...origins], document };
}
