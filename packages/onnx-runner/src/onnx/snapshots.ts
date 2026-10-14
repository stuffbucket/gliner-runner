import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { open, readdir, realpath, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { delimiter, join, resolve } from "node:path";

import type { KnownModel } from "../catalog.js";
import { RequestCancelledError } from "../errors.js";
import type { TemplateSourceFile } from "./templates.js";

/** Where a candidate snapshot was found, in discovery precedence order. */
export type SnapshotOrigin =
  | "explicit"
  | "GLINER_RUNNER_SNAPSHOT_DIRS"
  | "HF_HUB_CACHE"
  | "HF_HOME"
  | "xdg-huggingface-cache"
  | "gliner-runner-provider-store"
  | "models-dir-provider";

export interface SnapshotFileCheck {
  path: string;
  /** Expected size and SHA-256 (from the template). */
  size: number;
  sha256: string;
  verified: boolean;
  reason?: string;
}

export interface SnapshotCandidate {
  /** Directory that holds (or should hold) the snapshot files. */
  location: string;
  origin: SnapshotOrigin;
  layout: "direct" | "huggingface-cache" | "provider-store";
  /** Revision named by the directory layout; `undefined` for a directory passed in directly. */
  revision: string | undefined;
  /** True when every source file exists with the pinned size and SHA-256. */
  verified: boolean;
  /** Why the candidate was rejected (or not evaluated). */
  reason?: string;
  /** Per-file outcomes; empty when the candidate was rejected without looking at its files. */
  files: SnapshotFileCheck[];
}

export interface DiscoveryOptions {
  /** `--snapshot`: when given, only this directory is considered. */
  explicit?: string | undefined;
  env?: NodeJS.ProcessEnv | undefined;
  platform?: NodeJS.Platform | undefined;
  home?: string | undefined;
  /** The configured models directory, whose provider-store layouts are searched last. */
  modelsDir?: string | undefined;
  /** Check every candidate instead of stopping at the first verified one. */
  exhaustive?: boolean | undefined;
  signal?: AbortSignal | undefined;
}

export interface DiscoveryResult {
  /** Every candidate found, in precedence order. */
  candidates: SnapshotCandidate[];
  /** The first fully verified candidate. */
  selected: SnapshotCandidate | undefined;
}

const HASH_BLOCK_BYTES = 4 * 1024 * 1024;
const MAX_REVISION_DIRECTORIES = 64;

/** Results of hashing, valid while a file keeps its real path, size and modification time. */
const digests = new Map<string, Promise<string>>();

/** SHA-256 of a file, following symlinks (HF caches link snapshot files into `blobs/`); read only. */
async function digestOf(path: string, size: number, mtimeMs: number, signal?: AbortSignal): Promise<string> {
  const key = `${await realpath(path)}\0${size}\0${mtimeMs}`;
  let pending = digests.get(key);
  if (pending === undefined) {
    pending = (async () => {
      const handle = await open(path, constants.O_RDONLY);
      try {
        const hash = createHash("sha256");
        const block = Buffer.allocUnsafe(HASH_BLOCK_BYTES);
        for (let position = 0; ; ) {
          if (signal?.aborted) {
            throw new RequestCancelledError("snapshot verification was cancelled");
          }
          const { bytesRead } = await handle.read(block, 0, block.length, position);
          if (bytesRead === 0) {
            return hash.digest("hex");
          }
          hash.update(block.subarray(0, bytesRead));
          position += bytesRead;
        }
      } finally {
        await handle.close();
      }
    })();
    digests.set(key, pending);
    pending.catch(() => digests.delete(key));
  }
  return pending;
}

/** Forgets every cached digest (tests). */
export function clearSnapshotDigestCache(): void {
  digests.clear();
}

/**
 * Checks a directory against the template's source files: all must exist with the pinned size (checked for every
 * file before any hashing, so a wrong snapshot is rejected cheaply) and SHA-256. Never writes.
 */
export async function verifySnapshotFiles(
  directory: string,
  sourceFiles: readonly TemplateSourceFile[],
  signal?: AbortSignal,
): Promise<{ verified: boolean; reason: string | undefined; files: SnapshotFileCheck[] }> {
  const files: SnapshotFileCheck[] = sourceFiles.map((file) => ({
    path: file.path,
    size: file.size,
    sha256: file.sha256,
    verified: false,
  }));
  const stats = new Map<string, { size: number; mtimeMs: number }>();
  let reason: string | undefined;
  for (const [index, file] of sourceFiles.entries()) {
    const check = files[index] as SnapshotFileCheck;
    try {
      const status = await stat(join(directory, file.path));
      if (!status.isFile()) {
        check.reason = `${file.path} is not a regular file`;
      } else if (status.size !== file.size) {
        check.reason = `${file.path} has ${status.size} bytes, expected ${file.size}`;
      } else {
        stats.set(file.path, { size: status.size, mtimeMs: status.mtimeMs });
      }
    } catch (error) {
      const code = (error as { code?: unknown }).code;
      check.reason = code === "ENOENT" || code === "ENOTDIR" ? `missing ${file.path}` : `cannot read ${file.path}: ${String(error)}`;
    }
    reason ??= check.reason;
  }
  if (reason !== undefined) {
    return { verified: false, reason, files };
  }
  for (const [index, file] of sourceFiles.entries()) {
    const check = files[index] as SnapshotFileCheck;
    const status = stats.get(file.path) as { size: number; mtimeMs: number };
    try {
      const actual = await digestOf(join(directory, file.path), status.size, status.mtimeMs, signal);
      if (actual === file.sha256) {
        check.verified = true;
      } else {
        check.reason = `sha256 mismatch for ${file.path}`;
      }
    } catch (error) {
      if (error instanceof RequestCancelledError) {
        throw error;
      }
      check.reason = `cannot read ${file.path}: ${String(error)}`;
    }
    reason ??= check.reason;
  }
  return { verified: reason === undefined, reason, files };
}

/** The user cache directory the Python runner's `platformdirs.user_cache_path("gliner-runner")` resolves to. */
export function pythonUserCacheDir(env: NodeJS.ProcessEnv, platform: NodeJS.Platform, home: string): string {
  if (platform === "darwin") {
    return join(home, "Library", "Caches", "gliner-runner");
  }
  if (platform === "win32") {
    return join(env["LOCALAPPDATA"] ?? join(home, "AppData", "Local"), "gliner-runner", "Cache");
  }
  return join(env["XDG_CACHE_HOME"] ?? join(home, ".cache"), "gliner-runner");
}

interface Root {
  origin: SnapshotOrigin;
  /** Directory to look in. */
  path: string;
  layouts: readonly SnapshotCandidate["layout"][];
}

/** Standard locations in precedence order (see `discoverSnapshots`). */
export function snapshotRoots(options: DiscoveryOptions): Root[] {
  const env = options.env ?? process.env;
  const platform = options.platform ?? process.platform;
  const home = options.home ?? env["HOME"] ?? env["USERPROFILE"] ?? homedir();
  const roots: Root[] = [];
  const add = (origin: SnapshotOrigin, path: string | undefined, layouts: Root["layouts"]): void => {
    if (path !== undefined && path !== "") {
      roots.push({ origin, path: resolve(path), layouts });
    }
  };
  for (const entry of (env["GLINER_RUNNER_SNAPSHOT_DIRS"] ?? "").split(delimiter)) {
    add("GLINER_RUNNER_SNAPSHOT_DIRS", entry, ["direct", "provider-store", "huggingface-cache"]);
  }
  add("HF_HUB_CACHE", env["HF_HUB_CACHE"] ?? env["HUGGINGFACE_HUB_CACHE"], ["huggingface-cache"]);
  add("HF_HOME", env["HF_HOME"] === undefined || env["HF_HOME"] === "" ? undefined : join(env["HF_HOME"], "hub"), ["huggingface-cache"]);
  add("xdg-huggingface-cache", join(env["XDG_CACHE_HOME"] ?? join(home, ".cache"), "huggingface", "hub"), ["huggingface-cache"]);
  add("gliner-runner-provider-store", join(pythonUserCacheDir(env, platform, home), "models", "providers", "huggingface"), ["provider-store"]);
  if (options.modelsDir !== undefined) {
    add("models-dir-provider", join(options.modelsDir, "providers", "huggingface"), ["provider-store"]);
    add("models-dir-provider", options.modelsDir, ["provider-store"]);
  }
  return roots;
}

async function listDirectories(path: string): Promise<string[]> {
  try {
    return (await readdir(path, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory() || entry.isSymbolicLink())
      .map((entry) => entry.name)
      .sort();
  } catch {
    return [];
  }
}

/**
 * Looks for the pinned snapshot of `model` in the standard places, in this order: the explicit `--snapshot`
 * directory; `GLINER_RUNNER_SNAPSHOT_DIRS`; `HF_HUB_CACHE`; `$HF_HOME/hub`; `${XDG_CACHE_HOME:-~/.cache}/huggingface/hub`;
 * the Python runner's provider store; the models directory's own provider layouts. A candidate qualifies only
 * when every source file exists with the pinned size and SHA-256. Symlinks (HF caches link into `blobs/`) are
 * followed for reading; nothing is ever written. Other revisions of the same repository found along the way are
 * reported as rejected.
 */
export async function discoverSnapshots(
  model: KnownModel,
  revision: string,
  sourceFiles: readonly TemplateSourceFile[],
  options: DiscoveryOptions = {},
): Promise<DiscoveryResult> {
  const slug = model.repository.replace("/", "--");
  const found: Omit<SnapshotCandidate, "verified" | "files">[] = [];
  const seen = new Set<string>();
  const consider = async (
    origin: SnapshotOrigin,
    layout: SnapshotCandidate["layout"],
    location: string,
    candidateRevision: string | undefined,
  ): Promise<void> => {
    let key: string;
    try {
      key = await realpath(location);
      if (!(await stat(location)).isDirectory()) {
        return;
      }
    } catch {
      return;
    }
    if (!seen.has(key)) {
      seen.add(key);
      found.push({ location, origin, layout, revision: candidateRevision });
    }
  };

  if (options.explicit !== undefined) {
    await consider("explicit", "direct", resolve(options.explicit), undefined);
  } else {
    for (const root of snapshotRoots(options)) {
      for (const layout of root.layouts) {
        if (layout === "direct") {
          // A search root is only a candidate itself when it directly holds source files.
          const holdsSources = await Promise.any(
            sourceFiles.map((file) => stat(join(root.path, file.path))),
          ).then(
            () => true,
            () => false,
          );
          if (holdsSources) {
            await consider(root.origin, layout, root.path, undefined);
          }
        } else {
          const parent = layout === "huggingface-cache" ? join(root.path, `models--${slug}`, "snapshots") : join(root.path, slug);
          const revisions = (await listDirectories(parent)).slice(0, MAX_REVISION_DIRECTORIES);
          // The pinned revision first, then whatever else is there (to be reported as the wrong revision).
          for (const name of [...revisions.filter((entry) => entry === revision), ...revisions.filter((entry) => entry !== revision)]) {
            await consider(root.origin, layout, join(parent, name), name);
          }
        }
      }
    }
  }

  const candidates: SnapshotCandidate[] = [];
  let selected: SnapshotCandidate | undefined;
  for (const candidate of found) {
    if (candidate.revision !== undefined && candidate.revision !== revision) {
      candidates.push({
        ...candidate,
        verified: false,
        reason: `revision ${candidate.revision} is not the pinned revision ${revision}`,
        files: [],
      });
      continue;
    }
    if (selected !== undefined && options.exhaustive !== true) {
      candidates.push({ ...candidate, verified: false, reason: "not checked: an earlier candidate was verified", files: [] });
      continue;
    }
    const outcome = await verifySnapshotFiles(candidate.location, sourceFiles, options.signal);
    const checked: SnapshotCandidate = {
      ...candidate,
      verified: outcome.verified,
      files: outcome.files,
      ...(outcome.reason === undefined ? {} : { reason: outcome.reason }),
    };
    candidates.push(checked);
    if (outcome.verified && selected === undefined) {
      selected = checked;
    }
  }
  return { candidates, selected };
}
