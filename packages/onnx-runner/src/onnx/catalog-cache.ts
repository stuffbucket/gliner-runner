import { readFileSync } from "node:fs";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { CatalogRefreshResult, KnownModel } from "../catalog.js";

export const CATALOG_CACHE_NAME = "catalog-cache.json";
const DISCOVERED_KEY = "_discovered_models";

/** The two Hub lookups `POST /v1/models/refresh` performs. */
export interface CatalogRemote {
  /** Current head revision of a repository. */
  modelSha(repository: string, signal?: AbortSignal): Promise<string>;
  /** Repository ids published by `author` that match `search`. */
  listModels(author: string, search: string, signal?: AbortSignal): Promise<string[]>;
}

type CacheEntry = Record<string, unknown>;

/** `json.dumps(value, indent=2, sort_keys=True)` as Python writes it (ASCII only, keys sorted recursively). */
export function pythonJson(value: unknown): string {
  const sorted = (input: unknown): unknown => {
    if (Array.isArray(input)) {
      return input.map(sorted);
    }
    if (typeof input === "object" && input !== null) {
      return Object.fromEntries(
        Object.entries(input)
          .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
          .map(([key, entry]) => [key, sorted(entry)]),
      );
    }
    return input;
  };
  return JSON.stringify(sorted(value), null, 2).replace(
    /[\u0080-￿]/g,
    (character) => `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
}

/**
 * The refresh cache the Python runner keeps at `<models dir>/catalog-cache.json`, in the same format:
 * `{"<model_id>": {discovered_revision, refreshed_at}, ..., "_discovered_models": {repositories, refreshed_at}}`.
 * Loaded once at construction; a corrupt or non-object file counts as empty.
 */
export class CatalogCache {
  readonly path: string;
  readonly #directory: string;
  #entries: Record<string, CacheEntry> = {};
  #lock: Promise<unknown> = Promise.resolve();

  constructor(modelsDir: string) {
    this.#directory = modelsDir;
    this.path = join(modelsDir, CATALOG_CACHE_NAME);
    try {
      const value: unknown = JSON.parse(readFileSync(this.path, "utf8"));
      if (typeof value === "object" && value !== null && !Array.isArray(value)) {
        this.#entries = value as Record<string, CacheEntry>;
      }
    } catch {
      // absent or corrupt: start empty
    }
  }

  discoveredRevision(modelId: string): string | undefined {
    const value = this.#entries[modelId]?.["discovered_revision"];
    return typeof value === "string" ? value : undefined;
  }

  discoveredModels(): string[] {
    const value = this.#entries[DISCOVERED_KEY]?.["repositories"];
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
  }

  /** Looks every curated model up on the Hub, records the result and persists it; calls are serialized. */
  refresh(
    models: readonly KnownModel[],
    remote: CatalogRemote,
    signal?: AbortSignal,
  ): Promise<CatalogRefreshResult> {
    const run = this.#lock.then(() => this.#refresh(models, remote, signal));
    this.#lock = run.catch(() => undefined);
    return run;
  }

  async #refresh(
    models: readonly KnownModel[],
    remote: CatalogRemote,
    signal: AbortSignal | undefined,
  ): Promise<CatalogRefreshResult> {
    const refreshedAt = Date.now() / 1000;
    const changed: string[] = [];
    const next: Record<string, CacheEntry> = { ...this.#entries };
    for (const model of models) {
      const sha = await remote.modelSha(model.repository, signal);
      if (this.discoveredRevision(model.modelId) !== sha) {
        changed.push(model.modelId);
      }
      next[model.modelId] = { discovered_revision: sha, refreshed_at: refreshedAt };
    }
    const known = new Set(models.map((model) => model.repository));
    const discovered = (await remote.listModels("fastino", "GLiNER2.5", signal))
      .filter((id) => id.toLowerCase().includes("decide") && !known.has(id))
      .sort((left, right) => (left < right ? -1 : left > right ? 1 : 0));
    next[DISCOVERED_KEY] = { repositories: discovered, refreshed_at: refreshedAt };

    await mkdir(this.#directory, { recursive: true });
    const temporary = join(this.#directory, "catalog-cache.tmp");
    await writeFile(temporary, `${pythonJson(next)}\n`);
    await rename(temporary, this.path);
    this.#entries = next;
    return {
      refreshedAt,
      modelsChecked: models.length,
      changedModels: changed,
      discoveredModels: discovered,
    };
  }
}
