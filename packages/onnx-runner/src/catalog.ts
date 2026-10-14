import { ModelDownloadRequiredError, ModelIntegrityError } from "./errors.js";
import type { PrecisionProfile } from "./contracts.js";

/** A curated model the runner knows how to serve. */
export interface KnownModel {
  modelId: string;
  repository: string;
  description: string;
  /** Immutable upstream revision (40 hex) that the runner supports by default. */
  pinnedRevision: string;
  parameterCount: number;
}

/** Live state of one known model, reported by `GET /v1/models`. */
export interface CatalogModel extends KnownModel {
  discoveredRevision: string;
  /** Sorted revisions whose artifacts are complete on disk. */
  localRevisions: readonly string[];
}

export interface CatalogInventory {
  /** Directory operators install artifacts into. */
  destination: string;
  /** Repository ids found by the last refresh that are not curated yet. */
  discoveredModels: readonly string[];
  models: readonly CatalogModel[];
}

/** A model whose artifacts are installed and verified, ready to hand to a backend. */
export interface ResolvedModel {
  modelId: string;
  repository: string;
  revision: string;
  /** Directory holding the verified artifacts. */
  path: string;
  parameterCount: number;
}

export interface CatalogRefreshResult {
  refreshedAt: number;
  modelsChecked: number;
  changedModels: readonly string[];
  discoveredModels: readonly string[];
}

export interface ProfileCharacterization {
  observedMaxAbsScoreDrift: number | null;
  driftOrderOfMagnitude: string | null;
  scope: string;
}

/**
 * The runner's view of model artifacts. Implementations own artifact layout, integrity
 * verification and any remote lookups; the runner never touches the filesystem itself.
 */
export interface ArtifactCatalog {
  inventory(): Promise<CatalogInventory>;

  /** Canonical model id for `<id-or-repository>[@revision]`, or `undefined` when unknown. */
  canonicalModelId(reference: string): string | undefined;

  /**
   * Resolves `<id-or-repository>[@revision]` (default: the pinned revision) to verified artifacts.
   * Throws `ModelIntegrityError` for unknown models or invalid revisions, and
   * `ModelDownloadRequiredError` when the revision is not installed. Called on every request, so
   * implementations should cache successful verification.
   */
  resolve(reference: string): Promise<ResolvedModel>;

  /** The only operation allowed to use the network. Any rejection becomes 502. */
  refresh(): Promise<CatalogRefreshResult>;

  /** Recorded parity evidence for a profile; absent evidence is reported as unknown. */
  characterize?(modelId: string, profile: PrecisionProfile): ProfileCharacterization | undefined;
}

export const KNOWN_MODELS: readonly KnownModel[] = [
  {
    modelId: "decide-340m",
    repository: "fastino/GLiNER2.5-Decide",
    pinnedRevision: "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90",
    parameterCount: 486_444_053,
    description: "English GLiNER2.5 classification model",
  },
  {
    modelId: "decide-multilingual",
    repository: "fastino/GLiNER2.5-multi-Decide",
    pinnedRevision: "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d",
    parameterCount: 287_355_159,
    description: "Multilingual GLiNER2.5 classification model",
  },
  {
    modelId: "decide-1b",
    repository: "fastino/GLiNER2.5-Decide-1B",
    pinnedRevision: "67af949e17bebd20f05fe020a971079889e7a462",
    parameterCount: 1_188_796_693,
    description: "One-billion-parameter English GLiNER2.5 classification model",
  },
];

const COMMIT_SHA = /^[a-f0-9]{40}$/;

export function isCommitSha(value: string): boolean {
  return COMMIT_SHA.test(value);
}

export function parseModelReference(reference: string): { id: string; revision: string | undefined } {
  const separator = reference.indexOf("@");
  return separator < 0
    ? { id: reference, revision: undefined }
    : { id: reference.slice(0, separator), revision: reference.slice(separator + 1) };
}

export interface InMemoryCatalogOptions {
  destination?: string;
  models?: readonly KnownModel[];
  /** Installed revisions by model id; each maps to the artifact directory. */
  installed?: Readonly<Record<string, Readonly<Record<string, string>>>>;
  discoveredRevisions?: Readonly<Record<string, string>>;
  discoveredModels?: readonly string[];
  refresh?: () => Promise<CatalogRefreshResult>;
  characterize?: ArtifactCatalog["characterize"];
}

/** Catalog with no filesystem access, for tests and for serving with nothing installed. */
export class InMemoryArtifactCatalog implements ArtifactCatalog {
  readonly #destination: string;
  readonly #models: readonly KnownModel[];
  readonly #lookup = new Map<string, KnownModel>();
  readonly #installed: Record<string, Record<string, string>>;
  readonly #discoveredRevisions: Record<string, string>;
  #discoveredModels: readonly string[];
  readonly #refresh: () => Promise<CatalogRefreshResult>;
  readonly #characterize: InMemoryCatalogOptions["characterize"];

  constructor(options: InMemoryCatalogOptions = {}) {
    this.#destination = options.destination ?? "/models";
    this.#models = options.models ?? KNOWN_MODELS;
    for (const model of this.#models) {
      this.#lookup.set(model.modelId, model);
      this.#lookup.set(model.repository, model);
    }
    this.#installed = Object.fromEntries(
      Object.entries(options.installed ?? {}).map(([id, revisions]) => [id, { ...revisions }]),
    );
    this.#discoveredRevisions = { ...options.discoveredRevisions };
    this.#discoveredModels = options.discoveredModels ?? [];
    this.#refresh =
      options.refresh ??
      (async () => ({
        refreshedAt: Date.now() / 1000,
        modelsChecked: this.#models.length,
        changedModels: [],
        discoveredModels: this.#discoveredModels,
      }));
    this.#characterize = options.characterize;
  }

  characterize(modelId: string, profile: PrecisionProfile): ProfileCharacterization | undefined {
    return this.#characterize?.(modelId, profile);
  }

  install(modelId: string, revision: string, path: string): void {
    (this.#installed[modelId] ??= {})[revision] = path;
  }

  async inventory(): Promise<CatalogInventory> {
    return {
      destination: this.#destination,
      discoveredModels: this.#discoveredModels,
      models: this.#models.map((model) => ({
        ...model,
        discoveredRevision: this.#discoveredRevisions[model.modelId] ?? model.pinnedRevision,
        localRevisions: Object.keys(this.#installed[model.modelId] ?? {}).sort(),
      })),
    };
  }

  canonicalModelId(reference: string): string | undefined {
    return this.#lookup.get(parseModelReference(reference).id)?.modelId;
  }

  async resolve(reference: string): Promise<ResolvedModel> {
    const { id, revision: requested } = parseModelReference(reference);
    const model = this.#lookup.get(id);
    if (model === undefined) {
      throw new ModelIntegrityError(`model '${reference}' is not in the GLiNER2.5 catalog`);
    }
    const revision = requested ?? model.pinnedRevision;
    if (!isCommitSha(revision)) {
      throw new ModelIntegrityError(
        `model '${reference}' has an invalid revision; expected 40 lowercase hex characters`,
      );
    }
    const path = this.#installed[model.modelId]?.[revision];
    if (path === undefined) {
      throw new ModelDownloadRequiredError(reference, revision, this.#destination);
    }
    return {
      modelId: model.modelId,
      repository: model.repository,
      revision,
      path,
      parameterCount: model.parameterCount,
    };
  }

  async refresh(): Promise<CatalogRefreshResult> {
    const result = await this.#refresh();
    this.#discoveredModels = result.discoveredModels;
    return result;
  }
}
