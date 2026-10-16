import { platform } from "#platform";

import {
  KNOWN_MODELS,
  isCommitSha,
  parseModelReference,
  type ArtifactCatalog,
  type CatalogInventory,
  type CatalogModel,
  type CatalogRefreshResult,
  type KnownModel,
  type ProfileCharacterization,
  type ResolvedModel,
} from "../catalog.js";
import type { PrecisionProfile } from "../contracts.js";
import { ModelDownloadRequiredError, ModelIntegrityError } from "../errors.js";
import { hashHandle, readHandleText } from "../platform/files.js";
import type { CatalogCacheLike } from "../platform/port.js";
import type { CatalogRemote } from "./catalog-cache.js";
import { MANIFEST_NAME, parseManifest, type OnnxManifest } from "./manifest.js";

const MAX_MANIFEST_BYTES = 1024 * 1024;
const HASH_CHUNK_BYTES = 4 * 1024 * 1024;

const join = (...parts: string[]): string => platform.join(...parts);

/** `<owner>/<name>` -> `<owner>--<name>`. */
export function repositoryDirectoryName(repository: string): string {
  return repository.replace("/", "--");
}

/** Directory of one artifact: `<modelsDir>/onnx/<owner>--<name>/<revision>`. */
export function artifactDirectory(modelsDir: string, repository: string, revision: string): string {
  return join(modelsDir, "onnx", repositoryDirectoryName(repository), revision);
}

interface FileStat {
  size: number;
  mtimeMs: number;
}

interface Inspection {
  manifest: OnnxManifest;
  /** Identity of everything the verification depends on: path, size and mtime of every file. */
  fingerprint: string;
}

/*
 * Artifact files are checked with `lstat` and read with `O_NOFOLLOW` (the platform's `read-nofollow`), so a
 * symlink (or any other non-regular entry) is never accepted in place of a listed file. This is a point-in-time
 * check: a file replaced between the check and a later read by the ONNX Runtime or tokenizer loader (which open
 * by path) is not detected; the artifact directory must not be writable by untrusted users.
 */

function isMissing(error: unknown): boolean {
  const code = (error as { code?: unknown } | null)?.code;
  return code === "ENOENT" || code === "ENOTDIR";
}

/**
 * Cheap structural validation of an installed artifact directory: manifest content, file set, and file
 * sizes. Does not hash. Throws `ModelDownloadRequiredError` when nothing is installed and
 * `ModelIntegrityError` when what is installed is wrong.
 */
export async function inspectArtifact(
  directory: string,
  model: KnownModel,
  revision: string,
  destination: string,
): Promise<Inspection> {
  const notInstalled = (): ModelDownloadRequiredError =>
    new ModelDownloadRequiredError(model.modelId, revision, destination);
  const invalid = (detail: string): ModelIntegrityError =>
    new ModelIntegrityError(`ONNX artifact for '${model.modelId}' at ${revision} is invalid: ${detail}`);

  try {
    const directoryStatus = await platform.fs.lstat(directory);
    if (!directoryStatus.isDirectory()) {
      throw invalid("the artifact directory is not a regular directory (symbolic links are refused)");
    }
  } catch (error) {
    if (isMissing(error)) {
      throw notInstalled();
    }
    throw error;
  }

  const manifestPath = join(directory, MANIFEST_NAME);
  let manifestStat: FileStat;
  let text: string;
  try {
    const status = await platform.fs.lstat(manifestPath);
    if (!status.isFile()) {
      throw invalid("manifest.json is not a regular file");
    }
    if (status.size > MAX_MANIFEST_BYTES) {
      throw invalid("manifest.json is implausibly large");
    }
    manifestStat = { size: status.size, mtimeMs: status.mtimeMs };
    const handle = await platform.fs.open(manifestPath, "read-nofollow");
    try {
      text = await readHandleText(handle);
    } finally {
      await handle.close();
    }
  } catch (error) {
    if (isMissing(error)) {
      throw notInstalled();
    }
    if (isSymlinkRefusal(error)) {
      throw invalid("manifest.json is not a regular file");
    }
    throw error;
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw invalid("manifest.json is not valid JSON");
  }
  const manifest = parseManifest(raw, model, revision);

  const present = new Set(await platform.fs.readdir(directory));
  present.delete(MANIFEST_NAME);
  const listed = new Set(manifest.files.map((file) => file.path));
  const unlisted = [...present].filter((name) => !listed.has(name)).sort();
  if (unlisted.length > 0) {
    throw invalid(`unlisted files present: ${unlisted.join(", ")}`);
  }
  const parts = [`${MANIFEST_NAME}:${manifestStat.size}:${manifestStat.mtimeMs}`];
  for (const file of manifest.files) {
    let status;
    try {
      status = await platform.fs.lstat(join(directory, file.path));
    } catch (error) {
      if (isMissing(error)) {
        throw invalid(`listed file ${file.path} is missing`);
      }
      throw error;
    }
    if (!status.isFile()) {
      throw invalid(`listed file ${file.path} is not a regular file`);
    }
    if (status.size !== file.size) {
      throw invalid(`${file.path} has ${status.size} bytes, manifest records ${file.size}`);
    }
    parts.push(`${file.path}:${status.size}:${status.mtimeMs}`);
  }
  return { manifest, fingerprint: `${directory}\n${parts.join("\n")}` };
}

/** ELOOP (or EMLINK on some systems) is what `O_NOFOLLOW` reports for a symbolic link. */
function isSymlinkRefusal(error: unknown): boolean {
  const code = (error as { code?: unknown } | null)?.code;
  return code === "ELOOP" || code === "EMLINK";
}

async function sha256OfFile(path: string): Promise<{ digest: string; bytes: number }> {
  const handle = await platform.fs.open(path, "read-nofollow");
  try {
    if (!(await handle.stat()).isFile()) {
      throw new NotRegularFileError();
    }
    const { bytes, sha256 } = await hashHandle(handle, HASH_CHUNK_BYTES);
    return { digest: sha256, bytes };
  } finally {
    await handle.close();
  }
}

class NotRegularFileError extends Error {}

/** Streams every listed file through SHA-256 and compares with the manifest. */
export async function verifyArtifactDigests(
  directory: string,
  manifest: OnnxManifest,
): Promise<void> {
  await Promise.all(
    manifest.files.map(async (file) => {
      let actual;
      try {
        actual = await sha256OfFile(join(directory, file.path));
      } catch (error) {
        if (isMissing(error)) {
          throw new ModelIntegrityError(
            `ONNX artifact for '${manifest.modelId}' at ${manifest.revision} is invalid: ` +
              `listed file ${file.path} is missing`,
          );
        }
        if (isSymlinkRefusal(error) || error instanceof NotRegularFileError) {
          throw new ModelIntegrityError(
            `ONNX artifact for '${manifest.modelId}' at ${manifest.revision} is invalid: ` +
              `listed file ${file.path} is not a regular file`,
          );
        }
        throw error;
      }
      if (actual.bytes !== file.size || actual.digest !== file.sha256) {
        throw new ModelIntegrityError(
          `ONNX artifact for '${manifest.modelId}' at ${manifest.revision} failed integrity ` +
            `verification: ${file.path} does not match the manifest (expected sha256 ${file.sha256}, ` +
            `found ${actual.digest})`,
        );
      }
    }),
  );
}

type Verification = { manifest: OnnxManifest } | { error: unknown };

/** `1e-5`-style decade of a drift, as the PyTorch runner reports it; `"0"` for exact. */
export function driftOrderOfMagnitude(drift: number): string {
  return drift === 0 ? "0" : `1e${Math.floor(Math.log10(drift))}`;
}

export interface OnnxArtifactCatalogOptions {
  models?: readonly KnownModel[];
  /** Hub lookups for `refresh()`. Without one, `refresh()` fails (and `POST /v1/models/refresh` returns 502). */
  remote?: CatalogRemote;
}

/**
 * Filesystem catalog of prepared ONNX artifacts under `<modelsDir>/onnx/<owner>--<name>/<revision>/`.
 * Listing performs cheap structural checks; `resolve` additionally verifies the SHA-256 of every file
 * once per process for as long as the files keep the same size and modification time.
 */
export class OnnxArtifactCatalog implements ArtifactCatalog {
  readonly #modelsDir: string;
  readonly #models: readonly KnownModel[];
  readonly #lookup = new Map<string, KnownModel>();
  /** Outcome per directory, valid while its fingerprint is unchanged. */
  readonly #verified = new Map<string, { fingerprint: string; outcome: Verification }>();
  readonly #inFlight = new Map<string, Promise<Verification>>();
  readonly #evidence = new Map<string, OnnxManifest>();
  readonly #installed = new Set<string>();
  readonly #cache: CatalogCacheLike;
  readonly #remote: CatalogRemote | undefined;

  constructor(modelsDir: string, options: OnnxArtifactCatalogOptions = {}) {
    this.#modelsDir = modelsDir;
    this.#cache = platform.catalogCache(modelsDir);
    this.#remote = options.remote;
    this.#models = options.models ?? KNOWN_MODELS;
    for (const model of this.#models) {
      this.#lookup.set(model.modelId, model);
      this.#lookup.set(model.repository, model);
    }
  }

  get modelsDir(): string {
    return this.#modelsDir;
  }

  async inventory(): Promise<CatalogInventory> {
    const models: CatalogModel[] = [];
    for (const model of this.#models) {
      const revisions = await this.#installedRevisions(model);
      if (revisions.includes(model.pinnedRevision)) {
        this.#installed.add(model.modelId);
      } else {
        this.#installed.delete(model.modelId);
        this.#evidence.delete(model.modelId);
      }
      models.push({
        ...model,
        discoveredRevision: this.#cache.discoveredRevision(model.modelId) ?? model.pinnedRevision,
        localRevisions: revisions,
      });
    }
    return { destination: this.#modelsDir, discoveredModels: this.#cache.discoveredModels(), models };
  }

  /** Revisions whose directories pass the cheap structural validation. */
  async #installedRevisions(model: KnownModel): Promise<string[]> {
    const parent = join(this.#modelsDir, "onnx", repositoryDirectoryName(model.repository));
    let names: string[];
    try {
      names = await platform.fs.readdir(parent);
    } catch (error) {
      if (isMissing(error)) {
        return [];
      }
      throw error;
    }
    const revisions: string[] = [];
    for (const name of names.filter(isCommitSha).sort()) {
      try {
        const inspected = await inspectArtifact(join(parent, name), model, name, this.#modelsDir);
        this.#evidence.set(model.modelId, inspected.manifest);
        revisions.push(name);
      } catch (error) {
        if (!(error instanceof ModelIntegrityError)) {
          throw error;
        }
      }
    }
    return revisions;
  }

  /**
   * Curated model ids whose pinned artifacts were last seen installed and intact (updated by `inventory`,
   * `refresh` and `resolve`), so backend capabilities can be reported synchronously.
   */
  installedModelIds(): readonly string[] {
    return this.#models.map((model) => model.modelId).filter((id) => this.#installed.has(id));
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
    const directory = artifactDirectory(this.#modelsDir, model.repository, revision);
    let outcome: Verification;
    try {
      const inspected = await inspectArtifact(directory, model, revision, this.#modelsDir);
      outcome = await this.#verify(directory, inspected);
    } catch (error) {
      this.#markUnavailable(model, revision);
      throw error;
    }
    if ("error" in outcome) {
      this.#markUnavailable(model, revision);
      throw outcome.error;
    }
    this.#evidence.set(model.modelId, outcome.manifest);
    this.#installed.add(model.modelId);
    return {
      modelId: model.modelId,
      repository: model.repository,
      revision,
      path: directory,
      parameterCount: model.parameterCount,
    };
  }

  #markUnavailable(model: KnownModel, revision: string): void {
    if (revision === model.pinnedRevision) {
      this.#installed.delete(model.modelId);
    }
  }

  #verify(directory: string, inspected: Inspection): Promise<Verification> {
    const cached = this.#verified.get(directory);
    if (cached?.fingerprint === inspected.fingerprint) {
      return Promise.resolve(cached.outcome);
    }
    const running = this.#inFlight.get(inspected.fingerprint);
    if (running !== undefined) {
      return running;
    }
    const task = (async (): Promise<Verification> => {
      let outcome: Verification;
      try {
        await verifyArtifactDigests(directory, inspected.manifest);
        outcome = { manifest: inspected.manifest };
      } catch (error) {
        if (!(error instanceof ModelIntegrityError)) {
          throw error;
        }
        outcome = { error };
      }
      this.#verified.set(directory, { fingerprint: inspected.fingerprint, outcome });
      return outcome;
    })().finally(() => this.#inFlight.delete(inspected.fingerprint));
    this.#inFlight.set(inspected.fingerprint, task);
    return task;
  }

  /**
   * The only operation that uses the network: looks every curated model (and the publisher's catalog) up on the
   * Hub, as the Python runner does, and records the result in `<models dir>/catalog-cache.json`. Cached
   * integrity verification is dropped so the next resolve re-checks the files.
   */
  async refresh(): Promise<CatalogRefreshResult> {
    if (this.#remote === undefined) {
      throw new Error("no remote model catalog is configured");
    }
    const result = await this.#cache.refresh(this.#models, this.#remote);
    this.#verified.clear();
    return result;
  }

  characterize(modelId: string, profile: PrecisionProfile): ProfileCharacterization | undefined {
    if (profile.device !== "cpu" || profile.precision !== "fp32") {
      return undefined;
    }
    const manifest = this.#evidence.get(modelId);
    if (manifest === undefined) {
      return undefined;
    }
    const { parity } = manifest;
    const drift = Math.max(parity.maxAbsLogitDifference, parity.onnxOfficialMaxAbsLogitDifference ?? 0);
    const rows = parity.rows === null ? "" : `${parity.rows}-row `;
    const suite = parity.fixtureSuite === null ? "" : ` ${parity.fixtureSuite}`;
    return {
      observedMaxAbsScoreDrift: drift,
      driftOrderOfMagnitude: driftOrderOfMagnitude(drift),
      scope:
        `${rows}${suite.trim() || "prepared-artifact"} raw-logit characterization recorded by the ONNX ` +
        "preparation tool (ONNX Runtime CPU fp32 against the PyTorch reference); not a universal " +
        "accuracy guarantee.",
    };
  }
}
