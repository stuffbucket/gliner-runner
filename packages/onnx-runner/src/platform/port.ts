import type * as Ort from "onnxruntime-node";

import type { CatalogRefreshResult, KnownModel } from "../catalog.js";
import type { Gliner2Tokenizer } from "../gliner2/index.js";
import type { CatalogRemote } from "../onnx/catalog-cache.js";
import type { DiscoveryOptions, DiscoveryResult } from "../onnx/snapshots.js";
import type { TemplateSourceFile } from "../onnx/templates.js";

/**
 * Everything the runner needs from its host, so one codebase runs under Node (`node.ts`, the default) and in a
 * browser Web Worker (an implementation installed into `slot.ts`, selected by the `browser` export condition of
 * the `#platform` import). The Node implementation is the code the runner always used; the browser one maps the
 * same operations onto the origin private file system, WebCrypto-free hashing and onnxruntime-web.
 */

/**
 * `read` follows symbolic links (snapshot directories legitimately hold links into a blob store); `read-nofollow`
 * refuses them (artifact and template files). `write-exclusive` fails if the file exists; `write-truncate`
 * creates or empties it. Files are created with mode 0644 where the host has modes.
 */
export type OpenMode = "read" | "read-nofollow" | "write-exclusive" | "write-truncate";

export interface FileStatus {
  isFile(): boolean;
  isDirectory(): boolean;
  size: number;
  mtimeMs: number;
}

/** A positioned file handle. Reads and writes may be short; callers loop. */
export interface PlatformFile {
  read(buffer: Uint8Array, offset: number, length: number, position: number): Promise<{ bytesRead: number }>;
  write(buffer: Uint8Array, offset: number, length: number, position: number): Promise<{ bytesWritten: number }>;
  stat(): Promise<FileStatus>;
  sync(): Promise<void>;
  close(): Promise<void>;
}

/**
 * The file operations the runner uses. Failures carry Node-style `code` values (`ENOENT`, `ENOTDIR`, `EEXIST`,
 * `ELOOP`, ...), which callers inspect.
 */
export interface PlatformFileSystem {
  open(path: string, mode: OpenMode): Promise<PlatformFile>;
  /** Whole file as UTF-8 text. */
  readFile(path: string): Promise<string>;
  writeFile(path: string, text: string): Promise<void>;
  lstat(path: string): Promise<FileStatus>;
  readdir(path: string): Promise<string[]>;
  mkdir(path: string, options?: { recursive?: boolean; mode?: number }): Promise<void>;
  rename(from: string, to: string): Promise<void>;
  rm(path: string, options?: { recursive?: boolean; force?: boolean }): Promise<void>;
  /** Copies a file; fails if `to` exists. */
  copyFile(from: string, to: string): Promise<void>;
}

/** Incremental SHA-256; `digest` returns lowercase hex. */
export interface Sha256 {
  update(data: Uint8Array | string): void;
  digest(): string;
}

/** The `CatalogCache` surface the artifact catalog uses (`<models dir>/catalog-cache.json` under Node). */
export interface CatalogCacheLike {
  discoveredRevision(modelId: string): string | undefined;
  discoveredModels(): string[];
  refresh(models: readonly KnownModel[], remote: CatalogRemote): Promise<CatalogRefreshResult>;
}

export type TokenizerConstructor = new (tokenizerJson: object, tokenizerConfig: object) => Gliner2Tokenizer;

export interface SessionOptions {
  /** Intra-op threads; `undefined` lets the runtime decide. */
  intraOpThreads?: number | undefined;
}

/** An ONNX Runtime module (onnxruntime-node or onnxruntime-web, which share this API) and a session on it. */
export interface PlatformSession {
  ort: typeof Ort;
  session: Ort.InferenceSession;
}

export interface RunnerPlatform {
  readonly fs: PlatformFileSystem;
  /** Joins path segments with the host's separator. */
  join(...parts: string[]): string;
  createSha256(): Promise<Sha256>;
  /** Free bytes on the storage holding `directory`; `undefined` when unknown (the check is then skipped). */
  freeBytes(directory: string): Promise<number | undefined>;
  physicalMemoryBytes(): number;
  processRssBytes(): number;
  /** Environment variables (`process.env` under Node, empty in a browser). */
  env(): Readonly<Record<string, string | undefined>>;
  /** The bundled artifact templates. */
  defaultTemplatesDir(): string;
  /** The runner's package version. */
  runnerVersion(): string;
  /** `~` expansion and symlink resolution, as Python's `Path.expanduser().resolve()` does. */
  resolvePath(path: string): Promise<string>;
  /** Local snapshot discovery (Hugging Face caches and the like); a browser has none. */
  discoverSnapshots(
    model: KnownModel,
    revision: string,
    files: readonly TemplateSourceFile[],
    options: DiscoveryOptions,
  ): Promise<DiscoveryResult>;
  catalogCache(modelsDir: string): CatalogCacheLike;
  importTokenizers(): Promise<TokenizerConstructor>;
  /** Opens `model.onnx` (and its external data) from an artifact directory. */
  createSession(modelDirectory: string, options: SessionOptions): Promise<PlatformSession>;
}
