import { createHash } from "node:crypto";
import { constants, readFileSync } from "node:fs";
import { copyFile, lstat, mkdir, open, readFile, readdir, realpath, rename, rm, statfs, writeFile } from "node:fs/promises";
import { homedir, totalmem } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { CatalogCache } from "../onnx/catalog-cache.js";
import { GRAPH_NAME } from "../onnx/manifest.js";
import type { OpenMode, PlatformFile, PlatformSession, RunnerPlatform, TokenizerConstructor } from "./port.js";

import type * as Ort from "onnxruntime-node";

// `@huggingface/tokenizers@0.2.0` ships declarations that NodeNext rejects without `skipLibCheck`, and both it
// and onnxruntime-node are only needed once a model is actually loaded, so they are imported lazily and typed here.
const TOKENIZERS_SPECIFIER = "@huggingface/tokenizers";
const ORT_SPECIFIER = "onnxruntime-node";

/** `O_NOFOLLOW` is missing on Windows, where symbolic links need privileges to create. */
const O_NOFOLLOW = constants.O_NOFOLLOW ?? 0;

const FLAGS: Readonly<Record<OpenMode, number>> = {
  read: constants.O_RDONLY,
  "read-nofollow": constants.O_RDONLY | O_NOFOLLOW,
  "write-exclusive": constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL,
  "write-truncate": constants.O_WRONLY | constants.O_CREAT | constants.O_TRUNC,
};

let version: string | undefined;

/** The Node.js host: the local filesystem, `node:crypto`, `statfs`, `process.env` and onnxruntime-node. */
export const platform: RunnerPlatform = {
  fs: {
    async open(path, mode): Promise<PlatformFile> {
      const handle = await open(path, FLAGS[mode], 0o644);
      return {
        read: (buffer, offset, length, position) => handle.read(buffer, offset, length, position),
        write: (buffer, offset, length, position) => handle.write(buffer, offset, length, position),
        stat: () => handle.stat(),
        sync: () => handle.sync(),
        close: () => handle.close(),
      };
    },
    readFile: (path) => readFile(path, "utf8"),
    writeFile: (path, text) => writeFile(path, text),
    lstat: (path) => lstat(path),
    readdir: (path) => readdir(path),
    mkdir: async (path, options) => {
      await mkdir(path, options);
    },
    rename: (from, to) => rename(from, to),
    rm: (path, options) => rm(path, options),
    copyFile: (from, to) => copyFile(from, to, constants.COPYFILE_EXCL),
  },
  join,
  createSha256: () => {
    const hash = createHash("sha256");
    return Promise.resolve({
      update: (data) => void hash.update(data),
      digest: () => hash.digest("hex"),
    });
  },
  freeBytes: async (directory) => {
    try {
      const status = await statfs(directory);
      return Number(status.bavail) * Number(status.bsize);
    } catch {
      return undefined;
    }
  },
  physicalMemoryBytes: () => totalmem(),
  processRssBytes: () => process.memoryUsage.rss(),
  env: () => process.env,
  defaultTemplatesDir: () => fileURLToPath(new URL("../../../templates", import.meta.url)),
  runnerVersion: () => {
    if (version === undefined) {
      try {
        const raw = JSON.parse(readFileSync(fileURLToPath(new URL("../../../package.json", import.meta.url)), "utf8")) as { version?: unknown };
        version = typeof raw.version === "string" ? raw.version : "unknown";
      } catch {
        version = "unknown";
      }
    }
    return version;
  },
  resolvePath: async (path) => {
    const expanded = path === "~" ? homedir() : path.startsWith("~/") ? resolve(homedir(), path.slice(2)) : path;
    const absolute = resolve(expanded);
    try {
      return await realpath(absolute);
    } catch {
      return absolute;
    }
  },
  // Loaded on first use: snapshot discovery imports the template module, which imports this one.
  discoverSnapshots: async (model, revision, files, options) =>
    (await import("../onnx/snapshots.js")).discoverSnapshots(model, revision, files, options),
  catalogCache: (modelsDir) => new CatalogCache(modelsDir),
  importTokenizers: async () =>
    ((await import(TOKENIZERS_SPECIFIER)) as { Tokenizer: TokenizerConstructor }).Tokenizer,
  createSession: async (modelDirectory, options): Promise<PlatformSession> => {
    const ort = (await import(ORT_SPECIFIER)) as typeof Ort;
    const session = await ort.InferenceSession.create(join(modelDirectory, GRAPH_NAME), {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
      executionMode: "sequential",
      enableMemPattern: false,
      interOpNumThreads: 1,
      ...(options.intraOpThreads === undefined ? {} : { intraOpNumThreads: options.intraOpThreads }),
    });
    return { ort, session };
  },
};

/** Node always uses the built-in platform; only the browser build installs one. */
export function installPlatform(_platform: RunnerPlatform): never {
  throw new Error("the Node.js runner uses its built-in platform");
}
