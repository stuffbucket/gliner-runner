import type { FileStatus, OpenMode, PlatformFile, PlatformFileSystem } from "@gliner-runner/onnx-runner/portable";

/**
 * The runner's file operations on the origin private file system (OPFS), for a dedicated Web Worker. Paths are
 * POSIX-style and absolute (`/models/onnx/...`). Writes use `FileSystemSyncAccessHandle` (positioned, fast, and
 * the only writable API every engine supports in a worker, Safari included); reads use `File.slice`, which takes
 * no lock, so a file can be read by several readers while nothing writes it. Failures carry the Node error codes
 * the runner checks.
 */

type DirectoryHandle = FileSystemDirectoryHandle;

/** A Node-style filesystem error. */
function fsError(code: string, message: string): Error {
  return Object.assign(new Error(`${code}: ${message}`), { code });
}

/** Maps OPFS `DOMException`s to Node codes; anything else is passed through. */
function mapped(error: unknown, path: string): unknown {
  const name = (error as { name?: unknown } | null)?.name;
  switch (name) {
    case "NotFoundError":
      return fsError("ENOENT", `no such file or directory, '${path}'`);
    case "TypeMismatchError":
      return fsError("ENOTDIR", `not a directory (or not a file), '${path}'`);
    case "InvalidModificationError":
      return fsError("ENOTEMPTY", `directory not empty, '${path}'`);
    case "NoModificationAllowedError":
      return fsError("EBUSY", `resource busy or locked, '${path}'`);
    case "QuotaExceededError":
      return fsError("ENOSPC", `no space left for the browser's storage, '${path}'`);
    default:
      return error;
  }
}

export function splitPath(path: string): string[] {
  const parts: string[] = [];
  for (const part of path.split("/")) {
    if (part === "" || part === ".") {
      continue;
    }
    if (part === "..") {
      parts.pop();
    } else {
      parts.push(part);
    }
  }
  return parts;
}

export const joinPath = (...segments: string[]): string => `/${splitPath(segments.join("/")).join("/")}`;

/** Requests share one root handle; OPFS is per origin. */
let root: Promise<DirectoryHandle> | undefined;
const opfsRoot = (): Promise<DirectoryHandle> => (root ??= navigator.storage.getDirectory());

async function directory(parts: readonly string[], create = false): Promise<DirectoryHandle> {
  let current = await opfsRoot();
  for (const part of parts) {
    current = await current.getDirectoryHandle(part, { create });
  }
  return current;
}

/** The parent directory handle and the entry name of `path`. */
async function parentOf(path: string): Promise<{ parent: DirectoryHandle; name: string }> {
  const parts = splitPath(path);
  const name = parts.pop();
  if (name === undefined) {
    throw fsError("EISDIR", "the root cannot be used as a file");
  }
  try {
    return { parent: await directory(parts), name };
  } catch (error) {
    throw mapped(error, path);
  }
}

async function fileHandle(path: string, create = false): Promise<FileSystemFileHandle> {
  const { parent, name } = await parentOf(path);
  try {
    return await parent.getFileHandle(name, { create });
  } catch (error) {
    throw mapped(error, path);
  }
}

async function exists(path: string): Promise<boolean> {
  const { parent, name } = await parentOf(path).catch(() => ({ parent: undefined, name: "" }));
  if (parent === undefined) {
    return false;
  }
  for await (const key of parent.keys()) {
    if (key === name) {
      return true;
    }
  }
  return false;
}

const fileStatus = (size: number, mtimeMs: number, file: boolean): FileStatus => ({
  isFile: () => file,
  isDirectory: () => !file,
  size,
  mtimeMs,
});

function readerFor(handle: FileSystemFileHandle, path: string): PlatformFile {
  let file: Promise<File> | undefined;
  const current = (): Promise<File> => (file ??= handle.getFile());
  return {
    async read(buffer, offset, length, position) {
      const source = await current();
      const bytes = new Uint8Array(await source.slice(position, position + length).arrayBuffer());
      buffer.set(bytes, offset);
      return { bytesRead: bytes.length };
    },
    write() {
      return Promise.reject(fsError("EBADF", `'${path}' is open for reading`));
    },
    async stat() {
      const source = await current();
      return fileStatus(source.size, source.lastModified, true);
    },
    sync: () => Promise.resolve(),
    close: () => Promise.resolve(),
  };
}

function writerFor(access: FileSystemSyncAccessHandle): PlatformFile {
  let closed = false;
  return {
    read(buffer, offset, length, position) {
      return Promise.resolve({ bytesRead: access.read(buffer.subarray(offset, offset + length), { at: position }) });
    },
    write(buffer, offset, length, position) {
      return Promise.resolve({ bytesWritten: access.write(buffer.subarray(offset, offset + length), { at: position }) });
    },
    stat: () => Promise.resolve(fileStatus(access.getSize(), Date.now(), true)),
    sync() {
      access.flush();
      return Promise.resolve();
    },
    close() {
      if (!closed) {
        closed = true;
        access.close();
      }
      return Promise.resolve();
    },
  };
}

async function writeAll(path: string, bytes: Uint8Array): Promise<void> {
  const access = await (await fileHandle(path, true)).createSyncAccessHandle();
  try {
    access.truncate(0);
    let done = 0;
    while (done < bytes.length) {
      done += access.write(bytes.subarray(done), { at: done });
    }
    access.flush();
  } finally {
    access.close();
  }
}

/** Moves one file (cheap where `move` exists, a copy otherwise); an existing target is replaced. */
async function moveFile(from: string, to: string): Promise<void> {
  const source = await fileHandle(from);
  const target = await parentOf(to);
  await target.parent.removeEntry(target.name).catch(() => undefined);
  const movable = source as FileSystemFileHandle & { move?: (parent: DirectoryHandle, name: string) => Promise<void> };
  if (typeof movable.move === "function") {
    await movable.move(target.parent, target.name);
    return;
  }
  await writeAll(to, new Uint8Array(await (await source.getFile()).arrayBuffer()));
  const { parent, name } = await parentOf(from);
  await parent.removeEntry(name);
}

/**
 * Moves a directory file by file (OPFS directory moves are not available everywhere). `manifest.json` goes last,
 * so an interrupted move leaves a destination the runner reads as "not installed" rather than as a valid artifact.
 */
async function moveDirectory(from: string, to: string): Promise<void> {
  await directory(splitPath(to), true);
  const names: string[] = [];
  for await (const name of (await directory(splitPath(from))).keys()) {
    names.push(name);
  }
  names.sort((left, right) => Number(left === "manifest.json") - Number(right === "manifest.json"));
  for (const name of names) {
    const source = joinPath(from, name);
    if ((await opfs.lstat(source)).isDirectory()) {
      await moveDirectory(source, joinPath(to, name));
    } else {
      await moveFile(source, joinPath(to, name));
    }
  }
  const { parent, name } = await parentOf(from);
  await parent.removeEntry(name, { recursive: true });
}

export const opfs: PlatformFileSystem = {
  async open(path: string, mode: OpenMode): Promise<PlatformFile> {
    if (mode === "read" || mode === "read-nofollow") {
      return readerFor(await fileHandle(path), path); // OPFS has no symbolic links
    }
    if (mode === "write-exclusive" && (await exists(path))) {
      throw fsError("EEXIST", `file already exists, '${path}'`);
    }
    try {
      const access = await (await fileHandle(path, true)).createSyncAccessHandle();
      if (mode === "write-truncate") {
        access.truncate(0);
      }
      return writerFor(access);
    } catch (error) {
      throw mapped(error, path);
    }
  },

  async readFile(path) {
    try {
      return await (await (await fileHandle(path)).getFile()).text();
    } catch (error) {
      throw mapped(error, path);
    }
  },

  async writeFile(path, text) {
    try {
      await writeAll(path, new TextEncoder().encode(text));
    } catch (error) {
      throw mapped(error, path);
    }
  },

  async lstat(path) {
    const parts = splitPath(path);
    if (parts.length === 0) {
      return fileStatus(0, 0, false);
    }
    const { parent, name } = await parentOf(path);
    try {
      const file = await (await parent.getFileHandle(name)).getFile();
      return fileStatus(file.size, file.lastModified, true);
    } catch (error) {
      if ((error as { name?: unknown }).name !== "TypeMismatchError") {
        throw mapped(error, path);
      }
    }
    try {
      await parent.getDirectoryHandle(name);
      return fileStatus(0, 0, false);
    } catch (error) {
      throw mapped(error, path);
    }
  },

  async readdir(path) {
    try {
      const names: string[] = [];
      for await (const name of (await directory(splitPath(path))).keys()) {
        names.push(name);
      }
      return names;
    } catch (error) {
      throw mapped(error, path);
    }
  },

  async mkdir(path, options = {}) {
    const parts = splitPath(path);
    if (options.recursive === true) {
      try {
        await directory(parts, true);
        return;
      } catch (error) {
        throw mapped(error, path);
      }
    }
    if (await exists(path)) {
      throw fsError("EEXIST", `file already exists, '${path}'`);
    }
    const { parent, name } = await parentOf(path);
    try {
      await parent.getDirectoryHandle(name, { create: true });
    } catch (error) {
      throw mapped(error, path);
    }
  },

  async rename(from, to) {
    try {
      if ((await opfs.lstat(from)).isDirectory()) {
        await moveDirectory(from, to);
      } else {
        await moveFile(from, to);
      }
    } catch (error) {
      throw mapped(error, from);
    }
  },

  async rm(path, options = {}) {
    try {
      const { parent, name } = await parentOf(path);
      await parent.removeEntry(name, { recursive: options.recursive === true });
    } catch (error) {
      const result = mapped(error, path);
      if (options.force === true && (result as { code?: unknown }).code === "ENOENT") {
        return;
      }
      throw result;
    }
  },

  async copyFile(from, to) {
    if (await exists(to)) {
      throw fsError("EEXIST", `file already exists, '${to}'`);
    }
    try {
      const source = await (await fileHandle(from)).getFile();
      await writeAll(to, new Uint8Array(await source.arrayBuffer()));
    } catch (error) {
      throw mapped(error, from);
    }
  },
};
