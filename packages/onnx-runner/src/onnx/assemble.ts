import { platform } from "#platform";

import { ModelIntegrityError, RequestCancelledError } from "../errors.js";
import { hashHandle } from "../platform/files.js";
import type { PlatformFile } from "../platform/port.js";
import { randomHex } from "../timers.js";
import { artifactDirectory } from "./artifacts.js";
import { MANIFEST_NAME } from "./manifest.js";
import { SafetensorsError, SafetensorsFile } from "./safetensors.js";
import { WEIGHTS_FILE, type OnnxTemplate, type TemplateTensor } from "./templates.js";

/** Largest read or write issued for an identity tensor. */
const COPY_BLOCK_BYTES = 8 * 1024 * 1024;
/** Source rows are transposed in bands of at most this many bytes (the output band is the same size). */
const TRANSPOSE_BAND_BYTES = 32 * 1024 * 1024;
const TRANSPOSE_TILE = 32;
const HASH_BLOCK_BYTES = 4 * 1024 * 1024;
/** Headroom required on top of the bytes an operation is known to write. */
export const FREE_SPACE_MARGIN_BYTES = 64 * 1024 * 1024;

export class AssembleError extends ModelIntegrityError {}
export class ArtifactExistsError extends AssembleError {
  readonly path: string;

  constructor(path: string) {
    super(`ONNX artifact ${path} already exists; it is never overwritten (remove it first to rebuild)`);
    this.path = path;
  }
}
export class InsufficientSpaceError extends AssembleError {}

export interface Progress {
  stage: "verify-source" | "graph" | "weights" | "copy" | "verify-output" | "finalize";
  file?: string;
  done: number;
  total: number;
}

export interface AssembleOptions {
  template: OnnxTemplate;
  /** Directory holding the template's `source_files` under their plain names. */
  snapshotDir: string;
  /** Models root; the artifact lands in `<modelsDir>/onnx/<owner>--<name>/<revision>/`. */
  modelsDir: string;
  /** Verify size and SHA-256 of every source file first. Disable only for files this process just verified. */
  verifySources?: boolean;
  signal?: AbortSignal | undefined;
  onProgress?: ((progress: Progress) => void) | undefined;
  /** Test seam: free bytes available on the filesystem holding `directory`; `undefined` skips the check. */
  freeBytes?: ((directory: string) => Promise<number | undefined>) | undefined;
  /** Memory bounds; the defaults (8 MiB copy block, 32 MiB transpose band) suit production. */
  limits?: { copyBlockBytes?: number; transposeBandBytes?: number } | undefined;
}

interface Limits {
  copyBlockBytes: number;
  transposeBandBytes: number;
}

export function freeBytesOf(directory: string): Promise<number | undefined> {
  return platform.freeBytes(directory);
}

const join = (...parts: string[]): string => platform.join(...parts);

/** Whether typed arrays use little-endian byte order on this host. */
const littleEndian = (): boolean => new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** Sum of the bytes the artifact's outputs occupy (the manifest adds a few KiB, covered by the margin). */
export const outputBytes = (template: OnnxTemplate): number =>
  template.outputs.reduce((total, output) => total + output.size, 0);

export async function ensureFreeSpace(
  directory: string,
  requiredBytes: number,
  what: string,
  freeBytes: (directory: string) => Promise<number | undefined> = freeBytesOf,
): Promise<void> {
  const free = await freeBytes(directory);
  if (free !== undefined && free < requiredBytes + FREE_SPACE_MARGIN_BYTES) {
    throw new InsufficientSpaceError(
      `not enough free disk space for ${what} in ${directory}: ` +
        `${requiredBytes + FREE_SPACE_MARGIN_BYTES} bytes needed, ${free} available`,
    );
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new RequestCancelledError("model preparation was cancelled");
  }
}

async function writeFully(handle: PlatformFile, buffer: Uint8Array, position: number): Promise<void> {
  let done = 0;
  while (done < buffer.length) {
    const { bytesWritten } = await handle.write(buffer, done, buffer.length - done, position + done);
    if (bytesWritten === 0) {
      throw new AssembleError("write made no progress");
    }
    done += bytesWritten;
  }
}

export async function sha256OfFile(
  path: string,
  signal?: AbortSignal,
): Promise<{ bytes: number; sha256: string }> {
  const handle = await platform.fs.open(path, "read");
  try {
    if (!(await handle.stat()).isFile()) {
      throw new AssembleError(`${path} is not a regular file`);
    }
    return await hashHandle(handle, HASH_BLOCK_BYTES, () => throwIfAborted(signal));
  } finally {
    await handle.close();
  }
}

/**
 * Transposes a row-major `[rows, cols]` band of 32-bit words into a row-major `[cols, rows]` band. Words are
 * moved as integers, so every bit pattern (NaN payloads included) is preserved.
 */
export function transposeWords(source: Uint32Array, target: Uint32Array, rows: number, cols: number): void {
  for (let rowStart = 0; rowStart < rows; rowStart += TRANSPOSE_TILE) {
    const rowEnd = Math.min(rowStart + TRANSPOSE_TILE, rows);
    for (let colStart = 0; colStart < cols; colStart += TRANSPOSE_TILE) {
      const colEnd = Math.min(colStart + TRANSPOSE_TILE, cols);
      for (let row = rowStart; row < rowEnd; row += 1) {
        const base = row * cols;
        for (let col = colStart; col < colEnd; col += 1) {
          target[col * rows + row] = source[base + col] as number;
        }
      }
    }
  }
}

/** Writes one tensor of the data file at its recorded offset, reading the source in bounded blocks. */
async function writeTensor(
  weights: SafetensorsFile,
  tensor: TemplateTensor,
  output: PlatformFile,
  scratch: Uint8Array,
  bandBytes: number,
  signal: AbortSignal | undefined,
  advance: (bytes: number) => void,
): Promise<void> {
  if (tensor.transform === "identity") {
    for (let done = 0; done < tensor.length; ) {
      throwIfAborted(signal);
      const length = Math.min(scratch.length, tensor.length - done);
      const block = scratch.subarray(0, length);
      await weights.readTensorRange(tensor.source, done, length, block);
      await writeFully(output, block, tensor.offset + done);
      done += length;
      advance(length);
    }
    return;
  }
  const rows = tensor.sourceShape[0] as number;
  const cols = tensor.sourceShape[1] as number;
  if (rows === 0 || cols === 0) {
    return;
  }
  const rowsPerBand = Math.max(1, Math.min(rows, Math.floor(bandBytes / (cols * 4))));
  const sourceBuffer = new ArrayBuffer(rowsPerBand * cols * 4);
  const targetBuffer = new ArrayBuffer(rowsPerBand * cols * 4);
  const sourceBytes = new Uint8Array(sourceBuffer);
  const targetBytes = new Uint8Array(targetBuffer);
  for (let firstRow = 0; firstRow < rows; firstRow += rowsPerBand) {
    throwIfAborted(signal);
    const bandRows = Math.min(rowsPerBand, rows - firstRow);
    const chunkBytes = bandRows * cols * 4;
    await weights.readTensorRange(tensor.source, firstRow * cols * 4, chunkBytes, sourceBytes);
    transposeWords(new Uint32Array(sourceBuffer, 0, bandRows * cols), new Uint32Array(targetBuffer, 0, bandRows * cols), bandRows, cols);
    if (bandRows === rows) {
      await writeFully(output, targetBytes.subarray(0, chunkBytes), tensor.offset);
    } else {
      // Output row `col` is `rows` long; this band supplies its elements [firstRow, firstRow + bandRows).
      for (let col = 0; col < cols; col += 1) {
        await writeFully(
          output,
          targetBytes.subarray(col * bandRows * 4, (col + 1) * bandRows * 4),
          tensor.offset + (col * rows + firstRow) * 4,
        );
      }
    }
    advance(chunkBytes);
  }
}

/** Streams `model.onnx.data`: tensors in template order at their offsets, zero gaps, no trailing padding. */
async function writeData(
  template: OnnxTemplate,
  weights: SafetensorsFile,
  path: string,
  limits: Limits,
  signal: AbortSignal | undefined,
  progress: (done: number) => void,
): Promise<void> {
  const output = await platform.fs.open(path, "write-exclusive");
  try {
    const scratch = new Uint8Array(limits.copyBlockBytes);
    let position = 0;
    let written = 0;
    for (const tensor of template.data.tensors) {
      throwIfAborted(signal);
      if (tensor.offset < position) {
        throw new AssembleError(`tensor ${tensor.name} overlaps the previous tensor`);
      }
      for (let gap = tensor.offset - position; gap > 0; ) {
        const zeros = new Uint8Array(Math.min(gap, 4096));
        await writeFully(output, zeros, tensor.offset - gap);
        gap -= zeros.length;
      }
      await writeTensor(weights, tensor, output, scratch, limits.transposeBandBytes, signal, (bytes) => {
        written += bytes;
        progress(written);
      });
      position = tensor.offset + tensor.length;
    }
    if (position !== template.data.size) {
      throw new AssembleError(`data file would be ${position} bytes, template records ${template.data.size}`);
    }
    await output.sync();
  } finally {
    await output.close();
  }
}

/** Checks, before anything is written, that every tensor the template reads exists with the recorded layout. */
function assertWeightsMatch(template: OnnxTemplate, weights: SafetensorsFile): void {
  for (const tensor of template.data.tensors) {
    const source = weights.tensor(tensor.source);
    const sameShape =
      source.shape.length === tensor.sourceShape.length &&
      source.shape.every((dim, index) => dim === tensor.sourceShape[index]);
    if (source.dtype !== "F32" || !sameShape || source.byteLength !== tensor.length) {
      throw new AssembleError(
        `safetensors tensor ${JSON.stringify(tensor.source)} is ${source.dtype}${JSON.stringify(source.shape)}; ` +
          `the template requires F32${JSON.stringify(tensor.sourceShape)}`,
      );
    }
  }
}

async function verifySource(template: OnnxTemplate, snapshotDir: string, options: AssembleOptions): Promise<void> {
  const total = template.sourceFiles.reduce((sum, file) => sum + file.size, 0);
  let done = 0;
  for (const file of template.sourceFiles) {
    const path = join(snapshotDir, file.path);
    let actual;
    try {
      actual = await sha256OfFile(path, options.signal);
    } catch (error) {
      if ((error as { code?: unknown }).code === "ENOENT") {
        throw new AssembleError(`snapshot is missing ${file.path}`);
      }
      throw error;
    }
    if (actual.bytes !== file.size || actual.sha256 !== file.sha256) {
      throw new AssembleError(
        `snapshot file ${file.path} does not match the pinned revision ` +
          `(expected ${file.size} bytes sha256 ${file.sha256}, found ${actual.bytes} bytes sha256 ${actual.sha256})`,
      );
    }
    done += file.size;
    options.onProgress?.({ stage: "verify-source", file: file.path, done, total });
  }
}

/**
 * Builds the canonical artifact for `options.template` in a staging directory next to the destination, verifies
 * every output's size and SHA-256 against the template, writes `manifest.json` last and atomically renames the
 * directory into `<modelsDir>/onnx/<owner>--<name>/<revision>/`. Never overwrites an existing artifact and
 * removes its staging directory on any failure. Returns the artifact directory.
 */
export async function assembleArtifact(options: AssembleOptions): Promise<string> {
  const { template, snapshotDir, modelsDir, signal } = options;
  if (!littleEndian()) {
    throw new AssembleError("ONNX artifact preparation requires a little-endian host");
  }
  const destination = artifactDirectory(modelsDir, template.model.repository, template.revision);
  const parent = join(modelsDir, "onnx", template.model.repository.replace("/", "--"));
  const exists = async (): Promise<boolean> => {
    try {
      await platform.fs.lstat(destination);
      return true;
    } catch (error) {
      if ((error as { code?: unknown }).code === "ENOENT") {
        return false;
      }
      throw error;
    }
  };
  if (await exists()) {
    throw new ArtifactExistsError(destination);
  }
  throwIfAborted(signal);
  await platform.fs.mkdir(parent, { recursive: true });
  await ensureFreeSpace(parent, outputBytes(template), "the prepared artifact", options.freeBytes);

  if (options.verifySources ?? true) {
    await verifySource(template, snapshotDir, options);
  }
  const weights = await SafetensorsFile.open(join(snapshotDir, WEIGHTS_FILE)).catch((error: unknown) => {
    if (error instanceof SafetensorsError) {
      throw new AssembleError(`invalid ${WEIGHTS_FILE}: ${error.message}`);
    }
    if ((error as { code?: unknown }).code === "ENOENT") {
      throw new AssembleError(`snapshot is missing ${WEIGHTS_FILE}`);
    }
    throw error;
  });
  const limits: Limits = {
    copyBlockBytes: options.limits?.copyBlockBytes ?? COPY_BLOCK_BYTES,
    transposeBandBytes: options.limits?.transposeBandBytes ?? TRANSPOSE_BAND_BYTES,
  };
  const staging = join(parent, `.staging-${template.revision.slice(0, 12)}-${randomHex(4)}`);
  try {
    assertWeightsMatch(template, weights);
    await platform.fs.mkdir(staging, { mode: 0o755 });

    const dataSize = template.data.size;
    for (const file of template.templateFiles) {
      throwIfAborted(signal);
      if (file.output === MANIFEST_NAME) {
        continue; // written last
      }
      await platform.fs.copyFile(join(template.directory, file.path), join(staging, file.output));
      options.onProgress?.({ stage: "graph", file: file.output, done: file.size, total: file.size });
    }
    await writeData(template, weights, join(staging, template.data.path), limits, signal, (done) =>
      options.onProgress?.({ stage: "weights", file: template.data.path, done, total: dataSize }),
    );
    const templated = new Set(template.templateFiles.map((file) => file.output));
    for (const output of template.outputs) {
      if (output.path === template.data.path || templated.has(output.path)) {
        continue;
      }
      throwIfAborted(signal);
      await platform.fs.copyFile(join(snapshotDir, output.path), join(staging, output.path));
      options.onProgress?.({ stage: "copy", file: output.path, done: output.size, total: output.size });
    }
    await weights.close();

    // Every output is re-read from disk and checked before the manifest exists.
    const total = template.outputs.reduce((sum, output) => sum + output.size, 0);
    let verified = 0;
    for (const output of template.outputs) {
      const actual = await sha256OfFile(join(staging, output.path), signal);
      if (actual.bytes !== output.size || actual.sha256 !== output.sha256) {
        throw new AssembleError(
          `prepared ${output.path} does not match the template ` +
            `(expected ${output.size} bytes sha256 ${output.sha256}, got ${actual.bytes} bytes sha256 ${actual.sha256})`,
        );
      }
      verified += output.size;
      options.onProgress?.({ stage: "verify-output", file: output.path, done: verified, total });
    }

    const manifest = template.templateFiles.find((file) => file.output === MANIFEST_NAME);
    if (manifest === undefined) {
      throw new AssembleError("template does not provide manifest.json");
    }
    await platform.fs.copyFile(join(template.directory, manifest.path), join(staging, MANIFEST_NAME));
    const written = await sha256OfFile(join(staging, MANIFEST_NAME), signal);
    if (written.bytes !== manifest.size || written.sha256 !== manifest.sha256) {
      throw new AssembleError("prepared manifest.json does not match the template");
    }
    const expected = new Set([...template.outputs.map((output) => output.path), MANIFEST_NAME]);
    const present = await platform.fs.readdir(staging);
    if (present.length !== expected.size || present.some((name) => !expected.has(name))) {
      throw new AssembleError(`staging directory holds unexpected files: ${present.sort().join(", ")}`);
    }

    throwIfAborted(signal);
    if (await exists()) {
      throw new ArtifactExistsError(destination);
    }
    await platform.fs.rename(staging, destination);
    options.onProgress?.({ stage: "finalize", done: 1, total: 1 });
    return destination;
  } catch (error) {
    await weights.close().catch(() => undefined);
    await platform.fs.rm(staging, { recursive: true, force: true });
    throw error;
  }
}
