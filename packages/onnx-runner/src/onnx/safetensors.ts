import { platform } from "#platform";

import type { PlatformFile } from "../platform/port.js";

/**
 * Streaming reader for the safetensors container format: an 8-byte little-endian header length, a JSON header
 * mapping tensor names to `{dtype, shape, data_offsets}`, then a flat byte buffer. Only the header is ever
 * loaded; tensor bytes are read on demand as bounded ranges, so multi-gigabyte weight files are never held in
 * memory.
 */

/** Bytes per element of every dtype the format defines. Only `F32` is accepted by the assembler. */
const DTYPE_SIZES: Readonly<Record<string, number>> = {
  BOOL: 1,
  U8: 1,
  I8: 1,
  F8_E5M2: 1,
  F8_E4M3: 1,
  I16: 2,
  U16: 2,
  F16: 2,
  BF16: 2,
  I32: 4,
  U32: 4,
  F32: 4,
  F64: 8,
  I64: 8,
  U64: 8,
};

/** The reference implementation refuses headers above 100 MB; so do we. */
export const MAX_SAFETENSORS_HEADER_BYTES = 100_000_000;

export class SafetensorsError extends Error {
  override readonly name = "SafetensorsError";
}

export interface SafetensorsTensor {
  name: string;
  dtype: string;
  shape: readonly number[];
  /** Offset of the first byte, relative to the start of the data section. */
  begin: number;
  /** Offset one past the last byte, relative to the start of the data section. */
  end: number;
  /** `end - begin`. */
  byteLength: number;
  /** Number of elements (the product of `shape`; 1 for a scalar). */
  elements: number;
}

export interface SafetensorsHeader {
  /** Byte length of the JSON header. */
  headerBytes: number;
  /** Absolute file offset of the data section (8 + headerBytes). */
  dataStart: number;
  /** Byte length of the data section. */
  dataBytes: number;
  tensors: ReadonlyMap<string, SafetensorsTensor>;
  metadata: Readonly<Record<string, string>>;
}

type Json = Record<string, unknown>;

const isRecord = (value: unknown): value is Json =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * Validates a header exactly as the reference implementation does: `data_offsets` must tile the data section
 * from byte 0 to `dataBytes` with no gaps or overlaps (in offset order), and every tensor's byte length must
 * equal `product(shape) * sizeof(dtype)`.
 */
export function parseSafetensorsHeader(headerText: string, headerBytes: number, fileSize: number): SafetensorsHeader {
  const dataStart = 8 + headerBytes;
  const dataBytes = fileSize - dataStart;
  if (dataBytes < 0) {
    throw new SafetensorsError("safetensors header extends past the end of the file");
  }
  let raw: unknown;
  try {
    raw = JSON.parse(headerText);
  } catch {
    throw new SafetensorsError("safetensors header is not valid JSON");
  }
  if (!isRecord(raw)) {
    throw new SafetensorsError("safetensors header must be a JSON object");
  }
  const metadata: Record<string, string> = {};
  const tensors: SafetensorsTensor[] = [];
  for (const [name, entry] of Object.entries(raw)) {
    if (name === "__metadata__") {
      if (!isRecord(entry)) {
        throw new SafetensorsError("safetensors __metadata__ must be an object");
      }
      for (const [key, value] of Object.entries(entry)) {
        if (typeof value !== "string") {
          throw new SafetensorsError(`safetensors __metadata__ value for ${JSON.stringify(key)} must be a string`);
        }
        metadata[key] = value;
      }
      continue;
    }
    const where = `safetensors tensor ${JSON.stringify(name)}`;
    if (!isRecord(entry)) {
      throw new SafetensorsError(`${where} must be an object`);
    }
    const { dtype, shape, data_offsets: offsets } = entry;
    if (typeof dtype !== "string" || !Object.hasOwn(DTYPE_SIZES, dtype)) {
      throw new SafetensorsError(`${where} has unsupported dtype ${JSON.stringify(dtype)}`);
    }
    if (!Array.isArray(shape) || shape.some((dim) => typeof dim !== "number" || !Number.isSafeInteger(dim) || dim < 0)) {
      throw new SafetensorsError(`${where} has an invalid shape`);
    }
    if (
      !Array.isArray(offsets) ||
      offsets.length !== 2 ||
      offsets.some((value) => typeof value !== "number" || !Number.isSafeInteger(value) || value < 0)
    ) {
      throw new SafetensorsError(`${where} has invalid data_offsets`);
    }
    const [begin, end] = offsets as [number, number];
    if (end < begin) {
      throw new SafetensorsError(`${where} has data_offsets ending before they begin`);
    }
    let elements = 1;
    for (const dim of shape as number[]) {
      elements *= dim;
      if (!Number.isSafeInteger(elements)) {
        throw new SafetensorsError(`${where} has an overflowing shape`);
      }
    }
    const expected = elements * (DTYPE_SIZES[dtype] as number);
    if (!Number.isSafeInteger(expected) || end - begin !== expected) {
      throw new SafetensorsError(
        `${where} spans ${end - begin} bytes but its shape and dtype require ${expected}`,
      );
    }
    tensors.push({ name, dtype, shape: [...(shape as number[])], begin, end, byteLength: end - begin, elements });
  }

  const ordered = [...tensors].sort((left, right) => left.begin - right.begin || left.end - right.end);
  let cursor = 0;
  for (const tensor of ordered) {
    if (tensor.begin !== cursor) {
      throw new SafetensorsError(
        `safetensors data offsets are not contiguous at tensor ${JSON.stringify(tensor.name)} ` +
          `(expected offset ${cursor}, found ${tensor.begin})`,
      );
    }
    cursor = tensor.end;
  }
  if (cursor !== dataBytes) {
    throw new SafetensorsError(
      `safetensors tensors cover ${cursor} bytes but the data section has ${dataBytes}`,
    );
  }
  return {
    headerBytes,
    dataStart,
    dataBytes,
    tensors: new Map(tensors.map((tensor) => [tensor.name, tensor])),
    metadata,
  };
}

/** An open safetensors file. Always `close()` it. */
export class SafetensorsFile {
  readonly header: SafetensorsHeader;
  readonly #handle: PlatformFile;
  #closed = false;

  private constructor(handle: PlatformFile, header: SafetensorsHeader) {
    this.#handle = handle;
    this.header = header;
  }

  static async open(path: string): Promise<SafetensorsFile> {
    // Snapshot directories legitimately contain symlinks into a blob store, so links are followed; the
    // content is verified by hash before (or while) it is used.
    const handle = await platform.fs.open(path, "read");
    try {
      const status = await handle.stat();
      if (!status.isFile()) {
        throw new SafetensorsError(`${path} is not a regular file`);
      }
      if (status.size < 8) {
        throw new SafetensorsError("safetensors file is shorter than its 8-byte length prefix");
      }
      const prefix = new Uint8Array(8);
      await readFully(handle, prefix, 0, 0, 8);
      const declared = new DataView(prefix.buffer).getBigUint64(0, true);
      if (declared > BigInt(MAX_SAFETENSORS_HEADER_BYTES)) {
        throw new SafetensorsError(`safetensors header of ${declared} bytes exceeds the ${MAX_SAFETENSORS_HEADER_BYTES} byte limit`);
      }
      const headerBytes = Number(declared);
      if (8 + headerBytes > status.size) {
        throw new SafetensorsError("safetensors header extends past the end of the file");
      }
      const text = new Uint8Array(headerBytes);
      await readFully(handle, text, 0, 8, headerBytes);
      if (text[0] !== 0x7b) {
        throw new SafetensorsError("safetensors header must start with '{'");
      }
      let decoded: string;
      try {
        decoded = new TextDecoder("utf-8", { fatal: true }).decode(text);
      } catch {
        throw new SafetensorsError("safetensors header is not valid UTF-8");
      }
      return new SafetensorsFile(handle, parseSafetensorsHeader(decoded, headerBytes, status.size));
    } catch (error) {
      await handle.close();
      throw error;
    }
  }

  tensor(name: string): SafetensorsTensor {
    const tensor = this.header.tensors.get(name);
    if (tensor === undefined) {
      throw new SafetensorsError(`safetensors file has no tensor ${JSON.stringify(name)}`);
    }
    return tensor;
  }

  /**
   * Reads `length` bytes starting `offset` bytes into the named tensor into `target[targetOffset..]`.
   * Refuses ranges outside the tensor, so a caller bug can never read a neighbouring tensor.
   */
  async readTensorRange(
    name: string,
    offset: number,
    length: number,
    target: Uint8Array,
    targetOffset = 0,
  ): Promise<void> {
    const tensor = this.tensor(name);
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0) {
      throw new SafetensorsError("tensor range must be non-negative integers");
    }
    if (offset + length > tensor.byteLength) {
      throw new SafetensorsError(
        `range ${offset}+${length} is outside tensor ${JSON.stringify(name)} (${tensor.byteLength} bytes)`,
      );
    }
    if (targetOffset < 0 || targetOffset + length > target.length) {
      throw new SafetensorsError("tensor range does not fit in the target buffer");
    }
    await readFully(this.#handle, target, targetOffset, this.header.dataStart + tensor.begin + offset, length);
  }

  async close(): Promise<void> {
    if (!this.#closed) {
      this.#closed = true;
      await this.#handle.close();
    }
  }
}

/** `FileHandle.read` may return short; a short read before EOF is retried and EOF is an error. */
async function readFully(
  handle: PlatformFile,
  target: Uint8Array,
  targetOffset: number,
  position: number,
  length: number,
): Promise<void> {
  let done = 0;
  while (done < length) {
    const { bytesRead } = await handle.read(target, targetOffset + done, length - done, position + done);
    if (bytesRead === 0) {
      throw new SafetensorsError("unexpected end of safetensors file");
    }
    done += bytesRead;
  }
}
