import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  MAX_SAFETENSORS_HEADER_BYTES,
  SafetensorsError,
  SafetensorsFile,
  parseSafetensorsHeader,
} from "../../src/onnx/safetensors.js";
import { safetensorsBytes } from "./template-fixture.js";

const directory = mkdtempSync(join(tmpdir(), "gliner-safetensors-"));
let counter = 0;
function file(content: Buffer): string {
  const path = join(directory, `f${(counter += 1)}.safetensors`);
  writeFileSync(path, content);
  return path;
}

const f32 = (...values: number[]): Buffer => Buffer.from(new Float32Array(values).buffer);

/** Builds a raw container from a header object and data, bypassing the writer's consistency. */
function raw(header: unknown, data: Buffer, declared?: bigint): Buffer {
  const text = JSON.stringify(header);
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(declared ?? BigInt(Buffer.byteLength(text)));
  return Buffer.concat([prefix, Buffer.from(text), data]);
}

test("parses the header and reads tensor ranges without loading the file", async () => {
  const path = file(
    safetensorsBytes(
      [
        { name: "a", shape: [2, 2], data: f32(1, 2, 3, 4) },
        { name: "b", shape: [3], data: f32(5, 6, 7) },
        { name: "empty", shape: [0, 4], data: Buffer.alloc(0) },
      ],
      { format: "pt" },
    ),
  );
  const opened = await SafetensorsFile.open(path);
  try {
    assert.deepEqual(opened.header.metadata, { format: "pt" });
    assert.equal(opened.header.dataBytes, 28);
    assert.equal(opened.header.tensors.size, 3);
    const b = opened.tensor("b");
    assert.deepEqual([b.dtype, b.shape, b.begin, b.end, b.byteLength, b.elements], ["F32", [3], 16, 28, 12, 3]);
    const target = Buffer.alloc(8, 0xee);
    await opened.readTensorRange("a", 4, 8, target);
    assert.deepEqual([...new Float32Array(target.buffer, target.byteOffset, 2)], [2, 3]);
    const into = Buffer.alloc(8, 0xee);
    await opened.readTensorRange("b", 4, 4, into, 2);
    assert.deepEqual([...into.subarray(2, 6)], [...f32(6)]);
    assert.equal(into[0], 0xee);
    assert.equal(into[6], 0xee);
    await opened.readTensorRange("empty", 0, 0, Buffer.alloc(0));

    await assert.rejects(opened.readTensorRange("a", 12, 8, target), /outside tensor "a"/);
    await assert.rejects(opened.readTensorRange("a", -1, 4, target), SafetensorsError);
    await assert.rejects(opened.readTensorRange("a", 0, 16, Buffer.alloc(4)), /does not fit/);
    await assert.rejects(opened.readTensorRange("nope", 0, 4, target), /no tensor "nope"/);
  } finally {
    await opened.close();
    await opened.close();
  }
});

const rejects: [string, () => Buffer, RegExp][] = [
  ["shorter than the length prefix", () => Buffer.alloc(4), /shorter than its 8-byte/],
  ["header longer than the file", () => raw({}, Buffer.alloc(0), 1000n), /past the end of the file/],
  ["absurd header length", () => raw({}, Buffer.alloc(0), BigInt(MAX_SAFETENSORS_HEADER_BYTES) + 1n), /exceeds the 100000000 byte limit/],
  ["header not starting with a brace", () => {
    const body = Buffer.from(" {}");
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(body.length));
    return Buffer.concat([prefix, body]);
  }, /must start with '\{'/],
  ["invalid UTF-8", () => {
    const body = Buffer.concat([Buffer.from("{\"a\":\""), Buffer.from([0xff, 0xfe]), Buffer.from("\"}")]);
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(body.length));
    return Buffer.concat([prefix, body]);
  }, /not valid UTF-8/],
  ["invalid JSON", () => {
    const body = Buffer.from("{not json");
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(body.length));
    return Buffer.concat([prefix, body]);
  }, /not valid JSON/],
  ["unknown dtype", () => raw({ a: { dtype: "F33", shape: [1], data_offsets: [0, 4] } }, f32(1)), /unsupported dtype "F33"/],
  ["dtype inherited from Object.prototype", () => raw({ a: { dtype: "constructor", shape: [1], data_offsets: [0, 4] } }, f32(1)), /unsupported dtype/],
  ["negative shape", () => raw({ a: { dtype: "F32", shape: [-1], data_offsets: [0, 4] } }, f32(1)), /invalid shape/],
  ["fractional shape", () => raw({ a: { dtype: "F32", shape: [1.5], data_offsets: [0, 4] } }, f32(1)), /invalid shape/],
  ["missing offsets", () => raw({ a: { dtype: "F32", shape: [1] } }, f32(1)), /invalid data_offsets/],
  ["offsets of the wrong arity", () => raw({ a: { dtype: "F32", shape: [1], data_offsets: [0] } }, f32(1)), /invalid data_offsets/],
  ["reversed offsets", () => raw({ a: { dtype: "F32", shape: [1], data_offsets: [4, 0] } }, f32(1)), /ending before they begin/],
  ["length not matching the shape", () => raw({ a: { dtype: "F32", shape: [2], data_offsets: [0, 4] } }, f32(1)), /require 8/],
  ["overflowing shape", () => raw({ a: { dtype: "F32", shape: [2 ** 40, 2 ** 40], data_offsets: [0, 4] } }, f32(1)), /overflowing/],
  ["gap between tensors", () => raw({ a: { dtype: "F32", shape: [1], data_offsets: [0, 4] }, b: { dtype: "F32", shape: [1], data_offsets: [8, 12] } }, f32(1, 2, 3)), /not contiguous at tensor "b"/],
  ["overlapping tensors", () => raw({ a: { dtype: "F32", shape: [2], data_offsets: [0, 8] }, b: { dtype: "F32", shape: [1], data_offsets: [4, 8] } }, f32(1, 2)), /not contiguous/],
  ["trailing bytes beyond the tensors", () => raw({ a: { dtype: "F32", shape: [1], data_offsets: [0, 4] } }, f32(1, 2)), /cover 4 bytes but the data section has 8/],
  ["truncated data", () => raw({ a: { dtype: "F32", shape: [2], data_offsets: [0, 8] } }, f32(1)), /cover 8 bytes but the data section has 4/],
  ["header that is not an object", () => raw([1], Buffer.alloc(0)), /must start with|must be a JSON object/],
  ["tensor that is not an object", () => raw({ a: 3 }, Buffer.alloc(0)), /must be an object/],
  ["non-string metadata", () => raw({ __metadata__: { a: 1 } }, Buffer.alloc(0)), /__metadata__ value/],
  ["metadata that is not an object", () => raw({ __metadata__: [] }, Buffer.alloc(0)), /__metadata__ must be an object/],
];
for (const [name, build, message] of rejects) {
  test(`rejects: ${name}`, async () => {
    await assert.rejects(SafetensorsFile.open(file(build())), (error: unknown) => {
      assert.ok(error instanceof SafetensorsError, String(error));
      assert.match(error.message, message);
      return true;
    });
  });
}

test("rejects a directory and reports a missing file", async () => {
  await assert.rejects(SafetensorsFile.open(directory), SafetensorsError);
  await assert.rejects(SafetensorsFile.open(join(directory, "missing")), { code: "ENOENT" });
});

test("a file truncated after the header was read fails the range read", async () => {
  const path = file(safetensorsBytes([{ name: "a", shape: [4], data: f32(1, 2, 3, 4) }]));
  const opened = await SafetensorsFile.open(path);
  try {
    writeFileSync(path, safetensorsBytes([{ name: "a", shape: [4], data: f32(1, 2, 3, 4) }]).subarray(0, 8 + opened.header.headerBytes + 6));
    await assert.rejects(opened.readTensorRange("a", 0, 16, Buffer.alloc(16)), /unexpected end/);
  } finally {
    await opened.close();
  }
});

test("zero-length tensors and scalars validate", () => {
  const text = JSON.stringify({ s: { dtype: "F32", shape: [], data_offsets: [0, 4] }, z: { dtype: "F32", shape: [0], data_offsets: [4, 4] } });
  const header = parseSafetensorsHeader(text, text.length, 8 + text.length + 4);
  assert.equal(header.tensors.get("s")?.elements, 1);
  assert.equal(header.tensors.get("z")?.byteLength, 0);
});
