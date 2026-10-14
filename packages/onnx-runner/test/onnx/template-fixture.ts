import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { KnownModel } from "../../src/index.js";
import { MODEL, manifestFor, positionLogitsGraph, preprocessingConfig, syntheticTokenizer } from "./synthetic.js";

export const sha256 = (data: Buffer | string): string => createHash("sha256").update(data).digest("hex");

export interface FixtureTensor {
  name: string;
  /** Source shape in the safetensors file. */
  shape: number[];
  transform: "identity" | "transpose";
  /** Defaults to deterministic pseudo-random words (including NaN payloads and denormals). */
  words?: Uint32Array;
}

export function randomWords(count: number, seed: number): Uint32Array {
  const words = new Uint32Array(count);
  let state = (seed * 2654435761) >>> 0 || 1;
  for (let index = 0; index < count; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    words[index] = state;
  }
  if (count > 2) {
    words[0] = 0x7fc00001; // quiet NaN with a payload: must survive bit for bit
    words[1] = 0x00000001; // smallest denormal
  }
  return words;
}

/** Serialises tensors as a safetensors file (F32 unless `dtype` is given), padded like the reference writer. */
export function safetensorsBytes(
  tensors: { name: string; shape: number[]; data: Buffer; dtype?: string }[],
  metadata?: Record<string, string>,
): Buffer {
  const header: Record<string, unknown> = metadata === undefined ? {} : { __metadata__: metadata };
  let offset = 0;
  for (const tensor of tensors) {
    header[tensor.name] = {
      dtype: tensor.dtype ?? "F32",
      shape: tensor.shape,
      data_offsets: [offset, offset + tensor.data.length],
    };
    offset += tensor.data.length;
  }
  let text = JSON.stringify(header);
  text += " ".repeat((8 - ((text.length) % 8)) % 8);
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(Buffer.byteLength(text)));
  return Buffer.concat([prefix, Buffer.from(text), ...tensors.map((tensor) => tensor.data)]);
}

const wordsToBuffer = (words: Uint32Array): Buffer => Buffer.from(words.buffer, words.byteOffset, words.byteLength);

/** Independent reference: the bytes `model.onnx.data` must contain, computed element by element. */
export function expectedData(
  tensors: readonly (FixtureTensor & { words: Uint32Array })[],
  alignment: number,
): { data: Buffer; layout: { offset: number; length: number }[] } {
  const parts: Buffer[] = [];
  const layout: { offset: number; length: number }[] = [];
  let position = 0;
  for (const tensor of tensors) {
    const offset = Math.ceil(position / alignment) * alignment;
    parts.push(Buffer.alloc(offset - position));
    let output = tensor.words;
    if (tensor.transform === "transpose") {
      const [rows, cols] = tensor.shape as [number, number];
      output = new Uint32Array(rows * cols);
      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          output[col * rows + row] = tensor.words[row * cols + col] as number;
        }
      }
    }
    const bytes = wordsToBuffer(output);
    parts.push(bytes);
    layout.push({ offset, length: bytes.length });
    position = offset + bytes.length;
  }
  return { data: Buffer.concat(parts), layout };
}

export interface TemplateFixture {
  root: string;
  templatesDir: string;
  templateDir: string;
  snapshotDir: string;
  model: KnownModel;
  /** Files of the expected artifact by name (everything except manifest.json) and the manifest text. */
  artifact: Record<string, Buffer>;
  manifest: string;
  template: Record<string, any>;
  /** Source files by name, as served by a mock Hugging Face endpoint. */
  sources: Record<string, Buffer>;
}

export const DEFAULT_TENSORS: FixtureTensor[] = [
  { name: "embed", shape: [3, 5], transform: "identity" },
  { name: "proj", shape: [4, 6], transform: "transpose" },
  { name: "bias", shape: [7], transform: "identity" },
  { name: "wide", shape: [37, 70], transform: "transpose" },
  { name: "tail", shape: [1], transform: "identity" },
];

/**
 * Builds a self-consistent synthetic template package (template.json, graph.onnx, gliner2.json, manifest.json)
 * and a matching snapshot directory for the curated `decide-340m` pin, in fresh temporary directories.
 */
export function buildTemplateFixture(
  options: { tensors?: FixtureTensor[]; alignment?: number; model?: KnownModel; mutate?: (template: Record<string, any>) => void } = {},
): TemplateFixture {
  const model = options.model ?? MODEL;
  const alignment = options.alignment ?? 64;
  const tensors = (options.tensors ?? DEFAULT_TENSORS).map((tensor, index) => ({
    ...tensor,
    words: tensor.words ?? randomWords(tensor.shape.reduce((product, dim) => product * dim, 1), index + 1),
  }));
  const root = mkdtempSync(join(tmpdir(), "gliner-template-"));
  const snapshotDir = join(root, "snapshot");
  const templatesDir = join(root, "templates");
  const templateDir = join(templatesDir, model.repository.replace("/", "--"), model.pinnedRevision);
  mkdirSync(snapshotDir, { recursive: true });
  mkdirSync(templateDir, { recursive: true });

  const tokenizer = syntheticTokenizer();
  const sources: Record<string, Buffer> = {
    "model.safetensors": safetensorsBytes([
      // An unused tensor first, so the template does not simply read the file front to back.
      { name: "unused", shape: [2], data: Buffer.alloc(8, 1) },
      ...tensors.map((tensor) => ({ name: tensor.name, shape: tensor.shape, data: wordsToBuffer(tensor.words) })),
    ]),
    "tokenizer.json": Buffer.from(JSON.stringify(tokenizer.json)),
    "tokenizer_config.json": Buffer.from(JSON.stringify(tokenizer.config)),
  };
  for (const [name, content] of Object.entries(sources)) {
    writeFileSync(join(snapshotDir, name), content);
  }

  const { data, layout } = expectedData(tensors, alignment);
  const graph = positionLogitsGraph();
  const preprocessing = Buffer.from(JSON.stringify(preprocessingConfig(model)));
  const artifact: Record<string, Buffer> = {
    "gliner2.json": preprocessing,
    "model.onnx": graph,
    "model.onnx.data": data,
    "tokenizer.json": sources["tokenizer.json"] as Buffer,
    "tokenizer_config.json": sources["tokenizer_config.json"] as Buffer,
  };
  const manifest = manifestFor(model, artifact) as Record<string, any>;
  manifest["graph"].external_data = ["model.onnx.data"];
  const manifestText = JSON.stringify(manifest, null, 2);

  writeFileSync(join(templateDir, "graph.onnx"), graph);
  writeFileSync(join(templateDir, "gliner2.json"), preprocessing);
  writeFileSync(join(templateDir, "manifest.json"), manifestText);

  const describe = (path: string, content: Buffer | string) => ({
    path,
    size: Buffer.byteLength(content),
    sha256: sha256(Buffer.from(content)),
  });
  const template: Record<string, any> = {
    format_version: 1,
    kind: "gliner-runner-onnx-template",
    model_id: model.modelId,
    repository: model.repository,
    revision: model.pinnedRevision,
    source_files: [
      { ...describe("model.safetensors", sources["model.safetensors"] as Buffer), role: "weights" },
      { ...describe("tokenizer.json", sources["tokenizer.json"] as Buffer), role: "tokenizer" },
      { ...describe("tokenizer_config.json", sources["tokenizer_config.json"] as Buffer), role: "tokenizer" },
    ],
    template_files: [
      { ...describe("graph.onnx", graph), output: "model.onnx" },
      { ...describe("gliner2.json", preprocessing), output: "gliner2.json" },
      { ...describe("manifest.json", manifestText), output: "manifest.json" },
    ],
    data: {
      alignment,
      ...describe("model.onnx.data", data),
      tensors: tensors.map((tensor, index) => {
        const [rows, cols] = tensor.shape;
        return {
          name: `onnx::${tensor.name}`,
          source: tensor.name,
          transform: tensor.transform,
          dtype: "float32",
          source_shape: tensor.shape,
          shape: tensor.transform === "transpose" ? [cols, rows] : tensor.shape,
          offset: (layout[index] as { offset: number }).offset,
          length: (layout[index] as { length: number }).length,
        };
      }),
    },
    outputs: Object.entries(artifact).map(([path, content]) => describe(path, content)),
  };
  options.mutate?.(template);
  writeFileSync(join(templateDir, "template.json"), JSON.stringify(template, null, 2));
  return { root, templatesDir, templateDir, snapshotDir, model, artifact, manifest: manifestText, template, sources };
}

/** Writes `files` as a Hugging Face hub cache entry: `models--o--n/snapshots/<rev>/<file>` symlinked into `blobs/`. */
export function writeHfCache(
  hubDir: string,
  repository: string,
  revision: string,
  files: Record<string, Buffer>,
): string {
  const repoDir = join(hubDir, `models--${repository.replace("/", "--")}`);
  const snapshot = join(repoDir, "snapshots", revision);
  mkdirSync(snapshot, { recursive: true });
  mkdirSync(join(repoDir, "blobs"), { recursive: true });
  for (const [name, content] of Object.entries(files)) {
    const blob = join(repoDir, "blobs", sha256(content));
    writeFileSync(blob, content);
    symlinkSync(join("..", "..", "blobs", sha256(content)), join(snapshot, name));
  }
  return snapshot;
}

/** Writes `files` as a directory (Python provider layout or a plain snapshot directory). */
export function writeSnapshotDir(directory: string, files: Record<string, Buffer>): string {
  mkdirSync(directory, { recursive: true });
  for (const [name, content] of Object.entries(files)) {
    writeFileSync(join(directory, name), content);
  }
  return directory;
}
