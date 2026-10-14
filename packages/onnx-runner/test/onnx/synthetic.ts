import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { KNOWN_MODELS, type KnownModel } from "../../src/index.js";
import { SPECIAL_TOKENS } from "../../src/gliner2/index.js";

export const MODEL: KnownModel = KNOWN_MODELS[0] as KnownModel;

// --- minimal ONNX protobuf writer: logits = Cast(label_positions, to=float) -------------------------------------
const varint = (value: number): number[] => {
  const bytes: number[] = [];
  let rest = value;
  while (rest > 0x7f) {
    bytes.push((rest & 0x7f) | 0x80);
    rest = Math.floor(rest / 128);
  }
  bytes.push(rest);
  return bytes;
};
const tag = (field: number, wire: number): number[] => varint(field * 8 + wire);
const bytesField = (field: number, payload: readonly number[]): number[] => [
  ...tag(field, 2),
  ...varint(payload.length),
  ...payload,
];
const textField = (field: number, text: string): number[] => bytesField(field, [...Buffer.from(text, "utf8")]);
const intField = (field: number, value: number): number[] => [...tag(field, 0), ...varint(value)];

function valueInfo(name: string, elementType: number, dims: readonly string[]): number[] {
  const shape = dims.flatMap((dim) => bytesField(1, textField(2, dim)));
  const tensor = [...intField(1, elementType), ...bytesField(2, shape)];
  return [...textField(1, name), ...bytesField(2, bytesField(1, tensor))];
}

/** A valid ONNX graph with the classification contract whose logits are the label positions as float32. */
export function positionLogitsGraph(): Buffer {
  const node = [
    ...textField(1, "label_positions"),
    ...textField(2, "logits"),
    ...textField(3, "cast"),
    ...textField(4, "Cast"),
    ...bytesField(5, [...textField(1, "to"), ...intField(20, 2), ...intField(3, 1)]),
  ];
  const graph = [
    ...bytesField(1, node),
    ...textField(2, "positions"),
    ...bytesField(11, valueInfo("input_ids", 7, ["batch", "sequence"])),
    ...bytesField(11, valueInfo("attention_mask", 7, ["batch", "sequence"])),
    ...bytesField(11, valueInfo("label_positions", 7, ["batch", "labels"])),
    ...bytesField(11, valueInfo("label_mask", 9, ["batch", "labels"])),
    ...bytesField(12, valueInfo("logits", 1, ["batch", "labels"])),
  ];
  const model = [...intField(1, 9), ...bytesField(8, intField(2, 20)), ...bytesField(7, graph)];
  return Buffer.from(model);
}

const WORDS = ["billing", "support", "urgent", "my", "card", "was", "charged", "twice", "(", ")", "intent", "."];

export function syntheticTokenizer(): { json: object; config: object } {
  const vocab: Record<string, number> = { "[UNK]": 0 };
  const added = SPECIAL_TOKENS.map((content, index) => ({
    id: index + 1,
    content,
    single_word: false,
    lstrip: false,
    rstrip: false,
    normalized: false,
    special: true,
  }));
  WORDS.forEach((word, index) => {
    vocab[word] = index + 20;
  });
  return {
    json: {
      version: "1.0",
      truncation: null,
      padding: null,
      added_tokens: added,
      normalizer: null,
      pre_tokenizer: { type: "Whitespace" },
      post_processor: null,
      decoder: null,
      model: { type: "WordLevel", vocab, unk_token: "[UNK]" },
    },
    config: { tokenizer_class: "PreTrainedTokenizerFast", unk_token: "[UNK]" },
  };
}

export function preprocessingConfig(model: KnownModel): Record<string, unknown> {
  return {
    format_version: 1,
    kind: "gliner-runner-gliner2-preprocessing",
    model: { model_id: model.modelId, repository: model.repository, revision: model.pinnedRevision, token_pooling: "first" },
    tokenizer: {
      tokenizer_json: "tokenizer.json",
      tokenizer_config_json: "tokenizer_config.json",
      encode_each_word_independently: true,
      add_special_tokens: false,
      unknown_token: { token: "[UNK]", id: 0 },
    },
    special_tokens: Object.fromEntries(SPECIAL_TOKENS.map((token, index) => [token, index + 1])),
    text: {
      normalization: { empty_text: ".", terminal_characters: [".", "!", "?"], append_when_missing: "." },
      word_splitter: "whitespace",
      lowercase_words: true,
      lowercase_scope: "text-words-only",
      max_words: null,
    },
    batching: {
      padding_side: "right",
      input_ids_pad_value: 0,
      attention_mask_pad_value: 0,
      label_positions_pad_value: 0,
      label_mask_pad_value: false,
    },
    label_positions: { marker: "[L]" },
  };
}

export interface SyntheticFiles {
  [name: string]: Buffer | string;
}

export function manifestFor(model: KnownModel, files: SyntheticFiles): Record<string, unknown> {
  return {
    format_version: 1,
    kind: "gliner-runner-onnx-classifier",
    model_id: model.modelId,
    repository: model.repository,
    revision: model.pinnedRevision,
    parameter_count: model.parameterCount,
    license: "apache-2.0",
    operations: ["classify"],
    precision: "fp32",
    quantization: "none",
    provider: "CPUExecutionProvider",
    opset: 20,
    opset_imports: [{ domain: "ai.onnx", version: 20 }],
    preprocessing: "gliner2.json",
    graph: {
      path: "model.onnx",
      external_data: [],
      inputs: [
        { name: "input_ids", type: "tensor(int64)", shape: ["batch", "sequence"] },
        { name: "attention_mask", type: "tensor(int64)", shape: ["batch", "sequence"] },
        { name: "label_positions", type: "tensor(int64)", shape: ["batch", "labels"] },
        { name: "label_mask", type: "tensor(bool)", shape: ["batch", "labels"] },
      ],
      outputs: [{ name: "logits", type: "tensor(float)", shape: ["batch", "labels"] }],
    },
    parity: {
      absolute_tolerance: 0.0001,
      max_abs_logit_difference: 5e-6,
      onnx_official_max_abs_logit_difference: 5.2e-6,
      rows: 19,
      fixture_suite: "synthetic-v1",
    },
    files: Object.entries(files)
      .map(([path, content]) => {
        const buffer = Buffer.from(content);
        return { path, size: buffer.length, sha256: createHash("sha256").update(buffer).digest("hex") };
      })
      .sort((a, b) => (a.path < b.path ? -1 : 1)),
  };
}

export interface SyntheticArtifact {
  modelsDir: string;
  directory: string;
  files: SyntheticFiles;
}

/** Writes a complete, valid artifact for `MODEL` into a fresh temporary models directory. */
export function writeSyntheticArtifact(
  options: { model?: KnownModel; files?: SyntheticFiles; manifest?: (manifest: Record<string, unknown>) => void } = {},
): SyntheticArtifact {
  const model = options.model ?? MODEL;
  const tokenizer = syntheticTokenizer();
  const files: SyntheticFiles = options.files ?? {
    "gliner2.json": JSON.stringify(preprocessingConfig(model)),
    "model.onnx": positionLogitsGraph(),
    "tokenizer.json": JSON.stringify(tokenizer.json),
    "tokenizer_config.json": JSON.stringify(tokenizer.config),
  };
  const modelsDir = mkdtempSync(join(tmpdir(), "gliner-onnx-"));
  const directory = join(modelsDir, "onnx", model.repository.replace("/", "--"), model.pinnedRevision);
  mkdirSync(directory, { recursive: true });
  for (const [name, content] of Object.entries(files)) {
    writeFileSync(join(directory, name), content);
  }
  const manifest = manifestFor(model, files);
  options.manifest?.(manifest);
  writeFileSync(join(directory, "manifest.json"), JSON.stringify(manifest, null, 2));
  return { modelsDir, directory, files };
}
