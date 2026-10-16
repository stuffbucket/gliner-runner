import { platform } from "#platform";

import { SPECIAL_TOKENS, type SpecialTokenIds } from "../gliner2/index.js";
import { ModelIntegrityError } from "../errors.js";
import { PREPROCESSING_NAME } from "./manifest.js";

export interface PreprocessingConfig {
  specialTokenIds: SpecialTokenIds;
  unknownToken: string;
  unknownTokenId: number;
}

type Json = Record<string, unknown>;

const isRecord = (value: unknown): value is Json =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** What the TypeScript preprocessor implements; an artifact asking for anything else is not supported. */
const REQUIRED_BEHAVIOUR: ReadonlyArray<readonly [string, ReadonlyArray<string>, unknown]> = [
  ["format_version", ["format_version"], 1],
  ["kind", ["kind"], "gliner-runner-gliner2-preprocessing"],
  ["token pooling", ["model", "token_pooling"], "first"],
  ["tokenizer_json", ["tokenizer", "tokenizer_json"], "tokenizer.json"],
  ["tokenizer_config_json", ["tokenizer", "tokenizer_config_json"], "tokenizer_config.json"],
  ["encode_each_word_independently", ["tokenizer", "encode_each_word_independently"], true],
  ["add_special_tokens", ["tokenizer", "add_special_tokens"], false],
  ["word splitter", ["text", "word_splitter"], "whitespace"],
  ["lowercase_words", ["text", "lowercase_words"], true],
  ["lowercase_scope", ["text", "lowercase_scope"], "text-words-only"],
  ["max_words", ["text", "max_words"], null],
  ["empty_text", ["text", "normalization", "empty_text"], "."],
  ["terminal_characters", ["text", "normalization", "terminal_characters"], [".", "!", "?"]],
  ["append_when_missing", ["text", "normalization", "append_when_missing"], "."],
  ["padding_side", ["batching", "padding_side"], "right"],
  ["input_ids_pad_value", ["batching", "input_ids_pad_value"], 0],
  ["attention_mask_pad_value", ["batching", "attention_mask_pad_value"], 0],
  ["label_positions_pad_value", ["batching", "label_positions_pad_value"], 0],
  ["label_mask_pad_value", ["batching", "label_mask_pad_value"], false],
  ["label marker", ["label_positions", "marker"], "[L]"],
];

function lookup(root: Json, path: readonly string[]): unknown {
  let value: unknown = root;
  for (const key of path) {
    if (!isRecord(value)) {
      return undefined;
    }
    value = value[key];
  }
  return value;
}

/** Reads `gliner2.json` and checks it against the model identity and the implemented preprocessing. */
export async function readPreprocessingConfig(
  directory: string,
  expected: { modelId: string; repository: string; revision: string },
): Promise<PreprocessingConfig> {
  const fail = (detail: string): never => {
    throw new ModelIntegrityError(
      `ONNX artifact for '${expected.modelId}' at ${expected.revision} is invalid: ${PREPROCESSING_NAME} ${detail}`,
    );
  };
  let raw: unknown;
  try {
    raw = JSON.parse(await platform.fs.readFile(platform.join(directory, PREPROCESSING_NAME)));
  } catch {
    return fail("is not readable JSON");
  }
  if (!isRecord(raw)) {
    return fail("must be a JSON object");
  }
  for (const [label, path, value] of REQUIRED_BEHAVIOUR) {
    if (JSON.stringify(lookup(raw, path)) !== JSON.stringify(value)) {
      fail(`${label} must be ${JSON.stringify(value)}, found ${JSON.stringify(lookup(raw, path))}`);
    }
  }
  for (const key of ["model_id", "repository", "revision"] as const) {
    const wanted = key === "model_id" ? expected.modelId : expected[key];
    if (lookup(raw, ["model", key]) !== wanted) {
      fail(`model.${key} does not match ${wanted}`);
    }
  }
  const tokens = lookup(raw, ["special_tokens"]);
  if (!isRecord(tokens)) {
    return fail("special_tokens must be an object");
  }
  const ids: Record<string, number> = {};
  for (const token of SPECIAL_TOKENS) {
    const id = tokens[token];
    if (typeof id !== "number" || !Number.isSafeInteger(id) || id < 0) {
      fail(`special_tokens must give a non-negative id for ${token}`);
    }
    ids[token] = id as number;
  }
  if (Object.keys(tokens).length !== SPECIAL_TOKENS.length) {
    fail("special_tokens lists unexpected tokens");
  }
  const unknown = lookup(raw, ["tokenizer", "unknown_token", "id"]);
  if (typeof unknown !== "number" || !Number.isSafeInteger(unknown) || unknown < 0) {
    return fail("tokenizer.unknown_token.id must be a non-negative integer");
  }
  const unknownToken = lookup(raw, ["tokenizer", "unknown_token", "token"]);
  if (typeof unknownToken !== "string" || unknownToken.length === 0) {
    return fail("tokenizer.unknown_token.token must be a non-empty string");
  }
  return { specialTokenIds: ids as SpecialTokenIds, unknownToken, unknownTokenId: unknown };
}
