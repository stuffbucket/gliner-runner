/**
 * Prompt / token-sequence construction and batch padding (gliner2/processor.py, processing/word_splitter.py).
 * Spec: dist/port-spec/gliner2-classification.md sections 4 and 5.
 */
import { PY_WS_CLASS } from "./pyparity.js";
import type { CompiledSchema } from "./types.js";

// Python `\s` (NOT JS `\s`), `\w` plus combining marks, and the U+0130/U+0131 IGNORECASE quirk (spec section 4.2).
const WS = PY_WS_CLASS;
const WORD = String.raw`[\p{L}\p{N}_\p{M}‌‍]`;
const AZ = String.raw`a-zİı`;
export const WORD_PATTERN_SOURCE =
  String.raw`(?:https?://[^${WS}]+|www\.[^${WS}]+)` +
  String.raw`|[${AZ}0-9._%+-]+@[${AZ}0-9.-]+\.[${AZ}]{2,}` +
  String.raw`|@[${AZ}0-9_]+` +
  String.raw`|${WORD}+(?:[-_]${WORD}+)*` +
  String.raw`|[^${WS}]`;

/** Python `_normalize_text`: empty -> ".", append "." unless the text ends with . ! or ?. */
export function normalizeText(text: string): string {
  if (text.length === 0) return ".";
  return /[.!?]$/.test(text) ? text : text + ".";
}

/** The default "whitespace" word splitter: lowercased words of the original text. */
export function splitWords(text: string): string[] {
  const re = new RegExp(WORD_PATTERN_SOURCE, "giu");
  return Array.from(text.matchAll(re), (m) => m[0].toLowerCase());
}

/** Structural subset of `@huggingface/tokenizers` `Tokenizer` that the preprocessor needs. */
export interface Gliner2Tokenizer {
  encode(text: string, options?: { add_special_tokens?: boolean }): { readonly ids: ArrayLike<number | bigint> };
  token_to_id?(token: string): number | undefined;
}

export const SPECIAL_TOKENS = [
  "[SEP_STRUCT]",
  "[SEP_TEXT]",
  "[P]",
  "[C]",
  "[E]",
  "[R]",
  "[L]",
  "[EXAMPLE]",
  "[OUTPUT]",
  "[DESCRIPTION]",
] as const;
export type SpecialToken = (typeof SPECIAL_TOKENS)[number];
export type SpecialTokenIds = Readonly<Record<SpecialToken, number>>;

/** Raised when the tokenizer does not carry the ten GLiNER2 marker tokens as single-id added tokens. */
export class Gliner2TokenizerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "Gliner2TokenizerError";
  }
}

/** One encoded request (a single row of the batch). */
export interface Gliner2Row {
  /** Subword ids of the combined sequence (no CLS/SEP/BOS/EOS). */
  readonly inputIds: readonly number[];
  /** Position in `inputIds` of every `[L]` marker, all tasks concatenated in task then label order. */
  readonly labelPositions: readonly number[];
  /** Task index (declaration order) of each entry of `labelPositions`. */
  readonly labelGroups: readonly number[];
  /** Exact input token count for usage accounting: `inputIds.length` (= sum of the attention mask). */
  readonly inputTokens: number;
}

/**
 * Padded batch in ONNX input layout (right padding, pad value 0 for every tensor).
 * Tensor names: inputIds -> `input_ids`, attentionMask -> `attention_mask`, labelPositions -> `label_positions`,
 * labelMask -> `label_mask` (bool; `Uint8Array` 0/1).
 */
export interface Gliner2Batch {
  readonly batchSize: number;
  /** Padded sequence length S. */
  readonly seqLen: number;
  /** Padded label count L. */
  readonly labelLen: number;
  /** int64 [B, S] */
  readonly inputIds: BigInt64Array;
  /** int64 [B, S] */
  readonly attentionMask: BigInt64Array;
  /** int64 [B, L] */
  readonly labelPositions: BigInt64Array;
  /** bool [B, L] */
  readonly labelMask: Uint8Array;
  /** Per-row input token counts. */
  readonly inputTokenCounts: readonly number[];
  /** Per-row real label counts (`sum(labelMask[row])`); the logits of a row beyond this count are padding. */
  readonly labelCounts: readonly number[];
}

export interface Gliner2PreprocessorOptions {
  /** Max entries of the per-string token cache (Python `lru_cache`). Default 8192. */
  readonly cacheSize?: number;
}

/**
 * Builds model inputs for a compiled schema. Every element of the combined word sequence is tokenized
 * independently with `add_special_tokens: false`; there is no CLS/SEP and no truncation.
 */
export class Gliner2Preprocessor {
  /** Ids of the ten marker tokens, resolved and verified at construction. */
  readonly specialTokenIds: SpecialTokenIds;
  private readonly cache = new Map<string, readonly number[]>();
  private readonly cacheSize: number;

  constructor(
    private readonly tokenizer: Gliner2Tokenizer,
    options: Gliner2PreprocessorOptions = {},
  ) {
    this.cacheSize = options.cacheSize ?? 8192;
    const ids = {} as Record<SpecialToken, number>;
    for (const token of SPECIAL_TOKENS) {
      const encoded = this.encodeUncached(token);
      const looked = tokenizer.token_to_id?.(token);
      if (looked === undefined && tokenizer.token_to_id !== undefined) {
        throw new Gliner2TokenizerError(`tokenizer does not define the GLiNER2 marker token ${token}`);
      }
      const id = encoded[0];
      if (encoded.length !== 1 || id === undefined || (looked !== undefined && looked !== id)) {
        throw new Gliner2TokenizerError(
          `GLiNER2 marker token ${token} must encode to exactly one id (got [${encoded.join(", ")}])`,
        );
      }
      ids[token] = id;
    }
    this.specialTokenIds = ids;
  }

  private encodeUncached(token: string): number[] {
    return Array.from(this.tokenizer.encode(token, { add_special_tokens: false }).ids, Number);
  }

  private encode(token: string): readonly number[] {
    const hit = this.cache.get(token);
    if (hit !== undefined) return hit;
    const ids = this.encodeUncached(token);
    if (this.cache.size >= this.cacheSize) this.cache.clear();
    this.cache.set(token, ids);
    return ids;
  }

  /** The "words" of the combined sequence for one text (schema tokens for all tasks, separators, lowercased text words). */
  combinedWords(text: string, compiled: CompiledSchema): { words: string[]; markerGroup: Map<number, number> } {
    const groups = compiled.classifications.map((e) => {
      let prompt = e.prompt !== undefined ? `${e.task}: ${e.prompt}` : e.task;
      for (const [label, description] of e.labelDescriptions ?? []) {
        if (e.labels.includes(label)) prompt += ` [DESCRIPTION] ${label}: ${description}`;
      }
      for (const [input, output] of e.examples ?? []) {
        if (e.labels.includes(output)) prompt += ` [EXAMPLE] ${input} [OUTPUT] ${output}`;
      }
      const tokens: string[] = ["(", "[P]", prompt, "("];
      for (const label of e.labels) tokens.push("[L]", label);
      tokens.push(")", ")");
      return tokens;
    });
    const words: string[] = [];
    groups.forEach((g, i) => {
      words.push(...g);
      if (i < groups.length - 1) words.push("[SEP_STRUCT]");
    });
    words.push("[SEP_TEXT]");
    for (const w of splitWords(normalizeText(text))) words.push(w);

    // [L] markers of group g sit at group offset k = 4, 6, ..., len(g) - 3; offsets advance by len(g) + 1.
    const markerGroup = new Map<number, number>();
    let off = 0;
    groups.forEach((g, gi) => {
      for (let k = 4; k < g.length - 2; k += 2) markerGroup.set(off + k, gi);
      off += g.length + 1;
    });
    return { words, markerGroup };
  }

  /** Builds the id sequence and label positions of one text. */
  buildInput(text: string, compiled: CompiledSchema): Gliner2Row {
    const { words, markerGroup } = this.combinedWords(text, compiled);
    const inputIds: number[] = [];
    const labelPositions: number[] = [];
    const labelGroups: number[] = [];
    words.forEach((word, index) => {
      const group = markerGroup.get(index);
      if (group !== undefined) {
        labelPositions.push(inputIds.length);
        labelGroups.push(group);
      }
      for (const id of this.encode(word)) inputIds.push(id);
    });
    return { inputIds, labelPositions, labelGroups, inputTokens: inputIds.length };
  }

  /** Pads rows exactly like `SchemaTransformer._pad_batch` (right padding, pad value 0, no sorting). */
  collate(rows: readonly Gliner2Row[]): Gliner2Batch {
    return collateRows(rows);
  }
}

/** Standalone form of {@link Gliner2Preprocessor.collate}. */
export function collateRows(rows: readonly Gliner2Row[]): Gliner2Batch {
  if (rows.length === 0) throw new RangeError("cannot collate an empty batch");
  const batchSize = rows.length;
  let seqLen = 0;
  let labelLen = 0;
  for (const r of rows) {
    seqLen = Math.max(seqLen, r.inputIds.length);
    labelLen = Math.max(labelLen, r.labelPositions.length);
  }
  const inputIds = new BigInt64Array(batchSize * seqLen);
  const attentionMask = new BigInt64Array(batchSize * seqLen);
  const labelPositions = new BigInt64Array(batchSize * labelLen);
  const labelMask = new Uint8Array(batchSize * labelLen);
  rows.forEach((row, b) => {
    row.inputIds.forEach((id, i) => {
      inputIds[b * seqLen + i] = BigInt(id);
      attentionMask[b * seqLen + i] = 1n;
    });
    row.labelPositions.forEach((p, i) => {
      labelPositions[b * labelLen + i] = BigInt(p);
      labelMask[b * labelLen + i] = 1;
    });
  });
  return {
    batchSize,
    seqLen,
    labelLen,
    inputIds,
    attentionMask,
    labelPositions,
    labelMask,
    inputTokenCounts: rows.map((r) => r.inputIds.length),
    labelCounts: rows.map((r) => r.labelPositions.length),
  };
}
