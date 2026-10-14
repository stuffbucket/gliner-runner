import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  Gliner2Preprocessor,
  Gliner2TokenizerError,
  SPECIAL_TOKENS,
  collateRows,
  compileSchema,
  schemaFromDict,
} from "../../src/gliner2/index.js";
import type { CompiledSchema, Gliner2Tokenizer } from "../../src/gliner2/index.js";
import { SNAPSHOT_DIRS, loadFixture, loadTokenizer } from "./support.js";

interface InputCase {
  model: string;
  batch_example?: boolean;
  text?: string;
  classifications?: Array<{
    task: string;
    labels: string[];
    multi_label: boolean;
    cls_threshold: number;
    class_act: "auto" | "sigmoid" | "softmax";
    prompt?: string;
    label_descriptions?: Record<string, string>;
    examples?: Array<[string, string]>;
  }>;
  input_ids: number[] | number[][];
  attention_sum?: number;
  attention_mask?: number[][];
  cls_marker_indices: number[] | number[][];
  cls_marker_mask?: boolean[][];
  cls_group_index: number[] | number[][];
}
interface InputFixture {
  models: string[];
  cases: InputCase[];
}
const inputGolden = loadFixture<InputFixture>("input-ids.json");
const tokenizerGolden = loadFixture<Record<string, Array<[string, number[]]>>>("tokenizer-ids.json");

/** Rebuilds a runner-equivalent schema from the model-schema entries stored in the golden. */
function compiledFor(entries: NonNullable<InputCase["classifications"]>): CompiledSchema {
  const tasks = entries.map((e) => {
    const labels = e.label_descriptions
      ? new Map(e.labels.map((l) => [l, e.label_descriptions?.[l] ?? null] as const))
      : e.labels;
    return [
      e.task,
      {
        labels,
        min_labels: e.multi_label ? 0 : 1,
        max_labels: e.multi_label ? null : 1,
        threshold: e.cls_threshold,
        activation: e.class_act,
        ...(e.prompt !== undefined ? { instruction: e.prompt } : {}),
        ...(e.examples !== undefined ? { examples: e.examples } : {}),
      },
    ] as const;
  });
  return compileSchema(schemaFromDict({ tasks }));
}

/** A tokenizer stub: every distinct string gets its own id; markers have fixed ids. */
function fakeTokenizer(): Gliner2Tokenizer {
  const ids = new Map<string, number>(SPECIAL_TOKENS.map((t, i) => [t, 100 + i]));
  let next = 1000;
  return {
    encode(text: string) {
      if (text === "") return { ids: [] };
      let id = ids.get(text);
      if (id === undefined) {
        id = next++;
        ids.set(text, id);
      }
      return { ids: [id] };
    },
    token_to_id: (t: string) => ids.get(t),
  };
}

describe("gliner2 input construction", () => {
  it("builds ids, label positions and groups with a stub tokenizer", () => {
    const compiled = compileSchema(
      schemaFromDict({
        tasks: {
          intent: { labels: ["billing", "support"], min_labels: 1, max_labels: 1 },
          tags: { labels: ["refund", "bug", "ui"], min_labels: 0, max_labels: null, instruction: "Pick tags" },
        },
      }),
    );
    const pre = new Gliner2Preprocessor(fakeTokenizer());
    const { words, markerGroup } = pre.combinedWords("Charged twice", compiled);
    assert.deepEqual(words, [
      "(", "[P]", "intent", "(", "[L]", "billing", "[L]", "support", ")", ")",
      "[SEP_STRUCT]",
      "(", "[P]", "tags: Pick tags", "(", "[L]", "refund", "[L]", "bug", "[L]", "ui", ")", ")",
      "[SEP_TEXT]", "charged", "twice", ".",
    ]);
    assert.deepEqual([...markerGroup], [[4, 0], [6, 0], [15, 1], [17, 1], [19, 1]]);
    const row = pre.buildInput("Charged twice", compiled);
    assert.equal(row.inputTokens, words.length); // one id per word with the stub
    assert.deepEqual(row.labelPositions, [4, 6, 15, 17, 19]);
    assert.deepEqual(row.labelGroups, [0, 0, 1, 1, 1]);
    assert.equal(row.inputIds[4], pre.specialTokenIds["[L]"]);
    assert.equal(row.inputIds[10], pre.specialTokenIds["[SEP_STRUCT]"]);
    assert.equal(row.inputIds[23], pre.specialTokenIds["[SEP_TEXT]"]);
  });

  it("rejects a tokenizer without single-id marker tokens", () => {
    const broken: Gliner2Tokenizer = { encode: () => ({ ids: [1, 2] }) };
    assert.throws(() => new Gliner2Preprocessor(broken), Gliner2TokenizerError);
    const missing: Gliner2Tokenizer = { encode: () => ({ ids: [1] }), token_to_id: () => undefined };
    assert.throws(() => new Gliner2Preprocessor(missing), /does not define/);
  });

  it("collates rows with right padding, pad value 0 and per-row counts", () => {
    const batch = collateRows([
      { inputIds: [5, 6, 7], labelPositions: [1], labelGroups: [0], inputTokens: 3 },
      { inputIds: [8, 9, 10, 11, 12], labelPositions: [1, 3], labelGroups: [0, 1], inputTokens: 5 },
    ]);
    assert.equal(batch.batchSize, 2);
    assert.equal(batch.seqLen, 5);
    assert.equal(batch.labelLen, 2);
    assert.ok(batch.inputIds instanceof BigInt64Array);
    assert.ok(batch.attentionMask instanceof BigInt64Array);
    assert.ok(batch.labelPositions instanceof BigInt64Array);
    assert.ok(batch.labelMask instanceof Uint8Array);
    assert.deepEqual([...batch.inputIds], [5n, 6n, 7n, 0n, 0n, 8n, 9n, 10n, 11n, 12n]);
    assert.deepEqual([...batch.attentionMask], [1n, 1n, 1n, 0n, 0n, 1n, 1n, 1n, 1n, 1n]);
    assert.deepEqual([...batch.labelPositions], [1n, 0n, 1n, 3n]);
    assert.deepEqual([...batch.labelMask], [1, 0, 1, 1]);
    assert.deepEqual(batch.inputTokenCounts, [3, 5]);
    assert.deepEqual(batch.labelCounts, [1, 2]);
    assert.throws(() => collateRows([]), RangeError);
  });

  for (const model of ["small", "decide", "multi", "1b"]) {
    const tokenizer = loadTokenizer(model);
    const skip = tokenizer === null ? `snapshot for ${model} not found under ${SNAPSHOT_DIRS[model]}` : false;

    it(`${model}: tokenizer ids match Python for the sampled strings`, { skip }, () => {
      const tk = tokenizer as Gliner2Tokenizer;
      const cases = tokenizerGolden[model] as Array<[string, number[]]>;
      assert.ok(cases.length > 300);
      const bad: string[] = [];
      for (const [text, ids] of cases) {
        const got = Array.from(tk.encode(text, { add_special_tokens: false }).ids, Number);
        if (JSON.stringify(got) !== JSON.stringify(ids)) bad.push(JSON.stringify(text));
      }
      assert.deepEqual(bad, []);
    });

    it(`${model}: marker tokens are single-id added tokens`, { skip }, () => {
      const pre = new Gliner2Preprocessor(tokenizer as Gliner2Tokenizer);
      assert.equal(new Set(Object.values(pre.specialTokenIds)).size, SPECIAL_TOKENS.length);
    });

    it(`${model}: input_ids / label positions / groups match the reference`, { skip }, () => {
      const pre = new Gliner2Preprocessor(tokenizer as Gliner2Tokenizer);
      const cases = inputGolden.cases.filter((c) => c.model === model && c.batch_example !== true);
      assert.ok(cases.length > 0);
      for (const c of cases) {
        const compiled = compiledFor(c.classifications as NonNullable<InputCase["classifications"]>);
        // The compiled prompt entries equal what Python's compiler emitted.
        compiled.classifications.forEach((entry, i) => {
          const golden = (c.classifications as NonNullable<InputCase["classifications"]>)[i];
          assert.equal(entry.task, golden?.task);
          assert.deepEqual(entry.labels, golden?.labels);
          assert.equal(entry.prompt, golden?.prompt);
          assert.deepEqual(
            entry.labelDescriptions === undefined ? undefined : Object.fromEntries(entry.labelDescriptions),
            golden?.label_descriptions,
          );
          assert.deepEqual(entry.examples, golden?.examples);
        });
        const row = pre.buildInput(c.text as string, compiled);
        assert.deepEqual(row.inputIds, c.input_ids, `${model} ${JSON.stringify(c.text)}`);
        assert.deepEqual(row.labelPositions, c.cls_marker_indices);
        assert.deepEqual(row.labelGroups, c.cls_group_index);
        assert.equal(row.inputTokens, c.attention_sum);
      }
    });

    const batchCase = inputGolden.cases.find((x) => x.model === model && x.batch_example === true);
    it(`${model}: padded two-row batch matches the reference`, { skip: skip || (batchCase === undefined ? "no batch golden for this model" : false) }, () => {
      const pre = new Gliner2Preprocessor(tokenizer as Gliner2Tokenizer);
      const c = batchCase as InputCase;
      const compiled = compileSchema(schemaFromDict({ tasks: { intent: { labels: ["billing", "support"] } } }));
      const rows = ["Short one", "A much longer text with many more words in it, really."].map((t) => pre.buildInput(t, compiled));
      const batch = pre.collate(rows);
      const num = (a: ArrayLike<bigint | number>): number[] => Array.from(a, Number);
      assert.deepEqual(num(batch.inputIds), (c.input_ids as number[][]).flat());
      assert.deepEqual(num(batch.attentionMask), (c.attention_mask as number[][]).flat());
      assert.deepEqual(num(batch.labelPositions), (c.cls_marker_indices as number[][]).flat());
      assert.deepEqual(num(batch.labelMask), (c.cls_marker_mask as boolean[][]).flat().map(Number));
      assert.deepEqual(batch.inputTokenCounts, rows.map((r) => r.inputIds.length));
    });
  }
});
