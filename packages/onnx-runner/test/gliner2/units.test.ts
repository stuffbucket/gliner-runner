import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  OrderedObject,
  canUsePlainObject,
  cmpStr,
  compileSchema,
  decode,
  fromPlain,
  isIntegerLikeKey,
  normalizeText,
  pyRepr,
  pySum,
  schemaFromDict,
  serializeOrdered,
  splitWords,
  toPlain,
} from "../../src/gliner2/index.js";

describe("python parity helpers", () => {
  it("pySum is Neumaier-compensated like CPython 3.12", () => {
    assert.equal(pySum([]), 0);
    assert.equal(pySum([0.1, 0.2, 0.3]), 0.6); // plain left-to-right gives 0.6000000000000001
    assert.equal(0.1 + 0.2 + 0.3, 0.6000000000000001);
    assert.equal(pySum([1e100, 1.0, -1e100]), 1.0);
    assert.equal(pySum([-0.5]), -0.5);
  });

  it("cmpStr orders by code point, not UTF-16 code unit", () => {
    const astral = "\u{1F600}"; // surrogates D83D DE00 sort before U+FF5E in UTF-16
    const bmp = "～";
    assert.ok(astral < bmp, "sanity: JS compares code units");
    assert.equal(cmpStr(astral, bmp), 1);
    assert.equal(cmpStr(bmp, astral), -1);
    assert.equal(cmpStr("abc", "abc"), 0);
    assert.equal(cmpStr("ab", "abc"), -1);
    assert.equal(cmpStr("Z", "a"), -1);
    assert.deepEqual(["b", astral, bmp, "a"].sort(cmpStr), ["a", "b", bmp, astral]);
  });

  it("pyRepr follows Python quoting and escaping", () => {
    assert.equal(pyRepr("abc"), "'abc'");
    assert.equal(pyRepr("o'k"), `"o'k"`);
    assert.equal(pyRepr(`o'k"`), `'o\\'k"'`);
    assert.equal(pyRepr("a\nb\t\\"), "'a\\nb\\t\\\\'");
    assert.equal(pyRepr("\x00\x7f"), "'\\x00\\x7f'");
    assert.equal(pyRepr("é Ω 日本"), "'é Ω 日本'");
    assert.equal(pyRepr("​"), "'\\u200b'");
    assert.equal(pyRepr(" "), "'\\xa0'");
  });

  it("splitWords / normalizeText mirror the Python word splitter", () => {
    assert.equal(normalizeText(""), ".");
    assert.equal(normalizeText("ok"), "ok.");
    assert.equal(normalizeText("hi "), "hi .");
    assert.equal(normalizeText("Why?"), "Why?");
    assert.deepEqual(splitWords("Mail bob@Example.com, see https://x.y/z?q=1 or @user_1!"), [
      "mail",
      "bob@example.com",
      ",",
      "see",
      "https://x.y/z?q=1",
      "or",
      "@user_1",
      "!",
    ]);
    assert.deepEqual(splitWords("well-known snake_case; 3.5"), ["well-known", "snake_case", ";", "3", ".", "5"]);
    assert.deepEqual(splitWords("a b c\x1dd"), ["a", "b", "c", "d"]); // Python \s, not JS \s
    assert.deepEqual(splitWords("a﻿b"), ["a", "﻿", "b"]); // U+FEFF is not whitespace in Python
    assert.deepEqual(splitWords("İstanbul"), ["i̇stanbul"]);
  });
});

describe("ordered JSON", () => {
  it("serializes keys in insertion order, including integer-like keys", () => {
    const value = new OrderedObject([
      ["10", new OrderedObject([["b", 1], ["2", 2.5], ["1", null]])],
      ["2", ["x", true]],
    ]);
    assert.equal(serializeOrdered(value), '{"10":{"b":1,"2":2.5,"1":null},"2":["x",true]}');
    assert.equal(canUsePlainObject(value), false);
    assert.throws(() => JSON.stringify(value), /integer-like/);
    assert.notEqual(JSON.stringify(toPlain(value)), serializeOrdered(value)); // plain objects reorder them
  });

  it("plain-object round trip works for ordinary keys", () => {
    const value = fromPlain({ a: { z: 1, y: [1, 2] }, b: "s" });
    assert.equal(canUsePlainObject(value), true);
    assert.equal(JSON.stringify(value), serializeOrdered(value));
    assert.equal(isIntegerLikeKey("0"), true);
    assert.equal(isIntegerLikeKey("01"), false);
    assert.equal(isIntegerLikeKey("4294967295"), false);
    assert.equal(isIntegerLikeKey("-1"), false);
    assert.throws(() => serializeOrdered(Number.NaN), RangeError);
  });

  it("results keep Python key order for integer-like task and label names", () => {
    const compiled = compileSchema(
      schemaFromDict({
        tasks: [
          ["10", { labels: ["10", "2", "1"], min_labels: 0, max_labels: null, threshold: 0.5 }],
          ["2", { labels: new Map([["b", null], ["a", null]]), min_labels: 1, max_labels: 1 }],
        ],
      }),
    );
    const result = decode(compiled, [3, -3, 3, 0.2, 1.1], { candidateThreshold: 0.5, includeConfidence: true });
    const text = result.serialize();
    const parsedOrder = [...text.matchAll(/"(10|2|1|b|a|value|probabilities|_meta)":/g)].map((m) => m[1]);
    assert.deepEqual(parsedOrder.slice(0, 8), ["10", "value", "probabilities", "10", "2", "1", "2", "value"]);
    assert.ok(text.indexOf('"probabilities":{"10":') > 0);
    const plain = decode(compiled, [3, -3, 3, 0.2, 1.1], { candidateThreshold: 0.5, includeConfidence: false });
    assert.equal(plain.serialize(), '{"10":["10","1"],"2":"a"}');
  });
});

describe("result shapes", () => {
  it("scalar, list and empty shapes; _meta only when requested or infeasible", () => {
    const compiled = compileSchema(
      schemaFromDict({
        tasks: {
          one: { labels: ["a", "b"], min_labels: 1, max_labels: 1 },
          many: { labels: ["x", "y"], min_labels: 0, max_labels: null },
          withDefault: { labels: ["p", "q"], min_labels: 0, max_labels: null, default: "q" },
        },
      }),
    );
    const logits = [0.0, 1.0, -4, -4, -4, -1];
    const withConfidence = decode(compiled, logits, { candidateThreshold: 0.5, includeConfidence: true });
    const output = JSON.parse(withConfidence.serialize()) as Record<string, { value: unknown; confidence: number | null }>;
    assert.equal(output["one"]?.value, "b");
    assert.deepEqual(output["many"]?.value, []);
    assert.equal(output["many"]?.confidence, null); // empty multi-label selection without a default
    assert.equal(output["withDefault"]?.confidence !== null, true);
    assert.deepEqual(output["withDefault"]?.value, ["q"]);
    const plain = decode(compiled, logits, { candidateThreshold: 0.5, includeConfidence: false });
    assert.equal(plain.serialize(), '{"one":"b","many":[],"withDefault":["q"]}');
    assert.equal(plain.feasible, true);
  });

  it("an infeasible result always carries _meta, even without confidence", () => {
    const compiled = compileSchema(
      schemaFromDict({
        tasks: { t: { labels: ["a", "b"], min_labels: 1, max_labels: 2 }, u: { labels: ["x", "y"], min_labels: 1, max_labels: 2 } },
        constraints: [
          { type: "Implies", cond: { type: "LabelRef", task: "t", label: "a" }, then: { type: "LabelRef", task: "u", label: "x" } },
          { type: "Cardinality", task: "u", minimum: 0, maximum: 0 },
          { type: "LabelRef", task: "t", label: "a" },
        ],
      }),
    );
    const result = decode(compiled, [3, -3, 3, -3], { candidateThreshold: 0.5, includeConfidence: false });
    const out = JSON.parse(result.serialize()) as Record<string, unknown>;
    assert.deepEqual(Object.keys(out), ["t", "u", "_meta"]);
    const meta = out["_meta"] as { feasible: boolean; violations: string[]; decoder: string };
    assert.equal(meta.feasible, false);
    assert.equal(meta.decoder, "min_violations");
    assert.ok(meta.violations.every((v) => /^violated .+ \(tasks: [^)]+; weight 1\.0\)$/.test(v)));
  });
});

describe("decoder edge paths", () => {
  const config = { candidateThreshold: 0.0, includeConfidence: false } as const;

  it("maxCandidatesPerTask keeps only the highest-utility candidates", () => {
    const compiled = compileSchema(
      schemaFromDict({ tasks: { t: { labels: ["a", "b", "c", "d", "e"], min_labels: 0, max_labels: null } } }),
    );
    const logits = [5, 4, 3, -1, -2];
    assert.equal(decode(compiled, logits, config).serialize(), '{"t":["a","b","c"]}');
    assert.equal(decode(compiled, logits, { ...config, maxCandidatesPerTask: 2 }).serialize(), '{"t":["a","b"]}');
  });

  it("min_labels above the label count (unbounded task) falls back to the best achievable selection", () => {
    const compiled = compileSchema(
      schemaFromDict({ tasks: { t: { labels: ["a", "b"], min_labels: 3, max_labels: null } } }),
    );
    assert.equal(decode(compiled, [1, -1], config).serialize(), '{"t":["a","b"]}');
  });

  it("ordinal constraints, And and level labels appear in Python repr form in violations", () => {
    const compiled = compileSchema(
      schemaFromDict({
        tasks: { t: { labels: ["o'k", "b", "c"], ordered: true, min_labels: 0, max_labels: null } },
        constraints: [
          {
            type: "And",
            children: [
              { type: "MinLevel", task: "t", level: "c" },
              { type: "MaxLevel", task: "t", level: "o'k" },
            ],
          },
        ],
      }),
    );
    const result = decode(compiled, [-3, 2, -3], { candidateThreshold: 0.5, includeConfidence: false });
    assert.equal(result.feasible, false);
    assert.deepEqual(result.violations, [
      `violated And(children=(MinLevel(task='t', level='c'), MaxLevel(task='t', level="o'k"))) (tasks: t; weight 1.0)`,
    ]);
  });

  it("malformed label entries are reported with Python reprs", () => {
    assert.throws(() => schemaFromDict({ tasks: { t: { labels: [["a", 1]] } } }), /^Error: invalid label entry \['a', 1\]$|invalid label entry \['a', 1\]/);
    assert.throws(() => schemaFromDict({ tasks: { t: { labels: [null] } } }), /invalid label entry None/);
    assert.throws(() => schemaFromDict({ tasks: { t: { labels: ["a"], default: 5 } } }), /default 5 is not one of its labels/);
  });

  it("OrderedObject lookup helpers", () => {
    const o = new OrderedObject([["a", 1], ["b", [true]]]);
    assert.equal(o.get("a"), 1);
    assert.equal(o.get("zz"), undefined);
    assert.deepEqual(o.keys(), ["a", "b"]);
    assert.equal(o.serialize(), '{"a":1,"b":[true]}');
  });
});
