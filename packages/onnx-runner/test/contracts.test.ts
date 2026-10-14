import assert from "node:assert/strict";
import test from "node:test";

import { BatchRequestSchema, InferenceRequestSchema } from "../src/contracts.js";
import { parseJsonBody, parseValue } from "../src/validation.js";
import { wireRequest } from "./helpers.js";

type Mutate = (body: Record<string, any>) => void;

function mutated(mutate: Mutate): Record<string, any> {
  const body = structuredClone(wireRequest()) as Record<string, any>;
  mutate(body);
  return body;
}

function errorsFor(body: unknown) {
  const result = parseJsonBody(InferenceRequestSchema, JSON.stringify(body));
  assert.equal(result.success, false);
  return result.success ? [] : result.errors;
}

test("accepts the schema key and fills pydantic-compatible defaults", () => {
  const parsed = InferenceRequestSchema.parse(wireRequest());
  assert.match(parsed.request_id, /^[0-9a-f]{8}-[0-9a-f]{4}-/);
  assert.deepEqual(parsed.options, {
    threshold: 0.5,
    output_format: "native",
    include_confidence: true,
  });
  assert.equal("schema_" in parsed, false);
  assert.deepEqual(parsed.schema, {
    kind: "classification",
    constraints: [],
    tasks: {
      priority: {
        labels: ["urgent", "routine"],
        label_descriptions: null,
        min_labels: 1,
        max_labels: 1,
        ordered: false,
        threshold: 0.5,
        activation: "auto",
        temperature: 1,
        default: null,
        instruction: null,
      },
    },
  });
});

test("canonicalizes request ids to lowercase", () => {
  const parsed = InferenceRequestSchema.parse(
    wireRequest({ request_id: "955B971B-4295-4E39-9E45-9B404E32BB6E" }),
  );
  assert.equal(parsed.request_id, "955b971b-4295-4e39-9e45-9b404e32bb6e");
});

test("label descriptions accept partial maps and nulls", () => {
  const parsed = InferenceRequestSchema.parse(
    mutated((body) => {
      body["schema"].tasks.priority.label_descriptions = { urgent: "needs action", routine: null };
    }),
  );
  assert.equal(parsed.schema.kind, "classification");
});

test("label descriptions reject undeclared keys and blank values", () => {
  const undeclared = errorsFor(
    mutated((body) => {
      body["schema"].tasks.priority.label_descriptions = { zz: "d", aa: "d" };
    }),
  );
  assert.equal(undeclared[0]?.msg, "Value error, label description keys must be declared labels: aa, zz");
  const blank = errorsFor(
    mutated((body) => {
      body["schema"].tasks.priority.label_descriptions = { urgent: "  ", routine: "ok" };
    }),
  );
  assert.equal(
    blank[0]?.msg,
    "Value error, label descriptions must be non-blank strings or null: urgent",
  );
});

test("rejects blank model and text using Unicode whitespace", () => {
  for (const field of ["model", "text"]) {
    const errors = errorsFor(mutated((body) => (body[field] = " \t  ")));
    assert.deepEqual(errors[0]?.loc, ["body", field]);
    assert.equal(errors[0]?.msg, "Value error, must not be blank");
  }
});

test("threshold bounds differ between task and options", () => {
  const task = (threshold: number) =>
    parseValue(
      InferenceRequestSchema,
      mutated((body) => (body["schema"].tasks.priority.threshold = threshold)),
      "body",
    ).success;
  assert.equal(task(0), false);
  assert.equal(task(1), false);
  assert.equal(task(0.25), true);
  const option = (threshold: number) =>
    parseValue(
      InferenceRequestSchema,
      mutated((body) => (body["options"] = { threshold })),
      "body",
    ).success;
  assert.equal(option(0), true);
  assert.equal(option(1), true);
  assert.equal(option(1.01), false);
});

test("batch requests are bounded to 1..256 items", () => {
  assert.equal(parseValue(BatchRequestSchema, { requests: [] }, "body").success, false);
  const many = Array.from({ length: 257 }, () => wireRequest());
  assert.equal(parseValue(BatchRequestSchema, { requests: many }, "body").success, false);
  assert.equal(parseValue(BatchRequestSchema, { requests: many.slice(0, 256) }, "body").success, true);
});

// Expected values were captured from the Python server (FastAPI 0.141 / pydantic 2.13).
const pythonShapes: [string, Mutate, { type: string; loc: (string | number)[]; msg: string }][] = [
  [
    "empty text",
    (body) => (body["text"] = ""),
    { type: "string_too_short", loc: ["body", "text"], msg: "String should have at least 1 character" },
  ],
  [
    "unknown field",
    (body) => (body["extra"] = 1),
    { type: "extra_forbidden", loc: ["body", "extra"], msg: "Extra inputs are not permitted" },
  ],
  [
    "bad backend",
    (body) => (body["backend"] = "foo"),
    { type: "enum", loc: ["body", "backend"], msg: "Input should be 'pytorch', 'mlx' or 'onnx'" },
  ],
  [
    "operation mismatch",
    (body) => (body["operation"] = "extract_entities"),
    {
      type: "value_error",
      loc: ["body", "schema"],
      msg: "Value error, extract_entities requires a entities schema",
    },
  ],
  [
    "missing model",
    (body) => delete body["model"],
    { type: "missing", loc: ["body", "model"], msg: "Field required" },
  ],
  [
    "unknown schema kind",
    (body) => (body["schema"].kind = "zzz"),
    {
      type: "union_tag_invalid",
      loc: ["body", "schema"],
      msg:
        "Input tag 'zzz' found using 'kind' does not match any of the expected tags: " +
        "'classification', 'entities', 'relations', 'structured'",
    },
  ],
  [
    "missing schema kind",
    (body) => delete body["schema"].kind,
    {
      type: "union_tag_not_found",
      loc: ["body", "schema"],
      msg: "Unable to extract tag using discriminator 'kind'",
    },
  ],
  [
    "duplicate labels",
    (body) => (body["schema"].tasks.priority.labels = ["a", "a"]),
    {
      type: "value_error",
      loc: ["body", "schema", "classification", "tasks", "priority", "labels"],
      msg: "Value error, classification labels must be unique",
    },
  ],
  [
    "no labels",
    (body) => (body["schema"].tasks.priority.labels = []),
    {
      type: "too_short",
      loc: ["body", "schema", "classification", "tasks", "priority", "labels"],
      msg: "Tuple should have at least 1 item after validation, not 0",
    },
  ],
  [
    "threshold too high",
    (body) => (body["schema"].tasks.priority.threshold = 1.5),
    {
      type: "less_than",
      loc: ["body", "schema", "classification", "tasks", "priority", "threshold"],
      msg: "Input should be less than 1",
    },
  ],
  [
    "no tasks",
    (body) => (body["schema"].tasks = {}),
    {
      type: "too_short",
      loc: ["body", "schema", "classification", "tasks"],
      msg: "Dictionary should have at least 1 item after validation, not 0",
    },
  ],
  [
    "option threshold",
    (body) => (body["options"] = { threshold: 2 }),
    {
      type: "less_than_equal",
      loc: ["body", "options", "threshold"],
      msg: "Input should be less than or equal to 1",
    },
  ],
  [
    "output format",
    (body) => (body["options"] = { output_format: "x" }),
    { type: "enum", loc: ["body", "options", "output_format"], msg: "Input should be 'native'" },
  ],
];

for (const [name, mutate, expected] of pythonShapes) {
  test(`422 detail matches the Python server: ${name}`, () => {
    const [error, ...rest] = errorsFor(mutated(mutate));
    assert.equal(rest.length, 0);
    assert.ok(error);
    assert.deepEqual({ type: error.type, loc: error.loc, msg: error.msg }, expected);
    assert.ok("input" in error);
  });
}

test("422 detail echoes input and ctx like pydantic", () => {
  const [duplicate] = errorsFor(mutated((body) => (body["schema"].tasks.priority.labels = ["a", "a"])));
  assert.deepEqual(duplicate?.input, ["a", "a"]);
  assert.deepEqual(duplicate?.ctx, { error: {} });
  const [tag] = errorsFor(mutated((body) => (body["schema"].kind = "zzz")));
  assert.deepEqual(tag?.ctx, {
    discriminator: "'kind'",
    tag: "zzz",
    expected_tags: "'classification', 'entities', 'relations', 'structured'",
  });
  const [short] = errorsFor(mutated((body) => (body["text"] = "")));
  assert.deepEqual(short?.ctx, { min_length: 1 });
  const [missing] = errorsFor(mutated((body) => delete body["model"]));
  assert.equal((missing?.input as Record<string, unknown>)["text"], "An urgent invoice");
});

test("batch validation errors carry the item index in loc", () => {
  const result = parseJsonBody(
    BatchRequestSchema,
    JSON.stringify({ requests: [wireRequest({ text: "" })] }),
  );
  assert.equal(result.success, false);
  assert.deepEqual(result.success ? [] : result.errors[0]?.loc, ["body", "requests", 0, "text"]);
  const tagged = parseJsonBody(
    BatchRequestSchema,
    JSON.stringify({
      requests: [wireRequest(), wireRequest({ schema: { kind: "classification", tasks: { p: { labels: [] } } } })],
    }),
  );
  assert.deepEqual(tagged.success ? [] : tagged.errors[0]?.loc, [
    "body",
    "requests",
    1,
    "schema",
    "classification",
    "tasks",
    "p",
    "labels",
  ]);
});

test("malformed, empty and non-object bodies", () => {
  const malformed = parseJsonBody(InferenceRequestSchema, "{bad");
  assert.equal(malformed.success, false);
  const [invalid] = malformed.success ? [] : malformed.errors;
  assert.equal(invalid?.type, "json_invalid");
  assert.deepEqual(invalid?.loc, ["body", 1]);
  assert.equal(invalid?.msg, "JSON decode error");
  assert.deepEqual(invalid?.input, {});

  const empty = parseJsonBody(InferenceRequestSchema, "");
  assert.deepEqual(empty.success ? [] : empty.errors, [
    { type: "missing", loc: ["body"], msg: "Field required", input: null },
  ]);

  const array = parseJsonBody(InferenceRequestSchema, "[1]");
  const [notObject] = array.success ? [] : array.errors;
  assert.equal(notObject?.type, "model_attributes_type");
  assert.deepEqual(notObject?.input, [1]);
});

test("__proto__ keys survive tasks, label descriptions, constraints and structured schemas", () => {
  const raw =
    '{"model":"decide-340m","backend":"onnx","precision":"fp32","operation":"classify","text":"t","schema":' +
    '{"kind":"classification","tasks":{"__proto__":{"labels":["__proto__","b"],' +
    '"label_descriptions":{"__proto__":"d","b":null}},"\\u0001x":{"labels":["a"]}},' +
    '"constraints":[{"type":"AnySelected","__proto__":{"__proto__":1},"task":"t"}]}}';
  const result = parseJsonBody(InferenceRequestSchema, raw);
  assert.equal(result.success, true, JSON.stringify(result));
  const schema = (result.success ? result.data.schema : undefined) as any;
  assert.deepEqual(Object.keys(schema.tasks), ["__proto__", "\u0001x"]);
  assert.equal(Object.getPrototypeOf(schema.tasks), Object.prototype);
  assert.deepEqual(Object.keys(schema.tasks["__proto__"].label_descriptions), ["__proto__", "b"]);
  assert.equal(Object.getOwnPropertyDescriptor(schema.tasks["__proto__"].label_descriptions, "__proto__")?.value, "d");
  assert.deepEqual(Object.keys(schema.constraints[0]), ["type", "__proto__", "task"]);
  assert.deepEqual(Object.keys(Object.getOwnPropertyDescriptor(schema.constraints[0], "__proto__")?.value), ["__proto__"]);
  assert.equal(JSON.stringify(schema.tasks["__proto__"].labels), '["__proto__","b"]');

  const structured = parseJsonBody(
    InferenceRequestSchema,
    '{"model":"m","backend":"onnx","precision":"fp32","operation":"extract_structured","text":"t",' +
      '"schema":{"kind":"structured","schema":{"__proto__":{"a":"str"}}}}',
  );
  assert.equal(structured.success, true);
  assert.deepEqual(Object.keys((structured as any).data.schema.schema), ["__proto__"]);
});

test("validation errors inside __proto__ records report the key as sent", () => {
  const errors = errorsFor(
    mutated((body) => {
      body["schema"].tasks = JSON.parse('{"__proto__":{"labels":["a"],"label_descriptions":{"__proto__":5}}}');
    }),
  );
  assert.equal(errors.length, 1);
  assert.deepEqual(errors[0]?.loc, [
    "body",
    "schema",
    "classification",
    "tasks",
    "__proto__",
    "label_descriptions",
    "__proto__",
  ]);
  assert.equal(errors[0]?.type, "string_type");
});
