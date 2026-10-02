import assert from "node:assert/strict";
import test from "node:test";

import {
  GlinerClient,
  InferenceRequestSchema,
} from "./dist/index.js";

const request = {
  model: "decide-340m",
  backend: "pytorch",
  precision: "fp32",
  operation: "classify",
  text: "An urgent invoice",
  schema: {
    kind: "classification",
    tasks: { priority: { labels: ["urgent", "routine"] } },
  },
};

test("validates requests before transport", async () => {
  let called = false;
  const client = new GlinerClient({
    fetch: async () => {
      called = true;
      return new Response("{}");
    },
  });

  await assert.rejects(
    client.infer({ ...request, text: "" }),
    /text must not be blank/,
  );
  assert.equal(called, false);
});

test("validates successful server responses", async () => {
  const client = new GlinerClient({
    fetch: async () =>
      Response.json({
        request_id: "955b971b-4295-4e39-9e45-9b404e32bb6e",
        model: "decide-340m",
        backend: "pytorch",
        precision: "fp32",
        output: { priority: "urgent" },
        error: null,
        timing: { queue_ms: 1, inference_ms: 2 },
      }),
  });

  const response = await client.infer(request);
  assert.deepEqual(response.output, { priority: "urgent" });
});

test("rejects malformed successful responses", async () => {
  const client = new GlinerClient({
    fetch: async () => Response.json({ output: "missing metadata" }),
  });

  await assert.rejects(client.infer(request), /Invalid input/);
});

test("exports schemas for Electron boundaries", () => {
  assert.equal(InferenceRequestSchema.parse(request).operation, "classify");
  assert.throws(
    () =>
      InferenceRequestSchema.parse({
        ...request,
        operation: "extract_entities",
      }),
    /requires a entities schema/,
  );
});
