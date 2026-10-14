import { z } from "zod";

import { BatchRequestSchema, DownloadRequestSchema, InferenceRequestSchema } from "./contracts.js";

export const API_VERSION = "1.0.0";

function jsonSchema(schema: z.ZodType): Record<string, unknown> {
  const { $schema: _dialect, ...rest } = z.toJSONSchema(schema, {
    io: "input",
    unrepresentable: "any",
  }) as Record<string, unknown>;
  return rest;
}

const validationError = { description: "Validation Error" };
const domainError = (description: string) => ({ description });

function operation(
  summary: string,
  tag: string,
  responses: Record<string, unknown>,
  requestBody?: Record<string, unknown>,
): Record<string, unknown> {
  return {
    summary,
    tags: [tag],
    ...(requestBody === undefined
      ? {}
      : { requestBody: { required: true, content: { "application/json": { schema: requestBody } } } }),
    responses,
  };
}

const ok = { description: "Successful Response" };

/** Machine-readable route index; request bodies are derived from the runtime validators. */
export function buildOpenApiDocument(): Record<string, unknown> {
  const inferErrors = {
    "200": ok,
    "409": domainError("Unsupported explicit configuration"),
    "422": validationError,
    "429": domainError("Queue capacity exhausted"),
    "503": domainError("Runtime closed"),
  };
  return {
    openapi: "3.1.0",
    info: {
      title: "GLiNER Runner",
      summary: "Local, explicitly configured GLiNER inference",
      version: API_VERSION,
    },
    paths: {
      "/healthz": { get: operation("Health", "operations", { "200": ok }) },
      "/v1/capabilities": { get: operation("Capabilities", "inference", { "200": ok }) },
      "/v1/models": { get: operation("Models", "models", { "200": ok }) },
      "/v1/models/refresh": {
        post: operation("Refresh Models", "models", {
          "200": ok,
          "502": domainError("Catalog refresh failed"),
        }),
      },
      "/v1/models/{model_id}/downloads": {
        post: operation(
          "Create Download",
          "models",
          { "202": ok, "409": domainError("Downloads are not available"), "422": validationError },
          jsonSchema(DownloadRequestSchema),
        ),
      },
      "/v1/model-downloads/{job_id}": {
        get: operation("Download Status", "models", {
          "200": ok,
          "404": domainError("Unknown download job"),
          "422": validationError,
        }),
      },
      "/v1/infer": {
        post: operation("Infer", "inference", inferErrors, jsonSchema(InferenceRequestSchema)),
      },
      "/v1/batch": {
        post: operation("Batch", "inference", inferErrors, jsonSchema(BatchRequestSchema)),
      },
    },
  };
}
