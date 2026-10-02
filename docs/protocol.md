# Local HTTP protocol

The generated OpenAPI document at `/openapi.json` is authoritative. The same
document is checked in at [`openapi/openapi.json`](../openapi/openapi.json) for
client generation and non-Python implementations. CI regenerates it from the
server and rejects drift. The server uses JSON over HTTP, rejects unknown
request fields, and binds to `127.0.0.1:8090` by default.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Process admission health |
| `GET` | `/v1/capabilities` | Implemented backend capability matrix |
| `GET` | `/v1/models` | Curated remote/local/loaded model inventory |
| `POST` | `/v1/models/refresh` | Explicitly refresh remote revision metadata |
| `POST` | `/v1/models/{model_id}/downloads` | Start an approved download job |
| `GET` | `/v1/model-downloads/{job_id}` | Poll a download job |
| `POST` | `/v1/infer` | Submit one inference request |
| `POST` | `/v1/batch` | Submit 1-256 requests |
| `GET` | `/metrics` | Prometheus text metrics (not in OpenAPI) |

## Classification request

```json
{
  "request_id": "955b971b-4295-4e39-9e45-9b404e32bb6e",
  "model": "decide-340m",
  "backend": "pytorch",
  "precision": "fp32",
  "operation": "classify",
  "text": "The invoice needs urgent review.",
  "schema": {
    "kind": "classification",
    "tasks": {
      "priority": {
        "labels": ["urgent", "routine"],
        "min_labels": 1,
        "max_labels": 1,
        "threshold": 0.5,
        "activation": "auto",
        "temperature": 1.0
      }
    },
    "constraints": []
  },
  "options": {
    "threshold": 0.5,
    "output_format": "native",
    "include_confidence": true
  }
}
```

`request_id` and `options` may be omitted. Backend and precision are always
required so a client cannot accidentally accept a fallback. Device is an
operator setting, not a per-request routing input.

Classification tasks mirror the non-constraint portion of Fastino's public
classification schema. `constraints` accepts the upstream serialized
constraint objects and is validated again by Fastino before execution.

## Success

```json
{
  "request_id": "955b971b-4295-4e39-9e45-9b404e32bb6e",
  "model": "decide-340m",
  "backend": "pytorch",
  "precision": "fp32",
  "output": {
    "priority": {
      "value": "urgent",
      "confidence": 0.93,
      "probabilities": {"urgent": 0.93, "routine": 0.07}
    }
  },
  "error": null,
  "timing": {"queue_ms": 1.2, "inference_ms": 18.4},
  "usage": {"inputTokens": 31, "outputTokens": 0}
}
```

Output is Fastino's public `ClassificationResult.to_dict()` JSON. Input order
and request IDs are preserved across dynamic batches.

`usage.inputTokens` is the number of non-padding tokens in the actual encoder
input produced by Fastino's compiled classification schema and tokenizer path.
It therefore includes schema/prompt tokens as well as text tokens and reflects
any configured truncation. It is not a whitespace count or a separate raw-text
tokenizer estimate. Classification does not generate tokens, so
`usage.outputTokens` is exactly `0`. Every response in a dynamic batch carries
its own usage.

`POST /v1/batch` accepts `{"requests": [<request>, ...]}` and returns
`{"responses": [<response>, ...]}` in the same order.

## Errors

Runtime failures use FastAPI's `detail` envelope:

```json
{
  "detail": {
    "code": "queue_full",
    "message": "model worker queue is full",
    "retryable": true
  }
}
```

- `409`: the explicit configuration is unsupported, the memory ceiling is
  exceeded, or a known model requires an approved download;
- `422`: validation failure, unknown manifest, missing artifact, or checksum
  failure;
- `429`: bounded queue is full, with `Retry-After: 1`;
- `503`: the worker is closing.

Ordinary Pydantic request validation uses FastAPI's standard 422 detail list.
Messages are diagnostic; clients should branch on stable custom `code` fields
where present.

`model_download_required` includes an `event` object with `model`, exact
`revision`, `destination`, and `download_endpoint`. It is a notification, not
approval: inference never starts the download. `model_memory_limit` includes
the configured limit, estimated minimum, optional observed usage, profile, and
whether a just-loaded model was unloaded.

## Compatibility

Additive optional fields are allowed within `/v1`. Removing or reinterpreting
fields requires a new version. Clients should use `/v1/capabilities`; enum
membership does not imply that a backend is installed.

`usage` became a required response field in the initial `0.x` contract so the
runner can satisfy integrations that require exact token accounting. Clients
that strictly reject unknown response fields must update their schema; the
current TypeScript client requires and validates the exact
`{inputTokens, outputTokens}` object. Request JSON is unchanged.

The public contract deliberately avoids Python-specific serialization. A
future Rust or Go daemon must serve the same document and error semantics.
