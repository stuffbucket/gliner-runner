# Local HTTP protocol

The generated OpenAPI document at `/openapi.json` is authoritative. The server
uses JSON over HTTP, rejects unknown request fields, and binds to
`127.0.0.1:8090` by default.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Process admission health |
| `GET` | `/v1/capabilities` | Implemented backend capability matrix |
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
  "timing": {"queue_ms": 1.2, "inference_ms": 18.4}
}
```

Output is Fastino's public `ClassificationResult.to_dict()` JSON. Input order
and request IDs are preserved across dynamic batches.

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

- `409`: the explicit backend/device/precision/operation is unsupported;
- `422`: validation failure, unknown manifest, missing artifact, or checksum
  failure;
- `429`: bounded queue is full, with `Retry-After: 1`;
- `503`: the worker is closing.

Ordinary Pydantic request validation uses FastAPI's standard 422 detail list.
Messages are diagnostic; clients should branch on stable custom `code` fields
where present.

## Compatibility

Additive optional fields are allowed within `/v1`. Removing or reinterpreting
fields requires a new version. Clients should use `/v1/capabilities`; enum
membership does not imply that a backend is installed.
