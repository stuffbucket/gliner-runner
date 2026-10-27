# Local HTTP protocol

The generated OpenAPI document at `/openapi.json` is authoritative. The same
document is checked in at [`openapi/openapi.json`](../openapi/openapi.json) for
client generation and non-Python implementations. CI regenerates it from the
server and rejects drift. The server uses JSON over HTTP, rejects unknown
request fields, and binds to `127.0.0.1:8090` by default.

Two servers implement this contract: the Python server (`gliner-runner serve`),
which registers PyTorch, and the Python-free Node ONNX runner
(`gliner-runner-onnx serve`), which registers only the ONNX CPU/FP32 `classify`
backend. See [ONNX runtime](onnx-runtime.md). They share the same envelopes,
error codes, and usage semantics. Each advertises only what it implements in its
own `/v1/capabilities`. The Node runner implements the download endpoints too:
an approved `POST /v1/models/{model_id}/downloads` (destination equal to its
models directory) starts a job that finds a verified local snapshot or downloads
the pinned Hugging Face files and assembles the ONNX artifact from a bundled
template, with no Python; `GET /v1/model-downloads/{job_id}` reports it. Its
`POST /v1/models/refresh` looks up current revisions on the Hub and writes
`catalog-cache.json`, and never installs anything. With `--auto-download` (off by
default) an inference request for a missing model waits for a shared download job
instead of returning `model_download_required`. See
[ONNX runtime](onnx-runtime.md#installing-models).

The OpenAPI contract version is `1.1.0` and is independent of the runner's
package release version. Package patch and feature releases do not imply a
wire-contract change; incompatible protocol changes require an explicit API
version update.

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
| `POST` | `/v1/questions` | jev and Microsoft-Decision-1 questions, translated ([below](#translated-shapes)) |
| `POST` | `/v1/classify_text` | GLiNER2 `classify_text` arguments, translated ([below](#translated-shapes)) |
| `GET` | `/metrics` | Prometheus text metrics (not in OpenAPI) |

## Classification request

```json
{
  "request_id": "955b971b-4295-4e39-9e45-9b404e32bb6e",
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "operation": "classify",
  "text": "The invoice needs urgent review.",
  "schema": {
    "kind": "classification",
    "tasks": {
      "priority": {
        "labels": ["urgent", "routine", "unknown"],
        "label_descriptions": {
          "urgent": "Requires immediate handling",
          "routine": "Can follow the normal review queue",
          "unknown": null
        },
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
`label_descriptions` is optional, so existing labels-only requests are
unchanged. When present, every key must be one of the task's declared labels;
values are non-blank strings or `null`, and the mapping may describe only a
subset. The runner expands the mapping in declared label order and passes it to
Fastino as native described labels before schema compilation. Descriptions
therefore affect model semantics and `usage.inputTokens`.

## Success

```json
{
  "request_id": "955b971b-4295-4e39-9e45-9b404e32bb6e",
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "output": {
    "priority": {
      "value": "urgent",
      "confidence": 0.7207761650979043,
      "probabilities": {
        "urgent": 0.7207761650979043,
        "routine": 0.13768895688963823,
        "unknown": 0.14153487801245748
      }
    },
    "_meta": {
      "feasible": true,
      "decoder": "independent",
      "exact": true,
      "objective": 0.6589669585227966,
      "violations": []
    }
  },
  "error": null,
  "timing": {"queue_ms": 4.655458999997791, "inference_ms": 40.011875000000146},
  "usage": {"inputTokens": 34, "outputTokens": 0}
}
```

This is the Node ONNX runner's reply to the request above. Output is Fastino's
public `ClassificationResult.to_dict()` JSON: `probabilities` has every declared
label, and `_meta` describes how the labels were selected. Input order and
request IDs are preserved across dynamic batches.

`usage.inputTokens` is the number of non-padding tokens in the actual encoder
input produced by Fastino's compiled classification schema and tokenizer path.
It therefore includes schema/prompt tokens as well as text tokens and reflects
any configured truncation. It is not a whitespace count or a separate raw-text
tokenizer estimate. Classification does not generate tokens, so
`usage.outputTokens` is exactly `0`. Every response in a dynamic batch carries
its own usage.

`POST /v1/batch` accepts `{"requests": [<request>, ...]}` and returns
`{"responses": [<response>, ...]}` in the same order.

## Translated shapes

Two endpoints take the request shape of another decision API and translate it into ordinary
classification requests. The answers are computed only from those requests' probabilities, so a
client that sends the translated requests itself gets the same numbers. Both servers implement
them the same way. [API shapes](api-shapes.md) maps every function of TypeSafe's jev,
Microsoft-Decision-1 and GLiNER2 to the runner, with sources, and lists what is declined and why.
`model`, `backend` and `precision` are required as on `/v1/infer`; `request_id` is optional.

### `POST /v1/questions`

The `state` and `questions` of jev and Microsoft-Decision-1. `state` must be one non-blank string.
Each question has a `type`, optional `instructions` (one string or `null`) and `criteria`:

| `type` | `criteria` | Classification request | Answer |
| --- | --- | --- | --- |
| `noul` | optional `{"true": str or null, "false": str or null}` | labels `yes`, `no`; `criteria` as their descriptions | `{"type": "noul", "noul": <probability of yes>}` |
| `choice` | `{option: str or null}`, at least two | the options as labels, single-label | `{"type": "choice", "choice", "confidence", "probabilities"}` |
| `score` | `[lowest, ..., highest]`, at least two, unique | the levels as labels, single-label, `ordered: true` | `{"type": "score", "score", "confidence", "legend", "probabilities"}` |
| `multi_select` | `{option: str or null}`, at least one; optional `threshold` in (0, 1) | the options as labels, `min_labels: 0`, `max_labels: null`, `threshold` (also used as `options.threshold`) | `{"type": "multi_select", "selected", "probabilities"}` |

Each question becomes one `/v1/infer` request whose only task is named by the question id, with
`instructions` as the task `instruction`. The id is therefore part of the prompt. Questions never
share a request, so one cannot change another's numbers. `score` is the sum of level index times
probability. `confidence` uses jev's formulas: for a choice, (p_max − 1/n) / (1 − 1/n); for a
score, 1 minus the probability-weighted distance from the most likely level divided by the same
distance for an even spread, floored at 0. `multi_select` is not a jev type; its probabilities are
independent and do not sum to 1. `box` questions, `assets`, structured `state` and structured
`instructions` or `criteria` are rejected with a `422` that says why.

```json
{
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "state": "I have asked three times now. Can I please just talk to a real person?",
  "questions": {
    "wants_human": {"type": "noul", "instructions": "Is the customer asking for a human agent?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "returns": "Exchanges, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems"
      }
    }
  }
}
```

```json
{
  "request_id": "74cd9aad-4859-49bb-a4c7-85c1c7a7710f",
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "answers": {
    "wants_human": {"type": "noul", "noul": 0.9733915329310245},
    "team": {
      "type": "choice",
      "choice": "returns",
      "confidence": 0.11051912481238073,
      "probabilities": {
        "returns": 0.4070127498749205,
        "shipping": 0.29555396475895124,
        "billing": 0.2974332853661283
      }
    }
  },
  "usage": {"input_tokens": 107, "output_tokens": 0}
}
```

`usage` uses jev's field names and sums the questions' requests. The reply has no `timing`.

### `POST /v1/classify_text`

The arguments of GLiNER2 2.1.0 `classify_text` (`text`) or `batch_classify_text` (`texts`, 1-256),
with `tasks` as GLiNER2 takes them: a list of labels, or an object with `labels` (a list, or
`{name: description or null}`) and any of `multi_label`, `cls_threshold` (in (0, 1)), `top_k`,
`class_act` and `prompt`. `include_confidence` defaults to `false`. The top-level `threshold` is
accepted and has no effect, as in GLiNER2, where it is the span threshold.

All tasks of a text go into one `/v1/infer` request. A multi-label task becomes `min_labels: 1`,
`max_labels: top_k` (or unbounded) and `threshold: cls_threshold`; `options.threshold` is lowered
to the smallest `cls_threshold` below 0.5. `prompt` becomes `instruction` and `class_act` becomes
`activation`. A single-label task keeps the defaults. `examples`, a multi-label task with
`class_act: "softmax"`, and other GLiNER2 options are rejected with a `422`.

```json
{
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "text": "The battery is weak and the delivery was slow.",
  "tasks": {
    "sentiment": ["positive", "negative", "neutral"],
    "aspects": {"labels": ["battery", "screen", "price", "delivery"], "multi_label": true, "cls_threshold": 0.4}
  }
}
```

```json
{
  "request_id": "5322d861-f05b-4f8d-821d-3bf1e0225f4a",
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "result": {"sentiment": "negative", "aspects": ["battery", "delivery"]},
  "usage": {"inputTokens": 38, "outputTokens": 0}
}
```

`result` (or `results` for `texts`) is GLiNER2's formatted result: a label, or
`{"label", "confidence"}` with `include_confidence`, for a single-label task; a list of those for a
multi-label task, sorted by probability when `top_k` is set.

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

- `409`: the explicit configuration is unsupported (for example `onnx` with a
  precision other than `fp32`, or `pytorch` sent to the ONNX runner), the memory
  ceiling is exceeded, or a known model requires an approved download;
- `422`: validation failure, semantically invalid inference schema (`invalid_schema`),
  unknown manifest, missing artifact, or checksum failure;
- `413`: the Node ONNX runner rejects a request body above its configured limit
  (default 16 MiB) with `request_too_large`;
- `429`: bounded queue is full, with `Retry-After: 1`;
- `503`: the worker is closing.

Ordinary Pydantic request validation uses FastAPI's standard 422 detail list.
Messages are diagnostic; clients should branch on stable custom `code` fields
where present.

`model_download_required` includes an `event` object with `model`, exact
`revision`, `destination`, and `download_endpoint`. It is a notification, not
approval: inference never starts the download (unless the Node runner was
started with `--auto-download`; if that automatic install fails, the same error
is returned with the reason). `model_memory_limit` includes
the configured limit, estimated minimum, optional observed usage, profile, and
whether a just-loaded model was unloaded.

## Compatibility

Additive optional fields are allowed within `/v1`. Removing or reinterpreting
fields requires a new version. Clients should use `/v1/capabilities`; enum
membership does not imply that a backend is installed. The `backend` enum
includes `pytorch`, `mlx`, and `onnx`, but the Python server registers only
PyTorch and the Node runner registers only ONNX; `mlx` is registered by neither.
Selecting an unregistered backend returns `409`, never a fallback.

`usage` became a required response field in the initial `0.x` contract so the
runner can satisfy integrations that require exact token accounting. Clients
that strictly reject unknown response fields must update their schema; the
current TypeScript client requires and validates the exact
`{inputTokens, outputTokens}` object. Existing request fields are unchanged.

`label_descriptions` is an additive optional request field. Labels-only clients
retain their existing behavior and wire shape.

Contract version `1.1.0` adds `POST /v1/questions` and `POST /v1/classify_text`. Nothing about the
`1.0.0` endpoints changed.

The public contract deliberately avoids Python-specific serialization. A
future Rust or Go daemon must serve the same document and error semantics.
