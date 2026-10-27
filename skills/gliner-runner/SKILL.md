---
name: gliner-runner
description: >
  Use GLiNER Runner, a local zero-shot text classifier that runs on the user's own computer
  (CLI, HTTP server on 127.0.0.1:8090, browser playground). It takes one text plus labels you
  write and returns the chosen label or labels, a confidence, and a probability per label. Use
  it when code must sort text into categories it can name, when an LLM call only picks a label
  from a fixed list, when text must stay on the machine or work offline, or when each text needs
  several yes/no tags. Typical jobs: ticket routing, tagging, triage, moderation flags, intent,
  sentiment, language. It does not write, summarize or extract text.
---

# GLiNER Runner

GLiNER Runner reads one text and scores the labels you send. Your code owns the labels, the
thresholds and what happens next. It never generates text or returns a label you did not send.
The same request always gives the same answer.

## 1. Sources of truth

The running server wins over this skill: `GET /openapi.json` is the contract and
`GET /v1/capabilities` lists what this server runs. Docs: https://stuffbucket.github.io/gliner-runner/llms.txt
lists every page; each page has a Markdown copy at `<page>.md`, for example
`command-line-and-http.md` (curl, Node, many texts, errors), `how-it-works.md` (modes,
confidence, limits) and `docs/protocol.md` (every field and error code). If you cannot check,
use the shapes below and say so. Do not invent fields: unknown fields give 422.

## 2. Use it or not

- Pick exactly one of N labels you can list: a single-label task (the default).
- Decide which of N tags apply, any number or none: a several-labels task.
- Several questions about one text: several tasks in one request.
- Not this tool: summarize, rewrite, translate, answer questions, pull out a name, number or span
  (it can say whether a text contains one, not what or where), or categories you cannot list.
  Say so first, then offer what it can classify, for example "contains an order number: yes/no".

## 3. Install and start

```sh
# Homebrew (macOS on Apple Silicon, Linux). Skip `brew trust` if your Homebrew has no such command.
brew tap stuffbucket/tap
brew trust stuffbucket/tap
brew install gliner-runner
# without Homebrew (installs into ~/.local):
curl -fsSL https://github.com/stuffbucket/gliner-runner/releases/latest/download/install.sh | sh
gliner-runner --version                    # first line starts with gliner-runner-onnx
gliner-runner models install decide-340m   # one-time download, about 2.0 GB; asks first (--yes skips)
gliner-runner serve                        # http://127.0.0.1:8090; --port, --host, --models-dir
curl -s http://127.0.0.1:8090/healthz
```

- Ask the user before a model download. Inference never downloads (unless
  `serve --auto-download`): a missing model is `409 model_download_required`.
- Quick check without a server: `gliner-runner classify --labels a,b,c "text"` (`--multi`,
  `--threshold`, `--describe label=text`, `--json`, `--model`). It loads the model each call
  (about 1.5 s), so use `serve` for more than a few texts.

| `model` | Download | Use for |
| --- | --- | --- |
| `decide-340m` | about 2.0 GB | English. The default; start here |
| `decide-multilingual` | about 1.2 GB | Spanish, French, German and other languages |
| `decide-1b` | about 4.8 GB | English, when accuracy matters more than size |

## 4. The request

`POST /v1/infer` with `content-type: application/json`. When you show a request, show the
whole body, never a fragment. All fields are required except `request_id` (a UUID) and
`options`:

```json
{
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "operation": "classify",
  "text": "I was charged twice for my last order.",
  "schema": {
    "kind": "classification",
    "tasks": {
      "category": {
        "labels": ["billing", "support", "shipping"],
        "label_descriptions": {"billing": "Charges, invoices or refunds"}
      }
    }
  }
}
```

- `model`: the model id from section 3, for example `"decide-340m"`.
- `backend: "onnx"`, `precision: "fp32"`, `operation: "classify"`: the only values this runner
  runs. Others return 409; it never falls back.
- `text`: one non-empty text per request.
- `schema.kind`: always `"classification"`.
- `schema.tasks`: one entry per question. **The task name is model input**: name it in plain
  words (`category`, `urgency`), not `t1`. Renaming it, adding another task to the request or
  changing the task order changes the probabilities.
- `labels`: a JSON list of short, distinct names. The whole list is model input: adding or
  removing a label changes the other labels' probabilities.
- `label_descriptions`: optional `{label: text or null}`; keys must be declared labels. They
  set the boundary between labels and change the answer. Put explanations here, not in labels.
- No `(` or `)` in labels, descriptions, task names or `instruction` (422 `invalid_schema`).
- `min_labels` / `max_labels`: the mode (section 5). `max_labels` may not exceed the number of
  labels; `null` means no limit.
- `threshold`: per task, strictly between 0 and 1, default 0.5. Matters only with several labels.
- `options.threshold` (default 0.5) is a floor applied first. **To return labels below 0.5, set
  both `tasks.<name>.threshold` and `options.threshold`** to the value.
- `instruction`: optional task wording; it is model input, so test that it helps.
- Unknown fields give 422. The usual one is `labels` at the top level instead of in a task.

## 5. Single label or several: the response

| Need | Settings | `value` | `probabilities` | `confidence` |
| --- | --- | --- | --- | --- |
| Exactly one | default (`min_labels: 1`, `max_labels: 1`) | a string, always one label | softmax: sum to 1 | the chosen label's probability |
| Any number, or none | `"min_labels": 0, "max_labels": null` | a list, may be `[]` | independent sigmoid per label | one score for the whole selection; `null` when `value` is `[]` |
| Zero or one | `"min_labels": 0, "max_labels": 1` | a list of at most one | sigmoid per label | as above |

Response to that request without `label_descriptions` (`decide-340m`); read `output.<task name>`
and ignore `_meta`:

```json
{
  "request_id": "9d897bc2-6488-4b09-8d4e-c2ae7ec5dd70",
  "model": "decide-340m", "backend": "onnx", "precision": "fp32",
  "output": {
    "category": {
      "value": "billing",
      "confidence": 0.8887,
      "probabilities": {"billing": 0.8887, "support": 0.0485, "shipping": 0.0628}
    },
    "_meta": {"feasible": true, "decoder": "independent", "exact": true, "objective": 1.0048, "violations": []}
  },
  "error": null,
  "timing": {"queue_ms": 4.2, "inference_ms": 21.9},
  "usage": {"inputTokens": 22, "outputTokens": 0}
}
```

Several labels, task `aspects`, text "The battery is weak and the delivery was slow.":
`{"value": ["battery", "delivery"], "confidence": 0.9913, "probabilities": {"battery": 0.9916,
"screen": 0.0055, "price": 0.0058, "delivery": 0.9849}}`. Nothing above the threshold gives
`"value": [], "confidence": null`: a valid "no tag applies", not an error.

- Single label always picks something, even when nothing fits. To detect "none of these", add a
  label for it (`"other"`) and describe it. A single-label probability is a share among the
  labels sent: 0.41 can be the winner.
- A several-labels probability is a yes/no score for that label alone, returned in `value` when
  above the threshold; `probabilities` lists every label either way. `confidence` is the
  geometric mean over all labels of p (returned) or 1 - p (not returned); for one label's score
  read `probabilities[label]`.
- In several-labels mode `[]` already means "none": do not add a "none" label; it shifts the
  other scores. Name the task in the plural: on 30 test reviews, `topic`, `aspect` and `tag` got
  24 to 26 tag sets right, `topics`, `aspects` and `tags` 29 to 30. Leave out descriptions when
  the names are clear: in the same test, even short ones gave 3 of 10 unrelated texts a tag,
  against none without. Describe only labels that get confused.
- Thresholds change which labels are in `value`, not the probabilities.
- Confidence ranks answers; it is not the chance of being right. Calibrate on your data.

## 6. Thresholds live in code

Keep labels, descriptions and cutoffs as named constants in one place. Start with three bands
and tune them on 20 to 50 of the user's texts with known answers:

```ts
const ACT = 0.8;     // at or above: act automatically
const REVIEW = 0.5;  // between REVIEW and ACT: a person checks; below REVIEW: "none fits"
const c = data.output.category;
if (c.value === "other" || c.confidence < REVIEW) toFallback();
else if (c.confidence < ACT) toReview(c.value);
else route(c.value);
```

These are starting points, not model facts. A costlier mistake needs a higher cutoff. Changing
the task name, labels, descriptions or model shifts the scores; re-check the cutoffs after.

## 7. Many texts, long texts, web pages

- Many texts: one `serve` process. Send `POST /v1/batch` with `{"requests": [<infer request>, ...]}`
  (1 to 256, more gives 422; returns `{"responses": [...]}` in order), or concurrent `/v1/infer`
  calls, which the server batches. The queue holds 256 pending requests per model
  (`--queue-capacity`) and each batch item takes a slot. More in flight get `429 queue_full`,
  and one 429 fails the whole batch. So send one batch at a time or cap concurrency, and on 429
  wait `Retry-After` seconds and resend. Do not run `gliner-runner classify` per text.
- Long texts: best up to about 500 words; split longer ones into sections and combine in code.
- Web pages: the server sends CORS headers only to listed origins. Defaults:
  `http://localhost:3000, 4173, 4200, 4321, 5173, 8080`. For another origin run
  `gliner-runner cors add https://exact.origin` (scheme, host, port, no trailing slash) and
  restart `serve`, or start it with `--cors-origin <origin>`. A Vite dev proxy
  (`server.proxy: {"/v1": "http://127.0.0.1:8090"}`) avoids CORS. Example app:
  `examples/react-classifier`, explained on `use-in-react/`.
- The site's playground runs the model inside the page (`decide-340m` only); that package and
  the TypeScript client in `packages/client` are not on npm. Apps call `serve` over HTTP with
  plain `fetch`.

## 8. Errors

Branch on `detail.code`, not the message. Request validation errors have `detail` as a list
with `loc`.

| Status / code | Meaning | Do |
| --- | --- | --- |
| 422 list, `extra_forbidden` | Unknown field (often top-level `labels`) | Move it into `schema.tasks.<name>` |
| 422 list, `missing` / `enum` | Required field missing or wrong | Send all six fields of section 4 |
| 422 `invalid_schema` | `(`/`)` in a label, `max_labels` above the label count, `min_labels > max_labels` | Fix as the message says |
| 422 `model_integrity` | Unknown model id or damaged model | `decide-340m`, `decide-multilingual`, `decide-1b` |
| 409 `unsupported_configuration` | Backend, precision or operation not run here (`fp16`, `pytorch`, `extract_entities`) | `onnx` / `fp32` / `classify` |
| 409 `model_download_required` | Model not installed | Ask the user, then `gliner-runner models install <id>` |
| 409 `model_memory_limit` | Model above the memory ceiling | Smaller model or `--memory-limit-bytes` |
| 413 `request_too_large` | Body above 16 MiB | Smaller batches |
| 429 `queue_full` | Queue full | Retry after `Retry-After` |
| 503 `runtime_closed` | Shutting down | Restart `serve` |
| 404 | Wrong path | `/v1/infer` or `/v1/batch` |

## 9. When answers look wrong

Similar probabilities: the labels overlap or the text lacks the evidence; sharpen descriptions.
Always the catch-all label: its description is too broad. A long document decided by one
sentence: split it.
