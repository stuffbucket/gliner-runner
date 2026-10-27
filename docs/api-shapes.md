# Shapes from other decision APIs

People reach GLiNER Runner from other decision APIs. Where one of their functions is a job these
models can do and only the request or reply shape differs, the runner accepts their shape and
translates it. Each translation unpacks into ordinary `POST /v1/infer` classification requests, so
the numbers are the ones you would get by writing those requests yourself. Where the models cannot
do the job, the runner says so in a `422` and says what to do instead.

Both runners implement the translations the same way: the Python server
([`src/gliner_runner/shapes.py`](../src/gliner_runner/shapes.py)) and the Node ONNX runner
([`packages/onnx-runner/src/shapes.ts`](../packages/onnx-runner/src/shapes.ts)), which the browser
worker shares. The cases in [`tests/fixtures/api-shapes/cases.json`](../tests/fixtures/api-shapes/cases.json)
run in both test suites and must give the same requests, the same answers and the same validation
messages. The wire contract is in [protocol.md](protocol.md#translated-shapes).

## Sources

Every claim below about another API comes from these, read on 2026-10-21.

| API | Source | Version |
| --- | --- | --- |
| TypeSafe jev | [HTTP API](https://docs.typesafe.ai/api/system-one.md), [OpenAPI](https://api.typesafe.ai/openapi.json) | OpenAPI `info.version` 0.2.0 (file sha256 `ee599512cbea3a19…c057c3`); doc examples answer as `jev-1.13.0` |
| TypeSafe jev | [Noul](https://docs.typesafe.ai/primitives/noul.md), [Choice](https://docs.typesafe.ai/primitives/choice.md), [Score](https://docs.typesafe.ai/primitives/score.md), [Box](https://docs.typesafe.ai/primitives/box.md), [Confidence](https://docs.typesafe.ai/confidence.md) | live docs, 2026-10-21 |
| TypeSafe agent skill | [typesafe-ai/skills](https://github.com/typesafe-ai/skills) | commit `65a39f393687675ce170e6094757de20370365b9` (v0.5.7) |
| Microsoft-Decision-1 | [Deploy and use Microsoft-Decision-1 in Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision) | `ms.date` 2026-10-07, updated 2026-10-09 |
| Microsoft-Decision-1 | [announcement](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/) | 2026-10-09 |
| GLiNER2 | [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2/tree/v2.1.0) | tag v2.1.0, commit `b86d35c02f2c1a4d9840ea50197bd1b6d13b031e` |
| GLiNER2.5 Decide cards | [Decide](https://huggingface.co/fastino/GLiNER2.5-Decide/blob/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90/README.md), [multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide/blob/e173bca1f0c4217d7a55b3b0c705d5ece8a0218d/README.md), [Decide-1B](https://huggingface.co/fastino/GLiNER2.5-Decide-1B/blob/67af949e17bebd20f05fe020a971079889e7a462/README.md) | the revisions the runner pins |

GLiNER2 file references are to tag v2.1.0: `gliner2/inference/runtime.py` (`classify_text` line
1863, `batch_classify_text` line 1887, `_classification_schema` line 1972,
`_decode_classification_probabilities` line 914, `_format_classification` line 49),
`gliner2/inference/schema.py` (`Schema.classification` line 291) and `gliner2/api_client.py`.

## TypeSafe jev and Microsoft-Decision-1

The two APIs share one request shape. jev's endpoint is `https://api.typesafe.ai/v1/systemone`;
Microsoft-Decision-1's is `<foundry endpoint>/providers/microsoft/v1/systemone`. Both take
`{"model", "state", "questions": {id: {"type", "instructions", "criteria"}}}` and answer
`{"model", "answers": {id: {...}}, "usage"}` (jev HTTP API; Learn page, "Call the decision API").
Microsoft's page does not show a complete reply; it reads `answers[id]["type"]` and
`answers[id]["choice"]` and says the reply carries per-option probabilities and token usage. The
runner returns jev's documented field names for both.

| Their function | What it returns there | GLiNER Runner |
| --- | --- | --- |
| Noul (`type: "noul"`): a yes/no question, optional `criteria: {"true", "false"}` | `noul`: probability of yes, 0 to 1. No `confidence`. Near 0.5 means yes and no are about equally likely, not a medium degree (noul.md, "Reading a Noul") | `POST /v1/questions`, `type: "noul"`. One task with labels `yes` and `no`, `instructions` as the task instruction, `criteria.true`/`criteria.false` as the descriptions of `yes`/`no`. `noul` is the probability of `yes` (the two sum to 1). |
| Choice (`type: "choice"`): one of a set, `criteria: {option: description}` | `choice`, `probabilities` (sum to 1), `confidence` = (p_max − 1/n) / (1 − 1/n) (confidence.md, "Choice") | `type: "choice"`. One single-label task; option names are the labels, descriptions are `label_descriptions`. `choice` is the selected label, `probabilities` are the task's, `confidence` is jev's formula over those probabilities. |
| Score (`type: "score"`): ordered levels, `criteria: [lowest, ..., highest]` | `score` = Σ level × probability, `probabilities` and `legend` keyed `"0"`, `"1"`, ..., `confidence` from the spread around the most likely level (score.md; confidence.md, "Score") | `type: "score"`. One single-label task with `ordered: true` whose labels are the level descriptions. `score`, `confidence`, `legend` and `probabilities` are computed from the task's probabilities with jev's formulas. |
| Several labels that may all apply: jev has no type for this and says to ask one Noul per label (skill: "use one per label when several may apply") | one `noul` per label | Either one `noul` per label, or the runner's own `type: "multi_select"` with `criteria: {option: description}` and an optional `threshold` (default 0.5). That is one multi-label task (`min_labels: 0`, `max_labels: null`). The answer is `{"type": "multi_select", "selected": [...], "probabilities": {...}}`; each probability is that option's own yes probability, so they do not sum to 1. This type is not part of jev. |
| Box (`type: "box"`): where something is in an image | `box` edges and `confidence` (box.md) | Declined: `422`. The Decide models read text only. Use an image model. |
| Images in `state` or `instructions` (`assets`, the images header) | | Declined: `422`. Send the text instead. |
| `state` as a JSON object or array | the model reads the structure | Declined: `422`. `state` must be one string. Write the fields out as text, for example `"subject: ...\nmessage: ..."`. |
| Structured `instructions` or `criteria` (objects or arrays as values) | the model reads the structure | Declined: `422`. Write each one as a single string. |
| Backticked references such as `` `ticket.messages[0].text` `` | the model looks them up in `state` | Sent as plain text. There is no structured state to look them up in. |
| `GET /v1/models` | model names such as `jev-latest` | Use the runner's own `GET /v1/models`; `model` must be a runner model id (`decide-340m`, `decide-multilingual`, `decide-1b`). |

What else differs, on purpose:

- **`backend` and `precision` are required**, as on every runner request, so a client never gets a
  fallback it did not ask for. A jev or Decision-1 body needs these two fields added and the
  runner's model id in `model`.
- **The question id is part of the prompt.** jev does not send the id to the model (choice.md:
  "The model never sees the question id"). The runner follows Fastino's model card, which asks a
  question as a task name plus a `prompt` (the card's "Question over a passage" example), so the id
  is the task name the model reads. Use descriptive ids such as `wants_human`, not `q1`. With
  decide-340m, the same message scored 0.92 under the id `is_human_escalation` and 0.97 under
  `wants_human`.
- **Questions do not affect each other.** jev evaluates questions in parallel and they "cannot see
  one another's answers" (skill). GLiNER2 reads every task of one request in one pass, so a second
  task changes the first one's numbers a little (0.99496 alone, 0.99959 next to a choice question,
  decide-340m). The runner therefore sends one classification request per question. Asking more
  questions costs one model pass each.
- **`usage`** uses jev's names, `input_tokens` and `output_tokens`. `input_tokens` is the sum over
  the questions' requests; `output_tokens` is 0.
- **The reply** also carries the runner's `request_id`, `backend` and `precision`, and has no
  `timing` (the questions run as separate requests).
- **The numbers are not jev's.** The formulas are the same; the probabilities come from a different
  model. Measure thresholds on your own data, as both vendors also say.

### What the answers looked like

The jev pages publish answers for fixed inputs. These are the runner's answers to the same inputs
through `POST /v1/questions` (ONNX, CPU, FP32, the pinned revisions), next to jev's.

Noul, "Is the customer asking for a human agent?" (noul.md, "Reading a Noul"):

| State | jev-1.13.0 | decide-340m | decide-1b | decide-multilingual |
| --- | --- | --- | --- | --- |
| Thanks, that fixed it! | 0.02 | 0.07 | 0.16 | 0.31 |
| How do I reset my password? | 0.07 | 0.16 | 0.23 | 0.30 |
| I need this sorted today, whatever it takes. | 0.26 | 0.62 | 0.52 | 0.48 |
| Are you a bot? | 0.40 | 0.21 | 0.30 | 0.21 |
| Is there any way to speak to someone about my invoice? | 0.84 | 0.72 | 0.55 | 0.30 |
| I have asked three times now. Can I please just talk to a real person? | 0.99 | 0.92 | 0.82 | 0.30 |

decide-340m and decide-1b order the clear cases correctly. decide-multilingual does not answer
this yes/no question usefully. Test a noul on your own examples before you rely on it.

Score, "How severe is the reported issue?" with jev's three bug-severity levels (score.md,
"Reading a Score"), as `score / confidence`:

| State (shortened) | jev-1.13.0 | decide-340m | decide-1b | decide-multilingual |
| --- | --- | --- | --- | --- |
| Button misaligned by a few pixels | 0.0 / 1.0 | 0.19 / 0.71 | 0.28 / 0.58 | 0.66 / 0.01 |
| PDF export does nothing; CSV works | 1.0 / 1.0 | 0.61 / 0.08 | 0.73 / 0.00 | 1.34 / 0.17 |
| Spinner never finishes; CSV works for some | 1.11 / 0.84 | 1.00 / 0.22 | 0.91 / 0.00 | 1.29 / 0.42 |
| Crashes in Safari, works in Chrome | 1.43 / 0.35 | 0.81 / 0.00 | 1.00 / 0.35 | 1.17 / 0.69 |
| Nobody can log in; 500 on every attempt | 2.0 / 1.0 | 1.45 / 0.18 | 1.70 / 0.55 | 1.74 / 0.61 |

The ends come out in the right order on every model, but the models spread probability over the
levels far more than jev, so confidence is low. The Decide models were not trained on ordered
levels: the 340M card says "An ordinal scale trains as plain classes". The score is an honest
probability-weighted position over the labels, not a calibrated rating.

We also tried other encodings. For noul: the question itself as the task name, with no
instruction, did better on the two clear yes cases with decide-340m (0.90 and 0.995, against 0.72
and 0.92) and about the same on the rest; the runner keeps the model card's form, id plus
`prompt`, so that the id-to-task mapping stays one to one (see open questions). A single label with
its own sigmoid probability gave 0.43 to 0.77 for every message, which is useless; `true`/`false`
labels behaved like `yes`/`no`. For score: level descriptions as label names beat labels `0`, `1`,
`2` with the descriptions attached, on both decide-340m and decide-1b.

## GLiNER2

GLiNER2's Python API runs in-process, so the runner cannot take the call itself. It takes the same
arguments as JSON and returns the same result.

| Their function | What it returns there | GLiNER Runner |
| --- | --- | --- |
| `classify_text(text, tasks, threshold=0.5, include_confidence=False, ...)` | single-label task: `"label"`, or `{"label", "confidence"}`; multi-label task: `["a", "b"]`, or a list of `{"label", "confidence"}` (`_format_classification`) | `POST /v1/classify_text` with `text`, `tasks` and `include_confidence`; the reply's `result` is GLiNER2's dict. All tasks go into one classification request, as GLiNER2 scores them in one pass. |
| `batch_classify_text(texts, tasks, batch_size=8, ...)` | a list of those dicts | `POST /v1/classify_text` with `texts` (1 to 256); the reply's `results` is the list. One classification request per text; the server batches them. |
| A task as a list of labels, or `{"labels": [...] or {name: description}}` | | `labels`, and `label_descriptions` for described labels |
| `multi_label: true`, `cls_threshold` | every label with probability ≥ `cls_threshold` (sigmoid); the best label alone if none is (`_decode_classification_probabilities`) | `min_labels: 1`, `max_labels` unbounded, `threshold: cls_threshold`. The request's candidate floor (`options.threshold`) is lowered to the smallest `cls_threshold` below 0.5 so no label that should pass is dropped. |
| `top_k` | at most `top_k` labels, sorted by probability | `max_labels: top_k` (at most the label count), and the reply lists the labels by probability, highest first |
| `class_act` (`auto`, `sigmoid`, `softmax`) | the activation | `activation` |
| `prompt` | a question or instruction for the task | `instruction`; both become `"{task}: {prompt}"` in the model input |
| Single-label tasks | the most probable label; `cls_threshold` and `top_k` have no effect | the runner default: exactly one label, the most probable |
| Top-level `threshold` | the span-extraction threshold; classification does not use it (runtime.py, `_decode_classification_probabilities` reads `cls_threshold` only) | accepted, no effect, for the same reason |
| `examples` (few-shot `(text, label)` pairs) | added to the prompt as `[EXAMPLE] text [OUTPUT] label` | Declined: `422`. Not part of the runner's protocol yet (see open questions). |
| `multi_label: true` with `class_act: "softmax"` | labels whose softmax probability passes `cls_threshold` | Declined: `422`. The runner selects labels with sigmoid probabilities, so the selection would differ. Use `auto` or `sigmoid`. |
| `cls_threshold` of exactly 0 or 1 | every label, or the best label alone | Declined: `422`. Task thresholds must be strictly between 0 and 1. Use `top_k` instead. |
| `classify_text_long`, `batch_classify_text_long` | chunks long text (`chunk_size=384`, `chunk_overlap=64`) and merges the chunks | Not translated. Split long text yourself and send `texts`. |
| `extract_entities`, `extract_relations`, `extract_json` and their batch and long forms | spans and structures | Declined. The runners implement classification only (`/v1/capabilities` lists `classify`), and the Decide model cards document only `classify_text`. A `/v1/infer` request with an extraction operation returns `409 unsupported_configuration`. Use GLiNER2 in Python with an extraction checkpoint. |
| `include_spans`, `format_results=False`, `max_len`, `overlap_policy` | span positions, raw output, truncation | Not accepted (`422`, extra field). |
| `gliner2.classification`: `ClassificationSchema` with `task(...)` options, constraints, `ClassificationResult.to_dict()` | | Already the runner's native shape: `POST /v1/infer` with `schema.kind: "classification"` takes `ClassificationSchema.to_dict()` tasks and constraints and returns `to_dict()`. `examples` and `candidate_threshold` are the task options it does not expose. |

GLiNER2's hosted API client (`GLiNER2API`) posts `{"task", "text", "schema", "threshold",
"include_confidence", "include_spans", "format_results"}` to `<base>/gliner-2` and reads `result`
(`api_client.py`, `_make_request`). The runner's reply also keeps the result under `result`, but it
does not serve `/gliner-2`: that client cannot send `backend` and `precision` (see open questions).

## Microsoft Azure AI Foundry

The model is called **Microsoft-Decision-1**, not "decision-1". It is a Foundry model, version 1,
deployed with API type "Decision" (Learn page, "Microsoft-Decision-1 at a glance"), and Microsoft
built it by post-training Qwen3.5-9B (announcement). Its request is jev's: `state` and typed
`noul`, `choice` and `score` questions. Everything in the jev section applies. Differences in the
source request:

- the endpoint is `<foundry endpoint>/providers/microsoft/v1/systemone`, authenticated with an
  Entra token or `api-key`; the runner has no authentication and listens on `127.0.0.1`;
- `model` is the Foundry deployment name; for the runner it is a runner model id;
- `state` "can be text or JSON"; the runner takes text only.

## Open questions

- **Task name for a question.** The runner sends the question id as the task name and
  `instructions` as its prompt, as the Decide card does. Sending the `instructions` text as the task
  name instead would match jev's "the model never sees the question id" and scored a little better
  on the clear noul cases above, but then two questions with the same text would need separate
  handling and the id would no longer name the task.
- **Drop-in paths.** Serving `/v1/systemone` and `/gliner-2` would let the TypeSafe SDKs, a
  Decision-1 client and `GLiNER2API` point at the runner unchanged, but none of them can send
  `backend` and `precision`. That needs operator-configured defaults, which the protocol avoids
  today.
- **`examples`.** GLiNER2 and the TypeScript port already handle few-shot examples. Adding an
  optional `examples` field to classification tasks would let `/v1/classify_text` accept them; it
  needs a parity check against PyTorch first.
- **Structured `state`.** Writing a JSON object out as text is a translation the runner could do,
  but the Decide models were not trained on JSON input and the two runners would have to agree on
  one text form.
- **Long text.** `classify_text_long` could be translated as chunked requests merged by maximum, as
  GLiNER2 does.
