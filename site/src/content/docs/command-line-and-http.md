---
title: Use it from the command line or over HTTP
description: Every option of the classify command, and how to call the local server from curl or Node, read the reply, send many texts, and handle errors.
---

With the program and a model installed (see [Get started](/gliner-runner/start/)), you can run it
on your own text in two ways. In Terminal, `classify` takes one text and prints the answer. From
your own program, you send a request to the local server and read the reply. The server also takes
many texts in one request, and when a request fails, the reply says what went wrong.

## On the command line

`gliner-runner classify` classifies one text and prints the answer:

```sh
gliner-runner classify --labels weather,cooking,sports,money \
  "I will pay my rent on Friday, and then I will bake bread."
```

```text
money (confidence 0.783)
```

| Option | What it does |
| --- | --- |
| `--labels a,b,c` | The labels to choose from, separated by commas. Required. |
| `--multi` | Allow several labels, or none. Without it, the program picks exactly one label. |
| `--threshold 0.3` | The score a label needs with `--multi`. The default is 0.5. |
| `--describe label=text` | A short description of one label. Repeat it for more labels. |
| `--model decide-1b` | The model: `decide-340m` (the default), `decide-multilingual` or `decide-1b`. |
| `--json` | Print the full result, with the score of every label. |
| `--models-dir path` | Where the models are. The default is `GLINER_RUNNER_MODELS_DIR`, or else your user cache folder. |

With `--multi`, each chosen label is printed with its own score:

```sh
gliner-runner classify --multi --labels weather,cooking,sports,money \
  "I will pay my rent on Friday, and then I will bake bread."
```

```text
cooking (0.640), money (0.866)
```

When the text is `-` or left out, the command reads it from standard input:

```sh
echo "Where is my parcel?" |
  gliner-runner classify --labels billing,support,shipping
```

```text
shipping (confidence 0.906)
```

Each `classify` command loads the model again, which takes about 1.5 seconds (see
[Measured speed](/gliner-runner/how-it-works/#measured-speed)). To sort a file with one text per
line, use the loop from the [support ticket example](/gliner-runner/examples/support-tickets/).
For many texts, use the local server instead. It loads the model once.

## Over HTTP

Your program sends each text to a local server, which keeps the model loaded. This section goes
from starting the server to a working request from curl and from Node, how to read the reply, and
what to do when a request fails.

### Start the server

Start the local server and leave it running:

```sh
gliner-runner serve
```

```text
[gliner-runner-onnx] Application startup complete.
[gliner-runner-onnx] Listening on http://127.0.0.1:8090
```

It answers only on your own computer, at `127.0.0.1`, port 8090. `--port 8091` picks another
port. `gliner-runner serve --help` lists every option. In a second Terminal window, check that it
is running:

```sh
curl -s http://127.0.0.1:8090/healthz
```

```text
{"status":"ok","accepting_requests":true}
```

### Send a request with curl

Send a `POST` request to `/v1/infer`. This one asks for several labels:

```sh
curl -s http://127.0.0.1:8090/v1/infer \
  -H 'content-type: application/json' -d '{
  "model": "decide-340m", "backend": "onnx",
  "precision": "fp32", "operation": "classify",
  "text": "I will pay my rent on Friday, and then I will bake bread.",
  "schema": {"kind": "classification", "tasks": {
    "category": {
      "labels": ["weather", "cooking", "sports", "money"],
      "min_labels": 0, "max_labels": null
    }
  }}
}'
```

The server sends each reply on one line. On this page they are spread over several lines:

```json
{
  "request_id": "9d5e7db2-6929-4194-b634-aa06c8220faf",
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "output": {
    "category": {
      "value": ["cooking", "money"],
      "confidence": 0.8589424440847746,
      "probabilities": {
        "weather": 0.010178094737185768,
        "cooking": 0.6396000893475465,
        "sports": 0.0074523185172699884,
        "money": 0.8662421238746822
      }
    },
    "_meta": {
      "feasible": true,
      "decoder": "independent",
      "exact": true,
      "objective": 2.4417620301246643,
      "violations": []
    }
  },
  "error": null,
  "timing": {
    "queue_ms": 4.580874999999651,
    "inference_ms": 35.33479200000147
  },
  "usage": {"inputTokens": 30, "outputTokens": 0}
}
```

The request:

| Field | What it is |
| --- | --- |
| `model` | `decide-340m`, `decide-multilingual` or `decide-1b`. |
| `backend`, `precision`, `operation` | `"onnx"`, `"fp32"` and `"classify"`, as above. |
| `text` | The text to classify. One text per request, and it may not be empty. |
| `schema.kind` | Always `"classification"`. |
| `schema.tasks` | One entry for each question you ask about the text. The name (`category` here) is yours to choose, and the answer comes back under the same name. The model reads the name too, so use a plain word. See [The task name is part of the input](/gliner-runner/how-it-works/#the-task-name-is-part-of-the-input). |
| `labels` | The labels of a task. A label may not contain `(` or `)`. |
| `label_descriptions` | Optional. A description for some or all labels, such as `{"money": "Paying, spending or saving money"}`. |
| `min_labels`, `max_labels` | How many labels to return. The default is 1 and 1: exactly one. `0` and `null` mean any number, including none. |
| `threshold` | The score a label needs when more than one label is allowed. The default is 0.5. To go below 0.5, set `options.threshold` too. See [Lower the threshold](#lower-the-threshold). |
| `options.threshold` | Optional. The default is 0.5. A label that scores below it is never chosen. |
| `request_id` | Optional. A UUID that the reply repeats. The server makes one up when you leave it out. |

The server rejects a field it does not know, with status 422. See [Errors](#errors).

### Send a request from Node

Node 18 and later have `fetch` built in. Save this as `classify.mjs`:

```js
const reply = await fetch("http://127.0.0.1:8090/v1/infer", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    model: "decide-340m",
    backend: "onnx",
    precision: "fp32",
    operation: "classify",
    text: "I was charged twice for my last order.",
    schema: {
      kind: "classification",
      tasks: {
        category: { labels: ["billing", "support", "shipping"] },
      },
    },
  }),
});
const data = await reply.json();
if (!reply.ok) {
  throw new Error(`${reply.status}: ${JSON.stringify(data.detail)}`);
}
const answer = data.output.category;
console.log(answer.value, answer.confidence.toFixed(3));
console.log(answer.probabilities);
```

Run it while the server is running:

```sh
node classify.mjs
```

```text
billing 0.889
{
  billing: 0.8887295281128845,
  support: 0.04850131418933352,
  shipping: 0.06276915769778203
}
```

A web page can send the same request, but the browser first needs the server's permission. The
[React guide](/gliner-runner/use-in-react/) shows how.

### Read the reply

Check the status first. With status 200, the answer to each task is under `output`, under the
task's name. Any other status is an error, described under [Errors](#errors).

- **`value`.** With one label, the chosen label, as a string: `"billing"`. With several labels, a
  list: `["cooking", "money"]`. The list can be empty. An empty list is an answer: no label
  reached the threshold.
- **`confidence`.** With one label, the probability of the chosen label. With several labels, one
  number for the whole answer: the geometric mean, over every label you sent, of the probability of
  each chosen label and 1 minus the probability of each label left out. In the curl reply above,
  that is 0.640 and 0.866 for `cooking` and `money`, and 0.990 and 0.993 for `weather` and
  `sports`, which gives 0.859. It is `null` when `value` is empty. It is not the score of one
  label. For that, read `probabilities`.
- **`probabilities`.** Every label you sent, with its score.
- **`_meta`.** Details of how the labels were picked. You can ignore it.
- **`timing` and `usage`.** For the whole request: the time in milliseconds, and the number of
  tokens the model read.

[How it works](/gliner-runner/how-it-works/#what-the-numbers-mean) explains what the numbers mean,
and has an [example with two tasks](/gliner-runner/how-it-works/#what-it-can-and-cant-do) in one
request.

### Lower the threshold

With several labels, a label must pass two thresholds. `options.threshold` comes first: a label
that scores below it is never chosen. Of the labels left, those above the task's `threshold` are
chosen. Both are 0.5 unless you change them. To get labels that score below 0.5, set both.

Here the text is "I was charged twice, and now I cannot log in to see my invoices.", with five
labels. `billing` scores 0.347 and `account` 0.333:

| Thresholds you send | `value` |
| --- | --- |
| None (both 0.5) | `[]` |
| Task `threshold` 0.3 only | `[]` |
| `options.threshold` 0.3 only | `[]` |
| Both 0.3 | `["billing", "account"]` |

```sh
curl -s http://127.0.0.1:8090/v1/infer \
  -H 'content-type: application/json' -d '{
  "model": "decide-340m", "backend": "onnx",
  "precision": "fp32", "operation": "classify",
  "text":
    "I was charged twice, and now I cannot log in to see my invoices.",
  "schema": {"kind": "classification", "tasks": {
    "category": {
      "labels":
        ["billing", "technical", "shipping", "account", "cancellation"],
      "min_labels": 0, "max_labels": null, "threshold": 0.3
    }
  }},
  "options": {"threshold": 0.3}
}'
```

```json
{
  "request_id": "08bef1d5-6b22-475f-84f2-c98a89ea37aa",
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "output": {
    "category": {
      "value": ["billing", "account"],
      "confidence": 0.6063189535805477,
      "probabilities": {
        "billing": 0.34713420966455766,
        "technical": 0.2294686640049436,
        "shipping": 0.03179124858346374,
        "account": 0.333106979676832,
        "cancellation": 0.05012760635854453
      }
    },
    "_meta": {
      "feasible": true,
      "decoder": "independent",
      "exact": true,
      "objective": 0.3687696719233944,
      "violations": []
    }
  },
  "error": null,
  "timing": {
    "queue_ms": 4.611583999998402,
    "inference_ms": 39.28562500001863
  },
  "usage": {"inputTokens": 33, "outputTokens": 0}
}
```

The `--threshold` option of `classify` sets both.

### Many texts

Run one server and send all your texts to it. Do not run `classify` once per text: each run loads
the model again.

`POST /v1/batch` takes up to 256 requests at once, as `{"requests": [...]}`. The replies come back
as `{"responses": [...]}`, in the same order:

```sh
curl -s http://127.0.0.1:8090/v1/batch \
  -H 'content-type: application/json' -d '{"requests": [
  {"model": "decide-340m", "backend": "onnx",
   "precision": "fp32", "operation": "classify",
   "text": "I was charged twice for my last order.",
   "schema": {"kind": "classification", "tasks": {
     "category": {"labels": ["billing", "support", "shipping"]}}}},
  {"model": "decide-340m", "backend": "onnx",
   "precision": "fp32", "operation": "classify",
   "text": "Where is my parcel?",
   "schema": {"kind": "classification", "tasks": {
     "category": {"labels": ["billing", "support", "shipping"]}}}}
]}'
```

```json
{
  "responses": [
    {
      "request_id": "eca9528d-295d-46e1-9228-3a2e434fa8ca",
      "model": "decide-340m",
      "backend": "onnx",
      "precision": "fp32",
      "output": {
        "category": {
          "value": "billing",
          "confidence": 0.8887295281128845,
          "probabilities": {
            "billing": 0.8887295281128845,
            "support": 0.04850131418933352,
            "shipping": 0.06276915769778203
          }
        },
        "_meta": {
          "feasible": true,
          "decoder": "independent",
          "exact": true,
          "objective": 1.004750370979309,
          "violations": []
        }
      },
      "error": null,
      "timing": {
        "queue_ms": 4.667665999964811,
        "inference_ms": 35.94587500003399
      },
      "usage": {"inputTokens": 22, "outputTokens": 0}
    },
    {
      "request_id": "565deb97-8c48-497b-94a3-1608d972ebeb",
      "model": "decide-340m",
      "backend": "onnx",
      "precision": "fp32",
      "output": {
        "category": {
          "value": "shipping",
          "confidence": 0.9063632267191141,
          "probabilities": {
            "billing": 0.04140969515731491,
            "support": 0.05222707812357093,
            "shipping": 0.9063632267191141
          }
        },
        "_meta": {
          "feasible": true,
          "decoder": "independent",
          "exact": true,
          "objective": 1.1419072151184082,
          "violations": []
        }
      },
      "error": null,
      "timing": {
        "queue_ms": 4.609540999983437,
        "inference_ms": 35.94587500003399
      },
      "usage": {"inputTokens": 18, "outputTokens": 0}
    }
  ]
}
```

You can also send many `/v1/infer` requests at the same time. The server groups requests that
arrive together, and runs them as one batch.

Requests wait in a queue for the model. The queue holds 256 requests for each model. When it is
full, a new request gets status 429, code `queue_full`, and the header `Retry-After: 1`. Every
request of a batch goes into the queue, so a batch of 256 fills it alone. If one request of a batch
fails, the whole batch fails. In our test, two batches of 256 sent at the same time gave one reply
with status 200 and one with status 429. A batch of 257 is rejected with status 422.

To stay clear of the limit, send one batch at a time, and wait for its reply before you send the
next. If another program uses the same server, a 429 can still happen. Then wait the number of
seconds in `Retry-After` and send the same batch again. This Node function does both:

```js
const SERVER = "http://127.0.0.1:8090";
const task = {
  kind: "classification",
  tasks: { category: { labels: ["billing", "support", "shipping"] } },
};

async function classifyAll(texts) {
  const answers = [];
  for (let start = 0; start < texts.length; start += 256) {
    const requests = texts.slice(start, start + 256).map((text) => ({
      model: "decide-340m", backend: "onnx", precision: "fp32",
      operation: "classify", text, schema: task,
    }));
    let reply;
    for (;;) {
      reply = await fetch(`${SERVER}/v1/batch`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ requests }),
      });
      if (reply.status !== 429) break;
      const seconds = Number(reply.headers.get("retry-after") ?? 1);
      await new Promise((done) => setTimeout(done, seconds * 1000));
    }
    const data = await reply.json();
    if (!reply.ok) {
      const detail = JSON.stringify(data.detail);
      throw new Error(`${reply.status}: ${detail}`);
    }
    for (const response of data.responses) {
      answers.push(response.output.category.value);
    }
  }
  return answers;
}

const texts = [
  "I was charged twice for my last order.",
  "Where is my parcel?",
  "The app crashes when I log in.",
];
console.log(await classifyAll(texts));
```

```text
[ 'billing', 'shipping', 'support' ]
```

One batch of 256 one-sentence texts took 3.3 seconds on the computer from
[Measured speed](/gliner-runner/how-it-works/#measured-speed).

### Errors

When a request fails, the reply has a status other than 200 and a `detail`. Branch on
`detail.code`, not on the message. For example, a request with `labels` outside the task:

```json
{
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": ["body", "labels"],
      "msg": "Extra inputs are not permitted",
      "input": ["billing", "shipping"]
    }
  ]
}
```

When a field is wrong or missing, as here, `detail` is a list, and `loc` names the field.

| Reply | What happened | What to do |
| --- | --- | --- |
| 422, a list with `extra_forbidden` | A field the server does not know. Often `labels` outside the task. | Put `labels` under `schema.tasks.<name>`. |
| 422, a list with `missing` or `string_too_short` | A required field is missing, or `text` is empty. | Send every field of the request above. |
| 422 `invalid_schema` | The task is not valid, for example a label contains `(`. | Change it as the message says. |
| 422 `model_integrity` | The model name is not known, for example `decide-2b`. | Use `decide-340m`, `decide-multilingual` or `decide-1b`. |
| 409 `model_download_required` | The model is not installed. | Install it, for example `gliner-runner models install decide-340m`. |
| 409 `unsupported_configuration` | A `backend`, `precision` or `operation` other than `onnx`, `fp32` and `classify`. | Use those three values. |
| 409 `model_memory_limit` | The model needs more memory than the server may use. | Use a smaller model, or start the server with `--memory-limit-bytes`. |
| 413 `request_too_large` | The request is larger than 16 MiB. | Send fewer or shorter texts in each batch. |
| 429 `queue_full` | The queue is full. | Wait `Retry-After` seconds and send it again. |
| 503 `runtime_closed` | The server is shutting down. | Start `gliner-runner serve` again. |
| 404 | The path is wrong. | Use `/v1/infer` or `/v1/batch`. |

When the server is not running, there is no reply at all. curl prints nothing with `-s`, and exits
with code 7. Node throws `TypeError: fetch failed`, caused by `ECONNREFUSED`. Start
`gliner-runner serve`.

### Limits

- **One text per request.** It works best up to about 500 words. Split longer texts. See
  [How much text it can read](/gliner-runner/how-it-works/#how-much-text-it-can-read).
- **Up to 256 requests in a batch.**
- **Up to 256 waiting requests for each model.** `--queue-capacity` changes it.
- **Up to 16 MiB for one request.** `--max-body-bytes` changes it.
- **Only your own computer can reach the server.** It listens on `127.0.0.1`. `--host` changes it.
- **A web page needs permission.** The server answers web pages only from the addresses it allows.
  See [Use it in a React app](/gliner-runner/use-in-react/).

The full reference is the [local HTTP protocol](/gliner-runner/docs/protocol/). The server also
describes itself at `http://127.0.0.1:8090/openapi.json`.

## Coming from jev, GLiNER2 or Azure

If you already ask these questions with another API, the server takes your request in that API's
shape. It turns it into the ordinary requests above, so the numbers are the same as if you had
written those yourself. Add `"backend": "onnx"` and `"precision": "fp32"`, and put one of our
models in `model`.

| You use | Send it to | What changes |
| --- | --- | --- |
| TypeSafe jev: `noul`, `choice`, `score` | `POST /v1/questions` | `state` must be text. The question id is part of what the model reads, so give it a clear name. |
| Azure AI Foundry Microsoft-Decision-1 | `POST /v1/questions` | The same request as jev. |
| GLiNER2 `classify_text`, `batch_classify_text` | `POST /v1/classify_text` | Send `text` or `texts`. The result comes back under `result` or `results`. |

**jev or Microsoft-Decision-1.** Each answer has jev's fields: `noul` is the probability of yes,
and a value near 0.5 means the model cannot tell, not "a bit yes". `choice` and `score` come with
`probabilities` and jev's `confidence`. There is one more type, `multi_select`, for when several
options can apply.

```sh
curl -s http://127.0.0.1:8090/v1/questions \
  -H 'content-type: application/json' -d '{
  "model": "decide-340m", "backend": "onnx", "precision": "fp32",
  "state": "The API returns 500 on every call.",
  "questions": {"team": {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {
      "billing": "Charges, invoices, and refunds",
      "engineering": "Bugs, errors, and outages",
      "support": "How-to and account questions"
    }
  }}
}'
```

```json
{
  "request_id": "a1905374-994b-4f54-bbb6-f8932c49d72b",
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "engineering",
      "confidence": 0.48691054473256434,
      "probabilities": {
        "billing": 0.13386986595729453,
        "engineering": 0.6579403631550429,
        "support": 0.20818977088766247
      }
    }
  },
  "usage": {"input_tokens": 56, "output_tokens": 0}
}
```

**GLiNER2.** Send the `tasks` you pass to `classify_text`:

```sh
curl -s http://127.0.0.1:8090/v1/classify_text \
  -H 'content-type: application/json' -d '{
  "model": "decide-340m", "backend": "onnx", "precision": "fp32",
  "text": "The battery is weak and the delivery was slow.",
  "tasks": {
    "sentiment": ["positive", "negative", "neutral"],
    "aspects": {
      "labels": ["battery", "screen", "price", "delivery"],
      "multi_label": true, "cls_threshold": 0.4
    }
  }
}'
```

```json
{
  "request_id": "5322d861-f05b-4f8d-821d-3bf1e0225f4a",
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "result": {
    "sentiment": "negative",
    "aspects": ["battery", "delivery"]
  },
  "usage": {"inputTokens": 38, "outputTokens": 0}
}
```

Some things these models cannot do, and the server says so with status 422: finding things in
images (jev `box`), reading JSON `state`, and extracting entities, relations or JSON (GLiNER2
`extract_*`). [API shapes](/gliner-runner/docs/api-shapes/) lists every function of the three APIs
and what the server does with each.

## Next

[Use it in a React app](/gliner-runner/use-in-react/): a small web page that sends this request
from the browser.

[Use it from a coding agent](/gliner-runner/use-from-a-coding-agent/): a skill that teaches a coding
agent the same things as this page.
