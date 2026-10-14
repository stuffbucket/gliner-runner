---
title: GLiNER Runner
description: A local HTTP runner for GLiNER-family models with pinned models, explicit downloads, and published parity evidence.
---

GLiNER Runner sorts a piece of text into labels you choose: give it a message and a list of
labels such as `billing, support, shipping`, and it tells you which one fits, with a confidence
score. It runs on your own computer with no Python, and your text never leaves your machine. It
serves the three curated Fastino GLiNER2.5 Decide models, each pinned to an exact upstream revision
(see the [model pages](/gliner-runner/models/)).

## Install

With [Homebrew](https://brew.sh) (macOS on Apple Silicon, and Linux):

```sh
brew tap stuffbucket/tap
brew install gliner-runner
```

To update later:

```sh
brew update && brew upgrade gliner-runner
```

To test a release candidate instead, run `brew uninstall gliner-runner` and then
`brew install gliner-runner-rc`. The release candidate conflicts with the stable package, so
install only one of them.

Without Homebrew, this installs the same program into `~/.local`:

```sh
curl -fsSL https://github.com/stuffbucket/gliner-runner/releases/latest/download/install.sh | sh
```

Check that it works:

```sh
gliner-runner --version
```

If an older Python `gliner-runner` (from `pip`) is also on your `PATH`, it can shadow this one.
`which gliner-runner` shows which program runs; the version output above starts with
`gliner-runner-onnx` for the right one.

## Download a model

The program does not include a model. Download one once (this asks you to confirm, then
downloads from Hugging Face and prepares it):

```sh
gliner-runner models install decide-340m
```

| Model | Download | Use it for |
| --- | --- | --- |
| `decide-340m` | about 1.8 GiB | English text; the default |
| `decide-multilingual` | about 1.1 GiB | Spanish, French, German and other languages |
| `decide-1b` | about 4.4 GiB | English text, larger model |

Models are stored in your user cache (`~/Library/Caches/gliner-runner/models` on macOS,
`~/.cache/gliner-runner/models` on Linux). Add `--yes` to skip the confirmation question.

## Try it

Run a few built-in examples:

```sh
gliner-runner demo
```

```text
gliner-runner demo (model: decide-340m)

1. Support ticket routing
   Text:    I was charged twice for my last order and I want one of the payments refunded.
   Labels:  billing, support, shipping
   Result:  billing (confidence 0.866)

2. Support ticket routing
   Text:    My package has been in transit for two weeks and the tracking page has not changed.
   Labels:  billing, support, shipping
   Result:  shipping (confidence 0.921)

3. Several topics in one message (multi-label)
   Text:    The battery is weak and the delivery was slow.
   Labels:  battery, screen, price, delivery  (several allowed)
   Result:  battery (0.948), delivery (0.919)

4. Labels with descriptions
   Text:    Can I swap the blue jacket I bought for the same one in a larger size?
   Labels:  refund, exchange, question
            refund: The customer wants their money back
            exchange: The customer wants a different item or size
            question: The customer asks for information only
   Result:  exchange (confidence 0.833)

5. Sentiment
   Text:    The checkout was quick and the staff were friendly and helpful.
   Labels:  positive, neutral, negative
   Result:  positive (confidence 0.944)

6. Priority
   Text:    Our payment system is down and customers cannot check out. We need help right now.
   Labels:  urgent, normal, low
   Result:  urgent (confidence 0.855)

Classified 6 texts in 1.5 s.
Try your own: gliner-runner classify --labels billing,support,shipping "your text"
```

Then classify your own text. Pick the labels yourself.

**Route a support ticket** (one label):

```sh
gliner-runner classify --labels billing,support,shipping "My package never arrived and the tracking page has not updated in two weeks."
```

```text
shipping (confidence 0.942)
```

**Allow several labels** with `--multi`:

```sh
gliner-runner classify --labels battery,screen,price,delivery --multi "The battery is weak and the delivery was slow."
```

```text
battery (0.948), delivery (0.919)
```

**Describe the labels** when their names are not enough, with `--describe label=text`:

```sh
gliner-runner classify --labels refund,exchange,question \
  --describe "refund=The customer wants their money back" \
  --describe "exchange=The customer wants a different item or size" \
  --describe "question=The customer only asks for information" \
  "Can I swap the blue jacket for a larger size?"
```

```text
exchange (confidence 0.826)
```

**Other languages** need the multilingual model (`gliner-runner models install decide-multilingual`):

```sh
gliner-runner classify --model decide-multilingual --labels billing,support,shipping "Me cobraron dos veces el mismo pedido."
```

```text
billing (confidence 0.779)
```

```sh
gliner-runner classify --model decide-multilingual --labels billing,support,shipping "Mon colis n'est toujours pas arrivé après deux semaines."
```

```text
shipping (confidence 0.514)
```

Text can also come from a pipe, and `--json` prints the full result for scripts:

```sh
echo "The checkout was quick and the staff were friendly." | gliner-runner classify --labels positive,neutral,negative
```

```text
positive (confidence 0.935)
```

```sh
gliner-runner classify --labels urgent,normal,low --json "Our payment system is down and customers cannot check out."
```

```json
{"category":{"value":"urgent","confidence":0.6183409935537204,"probabilities":{"urgent":0.6183409935537204,"normal":0.18731622346263635,"low":0.19434278298364327}},"_meta":{"feasible":true,"decoder":"independent","exact":true,"objective":0.13862642645835876,"violations":[]}}
```

Confidence scores depend on the model and the wording; they are not percentages of accuracy.
If a model is not installed, the command prints the exact `gliner-runner models install` command to
run. `gliner-runner --help` lists every command.

## Run it as a local service

Applications can call the runner over HTTP. Start it and leave it running:

```sh
gliner-runner serve
```

It listens on `http://127.0.0.1:8090`. In another terminal:

```sh
curl -s http://127.0.0.1:8090/v1/infer -H 'content-type: application/json' -d '{
  "model": "decide-340m",
  "backend": "onnx",
  "precision": "fp32",
  "operation": "classify",
  "text": "I was charged twice for my last order.",
  "schema": {
    "kind": "classification",
    "tasks": {"category": {"labels": ["billing", "support", "shipping"]}}
  }
}'
```

The answer is JSON; the result is under `output.category.value` (here `"billing"`). Use
`--port` to change the port; see [ONNX runtime](/gliner-runner/docs/onnx-runtime/) for every option and the
[protocol](/gliner-runner/docs/protocol/) for the request and response shapes.

## Check for updates

With Homebrew, `brew update && brew upgrade gliner-runner` installs the newest release. `gliner-runner update check`
tells you whether a newer stable release exists and which command installs it; it never installs
anything.

## How it fits together

There are two servers behind one [HTTP protocol](/gliner-runner/docs/protocol/):

- **`gliner-runner`** (also installed as `gliner-runner-onnx`) is a Python-free Node.js runner using
  ONNX Runtime on CPU in FP32. It is the recommended way to install and run the models. See
  [ONNX runtime](/gliner-runner/docs/onnx-runtime/).
- **The Python server** has a PyTorch backend that is the correctness oracle. See
  [Backends](/gliner-runner/docs/backends/).

There is no silent fallback: a request for an unsupported backend, precision, device, or
operation fails with a structured error. Inference never downloads a model unless you opt in.
The [protocol reference](/gliner-runner/docs/protocol/) has the full request, response, and error
shapes.

## Where to go next

- [Models](/gliner-runner/models/): one card per curated model with pinned revision, files,
  checksums, and evaluation results.
- [Evaluation](/gliner-runner/evaluation/): the ONNX-versus-PyTorch parity matrix and the
  evaluation receipt.
- [Docs](/gliner-runner/docs/onnx-runtime/): operations, protocol, architecture, backends,
  benchmarks.
- [Releases](/gliner-runner/releases/): stable and release-candidate builds.

Model and evaluation pages are generated at build time from the repository's manifests, receipt,
and benchmark results, so they always match what the runner ships.
