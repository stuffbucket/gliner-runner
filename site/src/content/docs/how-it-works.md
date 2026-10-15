---
title: How it works
description: What the program can and cannot do, what you control, how fast it is, and when a chat model is the better tool.
sidebar:
  order: 1
---

## What a classifier does

A classifier reads a piece of text and picks from a list. You write the list. The program does
not invent new categories and it does not write anything back. It only decides which of your
labels fit.

```sh
gliner-runner classify --labels billing,support,shipping "I was charged twice for my last order."
```

```text
billing (confidence 0.889)
```

The labels are `billing`, `support`, and `shipping`. The answer is `billing`.

## What it can and can't do

The Decide models in this program do one thing: they classify text into labels you choose.

What it can do:

- **Pick one label** from your list, with a confidence score.
- **Pick several labels** when a text fits more than one, with `--multi`.
- **Pick no label** when nothing fits. See [One label or many](#one-label-or-many).
- **Use label descriptions.** A short description next to a label can change the answer. See
  [Descriptions help](#descriptions-help).
- **Answer several questions about the same text in one request.** This works through the local
  server (`gliner-runner serve`), not through the `classify` command. Each question is a task with
  its own labels. Here the text is classified for its topic and for its urgency:

```sh
curl -s http://127.0.0.1:8090/v1/infer -H 'content-type: application/json' -d '{
  "model": "decide-340m", "backend": "onnx", "precision": "fp32", "operation": "classify",
  "text": "When you have time, could you update the address on my account?",
  "schema": {"kind": "classification", "tasks": {
    "topic": {"labels": ["billing", "technical", "shipping", "account"]},
    "urgency": {"labels": ["urgent", "can wait"]}
  }}
}'
```

```text
{"request_id":"6f4f0a75-54f0-4632-a2d8-3b8d3f0876ff","model":"decide-340m","backend":"onnx","precision":"fp32","output":{"topic":{"value":"account","confidence":0.8588331120201168,"probabilities":{"billing":0.02427913726466854,"technical":0.02700261628763882,"shipping":0.08988513442757576,"account":0.8588331120201168}},"urgency":{"value":"can wait","confidence":0.8783437883969512,"probabilities":{"urgent":0.12165621160304882,"can wait":0.8783437883969512}},"_meta":{"feasible":true,"decoder":"independent","exact":true,"objective":2.023317337036133,"violations":[]}},"error":null,"timing":{"queue_ms":4.5986660000016855,"inference_ms":28.384084000001167},"usage":{"inputTokens":41,"outputTokens":0}}
```

The answer has one entry for each task: `topic` is `account` and `urgency` is `can wait`. The
whole request took about 28 milliseconds.

What it cannot do:

- **Find or pull out pieces of text.** It cannot return a name, a number, or a date from a message.
  It can tell you whether a message contains a phone number. It cannot point at the phone number
  or remove it. The [personal information example](/gliner-runner/examples/personal-information/)
  shows what that means in practice.
- **Extract relationships.** It cannot say who did what to whom.
- **Write text.** It does not draft, summarize, or rewrite. It only chooses from your labels.

## You pick the labels

The labels can be anything: `urgent, can wait`, `positive, mixed, negative`, `invoice, contract,
receipt`. Short, plain words work best. Pick labels that do not overlap, so that a person could
decide between them without arguing.

## Descriptions help

A label name alone can be ambiguous. Is `account` about logging in, or about money? Add a short
description to say what you mean, with `--describe label=description`:

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

Descriptions can change the answer. If the answers are not what you expect, rewrite the
descriptions before anything else.

## One label or many

By default the program picks exactly one label. Add `--multi` when a text can fit several labels,
or none:

```sh
gliner-runner classify --multi --labels battery,screen,price,delivery "The battery is weak and the delivery was slow."
```

```text
battery (0.948), delivery (0.919)
```

With `--multi`, every label is scored on its own, and every label whose score reaches the
threshold is returned. If none does, the program says so. This text is about knitting, and none of
the labels is about knitting:

```sh
gliner-runner classify --multi --labels weather,cooking,sports,money "My grandmother taught me to knit a scarf when I was seven."
```

```text
no label reached the 0.5 threshold
```

Without `--multi`, the program always picks one label, even when nothing fits. The only sign is a
low confidence:

```sh
gliner-runner classify --labels weather,cooking,sports,money "My grandmother taught me to knit a scarf when I was seven."
```

```text
cooking (confidence 0.345)
```

To get "none" with one label per text, add a label for it:

```sh
gliner-runner classify --labels weather,cooking,sports,money,"none of these" "My grandmother taught me to knit a scarf when I was seven."
```

```text
none of these (confidence 0.404)
```

## Confidence

Each answer comes with a number between 0 and 1. A higher number means the program is surer.
It is not a percentage of how often the answer is right. Use it to compare answers: send low
numbers to a person for review.

In many-label mode, a label is returned when its score reaches the threshold, which is 0.5 unless
you change it with `--threshold`:

```sh
gliner-runner classify --multi --threshold 0.3 --labels billing,technical,shipping,account,cancellation "I was charged twice, and now I cannot log in to see my invoices."
```

```text
billing (0.347), account (0.333)
```

With the default threshold of 0.5, the same text returns no label. A lower threshold returns
more labels, including more wrong ones. Add `--json` to see the score of every label:

```sh
gliner-runner classify --json --labels urgent,normal,low "Our payment system is down and customers cannot check out."
```

```text
{"category":{"value":"urgent","confidence":0.6183409935537204,"probabilities":{"urgent":0.6183409935537204,"normal":0.18731622346263635,"low":0.19434278298364327}},"_meta":{"feasible":true,"decoder":"independent","exact":true,"objective":0.13862642645835876,"violations":[]}}
```

## How much text it can read

The model reads the whole text you give it. It does not cut it off. But the time it takes grows
quickly as the text gets longer, so the practical limit is time, not a hard stop.

We classified one text about a billing problem at four lengths, with the labels
`billing, cooking, weather, sports` and the `decide-340m` model. The texts were built from ten
different sentences about the same problem, so they stay on one topic. The computer is the one
described under [Measured speed](#measured-speed).

```sh
gliner-runner classify --labels billing,cooking,weather,sports "$(cat text-500-words.txt)"
```

```text
billing (confidence 0.973)
```

| Text | Words | Tokens | Model time on the server (3 runs) | Whole `classify` command (1 run) | Answer |
| --- | --- | --- | --- | --- | --- |
| One sentence | 16 | 32 | 18 to 29 ms | 1.75 s | billing (0.927) |
| One paragraph | 102 | 125 | 52 to 61 ms | 1.52 s | billing (0.996) |
| About 500 words | 500 | 554 | 265 to 282 ms | 1.76 s | billing (0.973) |
| About 2,000 words | 2,002 | 2,171 | 1.96 to 2.23 s | 3.72 s | billing (0.907) |

The answer stayed correct at every length. Two longer runs on the server, one run each: 4,000 words
(4,327 tokens) took 8.0 seconds and still answered `billing (0.841)`. A text of 8,000 words had not
finished after several minutes, and we stopped it.

A long text that covers more than one topic is a different case. We took cooking text and added
one sentence about a double charge, with the same four labels. A plain cooking text of 100 words
answered `cooking (0.997)`. With the billing sentence added at the end, the answer was
`billing (0.523)`. The same happened at 500 words (`billing (0.789)`) and at 2,000 words
(`billing (0.957)`), and when we put the sentence at the start instead of the end. One sentence
about another topic can decide the answer for a long text.

What this means in practice:

- **A sentence up to a few paragraphs (about 500 words) works best.** It is fast and the answers
  were right.
- **Up to about 2,000 words works, but is slower.** Expect a couple of seconds for each text.
- **Beyond that, split the text.** Split a long document into paragraphs or sections and classify
  each part. This is faster, and it also keeps a mixed document from being decided by one passage.

Your own texts will behave differently from ours. Try a few of your own at the length you need.

## Why it is small and fast

The models are small compared with a chat model. The largest one is about 4.8 GB on disk, the
default is about 2.0 GB. The program runs the model on your computer's processor. It does not
send your text anywhere and it does not need an internet connection after the models are
downloaded.

- **Private.** The text stays on your machine.
- **Same input, same answer.** Run the same text with the same labels again and you get the same
  score. We ran one message three times and got `urgent (confidence 0.678)` each time.
- **Cheap.** There is no fee per request. The cost is the disk space and the electricity.
- **Works offline.** After the download, turn off the network and it keeps working.

### Measured speed

These numbers were measured on an Apple M5 Max laptop (18 cores, 128 GB memory), using the
processor only, with the `decide-340m` model, short English texts (one sentence), and three
labels. Your computer will give different numbers. An older or smaller machine will be slower,
and longer texts and more labels take longer.

| What we ran | Result |
| --- | --- |
| One `classify` command, start to finish (5 runs) | 1.4 to 1.5 seconds each |
| `gliner-runner demo` (6 texts, 3 runs) | 1.3 to 1.5 seconds in total |
| `serve`, one request at a time (100 requests, 3 runs) | 37 to 40 requests per second; median 25 ms per request, 95th percentile 29 to 30 ms |
| `serve`, 4 requests at once (100 requests, 3 runs) | 65 to 66 requests per second; median 60 ms per request |
| `serve`, 16 requests at once (100 requests, 3 runs) | 82 to 83 requests per second; median 190 ms per request |

The one-command time is mostly spent starting the program and loading the model, which happens
every time you run `classify`. The demo loads the model once and classifies six texts in about
the same time. If you have many texts to sort, run the program as a service with
`gliner-runner serve`. It loads the model once and then answers each request in about 25
milliseconds. The [React guide](/gliner-runner/use-in-react/) uses the service.

## When a chat model is the better tool

Use a chat model such as Claude when you need:

- **Writing.** Drafting a reply, summarizing a document, rewriting text.
- **Open-ended answers.** "What is this customer really upset about?"
- **Several steps of reasoning.** Working through a problem, comparing documents.
- **Labels you cannot list.** If you do not know the categories in advance, this program has
  nothing to choose from.

## When this program is the better tool

- **Fixed categories.** You know the labels, and the text must go into one or more of them.
- **Large volumes.** Thousands of messages, with no per-message fee.
- **Speed.** An answer in tens of milliseconds, not seconds.
- **Privacy.** The text cannot leave the computer.
- **Consistency.** The same text must always get the same label.
- **Offline use.** No internet connection, or none you can rely on.
