---
title: How it works
description: What the program does, what you control, how fast it is, and when a chat model is the better tool.
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
threshold is returned. If none does, the program says so:

```text
no label reached the 0.5 threshold
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

## What it does not do

This program classifies text. It does not find or cut out pieces of text, so it cannot pull a name,
a date, or an account number out of a message, and it cannot black out parts of a document. It
can tell you whether a message contains a phone number. It cannot point at the phone number. The
[personal information example](/gliner-runner/examples/personal-information/) explains what that
means in practice.
