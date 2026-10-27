---
title: Accuracy
description: How often each model picks the right label on public datasets that already have the answers, where it does badly, and how to check it yourself.
---

We sent the three models texts from eight public datasets whose right answers are known, and
counted how often they agreed. Each text went through a runner on one computer, the same way
the [command line and HTTP](/gliner-runner/command-line-and-http/) send it. This page gives
the results in plain terms. The
[full report](https://github.com/stuffbucket/gliner-runner/blob/main/benchmark-results/2026-10-21-accuracy/REPORT.md)
has every number, the method, and the per-label tables.

The [Evaluation](/gliner-runner/evaluation/) page answers a different question: whether the
runner gives the same answers as the models' original Python library. It does. This page
asks whether those answers are right.

## The short answer

- **Banking questions, 77 possible intents:** `decide-340m` picks the right one for 71 of
  every 100.
- **Product reviews, positive or negative:** right for 93 of every 100.
- **News headlines, 42 sections:** right for 42 of every 100. Many sections overlap.
- **fastino's own test of everyday decisions** (support, banking, triage, sentiment and
  more): 63 of every 100 on the public half. fastino reports 60 on the half it keeps back.
- **Language of a message, 10 languages:** `decide-340m` gets 85 of every 100. Use it rather
  than `decide-multilingual`, which gets 58.
- **Flagging toxic comments:** it catches nearly all of them, but it also flags more than
  half of the ordinary ones. Do not use it alone to remove posts.
- **Several labels at once:** at the default threshold it often gives no label at all. Lower
  the threshold, and check the result on your own texts.
- **Descriptions:** short descriptions written from the label names did not make answers
  more accurate on any dataset, and they made requests slower.

## How to read the numbers

- **Accuracy** is how many texts out of 100 got the right label.
- **Most common label** is what you would get by always giving the most frequent answer. A
  model has to beat it to be useful. **Random** is a guess among the labels.
- **F1** is used when a text can have several labels. It balances labels the model missed
  against labels it added that were wrong. 100 is perfect.
- **Calibration error** says how far the confidence is from how often the model is right.
  0.05 means that when it says 80%, it is right about 75 to 85 times in 100. Lower is better.
- **Sample size.** Most rows use 1,000 texts, so a number can be off by about 3 either way.
  Rows marked 200 can be off by about 6.

All times are for one request at a time, on an Apple M5 Max, on the CPU.

## Sort support messages and intents

| Dataset | Labels | Texts | `decide-340m` | with descriptions | `decide-1b` | `decide-multilingual` | Most common label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Banking questions (BANKING77) | 77 | 770 | 71.3 | | | | 1.3 |
| Banking questions, same 200 texts | 77 | 200 | 73.5 | 71.0 | 72.0 | | |
| fastino suite: support intent | 28 | 100 | 76.0 | 70.0 | | 65.0 | |
| fastino suite: banking intent | 16 | 100 | 64.0 | 61.0 | | 58.0 | |
| fastino suite: ticket queue | 16 | 100 | 58.0 | 58.0 | | 57.0 | |
| Help-desk tickets: queue | 10 | 1,000 | 22.4 | 19.2 | 24.6 | 18.8 | 29.2 |

The help-desk tickets are synthetic, and many of their labels look arbitrary when you read
the tickets. Every model scored below always answering "Technical Support". Treat that row as
a warning about the dataset more than about the models.

## Tell what is urgent

| Dataset | Labels | Texts | `decide-340m` | with descriptions | `decide-1b` | `decide-multilingual` | Most common label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Help-desk tickets: priority | 3 | 1,000 | 39.6 | 40.3 | 32.4 | 42.9 | 40.5 |
| fastino suite: ticket urgency | 4 | 100 | 34.0 | 40.0 | | 35.0 | |
| fastino suite: clinic request is urgent | 2 | 100 | 72.0 | 72.0 | | 76.0 | |

Urgency is the weakest job measured here. On the help-desk tickets only `decide-multilingual`
came out above always answering "medium" (42.9 against 40.5), by less than its margin of
error.

## Sort by topic

| Dataset | Labels | Texts | `decide-340m` | with descriptions | `decide-multilingual` | Most common label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| News headlines (HuffPost News Category) | 42 | 1,000 | 41.5 | 40.5 | | 17.0 |
| fastino suite: news topic | 10 | 100 | 69.0 | 68.0 | 63.0 | |
| fastino suite: document type | 12 | 100 | 79.0 | 81.0 | 75.0 | |
| fastino suite: science paper field | 10 | 100 | 47.0 | 43.0 | 38.0 | |

## Read reviews and feelings

| Dataset | Labels | Texts | `decide-340m` | with descriptions | `decide-multilingual` | Most common label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Amazon reviews, positive or negative | 2 | 1,000 | 92.8 | | | 50.0 |
| fastino suite: review sentiment | 3 | 100 | 84.0 | 83.0 | 81.0 | |
| Reddit comments, 28 emotions, several at once (F1, threshold 0.5) | 28 | 1,000 | 15.2 | 6.5 | | 31.8 |
| Same, threshold 0.19 chosen on separate texts (F1) | 28 | 1,000 | 33.1 | 28.0 (at 0.16) | | 31.8 |

The emotion rows show the main thing to know about several labels at once: at the default
threshold of 0.5, 847 of the 1,000 comments got no label at all. With a lower threshold the
model does about as well as always answering "neutral". Pick a threshold on some of your own
texts that you have labelled, not on the ones you report.

## Flag things to check

| Dataset | Labels | Texts | `decide-340m` | with descriptions | Most common label |
| --- | ---: | ---: | ---: | ---: | ---: |
| News-site comments, toxic or not (Civil Comments) | 2 | 1,000 | 46.0 | 47.6 | 92.0 |

80 of the 1,000 comments are toxic. The model flagged 610 of them, including 75 of the 80
toxic ones, and 535 ordinary ones. It is better used to sort a review queue than to remove
anything on its own.

## Other languages

The same 100 short requests to a voice assistant, in ten languages (MASSIVE). "Topic" is
which of 18 kinds of request it is; "language" is which of the ten languages it is in.

| Language | Topic, `decide-340m` | Topic, `decide-multilingual` | Language, `decide-340m` | Language, `decide-multilingual` |
| --- | ---: | ---: | ---: | ---: |
| English | 57 | 47 | 95 | 34 |
| German | 43 | 46 | 98 | 46 |
| Spanish | 45 | 48 | 92 | 97 |
| French | 52 | 51 | 95 | 93 |
| Italian | 46 | 46 | 92 | 85 |
| Portuguese | 47 | 43 | 73 | 69 |
| Dutch | 46 | 48 | 83 | 67 |
| Russian | 41 | 47 | 99 | 43 |
| Japanese | 54 | 53 | 50 | 19 |
| Chinese | 40 | 53 | 73 | 25 |
| All 1,000 | 47.1 | 48.2 | 85.0 | 57.8 |

Each language cell is 100 texts, so it can be off by about 10. The multilingual model holds up
better on Russian and Chinese topics. For naming the language, `decide-340m` is much better,
except on Japanese, which it often calls Russian.

## Against the numbers fastino publishes

fastino tests the Decide models on its own suite, Fast Decisions: 17 kinds of everyday
decision. It publishes the scores for a test half it keeps private. We can only run the public
half, so the two are not the same test.

| Model | fastino, private half | This run, public half |
| --- | ---: | ---: |
| `decide-340m` | 60.2 | 63.4 (all 1,700 texts) |
| `decide-1b` | 59.6 | 60.8 (200 texts) |
| `decide-multilingual` | 56.7 | 58.6 (all 1,700 texts) |

On the same 200 texts, `decide-340m` scored 59.7 and `decide-multilingual` 58.1. fastino's
blog also gives 75.3 for support intent and 64.3 for banking intent; we got 76.0 and 64.0 on
100 texts each.

## Where it does badly, and why

- **Several labels at once.** Each label gets its own score, and the scores are low, so the
  default threshold of 0.5 leaves many texts with nothing. "I despise [NAME] and all his
  cronies" scored anger 0.48 and disgust 0.47 and got no label.
- **Flagging.** It leans hard towards "toxic". A comment that opens by citing a Washington
  Post article was flagged at 0.87.
- **Labels that overlap or are short codes.** STYLE against STYLE & BEAUTY, PARENTS against
  PARENTING, or a voice assistant's "iot" label for smart-home requests: "prepare coffee now"
  went to "cooking" at 1.00. Clear, separate label names help more than descriptions did.
- **Writing systems.** `decide-340m` called 48 of 100 Japanese requests Russian.
  `decide-multilingual` answered "Spanish" for 342 of 1,000 requests, including 55 of the
  100 English ones.
- **Urgency.** On the help-desk tickets, `decide-340m` and `decide-1b` did worse than always
  answering "medium", and `decide-multilingual` only slightly better.

## Descriptions and speed

Every label, and its description, is part of what the model reads, so more labels and longer
descriptions take longer. Median time per request:

| Job | Labels | `decide-340m` | with descriptions | `decide-multilingual` | `decide-1b` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ticket priority | 3 | 79 ms | 207 ms | 38 ms | 238 ms |
| Ticket queue | 10 | 93 ms | 132 ms | 54 ms | 308 ms |
| Reddit emotions | 28 | 104 ms | 345 ms | | |
| News sections | 42 | 270 ms | 1,452 ms | | |
| Banking intents | 77 | 640 ms | 5,241 ms | | 651 ms |

The descriptions here were written mechanically, once, from the label names or the datasets'
own definitions. Better descriptions might help your labels; these did not raise accuracy on
any dataset. On some datasets they made the confidence a better guide to whether the answer
is right (on fastino's suite, the calibration error fell from 0.16 to 0.02).

## Check it yourself

The datasets, their licences and pinned versions, the descriptions, and the scripts are in
[`benchmarks/accuracy/`](https://github.com/stuffbucket/gliner-runner/tree/main/benchmarks/accuracy).
With a runner on port 8102:

```sh
python -I benchmarks/accuracy/fetch.py --out /tmp/gr-work/datasets
python -I benchmarks/accuracy/evaluate.py sample --out benchmark-results/2026-10-21-accuracy
python -I benchmarks/accuracy/evaluate.py run --out benchmark-results/2026-10-21-accuracy \
  --model decide-340m banking77/intent
python -I benchmarks/accuracy/evaluate.py score --out benchmark-results/2026-10-21-accuracy
```

`fetch.py` needs `pyarrow`. The other scripts need only Python. The sample uses a fixed seed,
so you get the same texts we did. One of the eight datasets, the help-desk tickets, is for
non-commercial use only.

The best test is still your own texts. Label 50 to 100 of them by hand, run them through the
labels you plan to use, and count.

## Next

[Evaluation](/gliner-runner/evaluation/): how the runner's answers were checked against the
original implementation.
