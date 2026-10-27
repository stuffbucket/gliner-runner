---
title: More ideas
description: Other jobs you can do by sorting text, grouped by job, each with a starting set of labels.
sidebar:
  order: 4
---

Once the program runs, give it more to sort. Each idea below is a list of labels to try. Put the
labels after `--labels`, separated by commas, and
add `--multi` where a text can fit more than one. Add a short `--describe` for any label whose
meaning is not obvious. See [how it works](/gliner-runner/how-it-works/) for how.

The jobs we ran in full are under [Examples](/gliner-runner/examples/). The ones below are
starting points we have not run in full.

## Sort incoming messages

| Idea | What you classify | Labels to start with |
| --- | --- | --- |
| Inbox rules | Each incoming email | `needs reply, newsletter, receipt, calendar invite, other` |
| Form triage | Free-text answers on a contact form | `sales question, support request, press, job application, other` |
| Bug reports | Each new report | `crash, slow, wrong result, display problem, feature request` |
| Event questions | Questions sent before an event | `schedule, location, tickets, accessibility, food` |
| Job applications | A cover letter | `engineering, design, sales, operations, other` |
| Donation notes | Notes sent with a donation | `in memory of, in honor of, general, question` |
| Chat messages | Messages in a group chat | `question, announcement, joke, plan, other` |

## Find what matters

| Idea | What you classify | Labels to start with |
| --- | --- | --- |
| Meeting notes | Each line of your notes | `decision, action item, question, background` |
| Log lines | One line from an application log | `error, warning, routine, security concern` |

## Understand feedback

| Idea | What you classify | Labels to start with |
| --- | --- | --- |
| Survey answers | Open-ended survey answers | `praise, complaint, suggestion, question` |
| Product feedback | Customer comments | `price, quality, shipping, ease of use, support` (with `--multi`) |

## Clean up lists

| Idea | What you classify | Labels to start with |
| --- | --- | --- |
| File naming | The first paragraph of a document | `invoice, contract, resume, receipt, letter, other` |
| News headlines | A headline | `business, sports, science, politics, local, other` |
| Recipe notes | Your own recipe notes | `breakfast, main dish, side, dessert, snack` |

## A starting command

This is the pattern behind every row. Replace the text and the labels:

```sh
gliner-runner classify \
  --labels "invoice,contract,resume,receipt,letter,other" \
  "Invoice 2041. Amount due: 480.00. Payment terms: net 30 days."
```

```text
invoice (confidence 0.760)
```

To sort a whole file, one item per line, use the loop from the
[support ticket example](/gliner-runner/examples/support-tickets/) and end it with
`done < yourfile.txt`.

## Before you rely on it

These label sets are starting points. We ran the pages under [Examples](/gliner-runner/examples/)
in full. For this list we only tried a few items, and some did not go well. Short labels with no
descriptions are often not enough for harder jobs. Three that we ran:

```sh
gliner-runner classify \
  --labels "error,warning,routine,security concern" \
  "2024-03-02 11:04:12 database connection timed out after 30 seconds"
gliner-runner classify \
  --labels "error,warning,routine,security concern" \
  "Failed login for user admin from 203.0.113.9 (attempt 5)"
gliner-runner classify \
  --labels "decision,action item,question,background" \
  "We agreed to move the launch to June."
```

```text
error (confidence 0.354)
error (confidence 0.532)
background (confidence 0.372)
```

The first is right, but the program is not sure (0.354). The second is a possible attack, and a
person would call it a `security concern`, but the program said `error`. The third is a decision,
and the program said `background`. Confidence was low in all three.

Try ten of your own items first, look at the answers, and adjust the labels and descriptions until
they match how you would sort them yourself.

## Next

Try a set of labels on your own text in the [playground](/gliner-runner/playground/), or start
from one of the [examples](/gliner-runner/examples/) we ran in full.
