---
title: More ideas
description: Other everyday things you can sort, each with a starting set of labels.
sidebar:
  order: 4
---

Each idea is a list of labels to try. Put the labels after `--labels`, separated by commas, and
add `--multi` where a text can fit more than one. Add a short `--describe` for any label whose
meaning is not obvious. See [how it works](/gliner-runner/how-it-works/) for how.

| Idea | What you classify | Labels to start with |
| --- | --- | --- |
| Inbox rules | Each incoming email | `needs reply, newsletter, receipt, calendar invite, other` |
| Form triage | Free-text answers on a contact form | `sales question, support request, press, job application, other` |
| Survey answers | Open-ended survey answers | `praise, complaint, suggestion, question` |
| Log lines | One line from an application log | `error, warning, routine, security concern` |
| Meeting notes | Each line of your notes | `decision, action item, question, background` |
| File naming | The first paragraph of a document | `invoice, contract, resume, receipt, letter, other` |
| Bug reports | Each new report | `crash, slow, wrong result, display problem, feature request` |
| Product feedback | Customer comments | `price, quality, shipping, ease of use, support` (with `--multi`) |
| Job applications | A cover letter | `engineering, design, sales, operations, other` |
| Donation notes | Notes sent with a donation | `in memory of, in honor of, general, question` |
| Event questions | Questions sent before an event | `schedule, location, tickets, accessibility, food` |
| Chat messages | Messages in a group chat | `question, announcement, joke, plan, other` |
| News headlines | A headline | `business, sports, science, politics, local, other` |
| Recipe notes | Your own recipe notes | `breakfast, main dish, side, dessert, snack` |

## A starting command

This is the pattern behind every row. Replace the text and the labels:

```sh
gliner-runner classify --labels "invoice,contract,resume,receipt,letter,other" "Invoice 2041. Amount due: 480.00. Payment terms: net 30 days."
```

```text
invoice (confidence 0.760)
```

To sort a whole file, one item per line, use the loop from the
[support ticket example](/gliner-runner/examples/support-tickets/) and end it with
`done < yourfile.txt`.

## Before you rely on it

These label sets are starting points. We ran the pages under [Examples](/gliner-runner/examples/)
in full. For this list we only tried a few items, and some did not go well. A database timeout
log line was labelled `routine (confidence 0.335)`, and a log line about a fifth failed login was
labelled `error (confidence 0.529)` rather than `security concern`. Meeting notes with the four
labels above got confidences between 0.35 and 0.38. Short labels with no descriptions are often
not enough for harder jobs.

Try ten of your own items first, look at the answers, and adjust the labels and descriptions until
they match how you would sort them yourself.
