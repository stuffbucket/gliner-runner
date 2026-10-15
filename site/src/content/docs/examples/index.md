---
title: Examples
description: Copy-and-paste recipes that sort real kinds of text, with the output we got when we ran them.
sidebar:
  label: Overview
  order: 0
---

Each page below takes one everyday job, lists the labels it uses, gives a command to paste into
Terminal, and shows what we got when we ran it.

| Page | Job | Labels |
| --- | --- | --- |
| [Support ticket routing](/gliner-runner/examples/support-tickets/) | Send each ticket to the right team | billing, technical, shipping, account, cancellation |
| [Spotting personal information](/gliner-runner/examples/personal-information/) | Tell whether a message contains personal details, and which kind | phone number, email address, home address, payment card, none |
| [What matters in an email](/gliner-runner/examples/email-lines/) | Mark each line of an email | action request, question, deadline, information only, greeting or sign-off |
| [Text message triage](/gliner-runner/examples/text-messages/) | Sort texts by how fast they need attention | urgent, can wait, spam, personal |
| [Review mood and topic](/gliner-runner/examples/reviews/) | Read a review for its mood and what it is about | positive, mixed, negative; food, service, price, cleanliness, noise |
| [Other languages](/gliner-runner/examples/other-languages/) | Route Spanish, French, and German messages | billing, technical, shipping, account |
| [Content moderation flags](/gliner-runner/examples/moderation/) | Flag posts for a person to look at | harassment, spam, self-promotion, off-topic, ok |

## Before you start

Install the program and a model (see the [home page](/gliner-runner/)). The English examples use
`decide-340m`. The other-languages page uses `decide-multilingual`.

The commands work in Terminal on macOS and in any Linux shell. Paste each block whole.

Some pages use a loop that runs one command for each line of a list. Each pass takes about 1.5
seconds, because the program loads the model every time. For a long list, see
[how it works](/gliner-runner/how-it-works/#measured-speed) for the faster way.

## Reading the results

- The number is a confidence score, not a percentage of how often the answer is right.
- Our results are what we got on one computer on one day. A different model or a different
  wording of a label can change them. Run the examples yourself and check them against your own
  text.
- Where a result was wrong or uncertain, we say so on the page.
