---
title: Examples
description: Jobs you can do with it, with commands to paste and the output we got when we ran them.
sidebar:
  label: Overview
---

Every example is a job people already do by hand. Each page takes one job, lists the labels it
uses, gives a command to paste into Terminal, and shows what we got when we ran it. Underneath,
every job is the same thing: put a piece of text into one or more labels you choose.

## Sort incoming messages

| Job | Labels |
| --- | --- |
| [Send each support ticket to the right team](/gliner-runner/examples/support-tickets/) | billing, technical, shipping, account, cancellation |
| [Sort text messages by how fast they need attention](/gliner-runner/examples/text-messages/) | urgent, can wait, spam, personal |
| [Tell which language a message is in](/gliner-runner/examples/which-language/) | English, Spanish, French, German, Italian |
| [Route messages written in other languages](/gliner-runner/examples/other-languages/) | billing, technical, shipping, account |

## Find what matters

| Job | Labels |
| --- | --- |
| [Find the important lines in an email](/gliner-runner/examples/email-lines/) | action request, question, deadline, information only, greeting or sign-off |
| [Tell what is urgent and what can wait](/gliner-runner/examples/urgent-or-can-wait/) | urgent, can wait |

## Flag things to check

| Job | Labels |
| --- | --- |
| [Check a message for personal information](/gliner-runner/examples/personal-information/) | phone number, email address, home address, payment card, none |
| [Flag posts that break your rules](/gliner-runner/examples/moderation/) | harassment, spam, self-promotion, off-topic, ok |

## Understand feedback

| Job | Labels |
| --- | --- |
| [Read reviews for mood and topics](/gliner-runner/examples/reviews/) | positive, mixed, negative; food, service, price, cleanliness, noise |

## Clean up lists

| Job | Labels |
| --- | --- |
| [Decide what to keep and what to throw away](/gliner-runner/examples/keep-or-discard/) | important to keep, safe to throw away |

One job in this group does not work. The program cannot find duplicates, because it reads one
text at a time and has no way to compare two.

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
