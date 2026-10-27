---
title: Check a message for personal information
description: Tell whether a message contains personal details, and which kinds. This does not remove them.
---

## The situation

Before you share or store a message, you want to know whether it contains personal information:
a phone number, an email address, a home address, or a payment card.

## What this can and cannot do

This program can answer: does this message contain a phone number? An email address? A home
address? A payment card? It can say yes to more than one.

It cannot show you where in the message the information is, and it cannot remove it. It sorts a
whole message into labels. It does not find or cut out pieces of text. Blacking out or removing
personal information needs a different kind of model, one that finds spans of text (an entity
extraction model). This program does not offer one yet.

Use this page to decide which messages need a closer look, not to clean them.

## The labels

A message can contain several kinds at once, so this uses many labels per message with
`--multi`. The label `none` is for messages with nothing personal in them.

| Label | Description we give it |
| --- | --- |
| `phone number` | A telephone number |
| `email address` | An email address |
| `home address` | A street address where someone lives or receives mail |
| `payment card` | A credit or debit card number, expiry date, or security code |
| `none` | The message contains no personal information |

## The command

```sh
while IFS= read -r line; do
  printf '%s\n  ' "$line"
  gliner-runner classify --multi --threshold 0.4 \
    --labels "phone number,email address,home address,payment card,none" \
    --describe "phone number=A telephone number" \
    --describe "email address=An email address" \
    --describe "home address=A street address where someone lives or receives mail" \
    --describe "payment card=A credit or debit card number, expiry date, or security code" \
    --describe "none=The message contains no personal information" \
    "$line"
done <<'EOF'
Call me on 415-555-0132 after 5pm.
My email is jordan.lee@example.com, please send the receipt there.
Ship it to 42 Maple Street, Springfield, IL 62704.
My card number is 4111 1111 1111 1111, expiry 08/28.
Reach me at jordan.lee@example.com or on 415-555-0132.
The meeting moved to Thursday at 3pm.
The quarterly report is attached and ready for review.
EOF
```

The details in these messages are made up. The card number is a standard test number.

## What we got

```text
Call me on 415-555-0132 after 5pm.
  phone number (0.646)
My email is jordan.lee@example.com, please send the receipt there.
  email address (0.738)
Ship it to 42 Maple Street, Springfield, IL 62704.
  home address (0.678)
My card number is 4111 1111 1111 1111, expiry 08/28.
  payment card (0.625)
Reach me at jordan.lee@example.com or on 415-555-0132.
  phone number (0.622), email address (0.467)
The meeting moved to Thursday at 3pm.
  none (0.550)
The quarterly report is attached and ready for review.
  none (0.674)
```

The message with two kinds of details got both labels. The messages with no personal details got
`none`.

## About the threshold

We set `--threshold 0.4` for this example. With the default of 0.5, the message with a phone
number and an email address returned only `phone number (0.630)`, and the email address was missed.
The lower threshold caught both. A lower threshold also means more false alarms, so test it on
messages of your own. For personal information it is usually better to flag too much than too
little, and have a person look.

## What to try next

- **Try the larger model.** Add `--model decide-1b` (see the [model list](/gliner-runner/models/))
  and compare its answers with these on your own messages.
- **Add kinds** that matter to you, such as `date of birth` or `government ID number`, each with
  a description.
- **Use it as a gate.** Hold back any message that is not labelled `none` for a person to read.

## Next

See [all examples](/gliner-runner/examples/), or [more ideas](/gliner-runner/more-ideas/) for other
jobs to try.
