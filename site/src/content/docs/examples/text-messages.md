---
title: Text message triage
description: Sort text messages into urgent, can wait, spam, and personal.
sidebar:
  order: 4
---

## The situation

You have a pile of text messages. You want the urgent ones first, the spam out of the way, and
the friendly ones kept separate from the rest.

## The labels

Each message gets one label.

| Label | Description we give it |
| --- | --- |
| `urgent` | Needs a reply or action within the hour |
| `can wait` | Useful, but nothing is needed today |
| `spam` | Unwanted advertising, prizes, or scam attempts |
| `personal` | A friendly message from family or a friend |

## The command

```sh
while IFS= read -r line; do
  printf '%s\n  ' "$line"
  gliner-runner classify \
    --labels "urgent,can wait,spam,personal" \
    --describe "urgent=Needs a reply or action within the hour" \
    --describe "can wait=Useful, but nothing is needed today" \
    --describe "spam=Unwanted advertising, prizes, or scam attempts" \
    --describe "personal=A friendly message from family or a friend" \
    "$line"
done <<'EOF'
The school called. Your daughter has a fever and needs to be picked up now.
Congratulations! You won a $500 gift card. Tap the link to claim it now.
Are you still coming to dinner on Sunday? Mum is making lasagna.
The basement is flooding. Can you call me right now?
Your package will arrive between Tuesday and Thursday.
You have been selected for an exclusive offer. Reply YES to continue.
EOF
```

## What we got

```text
The school called. Your daughter has a fever and needs to be picked up now.
  urgent (confidence 0.619)
Congratulations! You won a $500 gift card. Tap the link to claim it now.
  spam (confidence 0.694)
Are you still coming to dinner on Sunday? Mum is making lasagna.
  personal (confidence 0.441)
The basement is flooding. Can you call me right now?
  urgent (confidence 0.663)
Your package will arrive between Tuesday and Thursday.
  can wait (confidence 0.556)
You have been selected for an exclusive offer. Reply YES to continue.
  spam (confidence 0.653)
```

All six are sorted the way a person would sort them. The dinner message has the lowest confidence
(0.441), because it is both personal and asks a question.

## What to try next

- **Show only the urgent ones.** Change the line `done <<'EOF'` to
  `done <<'EOF' | grep -B1 '^  urgent'`. It prints each urgent message with its label and hides
  the rest.
- **Make "urgent" fit your life.** If a pharmacy pickup that closes at 6pm should count as urgent,
  say so in the description, for example `urgent=Needs a reply or action today`. In our test with
  the message "Your pharmacy order is ready. It will be held until 6pm today." the answer stayed
  `can wait` either way, so check your own cases.
- **Add labels** such as `delivery update` or `appointment`.
- **Wrap it in a script** that runs on a list of messages exported from your phone.
