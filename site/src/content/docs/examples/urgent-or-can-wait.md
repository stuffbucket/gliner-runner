---
title: Tell what is urgent and what can wait
description: Mark each message as needing action today or able to wait, so you read the important ones first.
---

## The situation

Your inbox has a mix of messages. Some need you today. Others can be read later this week. You
want the first kind marked, so you can start with them.

## The labels

Each message gets one label.

| Label | Description we give it |
| --- | --- |
| `urgent` | Needs a reply or action today |
| `can wait` | Nothing is needed today; it can be read later this week |

## The command

```sh
while IFS= read -r line; do
  printf '%s\n  ' "$line"
  gliner-runner classify \
    --labels "urgent,can wait" \
    --describe "urgent=Needs a reply or action today" \
    --describe "can wait=Nothing is needed today; it can be read later this week" \
    "$line"
done <<'EOF'
The server is down and no customer can place an order.
The vendor needs your signature on the contract before 5pm today.
Here are the notes from last week's meeting, for your records.
Whenever you have a moment, could you look at the new logo options?
The fire alarm in building B is going off. Please leave the building now.
Our newsletter for next month is ready to review, no rush.
Your flight leaves in three hours and you have not checked in.
EOF
```

## What we got

```text
The server is down and no customer can place an order.
  urgent (confidence 0.666)
The vendor needs your signature on the contract before 5pm today.
  urgent (confidence 0.878)
Here are the notes from last week's meeting, for your records.
  can wait (confidence 0.888)
Whenever you have a moment, could you look at the new logo options?
  can wait (confidence 0.869)
The fire alarm in building B is going off. Please leave the building now.
  urgent (confidence 0.789)
Our newsletter for next month is ready to review, no rush.
  can wait (confidence 0.984)
Your flight leaves in three hours and you have not checked in.
  urgent (confidence 0.516)
```

All seven are marked the way a person would mark them. The flight message has the lowest
confidence, 0.516.
Messages that say "today" or "no rush" are clear. Check the ones with a confidence near 0.5.

## What to try next

- **Ask two questions at once.** To get the topic and the urgency of each message in one request,
  use the local server. See
  [What it can and can't do](/gliner-runner/how-it-works/#what-it-can-and-cant-do).
- **Show only the urgent ones.** Change `done <<'EOF'` to `done <<'EOF' | grep -B1 '^  urgent'`.
- **Send low confidence to a person.** Treat anything under 0.6 as "read it yourself".
- **Sort text messages** the same way. See
  [Sort text messages by how fast they need attention](/gliner-runner/examples/text-messages/).
