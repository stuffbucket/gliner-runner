---
title: Send each support ticket to the right team
description: Send each customer message to the team that should handle it.
---

## The situation

Customers write to a shared support address. Someone reads each message and forwards it to billing,
technical help, shipping, account help, or the team that handles cancellations. You want the program
to suggest the team.

## The labels

Only one team should get each ticket, so this uses one label per message.

| Label | Description we give it |
| --- | --- |
| `billing` | Charges, invoices, refunds, or payment problems |
| `technical` | Something does not work, crashes, or shows an error |
| `shipping` | Delivery, tracking, or a damaged or missing package |
| `account` | Logging in, passwords, or changing account details |
| `cancellation` | The customer wants to stop or cancel the service |

## The command

This reads five tickets, one per line, and prints the team for each:

```sh
while IFS= read -r line; do
  printf '%s\n  ' "$line"
  gliner-runner classify \
    --labels billing,technical,shipping,account,cancellation \
    --describe "billing=Charges, invoices, refunds, or payment problems" \
    --describe "technical=Something does not work, crashes, or shows an error" \
    --describe "shipping=Delivery, tracking, or a damaged or missing package" \
    --describe "account=Logging in, passwords, or changing account details" \
    --describe "cancellation=The customer wants to stop or cancel the service" \
    "$line"
done <<'EOF'
I was charged twice for my March order.
The app closes by itself when I open the settings page.
My parcel says delivered but nothing is at my door.
I cannot log in. The password reset email never arrives.
Please cancel my subscription before the next renewal.
EOF
```

## What we got

```text
I was charged twice for my March order.
  billing (confidence 0.764)
The app closes by itself when I open the settings page.
  technical (confidence 0.563)
My parcel says delivered but nothing is at my door.
  shipping (confidence 0.743)
I cannot log in. The password reset email never arrives.
  account (confidence 0.614)
Please cancel my subscription before the next renewal.
  cancellation (confidence 0.862)
```

All five went to the team a person would choose.

## What to try next

- **Tickets about two things.** Add `--multi` to allow several teams. With the default threshold
  of 0.5, a ticket such as "I was charged twice, and now I cannot log in to see my invoices."
  returned no label in our test. Adding `--threshold 0.3` returned
  `billing (0.347), account (0.333)`.
- **A review queue.** Send any ticket with a confidence below 0.5 to a person.
- **Your own teams.** Replace the labels and descriptions with the names your team uses.
- **Your own tickets.** Put them in a file, one per line, and end the loop with
  `done < tickets.txt` instead of the `<<'EOF'` block.

## Next

See [all examples](/gliner-runner/examples/), or [more ideas](/gliner-runner/more-ideas/) for other
jobs to try.
