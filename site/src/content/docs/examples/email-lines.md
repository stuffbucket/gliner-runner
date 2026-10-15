---
title: Find the important lines in an email
description: Mark each line of an email as a request, a question, a deadline, plain information, or a greeting.
---

## The situation

An email is long and only a few lines in it matter to you: the ones that ask you to do something,
ask you a question, or give a date. You want each line marked so you can see them at a glance.

The program classifies a piece of text as a whole. To mark the lines of an email, first split the
email into lines, then classify each line.

## The labels

Each line gets one label.

| Label | Description we give it |
| --- | --- |
| `action request` | Asks the reader to do something |
| `question` | Asks something the reader should answer |
| `deadline` | Says something must happen by a certain date or time |
| `information only` | Shares a fact and asks for nothing |
| `greeting or sign-off` | A hello, thanks, or closing line |

## Step 1: save the email and split it into sentences

This saves a short email to `email.txt`, then writes one sentence per line to `lines.txt`:

```sh
cat > email.txt <<'EOF'
Hi Sam,

Thanks for sending the draft over last week. Could you send me the updated budget numbers? The review meeting has moved to Friday at 10am. Please reply by Wednesday so I can book the room.

For your information, the office will be closed on Monday.

Best regards,
Dana
EOF

sed 's/\([.?!]\) /\1\n/g' email.txt | grep -v '^$' > lines.txt
cat lines.txt
```

The `sed` command starts a new line after every `.`, `?`, or `!` that is followed by a space.
`grep -v '^$'` removes empty lines.

```text
Hi Sam,
Thanks for sending the draft over last week.
Could you send me the updated budget numbers?
The review meeting has moved to Friday at 10am.
Please reply by Wednesday so I can book the room.
For your information, the office will be closed on Monday.
Best regards,
Dana
```

This split is simple. It will cut in the wrong place after an abbreviation such as `Dr.` or `etc.`.
For most emails it is good enough.

## Step 2: classify each line

```sh
while IFS= read -r line; do
  result=$(gliner-runner classify \
    --labels "action request,question,deadline,information only,greeting or sign-off" \
    --describe "action request=Asks the reader to do something" \
    --describe "question=Asks something the reader should answer" \
    --describe "deadline=Says something must happen by a certain date or time" \
    --describe "information only=Shares a fact and asks for nothing" \
    --describe "greeting or sign-off=A hello, thanks, or closing line" \
    "$line")
  printf '%s\n    %s\n' "$line" "$result"
done < lines.txt
```

## What we got

```text
Hi Sam,
    greeting or sign-off (confidence 0.558)
Thanks for sending the draft over last week.
    greeting or sign-off (confidence 0.643)
Could you send me the updated budget numbers?
    action request (confidence 0.387)
The review meeting has moved to Friday at 10am.
    deadline (confidence 0.601)
Please reply by Wednesday so I can book the room.
    deadline (confidence 0.872)
For your information, the office will be closed on Monday.
    information only (confidence 0.657)
Best regards,
    greeting or sign-off (confidence 0.589)
Dana
    greeting or sign-off (confidence 0.440)
```

Most lines are marked the way a person would mark them. Two need a second look:

- "Could you send me the updated budget numbers?" is both a question and a request. The program
  chose `action request` with a low confidence of 0.387.
- "The review meeting has moved to Friday at 10am." gives a time but is not a deadline. The
  program marked it `deadline`.

Low confidence is a signal to read the line yourself. Lines that are labelled `greeting or
sign-off` are the ones you can skip.

## What to try next

- **Only show what matters.** Add `| grep -v 'greeting or sign-off'` to the end of the loop to
  hide the greetings.
- **Let a line carry several labels.** Add `--multi` to the command. In our test, "Please reply
  by Wednesday so I can book the room." came back as `deadline (0.923)` and nothing else, even
  though it is also a request. Check how it behaves on your own emails.
- **Use other labels,** such as `good news`, `bad news`, `needs approval`.
- **Use it on meeting notes or chat messages.** Anything with one idea per line works the same
  way.
