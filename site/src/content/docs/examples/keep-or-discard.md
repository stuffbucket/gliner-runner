---
title: Decide what to keep and what to throw away
description: Go through a list of papers, files, or notes and mark which ones to keep and which are safe to throw away.
---

## The situation

You have a pile of papers or a long list of files and notes. You want a first pass: which ones are
worth keeping, and which are safe to throw away. You still check the answers yourself before you
throw anything away.

## The labels

Each item gets one label.

| Label | Description we give it |
| --- | --- |
| `important to keep` | A legal, tax, money, or warranty record that I may need later |
| `safe to throw away` | Advertising, expired offers, or things that are over |

## The command

Describe each item in a short line of text:

```sh
while IFS= read -r line; do
  printf '%s\n  ' "$line"
  gliner-runner classify \
    --labels "important to keep,safe to throw away" \
    --describe "important to keep=A legal, tax, money, or warranty record that I may need later" \
    --describe "safe to throw away=Advertising, expired offers, or things that are over" \
    "$line"
done <<'EOF'
Signed lease agreement for the apartment, valid until 2028.
Pizza coupon that expired last June.
Tax return for 2023 with all the supporting receipts.
Flyer for a concert that already happened.
Warranty card for the washing machine bought last year.
Junk mail catalogue with a free-gift offer.
Photo of the whiteboard from a meeting two years ago.
EOF
```

## What we got

```text
Signed lease agreement for the apartment, valid until 2028.
  important to keep (confidence 0.647)
Pizza coupon that expired last June.
  safe to throw away (confidence 0.989)
Tax return for 2023 with all the supporting receipts.
  important to keep (confidence 0.824)
Flyer for a concert that already happened.
  safe to throw away (confidence 0.973)
Warranty card for the washing machine bought last year.
  important to keep (confidence 0.772)
Junk mail catalogue with a free-gift offer.
  safe to throw away (confidence 0.903)
Photo of the whiteboard from a meeting two years ago.
  safe to throw away (confidence 0.512)
```

The first six are sorted the way a person would sort them. The whiteboard photo is a judgment
call, and its confidence is the lowest, 0.512. Do not throw away anything the program calls
`safe to throw away` without a look, especially when the confidence is low.

## The labels matter

Our first try used the labels `keep` and `discard`, with the descriptions "Something I may need
again, such as a contract, a tax paper, or a warranty" and "Something out of date, already used,
or of no further use". It got two of the important items wrong: the lease agreement and the
warranty card were both labelled `discard` (confidence 0.676 and 0.678). Plain words that name the
kind of thing, and descriptions that list examples, worked better. If the program is wrong about
your items, rewrite the labels and descriptions first.

## What it cannot do

It cannot find duplicates. It reads one text at a time and has no way to compare two items, so it
cannot tell you that two files are the same. Use a different tool for that.

## What to try next

- **Add a third label** such as `not sure`, for things a person should decide.
- **Work through a file.** Put one item per line in a file and end the loop with
  `done < items.txt` instead of the `<<'EOF'` block.
- **Change what counts as important.** Edit the descriptions to match what you keep.
