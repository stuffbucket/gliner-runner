---
title: Review mood and topic
description: Read a customer review for its mood and for the topics it mentions.
sidebar:
  order: 5
---

## The situation

You have restaurant reviews. You want to know two things about each one: is it positive or
negative, and what is it about?

These are two separate questions, so each review is classified twice with two sets of labels.

## The labels

**Mood** has one label per review.

| Label | Meaning |
| --- | --- |
| `positive` | Happy with the visit |
| `mixed` | Some good, some bad |
| `negative` | Unhappy with the visit |

**Topics** can have several labels per review, so this one uses `--multi`.

| Label | Meaning |
| --- | --- |
| `food` | The meals and drinks |
| `service` | The staff and the waiting |
| `price` | The cost |
| `cleanliness` | How clean the place was |
| `noise` | How loud it was |

## The command

```sh
while IFS= read -r line; do
  printf '%s\n' "$line"
  printf '  mood:   '
  gliner-runner classify --labels positive,mixed,negative "$line"
  printf '  topics: '
  gliner-runner classify --multi --labels "food,service,price,cleanliness,noise" "$line"
done <<'EOF'
The pasta was excellent and the waiter was very kind, but it was far too expensive.
We waited an hour for a table and the bathroom was dirty.
Lovely evening. The food was great and the room was quiet.
Too loud to talk, and the soup was cold.
EOF
```

## What we got

```text
The pasta was excellent and the waiter was very kind, but it was far too expensive.
  mood:   mixed (confidence 0.835)
  topics: food (0.654), service (0.502), price (0.790)
We waited an hour for a table and the bathroom was dirty.
  mood:   negative (confidence 0.798)
  topics: cleanliness (0.979)
Lovely evening. The food was great and the room was quiet.
  mood:   positive (confidence 0.966)
  topics: food (0.814), noise (0.607)
Too loud to talk, and the soup was cold.
  mood:   negative (confidence 0.860)
  topics: noise (0.945)
```

The moods are right for all four reviews. The first review mixes praise and a complaint, and
`mixed` fits it.

The topics are right but not complete. The second review also complains about the wait
(`service`), and the fourth also mentions the soup (`food`). Neither was returned at the default
threshold of 0.5. Lowering the threshold to 0.3 for the fourth review:

```sh
gliner-runner classify --multi --threshold 0.3 --labels "food,service,price,cleanliness,noise" "Too loud to talk, and the soup was cold."
```

```text
food (0.319), noise (0.945)
```

That one now includes `food`. A lower threshold finds more topics and also more wrong ones.

## What to try next

- **Count them.** Collect the labels for many reviews and count how often `price` or `noise` comes
  up.
- **Use `mixed` as a trigger.** Mixed reviews are often the most useful ones to read.
- **Add describing words** to labels that are missed, for example
  `service=The staff, and how long people waited`.
- **Try the larger model** with `--model decide-1b` and compare.
