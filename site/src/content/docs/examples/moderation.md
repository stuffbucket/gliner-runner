---
title: Flag posts that break your rules
description: "Flag posts for a person to look at: harassment, spam, self-promotion, or off-topic."
---

## The situation

You run a small forum about cooking. Members post every day. You want the posts that break the
rules flagged, so a person can look at them. The program only suggests; a person decides.

## The labels

A post can break more than one rule, so this uses many labels per post with `--multi`. The
label `ok` is for a normal post.

| Label | Description we give it |
| --- | --- |
| `harassment` | Insults or attacks aimed at a person |
| `spam` | Repeated or unwanted ads, scam links, or junk |
| `self-promotion` | Promotes the poster's own product, channel, or business |
| `off-topic` | About sports, phones, politics, or anything other than cooking |
| `ok` | A normal post about cooking that follows the rules |

Change `cooking` to your own subject. The `off-topic` and `ok` descriptions are the ones that
depend on it.

## The command

```sh
while IFS= read -r line; do
  printf '%s\n  ' "$line"
  gliner-runner classify --multi --threshold 0.4 \
    --labels "harassment,spam,self-promotion,off-topic,ok" \
    --describe "harassment=Insults or attacks aimed at a person" \
    --describe "spam=Repeated or unwanted ads, scam links, or junk" \
    --describe "self-promotion=Promotes the poster's own product, channel, or business" \
    --describe "off-topic=About sports, phones, politics, or anything other than cooking" \
    --describe "ok=A normal post about cooking that follows the rules" \
    "$line"
done <<'EOF'
Thanks for the tip. I tried it with less salt and it came out much better.
You are an idiot and nobody wants your recipes. Quit posting.
BUY CHEAP WATCHES NOW!!! Click http://example.com/deal for 90% off.
I just started a channel about baking bread. Please subscribe and check out my cookbook!
Who do you think will win the football game tonight?
EOF
```

## What we got

```text
Thanks for the tip. I tried it with less salt and it came out much better.
  ok (0.713)
You are an idiot and nobody wants your recipes. Quit posting.
  harassment (0.940)
BUY CHEAP WATCHES NOW!!! Click http://example.com/deal for 90% off.
  spam (0.726)
I just started a channel about baking bread. Please subscribe and check out my cookbook!
  self-promotion (0.473), ok (0.569)
Who do you think will win the football game tonight?
  off-topic (0.712)
```

Four of the five are clear. The channel post came back as both `self-promotion` and `ok`. Treat
a post with any flag, including a flag next to `ok`, as one for a person to read.

## Descriptions matter here

The first time we ran this, we described `off-topic` as "Has nothing to do with a cooking forum".
A question about phone plans was not flagged, and the program called it `ok`. Spelling out the
kinds of things that are off topic ("About sports, phones, politics, or anything other than
cooking") fixed it for our examples. If a rule is missed, rewrite its description with
specific examples.

## What to try next

- **Send flagged posts to a queue.** Hold back any post whose result does not start with `ok`.
- **Tune the threshold.** A lower value flags more posts and also more good ones. We used 0.4.
- **Add your own rules,** such as `personal attack on a member` or `medical advice`, each with a
  description.
- **Keep a person in the loop.** This is a first filter. It will make mistakes, and it should not
  remove posts without someone looking.
