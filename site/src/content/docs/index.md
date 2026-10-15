---
title: GLiNER Runner
description: Sort text into labels you choose, on your own computer, with no account and no internet connection after the first download.
---

GLiNER Runner sorts text into labels you choose. It runs on your own computer, and your text
never leaves it.

## What it does

You give it some text and a list of labels. It picks the label that fits best and says how
confident it is. Confidence is a number from 0 (not confident) to 1 (very confident).

For example, the text is a sentence about the weather, and the labels are `weather`, `cooking`,
`sports`, and `money`:

```sh
gliner-runner classify --labels weather,cooking,sports,money "Heavy rain is expected all weekend, with strong winds along the coast."
```

```text
weather (confidence 0.997)
```

The answer is `weather`, and the program is very sure. The labels can be any short words you
like. You write them. The program does not make up new ones.

### Texts that fit two labels, or none

Some texts fit one label, some fit two, and some fit none. This example runs each of three texts
twice. The first run picks one label. The second run adds `--multi`, which lets a text have
several labels, or none.

```sh
while IFS= read -r line; do
  printf '%s\n' "$line"
  printf '  one label: '
  gliner-runner classify --labels weather,cooking,sports,money "$line"
  printf '  --multi:   '
  gliner-runner classify --multi --labels weather,cooking,sports,money "$line"
done <<'EOF'
The home team won the final by two goals.
I will pay my rent on Friday, and then I will bake bread.
My grandmother taught me to knit a scarf when I was seven.
EOF
```

```text
The home team won the final by two goals.
  one label: sports (confidence 0.999)
  --multi:   sports (0.980)
I will pay my rent on Friday, and then I will bake bread.
  one label: money (confidence 0.783)
  --multi:   cooking (0.640), money (0.866)
My grandmother taught me to knit a scarf when I was seven.
  one label: cooking (confidence 0.345)
  --multi:   no label reached the 0.5 threshold
```

- **The first text fits one label.** Both runs say `sports`.
- **The second text fits two labels.** Paying rent is `money` and baking is `cooking`. With one
  label the program has to choose, and it chose `money`. With `--multi` it returns both.
- **The third text fits none.** It is about knitting, and no label is about knitting. With one
  label, the program always picks something. Here it picked `cooking`, but its confidence is only
  0.345. A low number like that means "none of these". With `--multi`, a label is returned only
  when its confidence reaches 0.5, so the program returned no label at all.

If you want to stay with one label per text, add a label such as `none of these`. For the knitting
text, `--labels weather,cooking,sports,money,"none of these"` answers
`none of these (confidence 0.404)`.

### How much text

It works best on a sentence, up to a few paragraphs (about 500 words). A text of 2,000 words
still gives a sensible answer, but takes a few seconds. Much longer texts take far too long: 4,000
words took 8 seconds on the model alone. For a longer document, split it into parts, such as
paragraphs or sections, and classify each part. The
[measurements are on the How it works page](/gliner-runner/how-it-works/#how-much-text-it-can-read).

To see what else it can and cannot do, read
[What it can and can't do](/gliner-runner/how-it-works/#what-it-can-and-cant-do).

## Install

With [Homebrew](https://brew.sh) (macOS on Apple Silicon, and Linux):

```sh
brew tap stuffbucket/tap
brew trust stuffbucket/tap
brew install gliner-runner
```

`brew trust` is needed on Homebrew 7 and later. If your Homebrew says the command does not exist,
skip that line.

Without Homebrew, this installs the same program into `~/.local`:

```sh
curl -fsSL https://github.com/stuffbucket/gliner-runner/releases/latest/download/install.sh | sh
```

Check that it works:

```sh
gliner-runner --version
```

The first line of the answer should start with `gliner-runner-onnx`. If it does not, an older
program with the same name is running instead. `which gliner-runner` shows which one.

## Get the models

The program does not include a model. A model is a large file that you download once.

| Model | Download | Use it for |
| --- | --- | --- |
| `decide-340m` | about 2.0 GB | English text. The default. |
| `decide-multilingual` | about 1.2 GB | Spanish, French, German, and other languages |
| `decide-1b` | about 4.8 GB | English text, when you want the most accurate answers |

To get one model, name it:

```sh
gliner-runner models install decide-340m
```

To get all three:

```sh
gliner-runner models install --all
```

`--all` first prints how much it will download and how much disk space you need, then asks before
it downloads anything. It tells you when a model is already installed, or when it found a copy
already on your computer. Add `--yes` to skip the question.

The models are stored in your user cache folder (`~/Library/Caches/gliner-runner/models` on macOS,
`~/.cache/gliner-runner/models` on Linux).

## Try it

```sh
gliner-runner demo
```

The demo classifies six sample texts. The first two:

```text
gliner-runner demo (model: decide-340m)

1. Support ticket routing
   Text:    I was charged twice for my last order and I want one of the payments refunded.
   Labels:  billing, support, shipping
   Result:  billing (confidence 0.866)

2. Support ticket routing
   Text:    My package has been in transit for two weeks and the tracking page has not changed.
   Labels:  billing, support, shipping
   Result:  shipping (confidence 0.921)
```

Then use your own text and your own labels:

```sh
gliner-runner classify --labels billing,support,shipping "My package never arrived."
```

## Next

- [How it works](/gliner-runner/how-it-works/): what the program can and cannot do, how fast it is,
  and when to use something else.
- [Examples](/gliner-runner/examples/): jobs you can do with it, such as sorting support tickets,
  finding what matters in an email, flagging personal information, and reading reviews.
- [Use it in a React app](/gliner-runner/use-in-react/): a small web page that calls the program.
- [Try it in your browser](/gliner-runner/playground/): classify text on this site with the runner
  on your computer.
- [More ideas](/gliner-runner/more-ideas/): other things to sort, grouped by job.
- [Reference](/gliner-runner/models/): the models, test results, and technical documents.
