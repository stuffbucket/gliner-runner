---
title: GLiNER Runner
description: Sort text into labels you choose, on your own computer, with no account and no internet connection after the first download.
---

GLiNER Runner reads a piece of text and tells you which of your labels fits it. You give it a
message and a list of labels such as `billing, support, shipping`, and it answers with the best
label and a confidence score. It runs on your own computer, and your text never leaves it.

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

- [How it works](/gliner-runner/how-it-works/): what the program does, how fast it is, and when to
  use something else.
- [Examples](/gliner-runner/examples/): copy-and-paste recipes for support tickets, personal
  information, email, text messages, reviews, other languages, and moderation.
- [Use it in a React app](/gliner-runner/use-in-react/): a small web page that calls the program.
- [More ideas](/gliner-runner/more-ideas/): other things to sort.
- [Reference](/gliner-runner/models/): the models, test results, and technical documents.
