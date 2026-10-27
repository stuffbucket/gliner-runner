# Copy lab

Writes short website copy by generate-and-test instead of a single "write plainly" prompt. For one slot (a short
stretch of text on a page) it drafts many candidates with several prompts and several models, throws out those
that break the slot's hard limits, has a disinterested reader retell each survivor, and has three judge models
compare the best ones pairwise. The output is a ranked shortlist to choose from; it never edits the site.

```sh
node tools/copy-lab/lab.mjs --slot fifteen-seconds
```

`--slot all` runs every slot in `slots.json`; `--slot a,b` runs some. The report lands in
`/tmp/gr-work/copy-lab/<slot>/report.md` unless `--out` says otherwise.

## Before you run it

- **A chat API.** OpenAI-compatible, by default `http://localhost:4141/v1` (`--api`). The models named in
  `slots.json` must be listed by its `/v1/models`.
- **promptfoo.** The retell stage runs `npx -y promptfoo@0.123.1` (the version is pinned in `package.json`,
  `config.promptfoo`). The first run on a machine fetches it, which takes a few minutes; after that `npx` has it.

Nothing else is installed. The folder is not a pnpm workspace package for the reason `tools/site-review/README.md`
gives (it would re-resolve the root lockfile through the npm mirror), and it needs no packages of its own.

## Slots

`slots.json` has `defaults` and a list of `slots`; each slot overrides any default. A slot is:

| Field | |
| --- | --- |
| `id` | The name used with `--slot` and for the output folder. |
| `where`, `purpose` | Told to the drafting models. |
| `maxWords`, `maxSentences`, `maxSentenceWords`, `maxGrade` | Hard limits. `maxGrade` is the Flesch-Kincaid grade. |
| `stallWords`, `meta` | Words and phrases a candidate must not contain. A stall word also matches its plural. |
| `facts` | What a reader must be able to retell. `optional: true` facts are reported and break ties, but are not in the score. |
| `baselines` | Texts to score next to the candidates (the current text, the user's own words). They are retold even when they break a limit. `reference: true` also puts one in the pairwise judging, for comparison. |
| `angles`, `drafters`, `perCall` | Drafting prompts (keys of `ANGLES` in `lib/prompts.mjs`), drafting models, candidates per call. |
| `maxRetell` | The most survivors to retell, taken in turn from each model and angle. |
| `personas`, `runs`, `grader` | The retell readers (one promptfoo provider each), runs per reader, and the grading model. |
| `judges`, `judgeTop` | The pairwise judge models and how many of the best retold candidates they compare. |
| `show` | Optional: overrides for a second pool scored alongside the slot, for a one-sentence lead-in to an embedded demo. It is retold but not judged, and appears in the slot's report as a layout option. |

To add a slot, copy one, change its text and limits, and run it. To add a drafting angle, add an entry to
`ANGLES` in `lib/prompts.mjs` and name it in `angles`.

## What it does

| Stage | Tool | What happens |
| --- | --- | --- |
| generate | chat API | Every angle goes to every drafting model, which returns `perCall` candidates as JSON. The prompt carries the product facts, the user's own words, the landing page headlines as the voice, the style rules and the hard limits. Duplicates are merged. |
| check | `lib/constraints.mjs` | Word count, sentence count, longest sentence, Flesch-Kincaid grade (0.39 × words per sentence + 11.8 × syllables per word − 15.59, with a vowel-group syllable heuristic), stall words and meta phrases. A candidate that breaks any limit is dropped; every reason is kept in `drops.json`. |
| retell | promptfoo | A "disinterested friend" reads only the candidate and answers: in one sentence, what does it do; is it like ChatGPT, and why; would you keep reading (told that they skip most pages, so that a yes means the lines were quick to get and made them curious: asked plainly, `claude-sonnet-5` said yes to almost everything and `gemini-3.8-flash` to nothing). Each persona model reads each candidate `runs` times (sampled at temperature 0.8). An `llm-rubric` grader marks which facts the answer retold. The retell score is the share of required facts retold plus the share of reads that would keep reading, from 0 to 2. |
| judge | chat API | The `judgeTop` best by retell score, plus the reference baseline, are compared in every pair, in both orders, by each judge model: "Which would a busy person who does not care about tech understand faster and remember? Which sounds like a person, not a brochure?" Each answer is half a game. The ranking is a Bradley-Terry fit; the win rate is shown too. |
| report | | The top 3 with their numbers and source, the baselines, every judged and retold candidate, the show-don't-tell option, the drops and the cost. |

## Cost and caching

Every chat reply is cached in `<out>/.cache` under a hash of the model, messages, temperature and run, and every
retell row under a hash of the text, the rubric, the reader and the run. A re-run pays only for what changed: a new
slot, a new angle, an edited rubric. Delete the folder for a fresh sample.

The calls for one slot are about: `angles × drafters` to draft; `(retold + baselines) × personas × runs × 2` to
retell (a read and a grade); `entrants × (entrants − 1) × judges` to judge. With the defaults that is about 20,
620 and 470. Everything runs `--concurrency` at a time (default 12). The report's Cost table gives the measured
calls and time per stage.

## Output

In `<out>/<slot>/`:

- `report.md`: the top 3 and everything behind them.
- `candidates.json`: every survivor and baseline with its checks, retell score (per fact, keep-reading rate, two
  sample answers) and judge numbers.
- `drops.json`: each dropped draft and why.
- `retell.json`: every read: reader, run, the answers, which facts the grader found, and the grader's reason.
- `judge.json`: every pairwise judgement.
- `show/`: the same files for the show-don't-tell pool, if the slot has one.
- `run.json`: calls, tokens and time.

## Limitations

- The readers and judges are models. They read more carefully than a busy person, and they know what ChatGPT is
  in a way a real reader may not. The scores rank candidates against each other; they do not predict a real
  reader's 15 seconds.
- The syllable count is a heuristic, so the grade is approximate; it is a filter, not a score.
- A small number of runs per reader means retell scores move between runs by about 0.1 to 0.2. Treat close scores
  as ties and read the top few.

## Tests

`node --test tools/copy-lab/lib/*.test.mjs` tests the checks (words, sentences, syllables, grade, reasons) and the
Bradley-Terry fit without a model.
