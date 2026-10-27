# gliner-runner skill eval: results

Two tiers. The easy tier (`tasks.yaml`, round 1) checks that an agent with the skill writes a valid request,
picks the right mode and explains the numbers. The hard tier (`hard/`, round 2) asks for a complete script
for a realistic job, runs it against a real runner, and scores what it prints. The skill now scores 100% on
the easy tier, so only the hard tier still ranks skills.

## Final result

The committed skill (`SKILL.md`, 1,800 words) is "B-lean final" below.

Hard tier, 3 runs (`--no-cache`), counts are tasks passed over tasks run:

| Variant | Words | Model | Tuning | Held-out |
| --- | --- | --- | --- | --- |
| A, no skill | | haiku | 0/27 (0%) | 0/12 (0%) |
| A, no skill | | sonnet | 0/27 (0%) | 0/12 (0%) |
| B, skill v3 (round 1) | 2,116 | haiku | 24/27 (89%) | 10/12 (83%) |
| B, skill v3 (round 1) | 2,116 | sonnet | 24/27 (89%) | 12/12 (100%) |
| B-lean, trimmed only | 1,708 | haiku | 25/27 (93%) | 10/12 (83%) |
| B-lean, trimmed only | 1,708 | sonnet | 25/27 (93%) | 12/12 (100%) |
| B-lean final, trimmed + fixes | 1,800 | haiku | 25/27 (93%) | 11/12 (92%) |
| B-lean final, trimmed + fixes | 1,800 | sonnet | 27/27 (100%) | 10/12 (83%) |

Both models together: B 48/54 tuning and 22/24 held-out; trimmed only 50/54 and 22/24; final 52/54 and
21/24. A fails every task. Without the skill both models invent the API (`POST /classify`, `/v1/decide`,
`{"text", "labels"}`), so most scripts get 404 or 422 and crash or print empty results. On the fix tasks,
where the broken script shows the request, A's scripts reach the runner but still read the reply wrong.

By task, both models, 3 runs:

| Task | Split | A | B | B-lean trimmed | B-lean final |
| --- | --- | --- | --- | --- | --- |
| route-tickets | tuning | 0/6 | 5/6 | 6/6 | 6/6 |
| tag-reviews | tuning | 0/6 | 4/6 | 3/6 | 6/6 |
| bulk-600 | tuning | 0/6 | 6/6 | 6/6 | 6/6 |
| close-labels | tuning | 0/6 | 6/6 | 5/6 | 6/6 |
| flag-bands | tuning | 0/6 | 6/6 | 6/6 | 6/6 |
| parse-modes | tuning | 0/6 | 6/6 | 6/6 | 6/6 |
| fix-broken | tuning | 0/6 | 6/6 | 6/6 | 6/6 |
| retry-429 | tuning | 0/6 | 3/6 | 6/6 | 4/6 |
| fix-threshold | tuning | 0/6 | 6/6 | 6/6 | 6/6 |
| shell-tsv | held out | 0/6 | 6/6 | 6/6 | 5/6 |
| route-bands | held out | 0/6 | 6/6 | 6/6 | 6/6 |
| top-two | held out | 0/6 | 6/6 | 6/6 | 6/6 |
| zero-or-one | held out | 0/6 | 4/6 | 4/6 | 4/6 |

Easy tier with the final skill, 3 runs, runner on 8099 (`RUNNER_URL=http://127.0.0.1:8099`): haiku 75/75,
sonnet 75/75 (tuning 108/108, held-out 42/42). No loss against skill v3.

Raw promptfoo output, logs and every script with its stdout and stderr were kept outside the repo (round 2:
`/tmp/gr-work/skill-eval2/out/` and `artifacts/`).

## Hard tier

### Setup

- Config `hard/promptfooconfig.yaml`, tasks `hard/tasks.yaml`, prompts `hard/prompts.cjs` (the A and B
  system prompts of `prompts.cjs`, plus run rules appended to each task). Same answering models, endpoint,
  temperature and grader settings as the easy tier; `max_tokens` is 8000 because answers are whole
  scripts. No hard task uses `llm-rubric`, so the grader is never called.
- The answer must be one complete script: Node.js 24 with built-in modules only, or bash with curl and jq.
  `hard/run.cjs` takes the longest runnable code block, writes it and the input file into a new temp dir,
  and runs it under macOS `sandbox-exec`: cwd and HOME in the temp dir, file writes only there and in the
  user temp dir, no network except 127.0.0.1:8099, a 2-minute timeout (3 for bulk-600), a minimal
  environment, `/bin/bash` for shell scripts. Scripts run one at a time (a lock dir), so the runner's
  `/metrics` counters (`requests_rejected_total`, `requests_submitted_total`) belong to one script.
- `hard/scorers.cjs` scores what the script prints. Answers stay in the scorer and are never copied into
  the temp dir. Two kinds of check:
  - accuracy against known answers, with a pass mark set from reference runs (route-tickets, tag-reviews,
    bulk-600, close-labels, retry-429, zero-or-one);
  - agreement within 0.001 with the runner's answer to the reference request, for tasks whose prompt fixes
    the model, task names, labels and their order (flag-bands, parse-modes, fix-broken, fix-threshold,
    shell-tsv, route-bands, top-two).
- `hard/selftest.mjs` checks the harness without a model: a reference answer for every task must pass and
  five wrong answers must fail (600 concurrent calls, one batch of 600, one batch with no retry under load,
  single-label tagging, a fetch outside the runner).
- 13 tasks, 4 held out (31%). Held-out failures were read only after the skill was final.

| Task | Split | The script must |
| --- | --- | --- |
| route-tickets | tuning | route 40 tickets to 4 teams or "other": 30/40 right and 5/8 small talk to other |
| tag-reviews | tuning | tag 30 posts with any of 4 topics: 25/30 exact and 9/10 off-topic posts with no tag |
| bulk-600 | tuning | label 600 texts, 570 right, causing no 429 (600 parallel calls got 71 to 213 refused in test runs; one batch of 600 gets 422) |
| close-labels | tuning | split complaint from feedback by a definition that differs from the names: 12/20 (names alone give 8) |
| flag-bands | tuning | put 4 moderation flags in yes (0.8 up) and unsure (0.3 to 0.8) bands from each flag's own score |
| parse-modes | tuning | one request with a single-label and a several-labels task; print value, confidence and every probability |
| fix-broken | tuning | fix wrong `schema.kind`, missing `model`, labels as a string, and reading `output.value` |
| retry-429 | tuning | label 120 texts while other traffic keeps the queue full for the first 15 s; every text labelled |
| fix-threshold | tuning | fix `max_labels` above the label count, a threshold set only on the task, and `confidence` used as each flag's score |
| shell-tsv | held out | bash with curl and jq over lines with quotes, `$`, backslashes and non-ASCII; TSV of label and confidence |
| route-bands | held out | single label with "other" and route, review and fallback bands |
| top-two | held out | the two most likely labels with their probabilities, and an ambiguity flag |
| zero-or-one | held out | at most one team or null: 16/20 right and 4/5 small talk null |

Run it (runner on 8099, model endpoint on 4141, macOS):

```sh
GLINER_RUNNER_MODELS_DIR=<models dir> gliner-runner serve --port 8099 &
cd hard
node selftest.mjs                                  # harness check, no model calls
npx -y promptfoo@0.123.1 eval -c promptfooconfig.yaml -o /tmp/hard.json --no-cache
node ../report.mjs /tmp/hard.json
node regrade.mjs in.json out.json --reason <regex>   # re-run saved answers after a harness fix
```

`HARD_ARTIFACTS=<dir>` keeps each script and its output. Do not run the easy tier against the same runner at
the same time: its requests would count in the hard tier's 429 check.

### How the tier was built

- First A/B run on the first 7 tuning tasks and 3 held-out tasks: A 0/10 on both models, B 9/10 on both
  (6/7 tuning). Too close to the ceiling, so three tasks were added: retry-429 and fix-threshold (tuning)
  and zero-or-one (held out).
- retry-429 load: other traffic (450 parallel requests on `decide-340m`) starts before the script and stops
  15 s after it starts. Under it, one batch with no retry labels 0/120. A script that sends one request at a
  time without retrying usually passes: its requests wait in the queue but are rarely refused.
- parse-modes wording was changed after the first final run. It said topics are "any number of: battery,
  screen, price, delivery, or none", and sonnet sent "none" as a fifth label in most variants. It now says
  "any number of the labels ..., zero included". The three final runs all use the new wording; the
  ablation run and the version runs below used the old one.
- After the final runs, the sandbox was opened to the user temp dir: macOS `mktemp -d` writes there
  whatever `TMPDIR` says, and one held-out shell-tsv answer (B-lean final, haiku) failed on it. Re-run with
  `regrade.mjs`, it still fails: it uses `mapfile`, which `/bin/bash` 3.2 on macOS does not have.

### What was cut, and the evidence

Ablation on the hard tier: each row removes one section of skill v3, one run, 9 tuning tasks (old
parse-modes wording, which made parse-modes fail through the wording rather than the skill).

| Removed from v3 | haiku | sonnet |
| --- | --- | --- |
| nothing | 8/9 | 8/9 |
| 1 sources of truth | 9/9 | 8/9 |
| 2 use it or not | 7/9 | 7/9 |
| 5 single vs several labels | 8/9 | 8/9 |
| 6 response example | 7/9 | 7/9 |
| 7 thresholds in code | 8/9 | 8/9 |
| 8 many texts, web pages | 9/9 | 7/9 |
| 9 errors table | 8/9 | 8/9 |
| 10 do and don't | 9/9 | 8/9 |
| 11 when answers look wrong | 7/9 | 8/9 |

Every failure in this table was tag-reviews or parse-modes, plus one retry-429 crash (section 9) and one
fix-broken (section 6). No section's removal cost more than 2 tasks of 18, which is within run-to-run noise
(the full v3 skill scored 7/9 to 9/9 per model across runs). With the round 1 easy-tier ablation (removing
section 1 cost 2 haiku tasks, section 8 cost cors-origin, section 9 cost debug-422-parens), the cuts were:

- Section 10, do and don't (107 words): every line repeats another section. Removed.
- Section 1: the page table became the llms.txt link and three Markdown pages; the `/openapi.json` and
  `/v1/capabilities` lines stay (removing the whole section cost 2 easy tasks).
- Section 2: the table became four bullets.
- Sections 5 and 6 merged: one mode table and the response example, without the bullets that restated it.
  Removing either alone cost nothing on the easy tier.
- Section 11 cut to three sentences; section 8 lost the long-text timing and merged the two npm notes; the
  errors table lost the browser CORS row (CORS is in section 7). CORS, install and the errors table stay.

The trimmed skill (1,708 words) scored 50/54 tuning and 22/24 held-out against v3's 48/54 and 22/24: the
cuts lost nothing measurable.

### What was added, and why

Each addition answers a tuning failure and was checked against the code or a run on 8099:

- Several-labels mode: no "none" label (`[]` already means none, and an extra label shifts the other scores);
  name the task in the plural (on the 30 tag-reviews texts, `topic`, `aspect`, `tag` got 24 to 26 tag sets
  right, `topics`, `aspects`, `tags` 29 to 30; a haiku answer named its task `topic` and returned one tag per
  post); leave out descriptions when names are clear (short descriptions gave 3 of 10 off-topic posts a tag,
  none without; the cause of most tag-reviews failures for B and the trimmed skill).
- No `(` or `)` in descriptions, task names or `instruction` either (`gliner2/schema.ts` `cleanPy`; a B
  answer to route-tickets got 422 for a description with brackets).
- `/v1/batch` takes at most 256 (257 gives 422), each item takes a queue slot, and one 429 fails the whole
  batch (`handler.ts` `inferAll`); on 429 wait `Retry-After` (1 s, `errors.ts`) and resend.
- `probabilities` lists every label in several-labels mode; thresholds change `value`, not the
  probabilities; adding a task or changing the task order shifts the scores (measured on 8099).
- Docs: `llms.txt` and the `<page>.md` copies (on branch `site/skill-support`).
- The response example's numbers are for the request without `label_descriptions` (with the description,
  `billing` is 0.714, not 0.889); the skill now says so.

Versions, one run each on all 13 tasks (tuning, held-out per model):

| Skill | Words | haiku | sonnet |
| --- | --- | --- | --- |
| lean2: trim + none label, brackets, batch, short descriptions | 1,794 | 7/9, 3/4 | 9/9, 4/4 |
| lean3: lean2 + leave out descriptions | 1,787 | 8/9, 3/4 | 8/9, 3/4 |
| lean4 (final): lean3 + plural task names | 1,800 | see the final result | |

### Limits of the hard tier

- Noise: one task flip moves a cell by 4 points. The final skill's margin over v3 on tuning is 4 tasks of 54,
  and most of it is tag-reviews (6/6 against 4/6), which the added text targets. retry-429 differences are
  mostly timing: several haiku scripts stopped themselves at their own 110-second deadline under load.
- Held-out: the final skill scored 21/24 against v3's 22/24. Two of its failures are zero-or-one on sonnet,
  where the answer added label descriptions to a zero-or-one task and every small-talk message got a team:
  the "leave out descriptions" advice is written for several-labels mode and did not carry over to
  `min_labels: 0, max_labels: 1`. That is the next thing to fix, and it needs a new held-out task to check.
- The held-out tasks were written by the same author at the same time as the tuning tasks.
- The harness needs macOS (`sandbox-exec`) and runs shell scripts with `/bin/bash` 3.2.
- The agents had no tools: they could not read the live docs, `/openapi.json`, or try a request first.

## Easy tier (round 1)

### Setup

- Tool: `npx -y promptfoo@0.123.1`, config `promptfooconfig.yaml`, tasks `tasks.yaml`, asserts `asserts.cjs`.
- Answering models, through an OpenAI-compatible endpoint at `http://localhost:4141/v1`, temperature 0:
  `claude-haiku-5-5-20260301` (small) and `claude-sonnet-5` (mid). Grader for `llm-rubric`: `claude-opus-5`.
- Variants: **A** = a one-line description of gliner-runner only. **B** = the same line plus `SKILL.md` as
  the system prompt. `SKILL_FILE=<path>` makes B read another copy of the skill (used for versions and
  ablations).
- 25 tasks in 12 categories. 18 are tuning tasks, read while editing the skill. 7 (28%) are held out
  (`metadata.split: heldout`). No skill edit was made after reading a held-out answer.
- Grading is deterministic where possible. `postBody` finds the request body in the answer (JSON, curl
  `-d`, a JS object or a Python dict, resolving simple constants), sends it to a real runner, and
  requires HTTP 200 plus task checks (mode, descriptions, both thresholds, labels in the result, model).
  `regex` checks commands and field names. `llm-rubric` is used only for explanations; each rubric
  lists the facts it grades against.

Run it (needs the runner with the three models, on 8097 or at `RUNNER_URL`, and the model endpoint on 4141):

```sh
GLINER_RUNNER_MODELS_DIR=<models dir> gliner-runner serve --port 8097 &
npx -y promptfoo@0.123.1 eval -c promptfooconfig.yaml -o /tmp/results.json
node report.mjs /tmp/results.json            # several files are summed
node regrade.mjs in.json out.json [tasks.json]   # re-run the deterministic asserts on saved answers
```

### Result: A vs B (skill v3), 3 runs per cell

One cached run plus two `--no-cache` runs. Counts are tasks passed over tasks run.

| Variant | Model | Tuning | Held-out | All |
| --- | --- | --- | --- | --- |
| A, no skill | haiku | 7/54 (13%) | 3/21 (14%) | 10/75 (13%) |
| A, no skill | sonnet | 8/54 (15%) | 3/21 (14%) | 11/75 (15%) |
| B, skill v3 | haiku | 54/54 (100%) | 21/21 (100%) | 75/75 (100%) |
| B, skill v3 | sonnet | 54/54 (100%) | 21/21 (100%) | 75/75 (100%) |

By category, both models, 3 runs:

| Category | A | B |
| --- | --- | --- |
| batch | 0/6 | 6/6 |
| browser | 0/12 | 12/12 |
| debug | 0/24 | 24/24 |
| http | 1/30 | 30/30 |
| install | 0/6 | 6/6 |
| interpret | 14/18 | 18/18 |
| limits | 0/6 | 6/6 |
| mode | 0/12 | 12/12 |
| model | 0/6 | 6/6 |
| no-match | 0/6 | 6/6 |
| not-llm | 6/12 | 12/12 |
| threshold | 0/12 | 12/12 |

Without the skill, both models invent the API (for example `POST /classify` with `{"text", "labels"}`),
guess install commands, treat the task key as a plain identifier or say they cannot tell, and suggest
ways to make `fp16` work. They do well only on conceptual questions (`interpret`) and on refusing to
summarize.

### Versions (one run each, final grader)

| Skill | Words | haiku tuning | haiku held-out | sonnet tuning | sonnet held-out |
| --- | --- | --- | --- | --- | --- |
| v1: first draft | 2,302 | 18/18 | 7/7 | 18/18 | 7/7 |
| v2: tighter wording; queue limit (256, 429); "show the whole body" | 2,086 | 18/18 | 7/7 | 18/18 | 7/7 |
| v3: field list names `model` and `schema.kind` | 2,101 | 18/18 | 7/7 | 18/18 | 7/7 |
| v3 + Homebrew commands on separate lines (the round 1 file) | 2,116 | 18/18 | 7/7 | 18/18 | 7/7 |

The full skill passed every task from v1 on, so the versions were compared by ablation instead.

### Ablations: what each part of the skill is worth

Each row removes one part of skill v2 (one run). Tuning tasks only; held-out counts are in brackets.

| Removed | haiku | sonnet | Tasks that failed |
| --- | --- | --- | --- |
| nothing (v2) | 18/18 (7/7) | 18/18 (7/7) | |
| the exact request example (section 4) | 8/18 (3/7) | 7/18 (3/7) | every task that builds a request: `model` and `schema.kind` go missing |
| install commands (section 3) | 17/18 (7/7) | 17/18 (7/7) | install-serve (both models) |
| "task name is model input" | 17/18 (7/7) | 17/18 (7/7) | task-name (both) |
| "set both `threshold` and `options.threshold`" | 17/18 (7/7) | 17/18 (7/7) | threshold-low (both) |
| sources of truth (section 1) | 16/18 (7/7) | 18/18 (7/7) | debug-422-parens, batch-many (haiku) |
| errors table (section 9) | 18/18 (7/7) | 17/18 (7/7) | debug-422-parens (sonnet) |
| many texts / web pages (section 8) | 17/18 (7/7) | 18/18 (6/7) | cors-origin (haiku) |
| use it or not (section 2) | 18/18 (7/7) | 18/18 (7/7) | |
| single vs several labels (section 5) | 18/18 (7/7) | 18/18 (7/7) | |
| response example (section 6) | 18/18 (7/7) | 18/18 (7/7) | |
| thresholds in code (section 7) | 18/18 (7/7) | 18/18 (7/7) | |

Sections 2, 5, 6 and 7 overlap: each fact in them also appears in another section, so removing one
alone costs nothing on these tasks.

v3 change, measured the same way: with the request example removed, v2 drops to 8/18 and 7/18,
v3 (field list now names `model` and `schema.kind`) to 18/18 and 17/18. The rules no longer depend
on the agent copying the example.

### Grader changes during the work

All were applied to A and B alike, and every saved run was re-graded with `regrade.mjs`. None changed
an A result.

- debug-422-parens: a schema-only fragment is accepted (`fillDefaults`), since the task asks only for
  the label fix.
- threshold-low: no longer requires `billing` and `account` in the result. The answer chooses the task
  name, and the task name shifts the scores, so a correct request could return only `account`.
- Request extraction: handles shell-escaped apostrophes in curl (`'"'"'`), constants such as
  `LABELS = [...]`, and calls such as `str(uuid.uuid4())`. These three were found by reading held-out
  failures after the last skill edit; before the fix, B scored 6/7 held-out on haiku in each run, all
  on python-client, where the answer's Python was correct but could not be parsed.

### Limits of the easy tier

- The same author wrote the skill and the tasks at the same time, so the held-out tasks test topics
  the skill covers. They do not test unexpected requests.
- B is at the ceiling. The tier can find missing facts but cannot rank two good skills; the hard tier
  exists for that.
- One sample per task per run at temperature 0; the three final runs agreed.
- The agents had no tools. The skill's advice to read the live docs and `/openapi.json` was not
  exercised.
- Skill triggering (whether an agent loads the skill from its description) was not tested.
