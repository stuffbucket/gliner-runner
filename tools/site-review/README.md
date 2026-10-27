# Site review

Reviews the documentation site the way a few real readers would, then reports what confused them, what they
could not find, what fails accessibility checks, and what to change. It is re-runnable: point it at a dev server,
a preview build or the deployed site, and compare two reports.

```sh
GLINER_RUNNER_MODELS_DIR=dist/onnx-models node tools/site-review/review.mjs --site http://127.0.0.1:4399/gliner-runner/
```

The report lands in `/tmp/gr-work/site-review/<timestamp>/report.md` unless `--out` says otherwise.

## Before you run it

- **The site.** Any URL that serves the site under its base path. For a local build:
  `pnpm site:build`, then `pnpm --filter @gliner-runner/site exec astro preview --host 127.0.0.1 --port 4399`
  (Astro 7 runs the preview in the background; stop it with `astro preview stop` in `site/`).
- **The runner and its models.** The review starts `gliner-runner serve` itself, from
  `packages/onnx-runner/dist/src/cli.js` (build it with `pnpm --filter "@gliner-runner/onnx-runner..." run build`,
  or pass `--runner-cli` to use one built elsewhere). Models come from `GLINER_RUNNER_MODELS_DIR` or `--models-dir`.
  It needs `decide-340m`.
- **A chat model.** An OpenAI-compatible API, by default `http://localhost:4141/v1` with `claude-sonnet-5`.
- **Chromium.** Playwright 1.63.0 uses Chromium revision 1243. If that is in the Playwright cache
  (`~/Library/Caches/ms-playwright/chromium-1243`), the review launches it by path. Otherwise run
  `npx playwright@1.63.0 install chromium` once, or pass `--chrome <path>`.

The first run installs the pinned packages (below) into `~/.cache/gliner-runner/site-review`, and `npx` fetches
promptfoo (about 3 minutes the first time, cached after).

## What it does

| Stage | Tool | What happens |
| --- | --- | --- |
| walk | Playwright, chat model | Each persona in `personas.json` starts at the landing page. For each page the script reads the visible text, headings and links and takes a screenshot. The model, in persona, returns JSON: what I understood, what confused me (with a quote), what I expected and did not find, what looks wrong, whether I reached my goal, and which link I click next. The script follows that link, up to `--steps` pages. Each quote is checked against the page text, so invented quotes are marked. |
| hands-on | Playwright, gliner-runner, chat model | Personas with a `handsOn` task open the playground connected to the local runner (`?mode=local&runner=…`; the runner is started with `--cors-origin` for the site's origin). The persona picks a text and labels, or a thing and clues for 20 questions; the script types and clicks, answers each guess honestly, then the persona says what they made of the result. |
| checks | Playwright, axe-core | Ten key pages plus every page a persona reached: axe (WCAG 2.0 to 2.2 A and AA rules), navigation timing and transfer size, and a 390px phone view that flags anything wider than the screen. Every internal link seen is requested once. The page text is saved for the next stage. |
| comprehension | promptfoo | `comprehension/questions.json` holds persona questions whose answers are on a given page ("what does probabilities contain", "is my text sent anywhere", "how do I install it"). The model answers from that page's text only; an `llm-rubric` grader and `javascript` fact checks compare the answer with what the page and code say. A failure means the page does not teach that fact to that reader. |
| classify | gliner-runner | Every confusion, missing item, visual note, failed question, axe rule, broken link, overflow and slow page becomes a finding. `decide-340m` classifies each one into the categories in `lib/findings.mjs` (several labels allowed, with label descriptions), and the scores are kept. Findings from checks also carry a known category; the report says how often the classifier agreed with it. Near-duplicates on the same page (same quote, or similar words) are merged with their evidence. |
| suggest | chat model | One call per category turns its findings into page-specific changes, each tied to the findings (persona, page, quote) it rests on. |

The persona chain (walk, then hands-on) and the page chain (checks, then comprehension) run at the same time.

## Cost

Measured on the baseline run against a local preview (4 personas, 8 steps, `claude-sonnet-5`, concurrency 4):

| Stage | Time | Model calls | Other work |
| --- | --- | --- | --- |
| walk | 54 s | 26 | 26 pages read, about 134,000 prompt tokens in all |
| hands-on | 14 s | 6 | 3 playground tasks (two classifications, one 20 questions game) |
| checks | 32 s | 0 | 14 pages through axe and a phone view, 73 links requested |
| comprehension (promptfoo) | 23 s | 40 | 20 questions, an answer and a grade each |
| classify (gliner-runner) | 28 s | 0 | 138 classifications with `decide-340m` |
| suggest | 29 s | 10 | one call per category |
| **total** | **135 s** | **82** | the walk and the checks run at the same time |

The first run on a machine adds about a minute to install the pinned packages and about 3 minutes for `npx` to
fetch promptfoo. More personas or steps scale the walk linearly; `--concurrency` sets how many run at once.

Each walk step sends about 5,000 prompt tokens on average (the page text, up to 15,000 characters, plus links and
one JPEG screenshot). A promptfoo question is two calls: the answer and the grade.

## Options

| Flag | Default | |
| --- | --- | --- |
| `--site` | `https://stuffbucket.github.io/gliner-runner/` | The site root, with its base path. |
| `--out` | `/tmp/gr-work/site-review/<timestamp>` | Where the report, data and screenshots go. |
| `--model` | `claude-sonnet-5` | The chat model for personas, comprehension answers and grading, and suggestions. |
| `--api` | `http://localhost:4141/v1` | The OpenAI-compatible API. |
| `--personas` | `personas.json` | Another persona file. `--only a,b` picks some of them by id. |
| `--steps` | `8` | Most pages per persona. |
| `--concurrency` | `4` | Personas, promptfoo tests, classifications and suggestion calls in flight. |
| `--stages` | all | A comma list, for example `checks,comprehension` for a run with no walking. |
| `--runner-cli`, `--models-dir` | the repo's build | Which runner to start and where its models are. |
| `--classifier-model` | `decide-340m` | The gliner-runner model that classifies findings. |
| `--chrome` | cached Chromium 1243 | A Chromium or Chrome executable. |
| `--external-links` | off | Also send a HEAD request to every external link. |
| `--headed` | off | Show the browser. |

## Output

- `report.md`: the summary, the top findings, suggestions by category with evidence, every finding by category,
  each persona's path, the hands-on results, the comprehension table, axe rules, page speed and phone layout,
  and links.
- `data.json`: everything: each page read, each model reply, classifier scores, promptfoo results, timings and
  call counts.
- `screenshots/`: one per walk step, before and after each hands-on task, and each key page at phone width.
- `pages/`: the text of each key page as the comprehension stage read it.
- `comprehension.json`: promptfoo's own output.

## Why these tools

- **Playwright** drives a real Chromium, so the review sees the site as built, including the playground, which
  renders only in the browser. Version 1.63.0 because its Chromium (revision 1243) is the one already cached here;
  the review launches it by path and never downloads a browser.
- **@axe-core/playwright** runs axe inside the same page. It finds contrast, labelling and keyboard problems a
  model reading screenshots would miss or guess at.
- **Navigation timing instead of Lighthouse.** Lighthouse would add a large dependency and 10 to 20 seconds per
  page for scores that matter little on a static documentation site. The browser's own timing entries (load,
  first paint, requests, bytes) and a phone-width overflow check cover the layout and speed questions cheaply.
- **promptfoo** runs the comprehension questions as a normal eval suite: a fixed config in the repo, rubric and
  fact assertions, and JSON results. The config is `comprehension/promptfooconfig.mjs` and reads the model and
  API from the environment `review.mjs` sets. It runs with `npx -y promptfoo@0.123.1`: promptfoo has hundreds
  of dependencies and is only needed here.
- **gliner-runner** classifies the findings because sorting short texts into fixed labels is what it does, and
  the review then doubles as a test of it on real text. The agreement line in the report shows how far to trust it.
- **The chat model** does what needs judgement: reading as a persona, choosing the next link, grading answers,
  and writing suggestions.

## Why it is not a pnpm workspace package

`package.json` pins Playwright, axe-core and @axe-core/playwright to exact versions, but the folder is not in
`pnpm-workspace.yaml`. Adding it re-resolves `pnpm-lock.yaml` through the npm mirror, which writes sha1 hashes and
tarball URLs that `check:registry` rejects (see `docs/registries.md`); repairing them is a manual job for a tool
that only developers run. Instead `lib/deps.mjs` installs the pinned versions with `npm install --prefix
~/.cache/gliner-runner/site-review` on first use (npm checks each tarball's integrity) and imports them from there.
promptfoo goes through `npx -y` with an exact version for the same reason.

## Limitations

- Personas are a model playing a role. They find real gaps, but also ask for things the site has elsewhere, and
  they know more than they should. Read the evidence before acting on a finding; quotes not found on the page are
  marked in `data.json` (`onPage: false`).
- One run is one sample. Model replies vary between runs, so compare counts and themes, not exact wording.
- The hands-on tasks use the playground's "on my computer" mode. The in-browser mode would download 1.95 GB per run.
- The classifier is a small zero-shot model; its category is a starting point. Merging is by page and word overlap,
  so the same idea on two pages stays as two findings (the suggestion step usually merges them).
- Against the deployed site, the hands-on tasks need Chrome's local network access permission for
  `https://stuffbucket.github.io` to reach `127.0.0.1`, which headless Chromium may not grant.
- Speed numbers from a local preview say little about the deployed site.

## Tests

`node --test tools/site-review/lib/*.test.mjs` tests the pure parts (finding collection, merging, the report, quote
matching) without a browser, a model or a runner.
