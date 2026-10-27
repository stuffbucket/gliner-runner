# Accuracy on labelled public datasets, 2026-10-21

This run measures how often GLiNER Runner's three ONNX models pick the right label on eight
labelled datasets that are free to use for evaluation. Every example went through a local
runner over HTTP, with the request shape the site and playground use. The earlier reports in
`benchmark-results/` check that the runner gives the same answers as PyTorch. This one checks
whether those answers are right.

Scope: `gliner-runner-onnx` 0.4.0, ONNX Runtime CPU, FP32, the `classify` operation, one
request at a time, one Apple M5 Max host. Models at the runner's pinned revisions:
`decide-340m` (fastino/GLiNER2.5-Decide @ `9d1bfb84`), `decide-multilingual`
(fastino/GLiNER2.5-multi-Decide @ `e173bca1`), `decide-1b` (fastino/GLiNER2.5-Decide-1B @
`67af949e`).

Files in this directory:

- `samples.json`: the sampled ids for every task, with the seed (20261021) and the method.
- `results.json`: every metric for every run, including per-label precision, recall and F1,
  reliability bins, baselines, latency and per-group accuracy.
- `predictions.jsonl.gz`: every answer the runner gave (run, example id, predicted labels, top
  probability, inference time). It holds no dataset text and no gold labels.

The harness is in `benchmarks/accuracy/` (`manifest.json`, `fetch.py`,
`make_descriptions.py`, `descriptions.json`, `evaluate.py`).

## Headline results

Accuracy in percent (several-label: micro-F1). "Desc" means the request also carried a
description for every label. ECE is expected calibration error, from 0 to 1, lower is better.
n is the number of scored decisions.

| Dataset (task) | n | Model | Plain | With desc | ECE plain | ECE desc | Most-common-label baseline | Random baseline |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast Decisions, dev split (domain average) | 2,900 | decide-340m | 63.4 | 61.5 | 0.161 | 0.019 | 25.8 | 20.2 |
| Fast Decisions, dev split (domain average) | 2,900 | decide-multilingual | 58.6 | – | 0.148 | – | 25.8 | 20.2 |
| Fast Decisions, 200-row subsample (domain average) | 342 | decide-340m | 59.7 | 58.2 | 0.211 | 0.066 | | |
| Fast Decisions, 200-row subsample (domain average) | 342 | decide-1b | 60.8 | – | 0.159 | – | | |
| Fast Decisions, 200-row subsample (domain average) | 342 | decide-multilingual | 58.1 | – | 0.172 | – | | |
| BANKING77 (77 intents) | 770 | decide-340m | 71.3 | – | 0.127 | – | 1.3 | 1.6 |
| BANKING77, 200-row subsample | 200 | decide-340m | 73.5 | 71.0 | 0.122 | 0.182 | | |
| BANKING77, 200-row subsample | 200 | decide-1b | 72.0 | – | 0.119 | – | | |
| Support tickets (10 queues) | 1,000 | decide-340m | 22.4 | 19.2 | 0.098 | 0.143 | 29.2 | 9.7 |
| Support tickets (10 queues) | 1,000 | decide-multilingual | 18.8 | 18.5 | 0.362 | 0.265 | 29.2 | 9.7 |
| Support tickets (10 queues) | 1,000 | decide-1b | 24.6 | – | 0.141 | – | 29.2 | 9.7 |
| Support tickets (priority) | 1,000 | decide-340m | 39.6 | 40.3 | 0.069 | 0.102 | 40.5 | 34.3 |
| Support tickets (priority) | 1,000 | decide-multilingual | 42.9 | 43.9 | 0.061 | 0.101 | 40.5 | 34.3 |
| Support tickets (priority) | 1,000 | decide-1b | 32.4 | – | 0.142 | – | 40.5 | 34.3 |
| News Category (42 sections) | 1,000 | decide-340m | 41.5 | 40.5 | 0.267 | 0.077 | 17.0 | 2.8 |
| Amazon reviews (positive, negative) | 1,000 | decide-340m | 92.8 | – | 0.040 | – | 50.0 | 49.8 |
| GoEmotions (28, several labels), threshold 0.5 | 1,000 | decide-340m | 15.2 | 6.5 | 0.017 | 0.014 | 31.8 | 12.8 |
| GoEmotions, threshold tuned on validation | 1,000 | decide-340m | 33.1 (t=0.19) | 28.0 (t=0.16) | | | 31.8 | 12.8 |
| Civil Comments (toxic, not toxic) | 1,000 | decide-340m | 46.0 | 47.6 | 0.350 | 0.233 | 92.0 | 51.0 |
| MASSIVE scenario (18, 10 languages) | 1,000 | decide-340m | 47.1 | 44.6 | 0.186 | 0.075 | 14.0 | 4.8 |
| MASSIVE scenario (18, 10 languages) | 1,000 | decide-multilingual | 48.2 | 42.6 | 0.066 | 0.093 | 14.0 | 4.8 |
| MASSIVE language (10 languages) | 1,000 | decide-340m | 85.0 | – | 0.119 | – | 10.0 | 8.7 |
| MASSIVE language (10 languages) | 1,000 | decide-multilingual | 57.8 | – | 0.138 | – | 10.0 | 8.7 |

95% intervals (Wilson) for every accuracy are in `results.json` and in the tables below. With
1,000 examples they are about ±3 points; with 200, about ±6.

What stands out:

1. On fastino's own suite, decide-340m scores 63.4% (domain average, development split).
   fastino publishes 60.2% on the held-out test split. On BANKING77 it scores 71.3%.
2. Label descriptions, written mechanically from the label names or from the datasets'
   documentation, did not raise accuracy on any task by more than its 95% interval, and
   lowered it on several (Fast Decisions 340m −1.9 points, MASSIVE scenario −2.5 and −5.6,
   support queue −3.2, GoEmotions at 0.5 −8.7). Calibration moved both ways: ECE fell on Fast
   Decisions (0.161 to 0.019), News (0.267 to 0.077), Civil Comments (0.350 to 0.233) and
   MASSIVE with 340m (0.186 to 0.075), and rose on the ticket queue, ticket priority,
   BANKING77 and MASSIVE with the multilingual model. Descriptions always cost time
   (BANKING77 median 640 ms to 5,241 ms, because 77 descriptions take the input from 534 to
   1,998 tokens).
3. Several-label mode at the default threshold of 0.5 mostly returns nothing on GoEmotions
   (847 of 1,000 empty). A threshold chosen on a separate validation sample (0.19) more than
   doubles micro-F1, to 33.1, which is about the same as always answering "neutral" (31.8).
4. On moderation, decide-340m flags most comments as toxic: it finds 75 of the 80 toxic
   comments but also flags 535 of the 920 that are not.
5. For telling languages apart, use decide-340m (85.0%), not decide-multilingual (57.8%).
   decide-340m confuses Japanese and Chinese with Russian; decide-multilingual answers
   "Spanish" for 342 of the 1,000 utterances.
6. On the synthetic support-ticket set every model is below the most-common-label baseline
   on queue. On priority only decide-multilingual is above it (42.9 and 43.9 against 40.5),
   inside its 95% interval.

## Datasets

All eight are pinned by revision and file SHA-256 in `benchmarks/accuracy/manifest.json`,
which also quotes each licence. Raw data is not committed.

| Dataset | Source and revision | Licence (as stated by the source) | Used | Task |
| --- | --- | --- | --- | --- |
| Fast Decisions | [fastino/fast-decisions](https://huggingface.co/datasets/fastino/fast-decisions) @ `35afb79a` | Apache-2.0 ("**License:** Apache 2.0") | all 1,700 rows (2,900 decisions); 200-row subsample for decide-1b | 17 domains, 29 heads: 26 single-label, 3 several-label |
| BANKING77 | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77); data from PolyAI-LDN/task-specific-datasets @ `57ec275d` | CC-BY-4.0 ("Creative Commons Attribution 4.0 International") | 770 of 3,080 test rows (10 per intent); 200-row subsample | 77 intents |
| Customer Support Tickets | [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) @ `ddf1c81a` | CC-BY-NC-4.0 ("license: cc-by-nc-4.0"), non-commercial | 1,000 of 19,998, stratified by language and queue (595 English, 405 German) | 10 queues; priority low, medium, high |
| News Category | [heegyu/news-category-dataset](https://huggingface.co/datasets/heegyu/news-category-dataset) @ `304a05a5` (mirror of R. Misra's Kaggle set) | CC-BY-4.0 ("license: cc-by-4.0") | 1,000 of 209,521, stratified by section | 42 sections |
| Amazon Review Polarity | [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) @ `9d9c45c1` | Apache-2.0 ("Apache License 2.0"); the card does not say how the underlying reviews were licensed | 1,000 of 400,000 test rows, stratified | positive, negative |
| GoEmotions (simplified) | [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) @ `add49224` | Apache-2.0 | 1,000 of 5,427 test rows (simple random); 500 of 5,426 validation rows for the threshold | 28 labels, several per text |
| Civil Comments | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) @ `f2970eb3` | CC0-1.0 ("released under CC0 1.0") | 1,000 of 97,320 test rows, stratified (80 toxic) | toxic (score 0.5 or more), not toxic |
| MASSIVE 1.1 | [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive), parquet conversion @ `ed58ac42` | CC-BY-4.0 ("license: cc-by-4.0") | 100 utterances, stratified by scenario, in each of 10 locales: 1,000 rows | 18 scenarios; the utterance's language |

The customer-support-tickets set is non-commercial. It is used only to measure the models and
publish aggregate scores. No text from it is committed, not even its card's label definitions,
which `fetch.py` reads from the card at run time. This report only paraphrases it.
Its card says the tickets are synthetic.

Sets looked at and not used: AG News, SST-2, tweet_eval and IMDB (licence listed as unknown or
other), papluca/language-identification (no licence, and built partly from the withdrawn
Amazon multilingual reviews corpus), amazon_reviews_multi (withdrawn), dbpedia_14 (CC-BY-SA,
not on the requested list), and CLINC150 (CC-BY-3.0; dropped to keep to eight sets, and
because its 151 labels would have made it the slowest set).

What fastino, TypeSafe and Microsoft publish: fastino's only published labelled set for the
Decide models is Fast Decisions, and only its development split is public. TypeSafe publishes
no evaluation set or accuracy figure for its decision model. The Azure AI Foundry
"Microsoft-Decision-1" model is a post-trained Qwen3.5-9B, not a Decide model; its blog
reports a mean over 36 benchmarks it does not name, and names no reusable dataset.

## Method

- Runner: `node packages/onnx-runner/dist/src/cli.js serve --port 8102` with
  `GLINER_RUNNER_MODELS_DIR` pointing at the local ONNX models. Version 0.4.0, identity
  SHA-256 `63f4033ace2f4ffe3b1eab8f930c5a01637fa3db633bc36dd1891d62b420f7cd`. Node 24.21.0,
  onnxruntime-node 1.30.0, macOS 26.7.1, Apple M5 Max (18 cores, 128 GB).
- Requests: `POST /v1/infer` with `backend: onnx`, `precision: fp32`, one classification task
  per request, the task named after the dataset's field (`intent`, `queue`, `priority`,
  `topic`, `sentiment`, `emotions`, `toxicity`, `scenario`, `language`, or the Fast Decisions
  head name), and the labels spelled exactly as the dataset spells them, sorted. Several-label
  tasks add `min_labels: 0, max_labels: null`. This is the shape of `classify()` in
  `site/scripts/runner.mjs`. One request at a time. A 429 would have been retried with
  backoff; none occurred.
- Fast Decisions follows the dataset card's own evaluation: one request per head, the head's
  labels in the given order, its `multi_label` flag passed through, and predicted and gold
  labels compared as sets. "Domain average" is the mean of the 17 per-domain accuracies;
  "overall" pools all decisions.
- Sampling: seed 20261021. Stratified samples use proportional allocation (largest
  remainder) and a seeded draw within each stratum. `samples.json` lists every id.
- Subsamples: BANKING77 with descriptions ran at 4.6 s per example, so it was stopped after
  287 examples and cut to a stratified 200-example subsample of the 770 (2 or 3 per intent).
  The 76 examples already done that fell in the subsample were kept, the other 124 were run,
  and the remaining 211 answers were discarded. decide-1b on BANKING77 and on Fast Decisions
  used the same kind of 200-example subsample (Fast Decisions: 11 or 12 rows per domain, 342
  decisions). Every full run is also scored on the same subsample ids, so the subsample rows
  compare like with like.
- Metrics, single-label: accuracy with a Wilson 95% interval; macro-F1 over the labels that
  occur as gold or prediction; for two-label tasks, ROC AUC of the rarer label's probability.
- Metrics, several-label: micro- and macro-F1 and exact match at the runner's default
  threshold of 0.5, and at the threshold with the best micro-F1 on the separate validation
  sample (grid 0.01 to 0.99). Recomputing the selection from the returned probabilities at
  0.5 reproduced the runner's own answer for all 1,000 GoEmotions examples.
- Calibration: expected calibration error over 10 equal-width bins. Single-label: the
  probability of the predicted label against whether it was right. Several-label: every
  (text, label) probability against whether the label is gold. That version is dominated by
  the many easy negatives, so it is low for GoEmotions and says little.
- Baselines: "most common label" always answers the label most frequent in the scored sample
  itself, which favours the baseline. "Random" picks uniformly among the candidate labels with
  a seeded generator; for GoEmotions it picks each label independently with its frequency in
  the validation sample, and for Fast Decisions several-label heads each label with p = 0.5.
- Latency: the runner's own `inference_ms` per request, median and 90th percentile, with one
  request in flight.
- Label descriptions (`benchmarks/accuracy/descriptions.json`, written once by
  `make_descriptions.py`): from the dataset's documentation where it defines its labels
  (GoEmotions paper appendix A, verbatim; the Civil Comments paper's definition of toxicity;
  the ticket card's queue and priority definitions, verbatim, read from the card by `fetch.py`
  and kept out of the repository), otherwise a fixed template
  from the label name (BANKING77 "The customer's message is about {label words}.", News "A
  news story from the {Label} section.", MASSIVE "A request to a voice assistant about
  {label}.", Fast Decisions "{head words}: {label words}"). No wording was tuned. Amazon
  polarity and the language task ran without descriptions, because their labels are already
  plain words.

## Results by task

### Fast Decisions (development split)

| Model | Descriptions | Decisions | Domain average | Overall | 95% CI (overall) | ECE | Median ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decide-340m | no | 2,900 | 63.4 | 61.9 | 60.2–63.7 | 0.161 | 95 |
| decide-340m | yes | 2,900 | 61.5 | 59.8 | 58.0–61.6 | 0.019 | 220 |
| decide-multilingual | no | 2,900 | 58.6 | 57.3 | 55.5–59.1 | 0.148 | 43 |
| decide-340m (200-row subsample) | no | 342 | 59.7 | 57.3 | 52.0–62.4 | 0.211 | 92 |
| decide-340m (200-row subsample) | yes | 342 | 58.2 | 55.6 | 50.3–60.7 | 0.066 | 223 |
| decide-1b (200-row subsample) | no | 342 | 60.8 | 57.9 | 52.6–63.0 | 0.159 | 243 |
| decide-multilingual (200-row subsample) | no | 342 | 58.1 | 55.6 | 50.3–60.7 | 0.172 | 42 |

Baselines on the full split (domain averages): most common answer per head 25.8, random 20.2.
ECE here is over the single-label heads.

| Domain | 340m | 340m + desc | multilingual |
| --- | ---: | ---: | ---: |
| agent_handoff | 76.0 | 75.0 | 66.0 |
| banking_intent | 64.0 | 61.0 | 58.0 |
| benefits_request | 75.0 | 76.5 | 70.5 |
| clinic_request | 64.5 | 62.0 | 67.0 |
| document_type | 79.0 | 81.0 | 75.0 |
| email_triage | 59.2 | 56.5 | 55.0 |
| news_topic | 69.0 | 68.0 | 63.0 |
| paper_field | 47.0 | 43.0 | 38.0 |
| product_feedback | 51.5 | 48.5 | 45.0 |
| restaurant_review | 53.5 | 46.0 | 39.5 |
| review_sentiment | 84.0 | 83.0 | 81.0 |
| screen_tags | 48.0 | 44.5 | 44.0 |
| sports_recap | 75.7 | 71.7 | 68.0 |
| support_intent | 76.0 | 70.0 | 65.0 |
| support_topic | 44.0 | 45.0 | 48.0 |
| ticket_route | 48.3 | 50.0 | 50.3 |
| travel_request | 63.0 | 63.0 | 62.0 |

The three several-label heads are the weakest for every model (exact set match, n=100 each):
`screen_tags/genres` 13, 10 and 8; `restaurant_review/aspects` 25, 10 and 6;
`product_feedback/product_area` 29, 25 and 24 (340m, 340m with descriptions, multilingual).
Per-head numbers for all 29 heads are in `results.json`.

### Support and intent

| Task | Model | Desc | n | Accuracy | 95% CI | Macro-F1 | ECE | Median ms | p90 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BANKING77 | decide-340m | no | 770 | 71.3 | 68.0–74.4 | 71.2 | 0.127 | 706 | 1,271 |
| BANKING77 (sub200) | decide-340m | no | 200 | 73.5 | 67.0–79.1 | 71.8 | 0.122 | 640 | 1,278 |
| BANKING77 (sub200) | decide-340m | yes | 200 | 71.0 | 64.4–76.8 | 67.5 | 0.182 | 5,241 | 5,944 |
| BANKING77 (sub200) | decide-1b | no | 200 | 72.0 | 65.4–77.8 | 71.0 | 0.119 | 651 | 868 |
| Ticket queue | decide-340m | no | 1,000 | 22.4 | 19.9–25.1 | 21.7 | 0.098 | 93 | 209 |
| Ticket queue | decide-340m | yes | 1,000 | 19.2 | 16.9–21.8 | 18.2 | 0.143 | 132 | 277 |
| Ticket queue | decide-multilingual | no | 1,000 | 18.8 | 16.5–21.3 | 18.0 | 0.362 | 54 | 107 |
| Ticket queue | decide-multilingual | yes | 1,000 | 18.5 | 16.2–21.0 | 18.4 | 0.265 | 63 | 99 |
| Ticket queue | decide-1b | no | 1,000 | 24.6 | 22.0–27.4 | 21.9 | 0.141 | 308 | 561 |
| Ticket priority | decide-340m | no | 1,000 | 39.6 | 36.6–42.7 | 34.7 | 0.069 | 79 | 210 |
| Ticket priority | decide-340m | yes | 1,000 | 40.3 | 37.3–43.4 | 37.2 | 0.102 | 207 | 450 |
| Ticket priority | decide-multilingual | no | 1,000 | 42.9 | 39.9–46.0 | 36.8 | 0.061 | 38 | 67 |
| Ticket priority | decide-multilingual | yes | 1,000 | 43.9 | 40.8–47.0 | 33.0 | 0.101 | 43 | 65 |
| Ticket priority | decide-1b | no | 1,000 | 32.4 | 29.6–35.4 | 31.3 | 0.142 | 238 | 474 |

Baselines: BANKING77 most common 1.3, random 1.6; ticket queue 29.2 (always Technical
Support) and 9.7; ticket priority 40.5 (always medium) and 34.3. Ticket results by language
(English n=595, German n=405) are in `results.json`; no run differs by more than 4.0 points
between the two.

### Topic

| Task | Model | Desc | n | Accuracy | 95% CI | Macro-F1 | ECE | Median ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| News Category | decide-340m | no | 1,000 | 41.5 | 38.5–44.6 | 28.9 | 0.267 | 270 |
| News Category | decide-340m | yes | 1,000 | 40.5 | 37.5–43.6 | 28.6 | 0.077 | 1,452 |

Baselines 17.0 (always POLITICS) and 2.8.

### Sentiment and emotion

| Task | Model | Desc | n | Result | ECE | Median ms |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Amazon polarity | decide-340m | no | 1,000 | accuracy 92.8 (91.0–94.2), AUC 0.970 | 0.040 | 162 |
| GoEmotions at 0.5 | decide-340m | no | 1,000 | micro-F1 15.2, macro-F1 21.9, 847 empty | 0.017 | 104 |
| GoEmotions at 0.19 (tuned) | decide-340m | no | 1,000 | micro-F1 33.1, macro-F1 32.8, exact 15.7 | | |
| GoEmotions at 0.5 | decide-340m | yes | 1,000 | micro-F1 6.5, macro-F1 10.0, 935 empty | 0.014 | 345 |
| GoEmotions at 0.16 (tuned) | decide-340m | yes | 1,000 | micro-F1 28.0, macro-F1 31.3, exact 11.5 | | |

Baselines: Amazon 50.0 and 49.8; GoEmotions micro-F1 31.8 (always "neutral") and 12.8. At
0.5 the 340m model's answers are mostly right when it gives one (micro precision 63.3) but it
rarely gives one (micro recall 8.6).

### Moderation

| Task | Model | Desc | n | Accuracy | Macro-F1 | Toxic precision | Toxic recall | AUC | ECE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Civil Comments | decide-340m | no | 1,000 | 46.0 | 40.3 | 12.3 | 93.8 | 0.728 | 0.350 |
| Civil Comments | decide-340m | yes | 1,000 | 47.6 | 41.4 | 12.8 | 95.0 | 0.762 | 0.233 |

80 of the 1,000 are toxic. Always answering "not toxic" scores 92.0.

### Language and other languages (MASSIVE, the same 100 utterances in each language)

| Language | Scenario 340m | 340m + desc | multilingual | multilingual + desc | Language ID 340m | Language ID multilingual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 57 | 54 | 47 | 47 | 95 | 34 |
| German | 43 | 44 | 46 | 43 | 98 | 46 |
| Spanish | 45 | 44 | 48 | 43 | 92 | 97 |
| French | 52 | 51 | 51 | 44 | 95 | 93 |
| Italian | 46 | 42 | 46 | 41 | 92 | 85 |
| Portuguese | 47 | 49 | 43 | 35 | 73 | 69 |
| Dutch | 46 | 44 | 48 | 42 | 83 | 67 |
| Russian | 41 | 35 | 47 | 44 | 99 | 43 |
| Japanese | 54 | 46 | 53 | 45 | 50 | 19 |
| Chinese | 40 | 37 | 53 | 42 | 73 | 25 |
| All (n=1,000) | 47.1 | 44.6 | 48.2 | 42.6 | 85.0 | 57.8 |

Each cell under a language has n=100, so its 95% interval is about ±10 points.

## Comparison with published numbers

| Source | Model | Dataset and split | Their setup | Published | This run |
| --- | --- | --- | --- | ---: | --- |
| [GLiNER2.5-Decide card](https://huggingface.co/fastino/GLiNER2.5-Decide) | decide-340m | Fast Decisions, held-out test (5,100) | exact match, default threshold, no descriptions | 60.2 | 63.4 domain average, 61.9 overall (development split, 2,900 decisions) |
| same table | decide-1b | same | same | 59.6 | 60.8 domain average (200-row development subsample, 342 decisions) |
| same table | decide-multilingual | same, "scored here on English" | same | 56.7 | 58.6 domain average (development split) |
| [fastino blog](https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model) | decide-340m | Fast Decisions test, support intent | same | 75.3 | 76.0 (development, n=100) |
| same | decide-340m | Fast Decisions test, banking intent | same | 64.3 | 64.0 (development, n=100) |
| [GLiNER2 paper](https://arxiv.org/abs/2507.18546), table 2 | GLiNER2 base (205M), not a Decide model | BANKING77, split not stated | not stated | 0.70 | decide-340m 71.3 (770 test rows) |

The differences in setup matter more than the gaps. fastino scores the held-out test split,
which is not public, and its card says not to report development-split scores as the
benchmark; the rows above are development-split scores and are not the benchmark. fastino
runs the PyTorch `gliner2` library; this run uses the ONNX runner, which the parity report in
`2026-10-14-onnx-runtime-parity-v1` shows matching PyTorch within 2.2e-5 on its corpora.
fastino's model cards were read at `main` on 2026-10-21. The blog gives the 340m average as
60.1 where the cards say 60.2, and the dataset card labels the 59.6 row "GLiNER2 XL (1B)"
where the model cards say GLiNER2.5-Decide-1B. The paper's number is for a different, older
model. TypeSafe and Microsoft publish no number for these models on any of these datasets.

## Where it does badly

1. **Several labels at once, at the default threshold.** In several-label mode each label gets
   its own probability, and on GoEmotions they are low: the median top probability is 0.26.
   At 0.5, 847 of 1,000 texts get no label at all. Examples (GoEmotions test, gold in
   brackets): "I despise [NAME] and all his cronies" (anger) gets anger 0.48 and disgust
   0.47, so no label; "Happy cake day!" (excitement) gets joy 0.27. Fast Decisions'
   several-label heads show the same thing, with exact set match of 13% to 29%: a review
   whose gold aspects are food, service, price, wait and cleanliness gets all of them except
   "wait" (0.20). A threshold chosen on held-out data (0.19 here) helps; descriptions made
   it worse.
2. **Moderation over-flags.** decide-340m called 610 of the 1,000 Civil Comments toxic; 80
   are. Comments it flagged with high confidence include one that opens by citing a
   Washington Post article (toxic 0.87) and a remark about church, state and common-law
   marriage (0.99). The
   toxic probability ranks comments only moderately well (AUC 0.73). Used as a filter, it
   would send most ordinary comments to review.
3. **Labels that overlap, or are short codes.** News Category has STYLE and STYLE & BEAUTY,
   PARENTS and PARENTING, WELLNESS and HEALTHY LIVING, and the model draws those lines
   differently from the editors (16 STYLE & BEAUTY stories answered STYLE). MASSIVE has both
   "play" and "music", and uses "iot" for smart-home requests: "prepare coffee now" goes to
   "cooking" with probability 1.00. BANKING77 confuses neighbours: "How do I receive my
   physical card" is gold `order_physical_card` and gets `get_physical_card` at 1.00.
4. **Scripts the English model does not know, and a multilingual model that cannot name
   languages.** decide-340m answers "Russian" for 48 of 100 Japanese and 24 of 100 Chinese
   utterances ("プレイリストを作成" gets Russian 0.68). decide-multilingual's top probability on
   the language task is usually below 0.2 across 10 labels, and it answers "Spanish" 342
   times, including for 55 of the 100 English utterances ("create a playlist").
5. **The synthetic ticket set.** Every model scores below always answering "Technical
   Support" on queue. On priority, decide-340m (39.6) and decide-1b (32.4) are below always
   answering "medium" (40.5), decide-1b is also below the random baseline (34.3), and
   decide-multilingual (42.9) is above it by less than its interval. Many gold labels in this
   set look arbitrary on
   reading (for example, tickets about unauthorised access to medical records marked low
   priority), so this measures agreement with these labels more than routing quality. It is
   the weakest evidence in this report.

## Speed

Median `inference_ms` per request, one request in flight, CPU FP32 on the M5 Max. It grows
with the number of labels and with descriptions, because every label, and its description,
is part of the model's input.

| Task | Labels | 340m | 340m + desc | multilingual | 1b |
| --- | ---: | ---: | ---: | ---: | ---: |
| Amazon polarity | 2 | 162 | | | |
| Civil Comments | 2 | 120 | 101 | | |
| Ticket priority | 3 | 79 | 207 | 38 | 238 |
| Ticket queue | 10 | 93 | 132 | 54 | 308 |
| MASSIVE scenario | 18 | 104 | 140 | 45 | |
| GoEmotions | 28 | 104 | 345 | | |
| News Category | 42 | 270 | 1,452 | | |
| BANKING77 (sub200) | 77 | 640 | 5,241 | | 651 |
| Fast Decisions | 2 to 28 | 95 | 220 | 43 | 243 (sub200) |

## Run time

Requests ran from 21:39:14 to 23:32:02 UTC and from 23:40:34 to 00:17:07 UTC: 2 h 29 min of
wall time, one request at a time. That includes about 15 minutes on BANKING77 with
descriptions whose 211 answers outside the subsample were discarded. The kept runs add up to
8,018 s.

## Limitations

- One host, CPU, FP32, one request at a time. Latency on other hardware or under load will
  differ.
- Samples, not full test sets: 1,000 examples give about ±3 points, the 200-example
  subsamples about ±6, and per-language cells (n=100) about ±10.
- decide-1b ran only on the support and intent sets, two of them on 200-example subsamples.
  decide-multilingual ran only on Fast Decisions, MASSIVE and the ticket set.
- Fast Decisions scores are on the public development split, not on fastino's held-out test
  split, so they are not the published benchmark.
- The most-common-label baseline uses the scored sample's own label counts, which favours
  the baseline.
- Descriptions were written mechanically and once. Better-written descriptions might help;
  this run does not show that they do.
- Several-label thresholds were tuned only for GoEmotions. Fast Decisions several-label heads
  use the default 0.5, as fastino's evaluation does.
- The support-ticket set is synthetic, non-commercial and, on reading, noisily labelled. The
  Amazon polarity licence is as stated on its card; the underlying reviews' terms are not
  stated there.

## Reproduce

```sh
# 1. Data: needs pyarrow; each dataset goes into its own directory outside the repo
python -I benchmarks/accuracy/fetch.py --out /tmp/gr-work/datasets
python -I benchmarks/accuracy/make_descriptions.py --data /tmp/gr-work/datasets  # rewrites descriptions.json

# 2. Samples: writes samples.json; the seed is fixed
python -I benchmarks/accuracy/evaluate.py sample --out benchmark-results/2026-10-21-accuracy

# 3. A runner
GLINER_RUNNER_MODELS_DIR=dist/onnx-models node packages/onnx-runner/dist/src/cli.js serve --port 8102

# 4. Runs, for example
OUT=benchmark-results/2026-10-21-accuracy
python -I benchmarks/accuracy/evaluate.py run --out $OUT --model decide-340m banking77/intent
python -I benchmarks/accuracy/evaluate.py run --out $OUT --model decide-340m --descriptions fast_decisions
python -I benchmarks/accuracy/evaluate.py run --out $OUT --model decide-1b --subsample sub200 banking77/intent

# 5. Scores: results.json and predictions.jsonl.gz
python -I benchmarks/accuracy/evaluate.py score --out $OUT
```

Full predictions with every probability go to `--work` (default `/tmp/gr-work/accuracy`),
outside the repo. `run` resumes where it stopped.
