# Production ONNX runtime: Node-versus-PyTorch end-to-end parity

This is the acceptance evidence for the Python-free ONNX runner
(`gliner-runner-onnx serve`, package `@gliner-runner/onnx-runner`). It replays
two fixed request corpora through the runner's real HTTP server and compares
every response with golden outputs produced by the PyTorch backend
(`gliner_runner.backends.pytorch.PyTorchBackend`, Fastino GLiNER2 2.1.0).

Scope is exactly: ONNX Runtime `CPUExecutionProvider`, FP32, no quantization,
the `classify` operation, the three curated model revisions below, and one
Apple M-series host. It does not support or imply MPS, CUDA, FP16, BF16,
quantized, other operations, other revisions, or other hosts.

## Method

- Corpora: `english-v1` and `multilingual-v1`
  (`tests/fixtures/onnx-parity/corpus-*.json`; synthetic text only). Each has 14
  cases: 10 successful classifications (described and plain labels, single and
  multi-label, thresholds and empty results, ordering, a feasible constraint,
  multiple tasks, defaults and temperature, softmax activation, a long
  mixed-length text, and a repeat) and 4 expected structured `422` errors (an
  infeasible constraint and three invalid schemas). Each model was run against
  both corpora, so the English corpus is also applied to the multilingual model
  and the multilingual corpus to the English models. Cross-language application
  is a parity probe, not a claim about language quality.
- Execution modes against one server per model: `singleton` (14 requests),
  `repeated` (each case twice, 28), `batch` (3 compatible requests in one
  `/v1/batch`), `concurrent-group` (3 concurrent compatible requests), and
  `concurrent-all` (all 14 concurrently). That is 62 requests per model and
  corpus.
- Authoritative comparison: exact equality of the semantic result (labels,
  ordering, cardinality, empty results, decoder metadata, structured error
  code, message, retryable flag, and HTTP status) and exact `usage` token
  accounting. The runner also checks
  that repeated runs are byte-identical and records how many backend batches the
  dynamic scheduler formed in each mode (see each `.md`).
- Numeric comparison: raw confidence and probability values are compared
  without rounding. The tolerance (absolute `1e-4`) is gating. A nonzero delta
  strictly below it is *additionally* counted as "synthetic zero"; that count is
  a diagnostic view only. Raw maximum drift is the authoritative number and every
  per-request raw maximum is retained in the JSON reports.

## Result matrix

Every one of the 6 model/corpus combinations passed: 62/62 requests exactly
matched, 0 semantic mismatches, 0 numeric values above tolerance, repeat runs
identical (372/372 requests in total).

| Model (revision) | Corpus | Requests | Exact semantic matches | Numeric values compared | Exact zero | Synthetic zero (<1e-4, nonzero raw delta) | Over 1e-4 | Raw max abs drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decide-340m` (`9d1bfb84`) | english-v1 | 62 | 62 | 284 | 12 | 272 | 0 | 7.153e-6 |
| `decide-340m` (`9d1bfb84`) | multilingual-v1 | 62 | 62 | 284 | 4 | 280 | 0 | 4.292e-6 |
| `decide-multilingual` (`e173bca1`) | english-v1 | 62 | 62 | 284 | 4 | 280 | 0 | 1.550e-5 |
| `decide-multilingual` (`e173bca1`) | multilingual-v1 | 62 | 62 | 284 | 4 | 280 | 0 | 2.189e-5 |
| `decide-1b` (`67af949e`) | english-v1 | 62 | 62 | 284 | 12 | 272 | 0 | 2.205e-6 |
| `decide-1b` (`67af949e`) | multilingual-v1 | 62 | 62 | 284 | 4 | 280 | 0 | 3.979e-6 |

"Exact zero" plus "synthetic zero" equals the numeric values compared. The
largest raw drift across the matrix is `2.189e-5`, under the `1e-4` gate.

Per-mode exact matches (identical for every combination):

| Mode | Requests per combination | Exact semantic matches |
| --- | ---: | ---: |
| singleton | 14 | 14 |
| repeated | 28 | 28 |
| batch | 3 | 3 |
| concurrent-group | 3 | 3 |
| concurrent-all | 14 | 14 |

Per-mode raw maximum absolute drift:

| Model | Corpus | singleton | repeated | batch | concurrent-group | concurrent-all |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `decide-340m` | english-v1 | 7.153e-6 | 7.153e-6 | 4.768e-7 | 8.345e-7 | 7.153e-6 |
| `decide-340m` | multilingual-v1 | 4.292e-6 | 4.292e-6 | 1.192e-6 | 1.192e-6 | 4.292e-6 |
| `decide-multilingual` | english-v1 | 1.550e-5 | 1.550e-5 | 3.517e-6 | 3.517e-6 | 1.550e-5 |
| `decide-multilingual` | multilingual-v1 | 2.189e-5 | 2.189e-5 | 3.219e-6 | 3.219e-6 | 2.189e-5 |
| `decide-1b` | english-v1 | 2.205e-6 | 2.205e-6 | 1.907e-6 | 1.431e-6 | 2.205e-6 |
| `decide-1b` | multilingual-v1 | 3.979e-6 | 3.979e-6 | 6.557e-7 | 6.557e-7 | 3.979e-6 |

## Runtime, reference, and host

| Item | Value |
| --- | --- |
| Runner under test | `@gliner-runner/onnx-runner` 0.2.1 on Node.js v24.21.0 (`darwin-arm64`) |
| Inference runtime | `onnxruntime-node` 1.30.0, `CPUExecutionProvider`, FP32 |
| Tokenizer | `@huggingface/tokenizers` 0.2.0 with the project's TypeScript GLiNER2 preprocessing |
| Reference | PyTorch CPU/FP32 through the runner's `PyTorchBackend`; Python 3.12.12, `gliner2` 2.1.0, `torch` 2.14.0, `transformers` 5.19.0 |
| Artifact exporter (build time only) | `gliner_runner` 0.2.1, `onnx` 1.23.2, `onnxscript` 0.7.2, Python `onnxruntime` 1.31.0, opset 20 |
| Host | Apple M5 Max (18 cores), 128 GiB unified memory, macOS 26.7.1, arm64. The reports record only `darwin-arm64`; the rest is operator-recorded. |
| Reports generated | 2026-10-14 |

The graphs are exported with Python ONNX Runtime 1.31.0 and served with
`onnxruntime-node` 1.30.0. The numbers above are for that pairing.

## Artifact digests

Each `<model_id>-manifest.json` is the artifact manifest produced by
`gliner-runner prepare-onnx` (identity, provenance, graph contract, exporter
versions, build-time logit parity, and file hashes). Weights are not included in
this directory. The runner refuses to serve an artifact whose files do not match
these SHA-256 values.

These are the canonical artifacts: every weight is external in `model.onnx.data`
at 64-byte-aligned offsets. The Python-free Node runner reproduces them byte for
byte from the bundled template in `packages/onnx-runner/templates/` and the
pinned `model.safetensors`, so `models install` and `models prepare` yield
exactly the files measured here. The parity run above was repeated on these
canonical files with identical results.

`decide-340m` (`fastino/GLiNER2.5-Decide@9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90`),
source snapshot sha256 `5081e1a9b994143fe7f98810c909c8f6498914016b9fbe58c46a5a1135e34946`:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.onnx` | 4990967 | `782fa7253b80f7035b66f7a068208cbe4f01a0ebc5101cbe789ecb1d76a26e89` |
| `model.onnx.data` | 1744089152 | `2a7533742fd3301c7e30775fc75eddf7a9f42053ae66d954fea81b8a4f388262` |
| `tokenizer.json` | 8333952 | `3ad87d9ffe669147063e70850927dd2da90249e2acc5c8527f1eb65df467bcc8` |
| `tokenizer_config.json` | 3356 | `323199a4e946039410899f3779f2aa3eaef1500213c512727ad0f623d4f21309` |
| `special_tokens_map.json` | 2414 | `84ea70143f533d7e99b393d87f20010887a9ac2cba955828ef313886e4e83f4f` |
| `gliner2.json` | 2551 | `f8428dc05fd2f89e772f9aa5b522f2e5ad9bdb9dbdd28c30928055225c3241f0` |

`decide-multilingual` (`fastino/GLiNER2.5-multi-Decide@e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`),
source snapshot sha256 `0fc359fda9ab12da8c1b3a94320746de9a25be9e6ef1eb92cbc43051e5a508cb`:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.onnx` | 2562370 | `b07af9ca98fa064adde2582ca9ddaae467bc582e565e36e5f6d5e3f598fc920f` |
| `model.onnx.data` | 1114878016 | `0b1be2d5c57980320ff3fef2e4cab74e106b098628798f170b6e9db1479b6888` |
| `tokenizer.json` | 16035853 | `c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a` |
| `tokenizer_config.json` | 646 | `fd4a31dc2f1f17e31638c5f0e783b81cdb2fbe6bddd116a8d9e5d50d78148cf1` |
| `gliner2.json` | 2548 | `f834a77d5e1e2e9c0cdac0d0a1a67d1153c9b030097ff5fe40ea30d9dd299c9a` |

`decide-1b` (`fastino/GLiNER2.5-Decide-1B@67af949e17bebd20f05fe020a971079889e7a462`),
source snapshot sha256 `6c183eb70b52b6683a1489a56df3eedb7fea7e12a2c5535c67fe00a1eb515d98`:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.onnx` | 3206205 | `ee6de4e15bb6d0e0dabf17264ba784fa1efad000718752a9694b2209bc04661e` |
| `model.onnx.data` | 4137993280 | `89ef00057048e6c45433608886964be3ffdb525536e73aefab923d5ee7cd8716` |
| `tokenizer.json` | 3585055 | `ddb379b6a4679ee16646bf0b726de9ec538d7c80ea9d4d4e33b069f19c9e1efb` |
| `tokenizer_config.json` | 559 | `8bb8d094c7cde84942866ef98fb5b24f7387245573a7f707c8349729484c27b3` |
| `gliner2.json` | 2528 | `30fd562d39a53f224a8447f4f8982af13b98c49f8adae589aec5f5a9d05214f2` |

Corpus digests (also each report's `corpus_sha256` and the digest of the
checked-in corpus file): `english-v1`
`10aa107ccc49c0b0eab283a6d2ad74b98057e08b1c3b61dbec790d10d45c9de6`;
`multilingual-v1`
`b97b2041d4dcc4caf194167bc0aad5d5d3604cd7119a61cef1267f8bcbeaceff`.

Build-time raw-logit parity recorded in the manifests (3 batches, 19 rows, 69
valid logits per model; absolute gate `1e-4`):

| Model | Fixture suite | Selected-label matches | Max abs logit diff (PyTorch wrapper vs ONNX) | Batch-invariance max diff |
| --- | --- | ---: | ---: | ---: |
| `decide-340m` | classification-logits-multitask-english-v1 | 19/19 | 5.007e-6 | 2.384e-6 |
| `decide-multilingual` | classification-logits-multitask-multilingual-v1 | 19/19 | 9.298e-6 | 7.629e-6 |
| `decide-1b` | classification-logits-multitask-english-v1 | 19/19 | 1.931e-5 | 1.878e-6 |

## Files

- `node-vs-pytorch-<model_id>-<corpus>.json` and `.md`: the 12 end-to-end
  reports (per-request records, per-mode statistics, batching evidence). The
  `golden_file` field was reduced to the golden's file name; no other field was
  changed.
- `<model_id>-<corpus>.json`: the 6 PyTorch golden outputs the reports compare
  against (`gliner-runner onnx-reference`).
- `<model_id>-manifest.json`: the 3 artifact manifests (hashes and metadata
  only).

This directory contains no model weights, graphs, `.onnx` or `.data` files,
absolute local paths, usernames, hostnames, or credentials. Corpus text is
synthetic.

## Reproduce

See [ONNX runtime](../../docs/onnx-runtime.md#reproducing-parity-evidence).
Reports are deterministic apart from `generated_at`. Numeric drift can differ
across ONNX Runtime versions, CPUs, or thread counts; such a result is not
interchangeable with this evidence without its own review.
