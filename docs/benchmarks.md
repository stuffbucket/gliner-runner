# Real-model characterization

`benchmarks/benchmark_real.py` characterizes the official PyTorch backend
through both the direct Fastino API and the runner's loopback HTTP API. It is
an opt-in local command and is not part of offline CI.

## Running

Install the runtime, benchmark, and development dependencies, then identify the
repository, immutable revision, expected GLiNER2 runtime, and matching local
Hugging Face cache snapshot explicitly:

```sh
python -m pip install -e '.[pytorch,benchmark,dev]'
export GLINER_RUNNER_BENCHMARK_REPOSITORY=fastino/GLiNER2.5-Decide
export GLINER_RUNNER_BENCHMARK_REVISION=9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90
export GLINER_RUNNER_BENCHMARK_GLINER2_VERSION=2.1.0
export GLINER_RUNNER_BENCHMARK_MODEL="$HOME/.cache/huggingface/hub/models--fastino--GLiNER2.5-Decide/snapshots/$GLINER_RUNNER_BENCHMARK_REVISION"
mise run benchmark:real
```

The revision must be a lowercase 40-character commit SHA, not a branch or tag.
The resolved snapshot path must either have the canonical
`models--OWNER--NAME/snapshots/REVISION` cache identity or contain a
`.gliner-runner.json` inventory whose repository, revision, complete active-file
set, file sizes, and SHA-256 digests all verify. Downloader administration under
`.cache/huggingface/` is ignored, is not part of snapshot identity, and cannot
satisfy an inventory entry. Marker-backed content digests cover only inventoried
active files, excluding the marker and downloader metadata. The harness also
requires the installed GLiNER2 version to equal the expected version before
loading the model. These checks prevent a local directory from being described
as a different upstream artifact.

The snapshot must remain outside the repository. The command writes local
results to `benchmark-results/local/` by default. Set
`GLINER_RUNNER_BENCHMARK_OUTPUT` to choose another directory. The destination
must not exist; the harness refuses to overwrite or reuse prior result paths.

The default profile is CPU/FP32 and the default workload is `english-v1`. For
the multilingual pinned model, select its multilingual characterization suite
explicitly:

```sh
export GLINER_RUNNER_BENCHMARK_WORKLOAD=multilingual-v1
```

The suite identity and language scope are recorded in the evidence, and results
from different suites cannot be used as performance baselines for one another.
To characterize the validated Apple Silicon profile and compare it with a prior
CPU result:

```sh
export GLINER_RUNNER_BENCHMARK_DEVICE=mps
export GLINER_RUNNER_BENCHMARK_PRECISION=fp16
export GLINER_RUNNER_BENCHMARK_BASELINE=benchmark-results/local/characterization.json
export GLINER_RUNNER_BENCHMARK_OUTPUT=benchmark-results/local/mps-fp16
mise run benchmark:real
```

Unsupported pairs are rejected before model loading and never rewritten.

To run every executable example in the same snapshot's model card on the
portable CPU/FP32 default:

```sh
export GLINER_RUNNER_BENCHMARK_OUTPUT=benchmark-results/local/card
mise run benchmark:card
```

Apple Silicon profiles are explicit opt-ins rather than a portable default:

```sh
export GLINER_RUNNER_BENCHMARK_CARD_PROFILES='cpu/fp32 mps/fp16 mps/fp32'
export GLINER_RUNNER_BENCHMARK_OUTPUT=benchmark-results/local/card-all-profiles
mise run benchmark:card
```

The card command uses the same four required model identity variables and the
same new-output-path protection. It fails closed if any discovered
`model.classify_text` call cannot be parsed, and records the discovered and
executed counts. Both characterization commands force Hugging Face and
Transformers offline modes before loading the local snapshot.

The benchmark records:

- fresh-process startup, model-load, first-inference, and RSS measurements;
- MPS driver-allocated memory when the selected device exposes it;
- warm HTTP wall-clock latency and throughput for batch sizes 1, 2, 4, and 8
  at encoded token targets 16, 64, and 256;
- per-request runner `inputTokens` from Fastino's compiled-schema encoder
  attention mask, separately from the raw-text tokenizer target, and
  classification `outputTokens=0`;
- dynamic batch-window, schema grouping, and logical-model grouping behavior;
- direct official-oracle output parity and expected labels for the selected
  suite's transparent, language-tagged examples;
- a deterministic runner compatibility corpus covering described and plain
  labels, single- and multi-label output, ordering, thresholds and empty output,
  feasible and infeasible constraints, repeated inference, mixed-length
  singleton/batch execution, decoder metadata, and exact token accounting;
- a shared described-label schema on both the direct oracle and runner paths,
  with the complete compatibility corpus recorded in the workload identity so
  incompatible runs cannot be compared;
- client cancellation and bounded-queue backpressure under concurrent load;
- exact hardware, OS, runtime versions, model revision, content digest,
  precision, sample counts, and timing methodology.

Results are emitted as full JSON, a latency CSV, and a generated Markdown
summary. Output excludes model paths, input text, hostnames, usernames, and
credentials.

New real-model runs use benchmark format version 5, and new model-card runs use
format version 4. They record the explicit repository, revision, expected
GLiNER2 version, snapshot identity validation method, snapshot content digest,
and a deterministic behavior-workload identity. Real-model format version 5
adds the pinned workload-suite ID, complete workload digest, language scope, and
language-tagged transparent cases. Format version 4 added the structured
compatibility corpus and per-mode usage/decoder evidence; format version 2 added
`parity_usage`, `runner_input_tokens`, `output_tokens`, and
`schema_label_descriptions`.
Previously published artifacts remain immutable historical measurements and are
not rewritten to claim identity metadata that was not captured at run time.

## Research-only ONNX feasibility export

The ONNX tool exports only the encoder and classification head. Tokenization,
schema compilation, constrained decoding, response formatting, and usage
accounting remain in Python. This is an opt-in feasibility harness, not the
production path: serving uses the artifacts produced by `gliner-runner
prepare-onnx` and the Python-free Node runner described in
[ONNX runtime](onnx-runtime.md), whose end-to-end evidence is in the
[production ONNX parity report](../benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md).

Install its exact qualified research environment and select one reviewed,
immutable local snapshot:

```sh
python -m pip install -e '.[onnx-export,dev]'

export GLINER_RUNNER_ONNX_SMALL_MODEL=/absolute/path/to/gliner2.5-small-v1/df5910e44bc4ffdb0d95399a83b0ca4516349aa5
mise run benchmark:onnx-export

export GLINER_RUNNER_ONNX_DECIDE_MODEL=/absolute/path/to/GLiNER2.5-Decide/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90
mise run benchmark:onnx-export:decide

export GLINER_RUNNER_ONNX_MULTILINGUAL_MODEL=/absolute/path/to/GLiNER2.5-multi-Decide/e173bca1f0c4217d7a55b3b0c705d5ece8a0218d
mise run benchmark:onnx-export:multilingual

export GLINER_RUNNER_ONNX_1B_MODEL=/absolute/path/to/GLiNER2.5-Decide-1B/67af949e17bebd20f05fe020a971079889e7a462
mise run benchmark:onnx-export:1b
```

To add a diagnostic comparison view that represents accepted sub-threshold
PyTorch/ONNX Runtime differences as synthetic zero, opt in explicitly:

```sh
export GLINER_RUNNER_ONNX_SYNTHETIC_ZERO_THRESHOLD=0.0001
mise run benchmark:onnx-export:decide
```

For each valid, unpadded logit pair this records a normalized delta of zero only
when the raw absolute difference is strictly less than the configured threshold;
a difference equal to the threshold remains nonzero. The threshold must be finite
and in `(0, 0.0001]`. Raw logits, raw maximum drift, argmax selection, and the
`GLINER_RUNNER_ONNX_ATOL` acceptance gate remain unchanged. The option is absent
by default, does not round or clamp API output, and applies only to research
PyTorch-wrapper-versus-ONNX comparisons—not the independent public GLiNER2
scorer comparison or the production serving path (whose evidence reports raw and
synthetic-zero counts together). New exporter evidence uses format
version 3 to record this policy; earlier evidence remains immutable.

The reviewed exporter target list is closed over those four exact identities;
Small is a research control and is not a curated runtime model. The respective
default outputs are ignored, new-only directories named `onnx-small`,
`onnx-decide`, `onnx-decide-multilingual`, and `onnx-decide-1b` beneath
`benchmark-results/local/`. Override a task's destination with
`GLINER_RUNNER_ONNX_OUTPUT`; repository-local output is accepted only beneath
`benchmark-results/local/` or `dist/`, output inside the source snapshot is
rejected, and any existing destination is preserved. The exporter forces Hub
and Transformers offline mode before optional model imports. The 1B target
requires substantially more memory and disk and is expected to reference ONNX
external-data sidecars; it runs only through its dedicated task.

Evidence records and verifies:

- the exact GLiNER2 2.1.0, Torch 2.14.0, Transformers 5.19.0, ONNX 1.23.2,
  ONNX Runtime 1.31.0, and ONNX Script 0.7.2 environment;
- dynamic shared batch, sequence, and label axes for `input_ids`,
  `attention_mask`, `label_positions`, and `label_mask` to FP32 `logits`;
- standard ONNX opset 20 and every emitted opset import;
- finite valid and padded logits, masking behavior, and a finite reviewed
  absolute tolerance no larger than `0.001` (default `0.0001`);
- when explicitly enabled, raw and synthetic-zero delta statistics side by side,
  with normalization restricted to valid labels and excluded from all gates;
- batch-size-1 and batch-size-2 PyTorch/ONNX Runtime logit and selected-label
  parity; and
- independent agreement between the wrapper logits and GLiNER2's public
  `Classifier.batch_score` path on the same compiled fixtures.

Torch can emit a `.onnx.data` sidecar for large graphs even when inline external
data was requested. The evidence follows the graph references and records exact
sizes and SHA-256 hashes for the graph, every required sidecar, and the exporter
report. Graphs and weights remain local and ignored. Reviewed reports and
reproducibility manifests are published without those weights for the
[Small control](../benchmark-results/2026-10-13-onnx-small-research-v2/REPORT.md),
[English Decide](../benchmark-results/2026-10-13-onnx-decide-research-v2/REPORT.md),
[multilingual Decide](../benchmark-results/2026-10-13-onnx-multilingual-research-v2/REPORT.md),
and [Decide 1B](../benchmark-results/2026-10-13-onnx-decide-1b-research-v2/REPORT.md).

Set only the target variable for the export being checked and run
`python -m pytest -m 'integration and benchmark' tests/test_onnx_export_tools.py`.
The guarded tests use `GLINER_RUNNER_ONNX_SMALL_MODEL`,
`GLINER_RUNNER_ONNX_DECIDE_MODEL`,
`GLINER_RUNNER_ONNX_MULTILINGUAL_MODEL`, and
`GLINER_RUNNER_ONNX_1B_MODEL` independently; configuring a cheaper target never
implicitly launches the 1B export.

## Cross-runtime or revision behavior comparison

Performance comparison remains part of `benchmark_real.py`: supplying
`GLINER_RUNNER_BENCHMARK_BASELINE` compares latency and memory only after
verifying the same model content digest, CPU, and described-label schema. That
same-content baseline logic intentionally does not compare different model
revisions.

`benchmarks/compare_behavior.py` is a separate, inference-free comparator for
new characterization artifacts. It always requires the same repository,
workload identity, and case identities. Runtime, host, and device may differ;
revision or snapshot-content changes require the explicit
`GLINER_RUNNER_BENCHMARK_ALLOW_REVISION_CHANGE=1` opt-in. Comparison format 2
reports exact and score-insensitive output matches, token-usage parity, decoder
metadata parity, singleton/repeat/batch consistency, numeric path compatibility,
and maximum absolute numeric difference. It records measured denominators and
uses `null` rather than treating absent usage, decoder, mode, or numeric evidence
as a successful match. Run it with:

```sh
export GLINER_RUNNER_BENCHMARK_BEHAVIOR_BASELINE=benchmark-results/run-a/characterization.json
export GLINER_RUNNER_BENCHMARK_BEHAVIOR_CANDIDATE=benchmark-results/run-b/characterization.json
export GLINER_RUNNER_BENCHMARK_ALLOW_REVISION_CHANGE=1  # only for a deliberate revision change
mise run benchmark:compare-behavior
```

For model-card artifacts, set
`GLINER_RUNNER_BENCHMARK_BASELINE_PROFILE` and
`GLINER_RUNNER_BENCHMARK_CANDIDATE_PROFILE` to explicit `device/precision`
values whenever either artifact contains multiple profiles. Set
`GLINER_RUNNER_BENCHMARK_COMPARISON_OUTPUT` to write deterministic JSON instead
of printing it. Historical artifacts without a recorded workload identity are
not accepted by this comparator; rerun rather than inferring identity after the
fact.

## Interpreting results

These measurements characterize one machine and one small deterministic
workload. They are not general hardware or deployment recommendations.
Percentiles from the default three samples per matrix cell are useful for
reproducibility checks, not statistical performance claims. Exact JSON parity
can differ when floating-point confidence values vary by tiny amounts; selected
label parity and the maximum numeric difference are reported separately.

Arrival gaps wider than the configured fill window do not guarantee singleton
batches: requests that arrive while inference is active can accumulate in the
queue and be grouped when the worker becomes available.

## Reviewed evidence matrix

The following current-revision results were produced on one Apple M5 Max. A
validated native cell means 4/4 selected-label parity against the direct
CPU/FP32 oracle, 4/4 expected labels, matching compatibility behavior, usage,
and decoder structure, finite drift, and no fallback. Model-card counts are
execution and upstream *potential-output* matches, not accuracy measurements.
Research ONNX cells cover three raw-logit fixtures through ONNX Runtime's
CPUExecutionProvider; they are not end-to-end serving evidence. Production ONNX
cells replay the 14-case `english-v1` and `multilingual-v1` request corpora
(62 requests each across singleton, repeated, batch, concurrent-group, and
concurrent-all modes) through the Node runner's real HTTP server on ONNX Runtime
CPU/FP32 and compare them exactly with PyTorch CPU/FP32 goldens; "exact" means
identical labels, ordering, cardinality, empty results, decoder metadata,
structured errors, and usage, with raw numeric drift below the `1e-4` gate.

| Curated model and exact revision | Workload / languages | Native CPU/FP32 | Native MPS/FP16 | Native MPS/FP32 | Model card CPU/FP32 | Research ONNX logits | Production ONNX runtime (Node, CPU/FP32) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| English Decide `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90` | Preserved four-case English set; the format-v4 predecessor of `english-v1` | [validated](../benchmark-results/2026-10-13-gliner21-current-cpu-fp32-v4/REPORT.md), max drift `3.5762787e-7` | [validated](../benchmark-results/2026-10-13-gliner21-current-mps-fp16-v4/REPORT.md), max drift `0.001006961` | [validated](../benchmark-results/2026-10-13-gliner21-current-mps-fp32-v4/REPORT.md), max drift `3.8146973e-6` | [21/21 executed; 12/21 potential-output matches](../benchmark-results/2026-10-13-gliner21-current-model-card-v4/REPORT.md) | [3/3 selected labels](../benchmark-results/2026-10-13-onnx-decide-research-v2/REPORT.md), English fixtures, max drift `5.3048134e-6` | [124/124 exact](../benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md) on `english-v1` (max drift `7.153e-6`) and `multilingual-v1` (max drift `4.292e-6`) |
| Multilingual Decide `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d` | `multilingual-v1`: de, es, fr, ja | [validated](../benchmark-results/2026-10-13-multilingual-cpu-fp32-v5/REPORT.md), max drift `0` | [validated](../benchmark-results/2026-10-13-multilingual-mps-fp16-v5/REPORT.md), max drift `0.0053369999` | [validated](../benchmark-results/2026-10-13-multilingual-mps-fp32-v5/REPORT.md), max drift `2.2053719e-6` | [21/21 executed; 9/21 potential-output matches](../benchmark-results/2026-10-13-multilingual-model-card-v4/REPORT.md) | [3/3 selected labels](../benchmark-results/2026-10-13-onnx-multilingual-research-v2/REPORT.md), es/fr/de fixtures, max drift `2.3841858e-6` | [124/124 exact](../benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md) on `english-v1` (max drift `1.550e-5`) and `multilingual-v1` (max drift `2.189e-5`) |
| Decide 1B `67af949e17bebd20f05fe020a971079889e7a462` | `english-v1`: en | [validated](../benchmark-results/2026-10-13-decide-1b-cpu-fp32-v5/REPORT.md), max drift `4.7683716e-7` | [validated](../benchmark-results/2026-10-13-decide-1b-mps-fp16-v5/REPORT.md), max drift `0.0015290529` | [validated](../benchmark-results/2026-10-13-decide-1b-mps-fp32-v5/REPORT.md), max drift `7.7486038e-7` | [21/21 executed; 11/21 potential-output matches](../benchmark-results/2026-10-13-decide-1b-model-card-v4/REPORT.md) | [3/3 selected labels](../benchmark-results/2026-10-13-onnx-decide-1b-research-v2/REPORT.md), English fixtures, max drift `5.9604645e-6` | [124/124 exact](../benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md) on `english-v1` (max drift `2.205e-6`) and `multilingual-v1` (max drift `3.979e-6`) |

CUDA FP32, FP16, and BF16 were not evaluated on this host. Runtime hardware
predicates do not substitute for model-, operation-, language-, workload-, and
precision-specific parity evidence. Current model-card evidence is CPU-only;
the native MPS cells do not imply model-card MPS parity.

The production ONNX column is scoped to CPU/FP32 `classify` on one Apple M5 Max
and does not extend to MPS, CUDA, FP16, BF16, quantization, other revisions, or
other hosts. Its largest raw drift across all six model/corpus cells is
`2.189e-5`. The multilingual model's English cell and the English models'
multilingual cells are cross-language parity probes. The [production
report](../benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md) lists artifact digests, runtime versions, and host.

The non-catalog [Small ONNX result](../benchmark-results/2026-10-13-onnx-small-research-v2/REPORT.md)
is a research control only. Former GLiNER2 2.0 English measurements remain as
[immutable historical evidence](../benchmark-results/2026-10-02-apple-m5-max-cpu-fp32/),
but do not authorize profiles for the current revisions.
