# Backend research conclusions and gates

This document records project decisions, promotion gates, and links to bounded
measurements. Linked results apply only to their exact model, workload, language,
device, precision, runtime, and host scope.

## Fastino PyTorch reference

The supported target is Fastino GLiNER2 2.1's public PyTorch constrained
classification API. Python calls `Classifier.batch_classify` and
`ClassificationResult.to_dict`; it does not reimplement preprocessing or
decoding. The same adapter supports all three published Decide repositories:
`fastino/GLiNER2.5-Decide`, `fastino/GLiNER2.5-multi-Decide`, and
`fastino/GLiNER2.5-Decide-1B`. Their model snapshot revisions observed for this
release are respectively `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90`,
`e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`, and
`67af949e17bebd20f05fe020a971079889e7a462`.

Per-request input usage is captured from the attention mask returned by
Fastino's actual `collate_fn_inference` call while `batch_classify` processes
the compiled schema. The adapter fails if preprocessing does not expose one
count per request; it never estimates from text or substitutes zero.
Classification output usage is zero because no tokens are generated.
Optional runner `label_descriptions` are converted to Fastino's native
ordered label-to-description mapping before `ClassificationSchema.from_dict`.
This ensures descriptions participate in the compiled model prompt instead of
being discarded or treated as output-only metadata.

The package pins `gliner2[local]==2.1.0` and requires Transformers 5 plus
protobuf and sentencepiece so every curated tokenizer loads consistently.
Manifests must enumerate the complete
local snapshot because upstream revision forwarding could not be verified for
every weight load path. The runtime advertises CPU FP32 and locally available
MPS FP16/FP32 profiles; this capability discovery is separate from the
model-specific evidence gate below. CUDA FP32 is advertised when CUDA is
available; CUDA FP16 requires compute capability 5.3 or newer, and CUDA BF16
requires PyTorch's `is_bf16_supported()` check to pass. CUDA FP32, FP16, and
BF16 were not evaluated on the Apple host used for these reports.

All three current pins have reviewed native CPU/FP32, MPS/FP16, and MPS/FP32
characterizations on one Apple M5 Max with PyTorch 2.14.0 and GLiNER2 2.1.0:

- English Decide:
  [CPU FP32](../benchmark-results/2026-10-13-gliner21-current-cpu-fp32-v4/REPORT.md),
  [MPS FP16](../benchmark-results/2026-10-13-gliner21-current-mps-fp16-v4/REPORT.md),
  and [MPS FP32](../benchmark-results/2026-10-13-gliner21-current-mps-fp32-v4/REPORT.md)
  on the preserved four-case English workload;
- multilingual Decide:
  [CPU FP32](../benchmark-results/2026-10-13-multilingual-cpu-fp32-v5/REPORT.md),
  [MPS FP16](../benchmark-results/2026-10-13-multilingual-mps-fp16-v5/REPORT.md),
  and [MPS FP32](../benchmark-results/2026-10-13-multilingual-mps-fp32-v5/REPORT.md)
  on `multilingual-v1` across de, es, fr, and ja; and
- Decide 1B:
  [CPU FP32](../benchmark-results/2026-10-13-decide-1b-cpu-fp32-v5/REPORT.md),
  [MPS FP16](../benchmark-results/2026-10-13-decide-1b-mps-fp16-v5/REPORT.md),
  and [MPS FP32](../benchmark-results/2026-10-13-decide-1b-mps-fp32-v5/REPORT.md)
  on `english-v1`.

Fastino's constructor configures its scorer for the requested device but does
not transfer the underlying model, so the adapter also calls the public
`Classifier.to(device=..., dtype=...)` method. Without that explicit transfer,
upstream inference fails with `RuntimeError: Passed CPU tensor to MPS op`; the
runner never catches that failure to retry on CPU. Historical GLiNER2 2.0
reports remain immutable evidence for the former revision.

An implementation is releasable only after fixtures cover each advertised
operation, malformed inputs, empty results, batching, determinism expectations,
and device/precision rejection. Upstream behavior must be established from the
pinned code and artifacts, not inferred from a mutable README.

## Native MLX

MLX is a future Apple-silicon-native path. Its first bounded scope is English
span extraction. BF16 and INT8 are the baseline formats. INT4 is disabled by
default and requires an explicit request/configuration because quality loss may
be workload-dependent.

Promotion requires:

- project-owned conversion code tied to a pinned source artifact;
- tokenizer and preprocessing equivalence;
- tensor-shape and decoder tests;
- exact-match and score-difference parity reports on a redistributable or
  locally supplied evaluation set;
- separately documented tolerances for BF16, INT8, and INT4;
- rejection of universal Decide operations not yet implemented.

No request may move from MLX to PyTorch without an explicit caller policy.

## ONNX

ONNX is a production backend for constrained classification, served by the
Python-free Node runner (`gliner-runner-onnx serve`, package
`@gliner-runner/onnx-runner`). See [ONNX runtime](onnx-runtime.md) for artifact
preparation, packaging, and operation. The scope is exactly:

- ONNX Runtime `CPUExecutionProvider` (`onnxruntime-node` 1.30.0), FP32, no
  quantization;
- the `classify` operation only; universal Decide and span extraction are
  rejected;
- the three curated, immutable revisions above (`decide-340m`,
  `decide-multilingual`, `decide-1b`), each served only from a
  project-prepared artifact (built from a bundled template) whose manifest and
SHA-256 digests verify;
- parity evidence from one Apple M5 Max (macOS, `darwin-arm64`).

MPS, CUDA, FP16, BF16, quantized profiles, other hosts, and any other model or
revision are unsupported and rejected with `409`; there is no fallback to
PyTorch. The Python server does not register an ONNX backend, and its
`/v1/capabilities` does not advertise one. ONNX is available only from the Node
runner's own `/v1/capabilities`, which advertises `onnx`/`cpu`/`fp32`/`classify`.
The `onnx` value in the wire enumeration is shared by both servers.

Third-party or untraceable exports are insufficient. Artifacts are produced by
the project-owned `gliner-runner prepare-onnx` pipeline from a verified pinned
snapshot, at build time. The Node runner reproduces those artifacts byte for
byte at install time, without Python, from a bundled per-revision template and
the pinned Hugging Face safetensors (identity and transposed tensors), verifying
every source and output file by size and SHA-256; it never converts a model, and
nothing is prepared per request. A manifest records
the source snapshot digest, exporter and dependency versions, graph inputs,
outputs and dynamic axes, opset imports, provider, precision, quantization, and
the SHA-256 and size of every file. Tokenization, schema and prompt
compilation, constrained decoding, response formatting, and usage accounting run
in the runner's TypeScript implementation, which has no Python dependency and is
proven against the PyTorch reference rather than assumed equivalent.

Production parity is end to end, not raw-logit alone. The
[production parity evidence](../benchmark-results/2026-10-14-onnx-runtime-parity-v1/REPORT.md)
replays the `english-v1` and `multilingual-v1` request corpora through the real
HTTP server for each curated model in singleton, repeated, batch,
concurrent-group, and concurrent-all modes, against PyTorch CPU/FP32 goldens.
All 372/372 requests match exactly on labels, ordering, cardinality, empty
results, decoder metadata, structured errors, and usage; the largest raw
numeric difference in the matrix is `2.189e-5`, under the `1e-4` gate, and all
repeated runs are identical. These are measurements on this host and
revision/corpus set, not a claim of bitwise equality or of equivalence for any
other provider, precision, language, or operation.

The synthetic-zero comparison view (a nonzero absolute PyTorch/ONNX difference
strictly below `1e-4` counted as zero) is optional and diagnostic. It never
rounds or clamps API output, and raw logits, raw differences, label selection,
and the acceptance decision stay authoritative. The production evidence reports
it beside the raw maximum drift for every cell.

### Research exporter

The raw-logit research exporter (`benchmarks/export_classification_onnx.py`) is
separate from the production pipeline. It has reviewed feasibility evidence for
the [Small control](../benchmark-results/2026-10-13-onnx-small-research-v2/REPORT.md),
[English Decide](../benchmark-results/2026-10-13-onnx-decide-research-v2/REPORT.md),
[multilingual Decide](../benchmark-results/2026-10-13-onnx-multilingual-research-v2/REPORT.md),
and [Decide 1B](../benchmark-results/2026-10-13-onnx-decide-1b-research-v2/REPORT.md).
Those results are research evidence for raw logits only: the multilingual and 1B
graphs retained selected-label parity on all three reviewed fixture rows with
maximum PyTorch-to-ONNX Runtime logit differences of `2.3841858e-06` and
`5.9604645e-06`, and the 1B graph uses a verified 4.14 GB external-data
sidecar. They do not authorize serving; the production gate is the end-to-end
evidence above. The exporter's optional synthetic-zero view applies only to its
raw-logit comparisons.

## Core ML and Swift

Core ML and a native Swift integration are later work. No public protocol
should imply their presence. Their design begins only after the reference,
capability negotiation, artifact provenance, and parity harness are stable.

## Parity policy

Parity is operation-specific, model-specific, language-specific, and
precision-specific. Aggregate accuracy alone cannot authorize a capability.
Each report records reference/runtime versions, model digest, evaluation data
provenance, preprocessing settings, thresholds, and discrepancies. A failed or
missing gate means “unsupported,” not fallback. The reviewed Apple M5 Max MPS
FP16/FP32 maximum differences versus the direct CPU FP32 oracle were,
respectively, `0.00100696` / `0.0000038147` for English Decide,
`0.0053369999` / `0.0000022054` for multilingual Decide, and `0.0015290529` /
`0.0000007749` for Decide 1B; every cell retained 4/4 selected-label and
expected-label results. These are four-example, model- and workload-specific
measurements, not claims of universal numerical equivalence. Model-card
potential-output matches and the research-only raw-logit ONNX fixtures are
separate gates and cannot authorize a native profile.

The production ONNX profile (CPU/FP32, classify) has its own gate: exact
semantic, error, ordering, and usage equality with PyTorch CPU/FP32 across the
full request corpus in every execution mode, raw numeric drift within an
explicitly approved absolute bound of `1e-4`, and verified artifacts. Its
reviewed evidence covers three curated revisions, two corpora, and one Apple M5
Max host. It does not authorize MPS, CUDA, FP16, BF16, quantization, a new
revision, another operation, or another host, each of which needs its own
evidence.
