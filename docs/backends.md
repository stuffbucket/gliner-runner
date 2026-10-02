# Backend research conclusions and gates

This document records project decisions, not measured performance claims.

## Fastino PyTorch reference

The supported target is Fastino GLiNER2 2.0's public PyTorch constrained
classification API. Python calls `Classifier.batch_classify` and
`ClassificationResult.to_dict`; it does not reimplement preprocessing or
decoding. The same adapter supports all three published Decide repositories:
`fastino/GLiNER2.5-Decide`, `fastino/GLiNER2.5-multi-Decide`, and
`fastino/GLiNER2.5-Decide-1B`. Their model snapshot revisions observed for this
release are respectively `5a7adf72a23b4d311abae6ce050d7f0012bb3416`,
`a35a0cd3b7a0f00f2effc576f454cd48fa98aa5f`, and
`688cd7ba8917a0855ad3ce929cba5a9998932e79`.

The package pins `gliner2[local]==2.0.0`. Manifests must enumerate the complete
local snapshot because upstream revision forwarding could not be verified for
every weight load path. CPU FP32, CUDA FP32/FP16/BF16, and MPS FP16 are
advertised.

MPS FP16 is validated for the pinned English 340M revision on Apple M5 Max with
PyTorch 2.14.0 and GLiNER2 2.0.0. Fastino's constructor configures its scorer
for the requested device but does not transfer the underlying model, so the
adapter also calls the public `Classifier.to(device=..., dtype=...)` method.
Without that explicit transfer, upstream inference fails with
`RuntimeError: Passed CPU tensor to MPS op`; the runner never catches that
failure to retry on CPU. MPS FP32 also executed in the diagnostic probe, but is
not advertised because FP16 is the characterized profile. See the
[MPS characterization](../benchmark-results/2026-10-02-apple-m5-max-mps-fp16/REPORT.md).

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

The prospective ONNX backend is classification-only. Third-party or
untraceable exports are insufficient: serving stays disabled until exports
created and owned by this project are reproducible from pinned inputs and pass
the same preprocessing, output-shape, decoding, and parity gates. Dynamic axes,
opset, runtime providers, and quantization must be part of the artifact
identity. Universal Decide and span extraction must be rejected unless their
own designs are later accepted.

## Core ML and Swift

Core ML and a native Swift integration are later work. No public protocol
should imply their presence. Their design begins only after the reference,
capability negotiation, artifact provenance, and parity harness are stable.

## Parity policy

Parity is operation-specific, model-specific, language-specific, and
precision-specific. Aggregate accuracy alone cannot authorize a capability.
Each report records reference/runtime versions, model digest, evaluation data
provenance, preprocessing settings, thresholds, and discrepancies. A failed or
missing gate means “unsupported,” not fallback. MPS FP16 currently has
selected-label parity on the four transparent characterization examples with
a maximum observed numeric difference of 0.00184 versus CPU FP32; this is a
bounded characterization, not a claim of universal numerical equivalence.
