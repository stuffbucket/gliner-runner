# Production ONNX Goal

Build and ship a production ONNX inference path that can replace the current Python/PyTorch runner for every model and operation the runner advertises as supported, without requiring Python, PyTorch, Transformers, or GLiNER2 in the deployed inference runtime.

## Required model coverage

Support the same curated, immutable model revisions as the PyTorch runner:

- `fastino/GLiNER2.5-Decide`
- `fastino/GLiNER2.5-multi-Decide`
- `fastino/GLiNER2.5-Decide-1B`

Model support must be revision-specific and backed by verified artifacts and evidence. Support for one model, language, operation, provider, or precision must not be generalized to another without its own evidence.

## Model preparation and artifacts

Start from pinned Hugging Face snapshots and convert them through a project-owned, reproducible preparation pipeline. Conversion happens during build, release, installation, or explicit model preparation—not per request and preferably not on every server start.

Each production artifact must include or identify:

- the ONNX graph and every external-data sidecar;
- reusable tokenizer and configuration assets;
- source repository, immutable revision, and snapshot digest;
- graph and sidecar sizes and SHA-256 digests;
- exporter and dependency versions;
- graph inputs, outputs, dynamic axes, and opset imports;
- intended ONNX Runtime provider, precision, and quantization profile;
- provenance, licensing, and compatibility metadata.

The runtime must reject missing, altered, unsupported, or incorrectly paired artifacts. It must not silently fall back to PyTorch.

## Capability parity

Implement the complete behavior of every operation advertised by the PyTorch backend. The current production scope is constrained classification, including:

- plain and described labels;
- single-label and multi-label results;
- multiple tasks in one schema;
- task thresholds and threshold-driven empty results;
- minimum and maximum label constraints;
- ordered results;
- activation and temperature settings;
- defaults and instructions;
- feasible and infeasible constraints;
- independent and exact constrained decoding;
- confidence, probability, and decoder metadata;
- scalar, list, and empty result shapes;
- structured validation and inference errors;
- exact input-token and output-token accounting;
- singleton, repeated, and batched execution.

Tokenization, schema and prompt compilation, label-position construction, decoding, result formatting, and usage accounting may execute outside the ONNX graph, but their production implementation must not depend on Python and must be proven equivalent to the reference GLiNER2 path.

## Parity standard

For the same model revision and request, the ONNX path must:

- produce exactly the same semantic labels, ordering, cardinality, empty results, metadata, and structured errors;
- preserve singleton, repeated, and batch consistency;
- expose the same response and usage contract through the HTTP and client APIs;
- retain raw numerical measurements and remain within an explicitly approved error bound for each model, provider, precision, language, and operation;
- pass the complete PyTorch compatibility corpus and applicable multilingual matrix.

A synthetic-zero comparison view may represent an absolute PyTorch/ONNX delta below `1e-4` as zero for diagnostic comparison. It must remain optional and non-gating: raw logits, raw differences, label selection, and acceptance decisions remain authoritative and auditable.

Raw-logit parity alone is necessary but insufficient. Production parity requires end-to-end request, decoding, response, error, usage, batching, and lifecycle evidence.

## Production integration

Provide an ONNX backend that implements the complete backend lifecycle:

- capability reporting;
- validated model loading;
- batched inference;
- cancellation and bounded resource behavior;
- unloading and closing;
- explicit registry integration;
- provider, device, precision, and profile validation.

Exercise it through the same runtime, scheduler, HTTP, and client surfaces as PyTorch, including dynamic batching, schema and model grouping, cancellation, overload/backpressure, and no-fallback behavior.

## Definition of done

The goal is complete when, for every model and operation advertised as supported by the PyTorch runner, the production ONNX runner:

1. accepts the same valid requests;
2. rejects the same invalid or unsupported requests with equivalent structured errors;
3. produces the same semantic results and usage accounting;
4. stays within approved raw numerical tolerances;
5. passes the full operation, language, batching, scheduler, and API parity matrix;
6. loads only verified, reproducible model artifacts;
7. is packaged and shipped without a Python/PyTorch inference dependency; and
8. is documented and supported as a first-class backend rather than a research-only export path.
