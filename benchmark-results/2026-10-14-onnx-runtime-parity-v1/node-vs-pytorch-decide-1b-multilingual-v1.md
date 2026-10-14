# Node ONNX vs PyTorch parity: decide-1b / multilingual-v1

- Repository: `fastino/GLiNER2.5-Decide-1B` at `67af949e17bebd20f05fe020a971079889e7a462`
- Corpus sha256: `b97b2041d4dcc4caf194167bc0aad5d5d3604cd7119a61cef1267f8bcbeaceff`
- Runtime: node v24.21.0, platform darwin-arm64, onnxruntime-node 1.30.0, @huggingface/tokenizers 0.2.0
- Generated: 2026-10-14T20:39:30.610Z
- Numeric tolerance (absolute, gating): 0.0001; differences below it are counted as synthetic zero (diagnostic only)
- Overall: **PASS**

| Mode | Requests | Exact semantic matches | Mismatches | Numeric values | Exact zero | Synthetic zero (<1e-4) | Over tolerance | Max raw abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| singleton | 14 | 14 | 0 | 62 | 1 | 61 | 0 | 3.979e-6 |
| repeated | 28 | 28 | 0 | 124 | 2 | 122 | 0 | 3.979e-6 |
| batch | 3 | 3 | 0 | 18 | 0 | 18 | 0 | 6.557e-7 |
| concurrent-group | 3 | 3 | 0 | 18 | 0 | 18 | 0 | 6.557e-7 |
| concurrent-all | 14 | 14 | 0 | 62 | 1 | 61 | 0 | 3.979e-6 |

## Dynamic batching evidence

| Mode | Requests | Backend batches | Submitted requests |
| --- | ---: | ---: | ---: |
| batch | 3 | 1 | 3 |
| concurrent-group | 3 | 1 | 3 |
| concurrent-all | 14 | 12 | 14 |

## Overall

- Requests compared: 62; exact semantic matches: 62
- Max raw absolute numeric difference: 3.979e-6
- Repeat runs byte-identical: true

## Per-case maximum raw numeric difference

| Mode | Case | Max raw abs diff |
| --- | --- | ---: |
| singleton | described-single | 7.153e-7 |
| singleton | multi-label | 2.027e-6 |
| singleton | high-threshold-empty | 1.341e-7 |
| singleton | ordered-label | 4.768e-7 |
| singleton | feasible-constraint | 1.803e-6 |
| singleton | infeasible-constraint | 0.000e+0 |
| singleton | mixed-length-long | 6.557e-7 |
| singleton | described-single-repeat | 7.153e-7 |
| singleton | multi-task-described-instruction | 3.979e-6 |
| singleton | default-temperature-multi-label | 7.041e-7 |
| singleton | softmax-activation | 2.146e-6 |
| singleton | invalid-reserved-marker | 0.000e+0 |
| singleton | invalid-min-exceeds-max | 0.000e+0 |
| singleton | invalid-unknown-constraint-label | 0.000e+0 |
