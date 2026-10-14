# Node ONNX vs PyTorch parity: decide-1b / english-v1

- Repository: `fastino/GLiNER2.5-Decide-1B` at `67af949e17bebd20f05fe020a971079889e7a462`
- Corpus sha256: `10aa107ccc49c0b0eab283a6d2ad74b98057e08b1c3b61dbec790d10d45c9de6`
- Runtime: node v24.21.0, platform darwin-arm64, onnxruntime-node 1.30.0, @huggingface/tokenizers 0.2.0
- Generated: 2026-10-14T20:39:20.055Z
- Numeric tolerance (absolute, gating): 0.0001; differences below it are counted as synthetic zero (diagnostic only)
- Overall: **PASS**

| Mode | Requests | Exact semantic matches | Mismatches | Numeric values | Exact zero | Synthetic zero (<1e-4) | Over tolerance | Max raw abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| singleton | 14 | 14 | 0 | 62 | 3 | 59 | 0 | 2.205e-6 |
| repeated | 28 | 28 | 0 | 124 | 6 | 118 | 0 | 2.205e-6 |
| batch | 3 | 3 | 0 | 18 | 0 | 18 | 0 | 1.907e-6 |
| concurrent-group | 3 | 3 | 0 | 18 | 0 | 18 | 0 | 1.431e-6 |
| concurrent-all | 14 | 14 | 0 | 62 | 3 | 59 | 0 | 2.205e-6 |

## Dynamic batching evidence

| Mode | Requests | Backend batches | Submitted requests |
| --- | ---: | ---: | ---: |
| batch | 3 | 1 | 3 |
| concurrent-group | 3 | 1 | 3 |
| concurrent-all | 14 | 12 | 14 |

## Overall

- Requests compared: 62; exact semantic matches: 62
- Max raw absolute numeric difference: 2.205e-6
- Repeat runs byte-identical: true

## Per-case maximum raw numeric difference

| Mode | Case | Max raw abs diff |
| --- | --- | ---: |
| singleton | described-single | 1.490e-6 |
| singleton | multi-label | 9.239e-7 |
| singleton | high-threshold-empty | 2.539e-7 |
| singleton | ordered-label | 1.788e-7 |
| singleton | feasible-constraint | 4.172e-7 |
| singleton | infeasible-constraint | 0.000e+0 |
| singleton | mixed-length-long | 5.960e-7 |
| singleton | described-single-repeat | 1.490e-6 |
| singleton | multi-task-described-instruction | 2.205e-6 |
| singleton | default-temperature-multi-label | 1.788e-7 |
| singleton | softmax-activation | 7.153e-7 |
| singleton | invalid-reserved-marker | 0.000e+0 |
| singleton | invalid-min-exceeds-max | 0.000e+0 |
| singleton | invalid-unknown-constraint-label | 0.000e+0 |
