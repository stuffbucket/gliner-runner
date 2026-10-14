# Node ONNX vs PyTorch parity: decide-multilingual / multilingual-v1

- Repository: `fastino/GLiNER2.5-multi-Decide` at `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`
- Corpus sha256: `b97b2041d4dcc4caf194167bc0aad5d5d3604cd7119a61cef1267f8bcbeaceff`
- Runtime: node v24.21.0, platform darwin-arm64, onnxruntime-node 1.30.0, @huggingface/tokenizers 0.2.0
- Generated: 2026-10-14T20:39:48.912Z
- Numeric tolerance (absolute, gating): 0.0001; differences below it are counted as synthetic zero (diagnostic only)
- Overall: **PASS**

| Mode | Requests | Exact semantic matches | Mismatches | Numeric values | Exact zero | Synthetic zero (<1e-4) | Over tolerance | Max raw abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| singleton | 14 | 14 | 0 | 62 | 1 | 61 | 0 | 2.189e-5 |
| repeated | 28 | 28 | 0 | 124 | 2 | 122 | 0 | 2.189e-5 |
| batch | 3 | 3 | 0 | 18 | 0 | 18 | 0 | 3.219e-6 |
| concurrent-group | 3 | 3 | 0 | 18 | 0 | 18 | 0 | 3.219e-6 |
| concurrent-all | 14 | 14 | 0 | 62 | 1 | 61 | 0 | 2.189e-5 |

## Dynamic batching evidence

| Mode | Requests | Backend batches | Submitted requests |
| --- | ---: | ---: | ---: |
| batch | 3 | 1 | 3 |
| concurrent-group | 3 | 1 | 3 |
| concurrent-all | 14 | 12 | 14 |

## Overall

- Requests compared: 62; exact semantic matches: 62
- Max raw absolute numeric difference: 2.189e-5
- Repeat runs byte-identical: true

## Per-case maximum raw numeric difference

| Mode | Case | Max raw abs diff |
| --- | --- | ---: |
| singleton | described-single | 2.801e-6 |
| singleton | multi-label | 7.689e-6 |
| singleton | high-threshold-empty | 5.728e-7 |
| singleton | ordered-label | 1.669e-6 |
| singleton | feasible-constraint | 2.189e-5 |
| singleton | infeasible-constraint | 0.000e+0 |
| singleton | mixed-length-long | 3.219e-6 |
| singleton | described-single-repeat | 2.801e-6 |
| singleton | multi-task-described-instruction | 5.782e-6 |
| singleton | default-temperature-multi-label | 3.757e-7 |
| singleton | softmax-activation | 4.292e-6 |
| singleton | invalid-reserved-marker | 0.000e+0 |
| singleton | invalid-min-exceeds-max | 0.000e+0 |
| singleton | invalid-unknown-constraint-label | 0.000e+0 |
