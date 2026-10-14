# ONNX classification feasibility

This is research evidence for a raw classification-logit graph. Tokenization,
schema compilation, constrained decoding, formatting, and usage accounting
remain outside ONNX. It does not advertise or register a production backend.

## Source and runtime

- Model: `fastino/gliner2.5-small-v1@df5910e44bc4ffdb0d95399a83b0ca4516349aa5`
- Snapshot SHA-256: `77a62d0eba8110bba87ee32872e1b171a0283b11941f382b8729d9dd41c2aede`
- GLiNER2: 2.1.0
- PyTorch: 2.14.0
- ONNX: 1.23.2
- ONNX Runtime: 1.31.0
- Provider: `CPUExecutionProvider`
- Opset imports: [{"domain": "ai.onnx", "version": 20}]
- Absolute tolerance: 0.0001
- Quantization: `none`

## Contract and parity

- Inputs: input_ids, attention_mask, label_positions, label_mask
- Output: `logits`
- Valid logits compared: 9
- Selected-label parity: 3/3
- Maximum absolute logit difference: 1.001358e-05
- Official path: `gliner2.classification.Classifier.batch_score`
- Wrapper/official selected-label parity: 3/3
- Wrapper/official maximum absolute logit difference: 0
- Dynamic fixture shapes: B=1/S=41/K=3, B=2/S=30/K=4

## Generated artifacts

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `classifier.onnx` | 286267898 | `0cadd8e889bf5517a90903569aa52cd9afa59a97ede264cc16fca12a370b6941` |
| `export-report/onnx_export_2026-10-13_13-41-22-862677_success.md` | 440781 | `e403b2a81e5aa948e45a28a3111528b66027104f4e057c02fd0b232a9c097042` |
