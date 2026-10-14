# ONNX classification feasibility

This is research evidence for a raw classification-logit graph. Tokenization,
schema compilation, constrained decoding, formatting, and usage accounting
remain outside ONNX. It does not advertise or register a production backend.

## Source and runtime

- Model: `fastino/GLiNER2.5-Decide@9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90`
- Snapshot SHA-256: `5081e1a9b994143fe7f98810c909c8f6498914016b9fbe58c46a5a1135e34946`
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
- Maximum absolute logit difference: 5.3048134e-06
- Official path: `gliner2.classification.Classifier.batch_score`
- Wrapper/official selected-label parity: 3/3
- Wrapper/official maximum absolute logit difference: 0
- Dynamic fixture shapes: B=1/S=41/K=3, B=2/S=30/K=4

## Generated artifacts

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `classifier.onnx` | 4879709 | `46b14246c2c35b7f19b9184713beb55e4618335e086a3d81c2ddaf0b89635cb3` |
| `classifier.onnx.data` | 1744089088 | `f5aef45f7892cf3acf277516e225d5e653d6b4946918d728b3c3ab0882ac8d58` |
| `export-report/onnx_export_2026-10-13_13-41-42-878395_success.md` | 861913 | `aa3d291595579dfc61b49e01256ff194951fb082ef773d03801fd937cfe628a2` |
