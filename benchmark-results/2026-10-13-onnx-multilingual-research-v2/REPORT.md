# ONNX classification feasibility

This is research evidence for a raw classification-logit graph. Tokenization,
schema compilation, constrained decoding, formatting, and usage accounting
remain outside ONNX. It does not advertise or register a production backend.

## Source and runtime

- Model: `fastino/GLiNER2.5-multi-Decide@e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`
- Snapshot SHA-256: `0fc359fda9ab12da8c1b3a94320746de9a25be9e6ef1eb92cbc43051e5a508cb`
- GLiNER2: 2.1.0
- PyTorch: 2.14.0
- ONNX: 1.23.2
- ONNX Runtime: 1.31.0
- Provider: `CPUExecutionProvider`
- Opset imports: [{"domain": "ai.onnx", "version": 20}]
- Absolute tolerance: 0.0001
- Quantization: `none`
- Fixture suite: `classification-logits-multilingual-v1`
- Fixture languages: es, fr, de

## Contract and parity

- Inputs: input_ids, attention_mask, label_positions, label_mask
- Output: `logits`
- Valid logits compared: 9
- Selected-label parity: 3/3
- Maximum absolute logit difference: 2.3841858e-06
- Official path: `gliner2.classification.Classifier.batch_score`
- Wrapper/official selected-label parity: 3/3
- Wrapper/official maximum absolute logit difference: 0
- Dynamic fixture shapes: B=1/S=57/K=3, B=2/S=40/K=4

## Generated artifacts

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `classifier.onnx` | 1117368768 | `d51cb607467f93117c263f74869d813f70b71b1d5bdb8066edf4b1f53ba78bbe` |
| `export-report/onnx_export_2026-10-13_15-37-35-221720_success.md` | 441955 | `ab2a421ae414328390ad551e3a90df8a214f9ce7683a38288ea4839a4dc159b9` |
