# ONNX classification feasibility

This is research evidence for a raw classification-logit graph. Tokenization,
schema compilation, constrained decoding, formatting, and usage accounting
remain outside ONNX. It does not advertise or register a production backend.

## Source and runtime

- Model: `fastino/GLiNER2.5-Decide-1B@67af949e17bebd20f05fe020a971079889e7a462`
- Snapshot SHA-256: `6c183eb70b52b6683a1489a56df3eedb7fea7e12a2c5535c67fe00a1eb515d98`
- GLiNER2: 2.1.0
- PyTorch: 2.14.0
- ONNX: 1.23.2
- ONNX Runtime: 1.31.0
- Provider: `CPUExecutionProvider`
- Opset imports: [{"domain": "ai.onnx", "version": 20}]
- Absolute tolerance: 0.0001
- Quantization: `none`
- Fixture suite: `classification-logits-english-v1`
- Fixture languages: en

## Contract and parity

- Inputs: input_ids, attention_mask, label_positions, label_mask
- Output: `logits`
- Valid logits compared: 9
- Selected-label parity: 3/3
- Maximum absolute logit difference: 5.9604645e-06
- Official path: `gliner2.classification.Classifier.batch_score`
- Wrapper/official selected-label parity: 3/3
- Wrapper/official maximum absolute logit difference: 0
- Dynamic fixture shapes: B=1/S=52/K=3, B=2/S=32/K=4

## Generated artifacts

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `classifier.onnx` | 3099074 | `6a3363d18bc0eec6b68d36a0ee68041f45b30fb8b1b7d2d347ceb9a2fe337121` |
| `classifier.onnx.data` | 4138014720 | `9fd80d51728c2e7cfa12740fc773bc6cf7250f589a274108d06b9e5025eb7ace` |
| `export-report/onnx_export_2026-10-13_15-38-53-301823_success.md` | 539300 | `a204e7edd7f97ddc6b4afc2b254cac5f28eaefb43a3a1f5e8b2efe4b75a7289a` |
