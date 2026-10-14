# Real-model characterization: fastino/GLiNER2.5-multi-Decide (CPU FP32)

Generated from `characterization.json` by `benchmarks/benchmark_real.py`.

## Scope and caution

These are one-machine measurements, not general hardware or deployment
recommendations. The workload is a small deterministic characterization set,
the p95 values have only 3 batch samples per matrix
cell, and synthetic length probes are not representative of every production
distribution.

## Environment

- Hardware: Apple M5 Max, 18 physical /
  18 logical cores, 128.00 GiB
- OS: Darwin 26.7.1 (arm64)
- Python 3.12.12; PyTorch 2.14.0;
  GLiNER2 2.1.0; Transformers 5.19.0
- Candidate device/precision: CPU / FP32
- Correctness oracle: CPU / FP32
- Workload: `multilingual-v1` (de, es, fr, ja)
- Model: `fastino/GLiNER2.5-multi-Decide` at `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`
- Snapshot content SHA-256: `0fc359fda9ab12da8c1b3a94320746de9a25be9e6ef1eb92cbc43051e5a508cb`
- Snapshot: 9 files, 1.09 GiB

## Startup and memory

| Measurement | Value |
| --- | ---: |
| Oracle process startup | 90.4 ms |
| Direct official model load | 3590.2 ms |
| First direct batch (4 items) | 123.2 ms |
| Oracle starting RSS | 0.03 GiB |
| Oracle steady RSS | 2.27 GiB |
| Oracle peak RSS | 2.33 GiB |
| HTTP server ready before model load | 342.7 ms |
| Candidate backend load | 5575.3 ms |
| Cold first HTTP request including lazy load | 6174.5 ms |
| HTTP process starting RSS | 0.06 GiB |
| HTTP steady RSS, one loaded model | 2.42 GiB |
| HTTP peak RSS, including two-model probe | 3.34 GiB |


## Warm latency matrix

The full matrix is in `latency.csv`. Batch sizes 1, 2, 4, and 8 were tested at
encoded input targets 16, 64, 256.

- Highest observed throughput: 44.55 requests/s
  at batch 8, 106 runner input tokens
  (16 raw-text tokenizer tokens).
- Largest observed p95 batch wall time: 646.0 ms
  at batch 8,
  364 runner input tokens.

## Correctness and scheduling

- Direct-oracle exact output parity:
  4/4
  (100.0%).
- Selected-label parity:
  4/4
  (100.0%).
- Maximum numeric difference: 0.
- Expected-label accuracy on the 4 transparent
  `multilingual-v1` examples: 4/4.
- Mixed-schema batch observed: False.
- Interleaved schema batch sizes: [4, 4].
- Distinct logical models observed:
  2.
- Overload probe: 9 HTTP 429 responses and
  0 server-observed cancellations.
  The client task was cancelled, but the server recorded no scheduler cancellation; this loopback HTTP disconnect did not propagate to the in-flight route task.

### Fill-window observations

| Scenario | Arrival offsets (ms) | Observed batch sizes |
| --- | --- | --- |
| simultaneous | 0, 0, 0, 0, 0, 0, 0, 0 | 8 |
| inside_window | 0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 2.8 | 8 |
| outside_window | 0, 12, 24, 36 | 1, 3 |


## Methodology

The direct official `Classifier.batch_classify` oracle ran in a fresh process.
The runner ran through a real loopback Uvicorn/FastAPI server and the public
JSON endpoints. RSS was sampled every 20 ms from each worker process. Latency
uses wall-clock `perf_counter`; throughput is total successful requests divided
by summed batch wall time. On Apple unified memory, process RSS and MPS
driver-allocated memory are separate accounting views and must not be added
together. Runner input usage is counted from the attention mask produced by
Fastino's compiled-schema processor for the actual encoder batch;
classification output usage is zero. Direct-oracle and runner requests use the
same described-label schema recorded in result metadata. The model snapshot
stayed outside Git. Inputs are deterministic synthetic length probes plus the
4 transparent `multilingual-v1` examples
visible in the benchmark source.
