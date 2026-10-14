# Real-model characterization: fastino/GLiNER2.5-multi-Decide (MPS FP16)

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
- Candidate device/precision: MPS / FP16
- Correctness oracle: CPU / FP32
- Workload: `multilingual-v1` (de, es, fr, ja)
- Model: `fastino/GLiNER2.5-multi-Decide` at `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`
- Snapshot content SHA-256: `0fc359fda9ab12da8c1b3a94320746de9a25be9e6ef1eb92cbc43051e5a508cb`
- Snapshot: 9 files, 1.09 GiB

## Startup and memory

| Measurement | Value |
| --- | ---: |
| Oracle process startup | 102.7 ms |
| Direct official model load | 3553.3 ms |
| First direct batch (4 items) | 119.2 ms |
| Oracle starting RSS | 0.03 GiB |
| Oracle steady RSS | 2.27 GiB |
| Oracle peak RSS | 2.31 GiB |
| HTTP server ready before model load | 936.6 ms |
| Candidate backend load | 5680.6 ms |
| Cold first HTTP request including lazy load | 6846.1 ms |
| HTTP process starting RSS | 0.06 GiB |
| HTTP steady RSS, one loaded model | 1.18 GiB |
| HTTP peak RSS, including two-model probe | 3.17 GiB |
| Peak MPS driver-allocated memory | 1.03 GiB |


## Warm latency matrix

The full matrix is in `latency.csv`. Batch sizes 1, 2, 4, and 8 were tested at
encoded input targets 16, 64, 256.

- Highest observed throughput: 187.56 requests/s
  at batch 8, 106 runner input tokens
  (16 raw-text tokenizer tokens).
- Largest observed p95 batch wall time: 282.4 ms
  at batch 1,
  364 runner input tokens.

## Correctness and scheduling

- Direct-oracle exact output parity:
  0/4
  (0.0%).
- Selected-label parity:
  4/4
  (100.0%).
- Maximum numeric difference: 0.00534.
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
| outside_window | 0, 12, 24, 36 | 1, 1, 2 |

## CPU / FP32 versus MPS / FP16

Baseline: CPU / FP32.
Candidate: MPS / FP16.
Both measurements use the same model content digest and machine.

- Backend load: 5575.3 ms baseline
  versus 5680.6 ms candidate
  (0.98x baseline/candidate ratio).
- Cold first HTTP request: 6174.5 ms
  baseline versus 6846.1 ms candidate
  (0.90x baseline/candidate ratio).
- Steady RSS: 2.42 GiB baseline
  versus 1.18 GiB candidate.
- Peak process RSS: 3.34 GiB baseline
  versus 3.17 GiB candidate.

| Tokens | Batch | Baseline p50 ms | Candidate p50 ms | Latency speedup | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 56.1 | 21.5 | 2.61x | 2.04x |
| 16 | 2 | 73.0 | 22.4 | 3.26x | 3.12x |
| 16 | 4 | 112.0 | 24.1 | 4.65x | 3.74x |
| 16 | 8 | 178.1 | 37.4 | 4.77x | 4.21x |
| 64 | 1 | 61.7 | 16.2 | 3.81x | 3.20x |
| 64 | 2 | 93.7 | 27.0 | 3.48x | 3.70x |
| 64 | 4 | 146.4 | 27.3 | 5.37x | 4.84x |
| 64 | 8 | 258.8 | 44.6 | 5.80x | 5.45x |
| 256 | 1 | 110.0 | 24.7 | 4.45x | 0.92x |
| 256 | 2 | 185.4 | 40.3 | 4.60x | 4.23x |
| 256 | 4 | 328.1 | 81.0 | 4.05x | 4.08x |
| 256 | 8 | 608.8 | 131.3 | 4.64x | 4.68x |


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
