# Real-model characterization: fastino/GLiNER2.5-multi-Decide (MPS FP32)

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
- Candidate device/precision: MPS / FP32
- Correctness oracle: CPU / FP32
- Workload: `multilingual-v1` (de, es, fr, ja)
- Model: `fastino/GLiNER2.5-multi-Decide` at `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`
- Snapshot content SHA-256: `0fc359fda9ab12da8c1b3a94320746de9a25be9e6ef1eb92cbc43051e5a508cb`
- Snapshot: 9 files, 1.09 GiB

## Startup and memory

| Measurement | Value |
| --- | ---: |
| Oracle process startup | 102.8 ms |
| Direct official model load | 3541.6 ms |
| First direct batch (4 items) | 117.2 ms |
| Oracle starting RSS | 0.03 GiB |
| Oracle steady RSS | 2.27 GiB |
| Oracle peak RSS | 2.66 GiB |
| HTTP server ready before model load | 341.6 ms |
| Candidate backend load | 5628.4 ms |
| Cold first HTTP request including lazy load | 6638.8 ms |
| HTTP process starting RSS | 0.06 GiB |
| HTTP steady RSS, one loaded model | 1.17 GiB |
| HTTP peak RSS, including two-model probe | 2.61 GiB |
| Peak MPS driver-allocated memory | 2.22 GiB |


## Warm latency matrix

The full matrix is in `latency.csv`. Batch sizes 1, 2, 4, and 8 were tested at
encoded input targets 16, 64, 256.

- Highest observed throughput: 107.60 requests/s
  at batch 8, 158 runner input tokens
  (64 raw-text tokenizer tokens).
- Largest observed p95 batch wall time: 182.2 ms
  at batch 8,
  364 runner input tokens.

## Correctness and scheduling

- Direct-oracle exact output parity:
  0/4
  (0.0%).
- Selected-label parity:
  4/4
  (100.0%).
- Maximum numeric difference: 2.21e-06.
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
| outside_window | 0, 12, 24, 36 | 1, 2, 1 |

## CPU / FP32 versus MPS / FP32

Baseline: CPU / FP32.
Candidate: MPS / FP32.
Both measurements use the same model content digest and machine.

- Backend load: 5575.3 ms baseline
  versus 5628.4 ms candidate
  (0.99x baseline/candidate ratio).
- Cold first HTTP request: 6174.5 ms
  baseline versus 6638.8 ms candidate
  (0.93x baseline/candidate ratio).
- Steady RSS: 2.42 GiB baseline
  versus 1.17 GiB candidate.
- Peak process RSS: 3.34 GiB baseline
  versus 2.61 GiB candidate.

| Tokens | Batch | Baseline p50 ms | Candidate p50 ms | Latency speedup | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 56.1 | 24.7 | 2.27x | 2.05x |
| 16 | 2 | 73.0 | 33.9 | 2.15x | 1.72x |
| 16 | 4 | 112.0 | 36.4 | 3.08x | 1.58x |
| 16 | 8 | 178.1 | 51.3 | 3.47x | 1.95x |
| 64 | 1 | 61.7 | 18.2 | 3.39x | 2.63x |
| 64 | 2 | 93.7 | 32.6 | 2.87x | 1.88x |
| 64 | 4 | 146.4 | 40.7 | 3.60x | 3.64x |
| 64 | 8 | 258.8 | 66.6 | 3.89x | 3.49x |
| 256 | 1 | 110.0 | 40.0 | 2.75x | 2.52x |
| 256 | 2 | 185.4 | 55.3 | 3.35x | 3.26x |
| 256 | 4 | 328.1 | 111.3 | 2.95x | 2.79x |
| 256 | 8 | 608.8 | 179.7 | 3.39x | 3.46x |


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
