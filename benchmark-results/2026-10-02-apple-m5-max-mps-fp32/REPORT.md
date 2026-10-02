# Real-model characterization: GLiNER2.5-Decide 340M (MPS FP32)

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
- OS: Darwin 26.7 (arm64)
- Python 3.12.12; PyTorch 2.14.0;
  GLiNER2 2.0.0; Transformers 4.57.6
- Candidate device/precision: MPS / FP32
- Correctness oracle: CPU / FP32
- Model: `fastino/GLiNER2.5-Decide` at `5a7adf72a23b4d311abae6ce050d7f0012bb3416`
- Snapshot content SHA-256: `1395c304fb65d969409399e9937dfe31cb3b3e93e92eae29d40f6e084d414246`
- Snapshot: 10 files, 1.82 GiB

## Startup and memory

| Measurement | Value |
| --- | ---: |
| Oracle process startup | 229.1 ms |
| Direct official model load | 4245.8 ms |
| First direct batch (4 items) | 197.7 ms |
| Oracle starting RSS | 0.04 GiB |
| Oracle steady RSS | 2.34 GiB |
| Oracle peak RSS | 3.43 GiB |
| HTTP server ready before model load | 616.4 ms |
| Candidate backend load | 5983.2 ms |
| Cold first HTTP request including lazy load | 7027.2 ms |
| HTTP process starting RSS | 0.07 GiB |
| HTTP steady RSS, one loaded model | 1.23 GiB |
| HTTP peak RSS, including two-model probe | 4.08 GiB |
| Peak MPS driver-allocated memory | 4.07 GiB |


## Warm latency matrix

The full matrix is in `latency.csv`. Batch sizes 1, 2, 4, and 8 were tested at
encoded input targets 16, 64, 256.

- Highest observed throughput: 85.18 requests/s
  at batch 8, 17 encoded tokens.
- Largest observed p95 batch wall time: 323.6 ms
  at batch 8, 256 encoded tokens.

## Correctness and scheduling

- Direct-oracle exact output parity:
  0/4
  (0.0%).
- Selected-label parity:
  4/4
  (100.0%).
- Maximum numeric difference: 2.98e-06.
- Expected-label accuracy on the four transparent examples:
  4/4.
- Mixed-schema batch observed: False.
- Interleaved schema batch sizes: [4, 4].
- Distinct logical models observed:
  2.
- Overload probe: 8 HTTP 429 responses and
  0 server-observed cancellations.
  The client task was cancelled, but the server recorded no scheduler cancellation; this loopback HTTP disconnect did not propagate to the in-flight route task.

### Fill-window observations

| Scenario | Arrival offsets (ms) | Observed batch sizes |
| --- | --- | --- |
| simultaneous | 0, 0, 0, 0, 0, 0, 0, 0 | 8 |
| inside_window | 0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 2.8 | 8 |
| outside_window | 0, 12, 24, 36 | 1, 1, 2 |

## CPU / FP32 versus MPS / FP32

Baseline: CPU / FP32.
Candidate: MPS / FP32.
Both measurements use the same model content digest and machine.

- Backend load: 6659.6 ms baseline
  versus 5983.2 ms candidate
  (1.11x baseline/candidate ratio).
- Cold first HTTP request: 6797.8 ms
  baseline versus 7027.2 ms candidate
  (0.97x baseline/candidate ratio).
- Steady RSS: 2.34 GiB baseline
  versus 1.23 GiB candidate.
- Peak process RSS: 5.89 GiB baseline
  versus 4.08 GiB candidate.

| Tokens | Batch | Baseline p50 ms | Candidate p50 ms | Latency speedup | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 130.8 | 37.4 | 3.49x | 2.46x |
| 16 | 2 | 158.7 | 40.0 | 3.97x | 2.40x |
| 16 | 4 | 202.8 | 35.9 | 5.65x | 2.57x |
| 16 | 8 | 286.9 | 48.5 | 5.92x | 3.05x |
| 64 | 1 | 161.3 | 34.0 | 4.74x | 2.31x |
| 64 | 2 | 203.5 | 38.3 | 5.32x | 2.91x |
| 64 | 4 | 299.9 | 57.1 | 5.25x | 3.35x |
| 64 | 8 | 517.5 | 94.4 | 5.48x | 4.13x |
| 256 | 1 | 287.2 | 59.5 | 4.83x | 4.59x |
| 256 | 2 | 452.9 | 96.3 | 4.70x | 4.81x |
| 256 | 4 | 826.9 | 161.9 | 5.11x | 4.84x |
| 256 | 8 | 1557.8 | 306.2 | 5.09x | 5.07x |


## Methodology

The direct official `Classifier.batch_classify` oracle ran in a fresh process.
The runner ran through a real loopback Uvicorn/FastAPI server and the public
JSON endpoints. RSS was sampled every 20 ms from each worker process. Latency
uses wall-clock `perf_counter`; throughput is total successful requests divided
by summed batch wall time. On Apple unified memory, process RSS and MPS
driver-allocated memory are separate accounting views and must not be added
together. The model snapshot stayed outside Git. Inputs are deterministic
synthetic length probes plus the four examples visible in the benchmark source.
