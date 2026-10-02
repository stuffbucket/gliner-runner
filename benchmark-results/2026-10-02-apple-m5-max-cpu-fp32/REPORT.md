# Real-model characterization: GLiNER2.5-Decide 340M

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
- Device/precision: CPU / FP32
- Model: `fastino/GLiNER2.5-Decide` at `5a7adf72a23b4d311abae6ce050d7f0012bb3416`
- Snapshot content SHA-256: `1395c304fb65d969409399e9937dfe31cb3b3e93e92eae29d40f6e084d414246`
- Snapshot: 10 files, 1.82 GiB

## Startup and memory

| Measurement | Value |
| --- | ---: |
| Oracle process startup | 223.5 ms |
| Direct official model load | 4194.8 ms |
| First direct batch (4 items) | 185.6 ms |
| Oracle starting RSS | 0.04 GiB |
| Oracle steady RSS | 2.33 GiB |
| Oracle peak RSS | 3.75 GiB |
| HTTP server ready before model load | 613.4 ms |
| Cold first HTTP request including lazy load | 6797.8 ms |
| HTTP process starting RSS | 0.07 GiB |
| HTTP steady RSS, one loaded model | 2.34 GiB |
| HTTP peak RSS, including two-model probe | 5.89 GiB |

## Warm latency matrix

The full matrix is in `latency.csv`. Batch sizes 1, 2, 4, and 8 were tested at
encoded input targets 16, 64, 256.

- Highest observed throughput: 27.94 requests/s
  at batch 8, 17 encoded tokens.
- Largest observed p95 batch wall time: 1633.5 ms
  at batch 8, 256 encoded tokens.

## Correctness and scheduling

- Direct-oracle exact output parity:
  2/4
  (50.0%).
- Selected-label parity:
  4/4
  (100.0%).
- Maximum numeric difference: 3.58e-07.
- Expected-label accuracy on the four transparent examples:
  4/4.
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
by summed batch wall time. The model snapshot stayed outside Git. Inputs are
deterministic synthetic length probes plus the four examples visible in the
benchmark source.
