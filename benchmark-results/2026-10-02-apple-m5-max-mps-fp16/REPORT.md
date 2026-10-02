# Real-model characterization: GLiNER2.5-Decide 340M (MPS FP16)

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
- Candidate device/precision: MPS / FP16
- Correctness oracle: CPU / FP32
- Model: `fastino/GLiNER2.5-Decide` at `5a7adf72a23b4d311abae6ce050d7f0012bb3416`
- Snapshot content SHA-256: `1395c304fb65d969409399e9937dfe31cb3b3e93e92eae29d40f6e084d414246`
- Snapshot: 10 files, 1.82 GiB

## Startup and memory

| Measurement | Value |
| --- | ---: |
| Oracle process startup | 257.8 ms |
| Direct official model load | 4192.0 ms |
| First direct batch (4 items) | 178.3 ms |
| Oracle starting RSS | 0.04 GiB |
| Oracle steady RSS | 2.33 GiB |
| Oracle peak RSS | 3.87 GiB |
| HTTP server ready before model load | 589.3 ms |
| Candidate backend load | 6790.5 ms |
| Cold first HTTP request including lazy load | 6937.8 ms |
| HTTP process starting RSS | 0.07 GiB |
| HTTP steady RSS, one loaded model | 1.24 GiB |
| HTTP peak RSS, including two-model probe | 3.49 GiB |
| Peak MPS driver-allocated memory | 2.09 GiB |


## Warm latency matrix

The full matrix is in `latency.csv`. Batch sizes 1, 2, 4, and 8 were tested at
encoded input targets 16, 64, 256.

- Highest observed throughput: 212.26 requests/s
  at batch 8, 17 encoded tokens.
- Largest observed p95 batch wall time: 219.2 ms
  at batch 8, 256 encoded tokens.

## Correctness and scheduling

- Direct-oracle exact output parity:
  0/4
  (0.0%).
- Selected-label parity:
  4/4
  (100.0%).
- Maximum numeric difference: 0.00184.
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
| outside_window | 0, 12, 24, 36 | 1, 1, 2 |

## CPU versus MPS

Baseline: CPU / FP32.
Candidate: MPS / FP16.
Both measurements use the same model content digest and machine.

- Backend load: 6659.6 ms CPU versus
  6790.5 ms MPS
  (0.98x CPU/MPS ratio).
- Cold first HTTP request: 6797.8 ms CPU
  versus 6937.8 ms MPS
  (0.98x CPU/MPS ratio).
- Steady RSS: 2.34 GiB CPU
  versus 1.24 GiB MPS.
- Peak process RSS: 5.89 GiB CPU
  versus 3.49 GiB MPS.

| Tokens | Batch | CPU p50 ms | MPS p50 ms | Latency speedup | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 130.8 | 36.0 | 3.63x | 3.40x |
| 16 | 2 | 158.7 | 35.7 | 4.45x | 4.96x |
| 16 | 4 | 202.8 | 28.5 | 7.12x | 7.14x |
| 16 | 8 | 286.9 | 38.7 | 7.42x | 7.60x |
| 64 | 1 | 161.3 | 36.0 | 4.48x | 4.39x |
| 64 | 2 | 203.5 | 27.4 | 7.43x | 6.98x |
| 64 | 4 | 299.9 | 35.5 | 8.46x | 8.80x |
| 64 | 8 | 517.5 | 42.5 | 12.17x | 11.54x |
| 256 | 1 | 287.2 | 38.4 | 7.48x | 7.06x |
| 256 | 2 | 452.9 | 60.8 | 7.45x | 7.20x |
| 256 | 4 | 826.9 | 111.1 | 7.45x | 7.43x |
| 256 | 8 | 1557.8 | 191.7 | 8.13x | 7.83x |


## Methodology

The direct official `Classifier.batch_classify` oracle ran in a fresh process.
The runner ran through a real loopback Uvicorn/FastAPI server and the public
JSON endpoints. RSS was sampled every 20 ms from each worker process. Latency
uses wall-clock `perf_counter`; throughput is total successful requests divided
by summed batch wall time. On Apple unified memory, process RSS and MPS
driver-allocated memory are separate accounting views and must not be added
together. The model snapshot stayed outside Git. Inputs are deterministic
synthetic length probes plus the four examples visible in the benchmark source.
