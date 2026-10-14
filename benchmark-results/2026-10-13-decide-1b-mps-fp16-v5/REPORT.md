# Real-model characterization: fastino/GLiNER2.5-Decide-1B (MPS FP16)

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
- Workload: `english-v1` (en)
- Model: `fastino/GLiNER2.5-Decide-1B` at `67af949e17bebd20f05fe020a971079889e7a462`
- Snapshot content SHA-256: `6c183eb70b52b6683a1489a56df3eedb7fea7e12a2c5535c67fe00a1eb515d98`
- Snapshot: 8 files, 4.43 GiB

## Startup and memory

| Measurement | Value |
| --- | ---: |
| Oracle process startup | 103.7 ms |
| Direct official model load | 45743.9 ms |
| First direct batch (4 items) | 575.5 ms |
| Oracle starting RSS | 0.03 GiB |
| Oracle steady RSS | 5.42 GiB |
| Oracle peak RSS | 9.04 GiB |
| HTTP server ready before model load | 336.1 ms |
| Candidate backend load | 47807.5 ms |
| Cold first HTTP request including lazy load | 49055.9 ms |
| HTTP process starting RSS | 0.06 GiB |
| HTTP steady RSS, one loaded model | 2.85 GiB |
| HTTP peak RSS, including two-model probe | 10.97 GiB |
| Peak MPS driver-allocated memory | 3.02 GiB |


## Warm latency matrix

The full matrix is in `latency.csv`. Batch sizes 1, 2, 4, and 8 were tested at
encoded input targets 16, 64, 256.

- Highest observed throughput: 80.12 requests/s
  at batch 8, 102 runner input tokens
  (17 raw-text tokenizer tokens).
- Largest observed p95 batch wall time: 210.5 ms
  at batch 8,
  378 runner input tokens.

## Correctness and scheduling

- Direct-oracle exact output parity:
  0/4
  (0.0%).
- Selected-label parity:
  4/4
  (100.0%).
- Maximum numeric difference: 0.00153.
- Expected-label accuracy on the 4 transparent
  `english-v1` examples: 4/4.
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

## CPU / FP32 versus MPS / FP16

Baseline: CPU / FP32.
Candidate: MPS / FP16.
Both measurements use the same model content digest and machine.

- Backend load: 47128.2 ms baseline
  versus 47807.5 ms candidate
  (0.99x baseline/candidate ratio).
- Cold first HTTP request: 47853.8 ms
  baseline versus 49055.9 ms candidate
  (0.98x baseline/candidate ratio).
- Steady RSS: 5.09 GiB baseline
  versus 2.85 GiB candidate.
- Peak process RSS: 9.16 GiB baseline
  versus 10.97 GiB candidate.

| Tokens | Batch | Baseline p50 ms | Candidate p50 ms | Latency speedup | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 255.8 | 27.0 | 9.49x | 9.07x |
| 16 | 2 | 374.2 | 36.5 | 10.25x | 10.19x |
| 16 | 4 | 581.6 | 50.7 | 11.47x | 7.41x |
| 16 | 8 | 1033.0 | 68.2 | 15.14x | 10.41x |
| 64 | 1 | 291.8 | 30.2 | 9.66x | 10.05x |
| 64 | 2 | 479.8 | 37.5 | 12.79x | 12.83x |
| 64 | 4 | 819.1 | 54.4 | 15.04x | 14.06x |
| 64 | 8 | 1544.9 | 94.1 | 16.42x | 13.10x |
| 256 | 1 | 543.8 | 37.6 | 14.47x | 14.00x |
| 256 | 2 | 939.4 | 63.6 | 14.78x | 14.65x |
| 256 | 4 | 1859.5 | 108.3 | 17.17x | 17.10x |
| 256 | 8 | 3615.9 | 209.0 | 17.30x | 17.63x |


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
4 transparent `english-v1` examples
visible in the benchmark source.
