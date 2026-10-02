# Real-model characterization

`benchmarks/benchmark_real.py` characterizes the official PyTorch backend
through both the direct Fastino API and the runner's loopback HTTP API. It is
an opt-in local command and is not part of offline CI.

## Running

Install the runtime, benchmark, and development dependencies, then provide a
complete local snapshot of the pinned model revision:

```sh
python -m pip install -e '.[pytorch,benchmark,dev]'
export GLINER_RUNNER_BENCHMARK_MODEL=/path/to/pinned/model/snapshot
mise run benchmark:real
```

The snapshot must remain outside the repository. The command writes local
results to `benchmark-results/local/` by default. Set
`GLINER_RUNNER_BENCHMARK_OUTPUT` to choose another directory.

The default profile is CPU/FP32. To characterize the validated Apple Silicon
profile and compare it with a prior CPU result:

```sh
export GLINER_RUNNER_BENCHMARK_DEVICE=mps
export GLINER_RUNNER_BENCHMARK_PRECISION=fp16
export GLINER_RUNNER_BENCHMARK_BASELINE=benchmark-results/cpu/characterization.json
mise run benchmark:real
```

Unsupported pairs are rejected before model loading and never rewritten.

The benchmark records:

- fresh-process startup, model-load, first-inference, and RSS measurements;
- MPS driver-allocated memory when the selected device exposes it;
- warm HTTP wall-clock latency and throughput for batch sizes 1, 2, 4, and 8
  at encoded token targets 16, 64, and 256;
- dynamic batch-window, schema grouping, and logical-model grouping behavior;
- direct official-oracle output parity and expected labels for four transparent
  examples;
- client cancellation and bounded-queue backpressure under concurrent load;
- exact hardware, OS, runtime versions, model revision, content digest,
  precision, sample counts, and timing methodology.

Results are emitted as full JSON, a latency CSV, and a generated Markdown
summary. Output excludes model paths, input text, hostnames, usernames, and
credentials.

## Interpreting results

These measurements characterize one machine and one small deterministic
workload. They are not general hardware or deployment recommendations.
Percentiles from the default three samples per matrix cell are useful for
reproducibility checks, not statistical performance claims. Exact JSON parity
can differ when floating-point confidence values vary by tiny amounts; selected
label parity and the maximum numeric difference are reported separately.

Arrival gaps wider than the configured fill window do not guarantee singleton
batches: requests that arrive while inference is active can accumulate in the
queue and be grouped when the worker becomes available.

The published Apple M5 Max CPU/FP32 characterization is in
[`benchmark-results/2026-10-02-apple-m5-max-cpu-fp32/`](../benchmark-results/2026-10-02-apple-m5-max-cpu-fp32/).
The corresponding MPS/FP16 characterization and CPU comparison are in
[`benchmark-results/2026-10-02-apple-m5-max-mps-fp16/`](../benchmark-results/2026-10-02-apple-m5-max-mps-fp16/).
