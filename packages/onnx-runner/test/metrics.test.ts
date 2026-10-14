import assert from "node:assert/strict";
import test from "node:test";

import { Metrics } from "../src/index.js";

const HEADER = "# HELP gliner_runner_internal Runtime scheduler metrics.\n# TYPE gliner_runner_internal gauge\n";

test("an untouched registry renders only the family header", () => {
  assert.equal(new Metrics().prometheus(), HEADER);
});

test("counters render as integers, gauges as floats, sorted by name", () => {
  const metrics = new Metrics();
  metrics.increment("requests_completed_total", 2);
  metrics.increment("batches_total");
  metrics.increment("batches_total");
  metrics.gauge("queue_depth", 0);
  metrics.gauge("last_batch_size", 1);
  metrics.gauge("fractional", 0.25);
  assert.equal(
    metrics.prometheus(),
    `${HEADER}` +
      'gliner_runner_internal{metric="batches_total"} 2\n' +
      'gliner_runner_internal{metric="fractional"} 0.25\n' +
      'gliner_runner_internal{metric="last_batch_size"} 1.0\n' +
      'gliner_runner_internal{metric="queue_depth"} 0.0\n' +
      'gliner_runner_internal{metric="requests_completed_total"} 2\n',
  );
  assert.deepEqual(metrics.snapshot(), {
    requests_completed_total: 2,
    batches_total: 2,
    queue_depth: 0,
    last_batch_size: 1,
    fractional: 0.25,
  });
});
