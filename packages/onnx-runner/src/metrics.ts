/** Process-wide scheduler metrics rendered as one Prometheus gauge family. */
export class Metrics {
  readonly #counters = new Map<string, number>();
  readonly #gauges = new Map<string, number>();

  increment(name: string, amount = 1): void {
    this.#counters.set(name, (this.#counters.get(name) ?? 0) + amount);
  }

  gauge(name: string, value: number): void {
    this.#gauges.set(name, value);
  }

  /** Counters render as integers and gauges as floats (`1.0`), mirroring the Python runner. */
  snapshot(): Record<string, number> {
    return Object.fromEntries([...this.#counters, ...this.#gauges]);
  }

  prometheus(): string {
    const lines = [
      "# HELP gliner_runner_internal Runtime scheduler metrics.",
      "# TYPE gliner_runner_internal gauge",
    ];
    const entries = [
      ...[...this.#counters].map(([name, value]) => [name, String(value)] as const),
      ...[...this.#gauges].map(([name, value]) => [name, formatFloat(value)] as const),
    ];
    entries.sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0));
    for (const [name, value] of entries) {
      lines.push(`gliner_runner_internal{metric="${name}"} ${value}`);
    }
    return `${lines.join("\n")}\n`;
  }
}

function formatFloat(value: number): string {
  return Number.isInteger(value) ? value.toFixed(1) : String(value);
}
