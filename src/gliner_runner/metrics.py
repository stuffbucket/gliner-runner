from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Metrics:
    _counters: Counter[str] = field(default_factory=Counter)
    _gauges: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {**self._counters, **self._gauges}

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP gliner_runner_internal Runtime scheduler metrics.",
            "# TYPE gliner_runner_internal gauge",
        ]
        lines.extend(
            f'gliner_runner_internal{{metric="{name}"}} {value}'
            for name, value in sorted(snapshot.items())
        )
        return "\n".join(lines) + "\n"
