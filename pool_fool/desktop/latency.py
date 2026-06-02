from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class LatencyStats:
    samples_ms: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    last_ms: float = 0.0

    def record(self, ms: float) -> None:
        self.last_ms = ms
        self.samples_ms.append(ms)

    @property
    def avg_ms(self) -> float:
        if not self.samples_ms:
            return 0.0
        return sum(self.samples_ms) / len(self.samples_ms)

    @property
    def p95_ms(self) -> float:
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        idx = int(0.95 * (len(ordered) - 1))
        return ordered[idx]

    def format(self) -> str:
        return f"latency last={self.last_ms:.1f}ms avg={self.avg_ms:.1f}ms p95={self.p95_ms:.1f}ms"


class FrameTimer:
    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def tick(self) -> float:
        now = time.monotonic()
        dt = (now - self._t0) * 1000.0
        self._t0 = now
        return dt
