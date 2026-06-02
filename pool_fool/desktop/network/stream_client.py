from __future__ import annotations

import threading
import time
import urllib.request
from typing import Callable

import cv2
import numpy as np


class MjpegStreamClient:
    """Pull MJPEG from Pi edge (http://host:8080/stream.mjpg)."""

    def __init__(self, url: str, on_frame: Callable[[np.ndarray], None]) -> None:
        self.url = url
        self._on_frame = on_frame
        self._running = False
        self._thread: threading.Thread | None = None
        self._latency_ms: float = 0.0

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            try:
                self._read_stream()
            except Exception:
                time.sleep(0.5)

    def _read_stream(self) -> None:
        req = urllib.request.Request(self.url, headers={"User-Agent": "pool-fool"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            buf = b""
            while self._running:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2)
                    if start < 0 or end < 0:
                        break
                    jpg = buf[start : end + 2]
                    buf = buf[end + 2 :]
                    t0 = time.monotonic()
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._latency_ms = (time.monotonic() - t0) * 1000.0
                        self._on_frame(frame)
