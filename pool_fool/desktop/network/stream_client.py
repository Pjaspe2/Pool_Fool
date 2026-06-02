from __future__ import annotations

import threading
import time
import urllib.request
from typing import Callable

import cv2
import numpy as np


class MjpegStreamClient:
    """Pull MJPEG from Pi edge (http://host:8080/stream.mjpg)."""

    def __init__(
        self,
        url: str,
        on_frame: Callable[[np.ndarray], None],
        *,
        connect_timeout_s: float = 15.0,
    ) -> None:
        self.url = url
        self._on_frame = on_frame
        self._connect_timeout_s = connect_timeout_s
        self._running = False
        self._thread: threading.Thread | None = None
        self._latency_ms: float = 0.0
        self._frames_received = 0
        self._last_error: str | None = None

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

    @property
    def frames_received(self) -> int:
        return self._frames_received

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _loop(self) -> None:
        while self._running:
            try:
                print(f"Connecting to MJPEG stream: {self.url}")
                self._last_error = None
                self._read_stream()
            except Exception as e:
                self._last_error = str(e)
                print(f"Stream error ({self.url}): {e} — retrying in 0.5s")
                time.sleep(0.5)

    def _read_stream(self) -> None:
        req = urllib.request.Request(self.url, headers={"User-Agent": "pool-fool"})
        with urllib.request.urlopen(req, timeout=self._connect_timeout_s) as resp:
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
                        self._frames_received += 1
                        if self._frames_received == 1:
                            print(
                                f"Stream connected ({frame.shape[1]}x{frame.shape[0]}). "
                                "Look for the pool_fool_debug window."
                            )
                        self._on_frame(frame)
