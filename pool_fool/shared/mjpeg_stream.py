from __future__ import annotations

import threading
import time
import urllib.request

import cv2
import numpy as np


class LatestMjpegStream:
    """
    Read MJPEG over HTTP by scanning JPEG SOI/EOI (matches Pi edge server).

    OpenCV VideoCapture often fails on our multipart stream (boundary warnings / freeze).
    """

    def __init__(self, url: str, *, connect_timeout_s: float = 15.0) -> None:
        self.url = url
        self._connect_timeout_s = connect_timeout_s
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._frames_received = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def wait_first_frame(self, timeout_s: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.read() is not None:
                return True
            time.sleep(0.05)
        return False

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def _loop(self) -> None:
        while self._running:
            try:
                self._read_once()
            except Exception:
                time.sleep(0.5)

    def _read_once(self) -> None:
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
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._lock:
                            self._latest = frame
                            self._frames_received += 1


def read_one_mjpeg_frame(url: str, *, timeout_s: float = 20.0) -> np.ndarray:
    """Single-frame grab for calibration snapshot tools."""
    stream = LatestMjpegStream(url)
    stream.start()
    try:
        if not stream.wait_first_frame(timeout_s=timeout_s):
            raise TimeoutError(f"No frame from stream within {timeout_s}s: {url}")
        frame = stream.read()
        if frame is None:
            raise RuntimeError(f"Stream opened but no frame: {url}")
        return frame
    finally:
        stream.stop()
