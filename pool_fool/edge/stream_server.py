from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import IO


class FfmpegRtspPublisher:
    """
    Publish raw BGR frames to RTSP via ffmpeg (run on Pi).

    Requires ffmpeg installed on the edge device.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        *,
        rtsp_url: str = "rtsp://0.0.0.0:8554/pool",
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.rtsp_url = rtsp_url
        self._proc: subprocess.Popen[bytes] | None = None
        self._stdin: IO[bytes] | None = None

    def start(self) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-f",
            "rtsp",
            self.rtsp_url,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._stdin = self._proc.stdin

    def write_frame(self, frame) -> None:
        if self._stdin is None:
            return
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return
        try:
            self._stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def stop(self) -> None:
        if self._stdin:
            self._stdin.close()
        if self._proc:
            self._proc.wait(timeout=3)
        self._proc = None
        self._stdin = None


class MjpegHttpServer:
    """Lightweight MJPEG over HTTP when ffmpeg RTSP is unavailable."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self._latest: bytes | None = None
        self._lock = threading.Lock()

    def update_jpeg(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._latest = jpeg_bytes

    def get_latest(self) -> bytes | None:
        with self._lock:
            return self._latest
