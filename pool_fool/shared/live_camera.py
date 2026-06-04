from __future__ import annotations

from typing import Any

import numpy as np

from pool_fool.shared.camera import (
    CameraOpenError,
    is_stream_url,
    open_camera,
    read_frame_with_warmup,
)
from pool_fool.shared.mjpeg_stream import LatestMjpegStream


class LiveCamera:
    """Keep one capture open (macOS webcam or Pi MJPEG URL)."""

    def __init__(self, source: str | int, cam_cfg: dict[str, Any]) -> None:
        self.source = source
        self._stream: LatestMjpegStream | None = None
        self.cap = None
        if is_stream_url(source):
            self._stream = LatestMjpegStream(str(source))
            self._stream.start()
            if not self._stream.wait_first_frame(timeout_s=20.0):
                self._stream.stop()
                raise CameraOpenError(
                    f"Cannot read frames from stream: {source}",
                    index=-1,
                    hints=[
                        "Close browser tabs on /stream.mjpg.",
                        "Ensure pool-fool-edge is running on the Pi.",
                    ],
                )
        else:
            self.cap = open_camera(int(source), cam_cfg)

    def read(self) -> np.ndarray | None:
        if self._stream is not None:
            return self._stream.read()
        if self.cap is None:
            return None
        ret, frame = read_frame_with_warmup(self.cap, attempts=3, delay_s=0.01)
        return frame if ret else None

    def release(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> LiveCamera:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
