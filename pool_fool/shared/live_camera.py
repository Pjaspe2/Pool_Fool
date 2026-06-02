from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pool_fool.shared.camera import open_camera, read_frame_with_warmup


class LiveCamera:
    """Keep one VideoCapture open (required on macOS — do not reopen every frame)."""

    def __init__(self, index: int, cam_cfg: dict[str, Any]) -> None:
        self.index = index
        self.cap = open_camera(index, cam_cfg)

    def read(self) -> np.ndarray | None:
        ret, frame = read_frame_with_warmup(self.cap, attempts=3, delay_s=0.01)
        return frame if ret else None

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> LiveCamera:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
