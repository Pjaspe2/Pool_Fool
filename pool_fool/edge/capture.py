from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from pool_fool.shared.camera import configure_capture


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp_ms: int
    camera_id: int


class CameraCapture:
    def __init__(
        self,
        device_index: int,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        manual_exposure: bool = True,
        exposure: float = -6,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.manual_exposure = manual_exposure
        self.exposure = exposure
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            return False
        configure_capture(
            self._cap,
            {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "manual_exposure": self.manual_exposure,
                "exposure": self.exposure,
            },
        )
        return True

    def read(self) -> FramePacket | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return FramePacket(frame=frame, timestamp_ms=int(time.time() * 1000), camera_id=self.device_index)

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None


class DualCameraCapture:
    """Overhead + side USB cameras on Pi."""

    def __init__(self, overhead: CameraCapture, side: CameraCapture | None = None) -> None:
        self.overhead = overhead
        self.side = side

    def open(self) -> bool:
        if not self.overhead.open():
            return False
        if self.side and not self.side.open():
            self.overhead.release()
            return False
        return True

    def read_overhead(self) -> FramePacket | None:
        return self.overhead.read()

    def read_side(self) -> FramePacket | None:
        if self.side is None:
            return None
        return self.side.read()

    def release(self) -> None:
        self.overhead.release()
        if self.side:
            self.side.release()
