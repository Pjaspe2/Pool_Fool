from __future__ import annotations

import cv2
import numpy as np


class ScreenWarp:
    """Perspective-warp a skewed screen quad to a square rectangle for detection."""

    def __init__(self, out_width: int = 900, out_height: int = 700) -> None:
        self.out_size = (out_width, out_height)
        self.corners: list[tuple[int, int]] | None = None
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None

    @property
    def is_active(self) -> bool:
        return self._H is not None

    def set_corners(self, corners: list[tuple[int, int]]) -> None:
        if len(corners) != 4:
            self.corners = None
            self._H = None
            self._H_inv = None
            return
        self.corners = corners
        src = np.array(corners, dtype=np.float32)
        w, h = self.out_size
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        self._H = cv2.getPerspectiveTransform(src, dst)
        self._H_inv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, frame: np.ndarray) -> np.ndarray | None:
        if self._H is None:
            return None
        return cv2.warpPerspective(frame, self._H, self.out_size)

    def map_points_to_original(self, pts: np.ndarray) -> np.ndarray:
        """pts Nx1x2 in warped image -> original camera image."""
        if self._H_inv is None:
            return pts
        return cv2.perspectiveTransform(pts.astype(np.float32), self._H_inv)

    def draw_quad(self, frame: np.ndarray) -> None:
        if self.corners and len(self.corners) >= 2:
            pts = np.array(self.corners, dtype=np.int32)
            cv2.polylines(frame, [pts], len(self.corners) == 4, (255, 200, 0), 2)
            for i, (x, y) in enumerate(self.corners):
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(frame, str(i + 1), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
