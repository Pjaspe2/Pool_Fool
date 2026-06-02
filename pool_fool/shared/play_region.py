from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pool_fool.shared.homography import image_to_table, table_to_image


class PlayRegion:
    """Polygon on the table plane (mm). Detection ignores everything outside."""

    def __init__(self, corners_mm: np.ndarray) -> None:
        if corners_mm.shape != (4, 2):
            raise ValueError("PlayRegion requires 4 corners as (4, 2) mm")
        self.corners_mm = corners_mm.astype(np.float64)

    @classmethod
    def load(cls, path: Path) -> PlayRegion | None:
        if not path.exists():
            return None
        data = np.load(path)
        return cls(data["corners_mm"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, corners_mm=self.corners_mm)

    def contains(self, point_mm: np.ndarray) -> bool:
        poly = self.corners_mm.astype(np.float32)
        pt = (float(point_mm[0]), float(point_mm[1]))
        return cv2.pointPolygonTest(poly, pt, False) >= 0

    def pixel_mask(self, frame_shape: tuple[int, int], H_inv: np.ndarray) -> np.ndarray:
        h, w = frame_shape[:2]
        pts = np.array(
            [table_to_image(H_inv, c) for c in self.corners_mm],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        return mask

    def draw(self, frame: np.ndarray, H_inv: np.ndarray, *, color=(255, 180, 0), thickness: int = 2) -> None:
        pts = np.array(
            [table_to_image(H_inv, c) for c in self.corners_mm],
            dtype=np.int32,
        )
        cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)

    @classmethod
    def from_image_clicks(
        cls,
        image_corners: list[tuple[float, float]],
        H: np.ndarray,
    ) -> PlayRegion:
        mm = [image_to_table(H, pt) for pt in image_corners]
        return cls(np.array(mm, dtype=np.float64))
