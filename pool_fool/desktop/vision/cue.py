from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pool_fool.shared.homography import image_to_table
from pool_fool.shared.play_region import PlayRegion


@dataclass
class CueLine:
    """Unit aim direction in table mm (from cue ball toward tip)."""

    direction_mm: np.ndarray
    confidence: float
    tip_mm: np.ndarray | None = None


class CueDetector:
    def __init__(
        self,
        vision_cfg: dict,
        H: np.ndarray,
        play_region: PlayRegion | None = None,
    ) -> None:
        self.cfg = vision_cfg
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.play_region = play_region
        self._last_direction: np.ndarray | None = None

    def detect(
        self,
        frame: np.ndarray,
        cue_center_mm: np.ndarray,
        *,
        roi_radius_mm: float = 400.0,
        mm_per_px: float = 2.0,
    ) -> CueLine | None:
        if self.play_region is not None and not self.play_region.contains(cue_center_mm):
            return self._fallback()

        cue_px = cv2.perspectiveTransform(
            np.array([[cue_center_mm]], dtype=np.float32).reshape(1, 1, 2), self.H_inv
        ).reshape(2)
        r_px = int(roi_radius_mm / max(mm_per_px, 0.5))
        cx, cy = int(cue_px[0]), int(cue_px[1])
        h, w = frame.shape[:2]
        x0, x1 = max(0, cx - r_px), min(w, cx + r_px)
        y0, y1 = max(0, cy - r_px), min(h, cy + r_px)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return self._fallback()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.play_region is not None:
            full_mask = self.play_region.pixel_mask(frame.shape[:2], self.H_inv)
            roi_mask = full_mask[y0:y1, x0:x1]
            gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=int(self.cfg.get("cue_hough_threshold", 60)),
            minLineLength=int(self.cfg.get("cue_min_line_length_px", 80)),
            maxLineGap=int(self.cfg.get("cue_max_line_gap_px", 15)),
        )
        if lines is None:
            return self._fallback()

        best = None
        best_score = 0.0
        cue_px_roi = (cx - x0, cy - y0)

        for line in lines:
            x1l, y1l, x2l, y2l = line[0]
            dx, dy = x2l - x1l, y2l - y1l
            length = float(np.hypot(dx, dy))
            if length < 40:
                continue
            # Distance from cue to line segment
            dist = self._point_line_dist(cue_px_roi, (x1l, y1l), (x2l, y2l))
            if dist > 80:
                continue
            score = length / (1.0 + dist)
            if score > best_score:
                best_score = score
                best = (x1l + x0, y1l + y0, x2l + x0, y2l + y0)

        if best is None:
            return self._fallback()

        x1g, y1g, x2g, y2g = best
        p1 = image_to_table(self.H, (x1g, y1g))
        p2 = image_to_table(self.H, (x2g, y2g))
        # Direction from point farther from cue ball center
        d1 = float(np.linalg.norm(p1 - cue_center_mm))
        d2 = float(np.linalg.norm(p2 - cue_center_mm))
        tip = p1 if d1 < d2 else p2
        direction = tip - cue_center_mm
        if self.play_region is not None and not self.play_region.contains(tip):
            return self._fallback()
        n = float(np.linalg.norm(direction))
        if n < 1e-6:
            return self._fallback()
        direction = direction / n
        self._last_direction = direction
        return CueLine(direction_mm=direction, confidence=min(1.0, best_score / 200.0), tip_mm=tip)

    def _fallback(self) -> CueLine | None:
        if self._last_direction is None:
            return None
        return CueLine(
            direction_mm=self._last_direction.copy(),
            confidence=0.2,
            tip_mm=None,
        )

    @staticmethod
    def _point_line_dist(
        p: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        px, py = p
        ax, ay = a
        bx, by = b
        abx, aby = bx - ax, by - ay
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby + 1e-9)))
        cx, cy = ax + t * abx, ay + t * aby
        return float(np.hypot(px - cx, py - cy))
