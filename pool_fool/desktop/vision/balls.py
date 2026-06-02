from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pool_fool.desktop.vision.felt import build_felt_mask
from pool_fool.shared.homography import image_to_table
from pool_fool.shared.play_region import PlayRegion
from pool_fool.shared.table import TableSpec


@dataclass
class DetectedBall:
    center_px: tuple[float, float]
    center_mm: np.ndarray
    radius_px: float
    is_cue: bool
    brightness: float
    track_id: int = -1
    # Set by YOLO only — used to draw boxes so you can tell YOLO vs Hough circles
    bbox_px: tuple[float, float, float, float] | None = None


class BallDetector:
    def __init__(
        self,
        table: TableSpec,
        vision_cfg: dict,
        H: np.ndarray,
        play_region: PlayRegion | None = None,
    ) -> None:
        self.table = table
        self.cfg = vision_cfg
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.play_region = play_region
        self._cue_hint_mm: np.ndarray | None = None
        self._play_mask: np.ndarray | None = None
        self._mask_shape: tuple[int, int] | None = None

    def _play_mask_for(self, frame: np.ndarray) -> np.ndarray | None:
        if self.play_region is None:
            return None
        shape = frame.shape[:2]
        if self._play_mask is None or self._mask_shape != shape:
            self._play_mask = self.play_region.pixel_mask(shape, self.H_inv)
            self._mask_shape = shape
        return self._play_mask

    def set_cue_hint(self, center_mm: np.ndarray | None) -> None:
        self._cue_hint_mm = center_mm.copy() if center_mm is not None else None

    def _felt_mask(self, hsv: np.ndarray) -> np.ndarray:
        return build_felt_mask(hsv, self.cfg)

    def detect(self, frame: np.ndarray) -> list[DetectedBall]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        felt = self._felt_mask(hsv)
        not_felt = cv2.bitwise_not(felt)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.bitwise_and(gray, gray, mask=not_felt)
        play_mask = self._play_mask_for(frame)
        if play_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=play_mask)

        min_r = int(self.cfg.get("min_ball_radius_px", 8))
        max_r = int(self.cfg.get("max_ball_radius_px", 40))
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(self.cfg.get("hough_dp", 1.2)),
            minDist=float(self.cfg.get("hough_min_dist_px", 25)),
            param1=float(self.cfg.get("hough_param1", 80)),
            param2=float(self.cfg.get("hough_param2", 28)),
            minRadius=min_r,
            maxRadius=max_r,
        )

        balls: list[DetectedBall] = []
        if circles is None:
            return balls

        min_bright = float(self.cfg.get("cue_ball_min_brightness", 180))
        for c in circles[0]:
            x, y, r = float(c[0]), float(c[1]), float(c[2])
            xi, yi = int(x), int(y)
            if xi < 0 or yi < 0 or xi >= frame.shape[1] or yi >= frame.shape[0]:
                continue
            roi = gray[
                max(0, yi - 3) : min(gray.shape[0], yi + 4),
                max(0, xi - 3) : min(gray.shape[1], xi + 4),
            ]
            brightness = float(np.mean(roi)) if roi.size else 0.0
            mm = image_to_table(self.H, (x, y))
            if self.play_region is not None and not self.play_region.contains(mm):
                continue
            balls.append(
                DetectedBall(
                    center_px=(x, y),
                    center_mm=mm,
                    radius_px=r,
                    is_cue=brightness >= min_bright,
                    brightness=brightness,
                )
            )

        self._assign_cue_ball(balls)
        return balls

    def _assign_cue_ball(self, balls: list[DetectedBall]) -> None:
        if not balls:
            return
        for b in balls:
            b.is_cue = False

        if self._cue_hint_mm is not None:
            dists = [float(np.linalg.norm(b.center_mm - self._cue_hint_mm)) for b in balls]
            idx = int(np.argmin(dists))
            balls[idx].is_cue = True
            return

        # Brightest ball as cue
        brightest = max(balls, key=lambda b: b.brightness)
        brightest.is_cue = True

    def split_cue_and_objects(self, balls: list[DetectedBall]) -> tuple[DetectedBall | None, list[DetectedBall]]:
        cue = next((b for b in balls if b.is_cue), None)
        objects = [b for b in balls if not b.is_cue]
        return cue, objects
