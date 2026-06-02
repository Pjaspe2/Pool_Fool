from __future__ import annotations

import copy

import numpy as np

from pool_fool.desktop.vision.balls import DetectedBall
from pool_fool.desktop.vision.yolo_support import (
    cluster_detections,
    expected_ball_radius_px,
    filter_ball_sizes,
    preprocess_for_yolo,
)
from pool_fool.shared.homography import image_to_table
from pool_fool.shared.play_region import PlayRegion
from pool_fool.shared.table import TableSpec


class YoloBallDetector:
    """
    Ball detection via Ultralytics YOLO (COCO class 32 = sports ball).

    Install: pip install -e ".[yolo]"
    """

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

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError('YOLO requires: pip install -e ".[yolo]"') from e

        model_name = vision_cfg.get("yolo_model", "yolov8n.pt")
        self._model = YOLO(model_name)
        self._classes = vision_cfg.get("yolo_class_ids", [32])
        self._conf = float(vision_cfg.get("yolo_confidence", 0.25))
        self._imgsz = int(vision_cfg.get("yolo_imgsz", 640))
        self._iou = float(vision_cfg.get("yolo_iou", 0.55))
        self._max_det = int(vision_cfg.get("yolo_max_det", 16))
        self._stride = max(1, int(vision_cfg.get("yolo_frame_stride", 2)))
        self._clahe = bool(vision_cfg.get("yolo_clahe", True))
        self._merge_mm = float(vision_cfg.get("yolo_cluster_merge_mm", 55.0))
        self._min_white = float(vision_cfg.get("yolo_cue_min_brightness", 140.0))
        self._frame_i = 0
        self._cached: list[DetectedBall] = []

    def set_cue_hint(self, center_mm: np.ndarray | None) -> None:
        self._cue_hint_mm = center_mm.copy() if center_mm is not None else None

    def detect(self, frame: np.ndarray) -> list[DetectedBall]:
        self._frame_i += 1
        if self._stride > 1 and self._frame_i % self._stride != 0 and self._cached:
            return copy.deepcopy(self._cached)

        infer = preprocess_for_yolo(frame, clahe=self._clahe)
        results = self._model.predict(
            infer,
            conf=self._conf,
            iou=self._iou,
            max_det=self._max_det,
            classes=self._classes,
            imgsz=self._imgsz,
            verbose=False,
        )
        balls: list[DetectedBall] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                r_px = (x2 - x1 + y2 - y1) / 4.0
                mm = image_to_table(self.H, (cx, cy))
                if self.play_region is not None and not self.play_region.contains(mm):
                    continue
                xi, yi = int(cx), int(cy)
                gray_roi = frame[
                    max(0, yi - 4) : min(frame.shape[0], yi + 5),
                    max(0, xi - 4) : min(frame.shape[1], xi + 5),
                ]
                brightness = float(gray_roi.mean()) if gray_roi.size else 128.0
                balls.append(
                    DetectedBall(
                        center_px=(cx, cy),
                        center_mm=mm,
                        radius_px=r_px,
                        is_cue=False,
                        brightness=brightness,
                        bbox_px=(x1, y1, x2, y2),
                    )
                )

        balls = filter_ball_sizes(balls, H_inv=self.H_inv, table=self.table)
        balls = cluster_detections(balls, self._merge_mm)
        self._assign_cue(balls)
        self._cached = copy.deepcopy(balls)
        return balls

    def _assign_cue(self, balls: list[DetectedBall]) -> None:
        if not balls:
            return
        for b in balls:
            b.is_cue = False
        if self._cue_hint_mm is not None:
            idx = int(
                np.argmin([float(np.linalg.norm(b.center_mm - self._cue_hint_mm)) for b in balls])
            )
            balls[idx].is_cue = True
            return
        # White cue on red felt: prefer brightest detection above threshold
        bright = [b for b in balls if b.brightness >= self._min_white]
        pool = bright if bright else balls
        cue = max(pool, key=lambda b: b.brightness)
        cue.is_cue = True

    def split_cue_and_objects(self, balls: list[DetectedBall]):
        cue = next((b for b in balls if b.is_cue), None)
        objects = [b for b in balls if not b.is_cue]
        return cue, objects
