from __future__ import annotations

from typing import Protocol

import numpy as np

from pool_fool.desktop.vision.balls import DetectedBall
from pool_fool.shared.play_region import PlayRegion
from pool_fool.shared.table import TableSpec


class BallDetectorProtocol(Protocol):
    def detect(self, frame: np.ndarray) -> list[DetectedBall]: ...
    def set_cue_hint(self, center_mm: np.ndarray | None) -> None: ...
    def split_cue_and_objects(
        self, balls: list[DetectedBall]
    ) -> tuple[DetectedBall | None, list[DetectedBall]]: ...


def detector_mode_label(vision_cfg: dict) -> str:
    model = vision_cfg.get("yolo_model", "yolov8n.pt")
    conf = vision_cfg.get("yolo_confidence", 0.25)
    classes = vision_cfg.get("yolo_class_ids", [32])
    src = "COCO-32" if classes == [32] else "custom"
    return f"YOLO {model} conf={conf} ({src})"


def create_ball_detector(
    vision_cfg: dict,
    table: TableSpec,
    H: np.ndarray,
    play_region: PlayRegion | None,
) -> BallDetectorProtocol:
    from pool_fool.desktop.vision.yolo_balls import YoloBallDetector

    return YoloBallDetector(table, vision_cfg, H, play_region)
