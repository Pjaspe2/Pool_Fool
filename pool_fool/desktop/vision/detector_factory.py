from __future__ import annotations

from typing import Protocol

import numpy as np

from pool_fool.desktop.vision.balls import BallDetector, DetectedBall
from pool_fool.shared.play_region import PlayRegion
from pool_fool.shared.table import TableSpec


class BallDetectorProtocol(Protocol):
    def detect(self, frame: np.ndarray) -> list[DetectedBall]: ...
    def set_cue_hint(self, center_mm: np.ndarray | None) -> None: ...
    def split_cue_and_objects(
        self, balls: list[DetectedBall]
    ) -> tuple[DetectedBall | None, list[DetectedBall]]: ...


def create_ball_detector(
    vision_cfg: dict,
    table: TableSpec,
    H: np.ndarray,
    play_region: PlayRegion | None,
) -> BallDetectorProtocol:
    mode = str(vision_cfg.get("detector", "classical")).lower()
    if mode == "yolo":
        from pool_fool.desktop.vision.yolo_balls import YoloBallDetector

        return YoloBallDetector(table, vision_cfg, H, play_region)
    return BallDetector(table, vision_cfg, H, play_region)
