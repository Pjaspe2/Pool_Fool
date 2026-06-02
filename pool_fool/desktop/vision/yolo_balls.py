from __future__ import annotations

from pool_fool.desktop.vision.balls import DetectedBall
from pool_fool.shared.homography import image_to_table
from pool_fool.shared.play_region import PlayRegion
from pool_fool.shared.table import TableSpec


class YoloBallDetector:
    """
    Optional detector using Ultralytics YOLO (COCO class 32 = sports ball).

    Install: pip install -e ".[yolo]"
  Set vision.detector: yolo in config/default.yaml
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
        self.play_region = play_region
        self._cue_hint_mm = None

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                'YOLO mode requires: pip install -e ".[yolo]"'
            ) from e

        model_name = vision_cfg.get("yolo_model", "yolov8n.pt")
        self._model = YOLO(model_name)
        self._classes = vision_cfg.get("yolo_class_ids", [32])
        self._conf = float(vision_cfg.get("yolo_confidence", 0.35))

    def set_cue_hint(self, center_mm) -> None:
        import numpy as np

        self._cue_hint_mm = center_mm.copy() if center_mm is not None else None

    def detect(self, frame) -> list[DetectedBall]:
        import numpy as np

        results = self._model.predict(
            frame,
            conf=self._conf,
            classes=self._classes,
            verbose=False,
        )
        balls: list[DetectedBall] = []
        if not results:
            return balls
        r0 = results[0]
        if r0.boxes is None:
            return balls

        for box in r0.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            r_px = (x2 - x1 + y2 - y1) / 4.0
            mm = image_to_table(self.H, (cx, cy))
            if self.play_region is not None and not self.play_region.contains(mm):
                continue
            xi, yi = int(cx), int(cy)
            gray_roi = frame[
                max(0, yi - 2) : min(frame.shape[0], yi + 3),
                max(0, xi - 2) : min(frame.shape[1], xi + 3),
            ]
            brightness = float(gray_roi.mean()) if gray_roi.size else 128.0
            balls.append(
                DetectedBall(
                    center_px=(cx, cy),
                    center_mm=mm,
                    radius_px=r_px,
                    is_cue=False,
                    brightness=brightness,
                )
            )

        self._assign_cue(balls)
        return balls

    def _assign_cue(self, balls: list[DetectedBall]) -> None:
        if not balls:
            return
        for b in balls:
            b.is_cue = False
        if self._cue_hint_mm is not None:
            import numpy as np

            idx = int(
                np.argmin([float(np.linalg.norm(b.center_mm - self._cue_hint_mm)) for b in balls])
            )
            balls[idx].is_cue = True
            return
        brightest = max(balls, key=lambda b: b.brightness)
        brightest.is_cue = True

    def split_cue_and_objects(self, balls: list[DetectedBall]):
        cue = next((b for b in balls if b.is_cue), None)
        objects = [b for b in balls if not b.is_cue]
        return cue, objects
