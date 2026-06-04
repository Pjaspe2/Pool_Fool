import numpy as np

from pool_fool.desktop.vision.balls import DetectedBall
from pool_fool.desktop.vision.yolo_support import filter_stick_like_boxes


def test_reject_elongated_bbox():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    balls = [
        DetectedBall(
            center_px=(100, 100),
            center_mm=np.array([0.0, 0.0]),
            radius_px=10,
            is_cue=False,
            brightness=200,
            bbox_px=(10, 90, 190, 110),
        )
    ]
    out = filter_stick_like_boxes(balls, frame, {"yolo_min_bbox_aspect": 0.72})
    assert len(out) == 0


def test_keep_round_bbox():
    frame = np.full((200, 200, 3), 40, dtype=np.uint8)
    balls = [
        DetectedBall(
            center_px=(100, 100),
            center_mm=np.array([0.0, 0.0]),
            radius_px=12,
            is_cue=False,
            brightness=220,
            bbox_px=(88, 88, 112, 112),
        )
    ]
    out = filter_stick_like_boxes(balls, frame, {"yolo_reject_wood_hue": False})
    assert len(out) == 1
