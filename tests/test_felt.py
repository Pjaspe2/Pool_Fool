import cv2
import numpy as np

from pool_fool.desktop.vision.felt import build_felt_mask


def test_red_felt_mask():
    cfg = {
        "felt_color": "red",
        "felt_hsv_red": {
            "ranges": [
                {"low": [0, 80, 50], "high": [12, 255, 255]},
                {"low": [168, 80, 50], "high": [180, 255, 255]},
            ]
        },
    }
    hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    hsv[:, :] = (5, 200, 200)
    mask = build_felt_mask(hsv, cfg)
    assert mask.mean() > 200
