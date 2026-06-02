from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def build_felt_mask(hsv: np.ndarray, vision_cfg: dict[str, Any]) -> np.ndarray:
    """
    Mask of playing-surface felt (255 = felt).

    Config:
      felt_color: red | green | blue (default red if felt_hsv_red present)
      felt_hsv_<color>: { ranges: [ {low: [H,S,V], high: [...]}, ... ] }
      Legacy: felt_hsv_green / felt_hsv_blue single low/high
    """
    color = vision_cfg.get("felt_color", "red").lower()
    key = f"felt_hsv_{color}"
    spec = vision_cfg.get(key)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    if spec and "ranges" in spec:
        for band in spec["ranges"]:
            lo = np.array(band["low"], dtype=np.uint8)
            hi = np.array(band["high"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    elif spec and "low" in spec:
        lo = np.array(spec["low"], dtype=np.uint8)
        hi = np.array(spec["high"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
    else:
        # Legacy green + blue
        for legacy in ("felt_hsv_green", "felt_hsv_blue"):
            leg = vision_cfg.get(legacy)
            if not leg:
                continue
            lo = np.array(leg["low"], dtype=np.uint8)
            hi = np.array(leg["high"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
