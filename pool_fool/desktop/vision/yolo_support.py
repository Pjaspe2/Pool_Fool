from __future__ import annotations

import cv2
import numpy as np

from pool_fool.desktop.vision.balls import DetectedBall
from pool_fool.shared.homography import table_to_image
from pool_fool.shared.table import TableSpec


def preprocess_for_yolo(frame: np.ndarray, *, clahe: bool = True) -> np.ndarray:
    """Boost contrast on overhead pool shots (red felt, white balls)."""
    if not clahe:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = cl.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def expected_ball_radius_px(
    H_inv: np.ndarray, center_mm: np.ndarray, ball_radius_mm: float
) -> float:
    c = np.array(center_mm, dtype=np.float64)
    edge = c + np.array([ball_radius_mm, 0.0])
    p0 = table_to_image(H_inv, c)
    p1 = table_to_image(H_inv, edge)
    return float(np.hypot(p1[0] - p0[0], p1[1] - p1[1]))


def cluster_detections(balls: list[DetectedBall], merge_mm: float) -> list[DetectedBall]:
    """Merge duplicate YOLO boxes that jitter around one physical ball."""
    if len(balls) <= 1:
        return balls
    merged: list[DetectedBall] = []
    used = [False] * len(balls)
    for i, bi in enumerate(balls):
        if used[i]:
            continue
        group = [bi]
        used[i] = True
        for j in range(i + 1, len(balls)):
            if used[j]:
                continue
            d = float(np.linalg.norm(balls[j].center_mm - bi.center_mm))
            if d <= merge_mm:
                group.append(balls[j])
                used[j] = True
        if len(group) == 1:
            merged.append(group[0])
            continue
        cx = float(np.mean([b.center_px[0] for b in group]))
        cy = float(np.mean([b.center_px[1] for b in group]))
        mm = np.mean([b.center_mm for b in group], axis=0)
        r_px = float(np.mean([b.radius_px for b in group]))
        bright = float(np.mean([b.brightness for b in group]))
        x1 = float(np.mean([b.bbox_px[0] for b in group if b.bbox_px]))
        y1 = float(np.mean([b.bbox_px[1] for b in group if b.bbox_px]))
        x2 = float(np.mean([b.bbox_px[2] for b in group if b.bbox_px]))
        y2 = float(np.mean([b.bbox_px[3] for b in group if b.bbox_px]))
        merged.append(
            DetectedBall(
                center_px=(cx, cy),
                center_mm=mm,
                radius_px=r_px,
                is_cue=False,
                brightness=bright,
                bbox_px=(x1, y1, x2, y2) if group[0].bbox_px else None,
            )
        )
    return merged


def _resolve_allow_class_ids(vision_cfg: dict, model) -> set[int] | None:
    """
    If yolo_allow_class_names is set, only those YOLO classes become detections.
    Stops pockets/stick/wood from being forced into ball classes when using yolo_class_ids: all.
    """
    allow = vision_cfg.get("yolo_allow_class_names")
    if not allow:
        return None
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        name_map = {int(k): str(v).lower() for k, v in names.items()}
    else:
        name_map = {i: str(n).lower() for i, n in enumerate(names)}
    ids: set[int] = set()
    for sub in allow:
        sub = str(sub).lower()
        for cid, cname in name_map.items():
            if sub in cname or cname.replace("_", " ") == sub.replace("_", " "):
                ids.add(cid)
    return ids if ids else None


def filter_stick_like_boxes(
    balls: list[DetectedBall],
    frame: np.ndarray,
    vision_cfg: dict,
) -> list[DetectedBall]:
    """Drop elongated boxes and brown wood (cue shaft) mistaken for balls."""
    min_aspect = float(vision_cfg.get("yolo_min_bbox_aspect", 0.72))
    reject_wood = bool(vision_cfg.get("yolo_reject_wood_hue", True))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if reject_wood else None
    out: list[DetectedBall] = []
    for b in balls:
        if b.bbox_px is None:
            out.append(b)
            continue
        x1, y1, x2, y2 = b.bbox_px
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        aspect = min(w, h) / max(w, h)
        if aspect < min_aspect:
            continue
        if reject_wood and hsv is not None and not b.is_cue:
            xi = int((x1 + x2) / 2)
            yi = int((y1 + y2) / 2)
            pad = 6
            roi = hsv[
                max(0, yi - pad) : min(hsv.shape[0], yi + pad),
                max(0, xi - pad) : min(hsv.shape[1], xi + pad),
            ]
            if roi.size:
                h_med = float(np.median(roi[:, :, 0]))
                s_med = float(np.median(roi[:, :, 1]))
                v_med = float(np.median(roi[:, :, 2]))
                # Dark brown cue on red felt (not white/yellow balls)
                if v_med < 120 and 8 < h_med < 35 and s_med > 40:
                    continue
        out.append(b)
    return out


def filter_ball_sizes(
    balls: list[DetectedBall],
    *,
    H_inv: np.ndarray,
    table: TableSpec,
    min_scale: float = 0.45,
    max_scale: float = 2.2,
    min_aspect: float = 0.72,
) -> list[DetectedBall]:
    """Drop detections far from expected pool-ball diameter in image space."""
    out: list[DetectedBall] = []
    for b in balls:
        if b.bbox_px is None:
            continue
        x1, y1, x2, y2 = b.bbox_px
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        aspect = min(w, h) / max(w, h)
        if aspect < min_aspect:
            continue
        exp_r = expected_ball_radius_px(H_inv, b.center_mm, table.ball_radius_mm)
        if exp_r < 3:
            out.append(b)
            continue
        if min_scale * exp_r <= b.radius_px <= max_scale * exp_r:
            out.append(b)
    return out
