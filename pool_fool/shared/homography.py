from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def image_to_table(H: np.ndarray, pt: tuple[float, float] | np.ndarray) -> np.ndarray:
    p = np.array([[float(pt[0]), float(pt[1])]], dtype=np.float32)
    out = cv2.perspectiveTransform(p.reshape(1, 1, 2), H)
    return out.reshape(2).astype(np.float64)


def table_to_image(H_inv: np.ndarray, pt: np.ndarray) -> tuple[int, int]:
    p = np.array([[float(pt[0]), float(pt[1])]], dtype=np.float32)
    out = cv2.perspectiveTransform(p.reshape(1, 1, 2), H_inv)
    x, y = out.reshape(2)
    return int(round(x)), int(round(y))


def save_homography(
    path: Path,
    H: np.ndarray,
    *,
    kind: str = "table",
    region: str = "full",
    dst_corners_mm: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra: dict = {"region": region}
    if dst_corners_mm is not None:
        extra["dst_corners_mm"] = dst_corners_mm.astype(np.float64)
    np.savez(path, H=H, kind=kind, **extra)


def load_homography(path: Path) -> np.ndarray:
    data = np.load(path)
    return data["H"].astype(np.float64)


def compute_table_homography(
    image_corners: list[tuple[float, float]],
    table_corners_mm: list[tuple[float, float]],
) -> np.ndarray:
    """Map image pixels to table mm (length along x, width along y)."""
    src = np.array(image_corners, dtype=np.float32)
    dst = np.array(table_corners_mm, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("Homography solve failed")
    return H.astype(np.float64)


def default_table_corners_mm(length_mm: float, width_mm: float) -> list[tuple[float, float]]:
    return [
        (0.0, 0.0),
        (length_mm, 0.0),
        (length_mm, width_mm),
        (0.0, width_mm),
    ]
