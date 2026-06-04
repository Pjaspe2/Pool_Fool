from __future__ import annotations

import cv2
import numpy as np

from pool_fool.shared.homography import table_to_image
from pool_fool.shared.table_layout import TableLayout


def draw_table_layout(
    frame: np.ndarray,
    H_inv: np.ndarray,
    layout: TableLayout,
    *,
    rail_color: tuple[int, int, int] = (80, 80, 80),
    pocket_color: tuple[int, int, int] = (255, 220, 0),
    pocket_radius_px: int = 10,
    label_pockets: bool = False,
    draw_border: bool = True,
) -> None:
    """Draw table border (play region or config rect) and pocket centers."""
    if draw_border:
        poly = np.array(
            [table_to_image(H_inv, c) for c in layout.border_polygon_mm()],
            dtype=np.int32,
        )
        cv2.polylines(frame, [poly], True, rail_color, 1, cv2.LINE_AA)

    for pocket in layout.pockets:
        px = table_to_image(H_inv, np.array(pocket.center_mm, dtype=np.float64))
        cv2.circle(frame, px, pocket_radius_px, pocket_color, 2, cv2.LINE_AA)
        cv2.circle(frame, px, 2, pocket_color, -1, cv2.LINE_AA)
        if label_pockets:
            cv2.putText(
                frame,
                pocket.id.replace("corner_", "c").replace("side_", "s"),
                (px[0] + 12, px[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                pocket_color,
                1,
                cv2.LINE_AA,
            )
