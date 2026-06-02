from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pool_fool.desktop.physics.ghost_ball import GhostBallResult
from pool_fool.shared.homography import load_homography, table_to_image
from pool_fool.shared.schemas import ShotGuide
from pool_fool.shared.table import TableSpec


class OverlayRenderer:
    """Render guides in camera image or projector pixel space."""

    def __init__(
        self,
        table: TableSpec,
        overlay_cfg: dict,
        H_cam: np.ndarray,
        H_proj: np.ndarray | None = None,
    ) -> None:
        self.table = table
        self.cfg = overlay_cfg
        self.H_cam = H_cam
        self.H_cam_inv = np.linalg.inv(H_cam)
        self.H_proj = H_proj
        self.H_proj_inv = np.linalg.inv(H_proj) if H_proj is not None else None

    @classmethod
    def from_config(cls, cfg: dict, root: Path) -> OverlayRenderer:
        from pool_fool.shared.config import resolve_path, table_spec_from_config

        table = table_spec_from_config(cfg)
        cam_path = resolve_path(cfg, "table_homography", root)
        H_cam = load_homography(cam_path)
        H_proj = None
        proj_path = resolve_path(cfg, "projector_homography", root)
        if proj_path.exists():
            H_proj = load_homography(proj_path)
        return cls(table, cfg["overlay"], H_cam, H_proj)

    def shot_to_guide(self, result: GhostBallResult) -> ShotGuide:
        from pool_fool.shared.schemas import shot_from_arrays

        return shot_from_arrays(
            result.cue,
            result.ghost,
            result.object_ball,
            result.valid,
            object_index=result.object_index,
            blocked=result.blocked,
            message=result.message,
        )

    def draw_on_camera(self, frame: np.ndarray, result: GhostBallResult) -> np.ndarray:
        return draw_debug_frame(frame, self.H_cam_inv, result, self.cfg, self.table)

    def render_projector_frame(
        self,
        result: GhostBallResult,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Black canvas with bright lines in projector pixel coordinates."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if not result.valid or self.H_proj is None or self.H_proj_inv is None:
            return canvas

        line_color = tuple(self.cfg.get("line_color_bgr", [0, 255, 200]))
        ghost_color = tuple(self.cfg.get("ghost_ball_color_bgr", [0, 180, 255]))
        thickness = int(self.cfg.get("line_thickness_px", 3))
        gr = int(self.cfg.get("ghost_ball_radius_px", 12))

        pts_mm = [result.cue, result.ghost, result.object_ball]
        pts_px = [self._table_to_projector(p) for p in pts_mm]
        for i in range(len(pts_px) - 1):
            cv2.line(canvas, pts_px[i], pts_px[i + 1], line_color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, pts_px[1], gr, ghost_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pts_px[2], gr // 2, line_color, 2, cv2.LINE_AA)
        return canvas

    def _table_to_projector(self, pt_mm: np.ndarray) -> tuple[int, int]:
        assert self.H_proj_inv is not None
        return table_to_image(self.H_proj_inv, pt_mm)


def draw_debug_frame(
    frame: np.ndarray,
    H_cam_inv: np.ndarray,
    result: GhostBallResult,
    overlay_cfg: dict,
    table: TableSpec,
) -> np.ndarray:
    vis = frame.copy()
    line_color = tuple(overlay_cfg.get("line_color_bgr", [0, 255, 200]))
    ghost_color = tuple(overlay_cfg.get("ghost_ball_color_bgr", [0, 180, 255]))
    thickness = int(overlay_cfg.get("line_thickness_px", 3))
    gr = int(overlay_cfg.get("ghost_ball_radius_px", 12))

    if result.valid:
        pts = [result.cue, result.ghost, result.object_ball]
        px_pts = [table_to_image(H_cam_inv, p) for p in pts]
        for i in range(len(px_pts) - 1):
            cv2.line(vis, px_pts[i], px_pts[i + 1], line_color, thickness, cv2.LINE_AA)
        cv2.circle(vis, px_pts[1], gr, ghost_color, 2, cv2.LINE_AA)
        cv2.circle(vis, px_pts[2], gr // 2, line_color, 2, cv2.LINE_AA)
    elif result.message:
        cv2.putText(
            vis,
            result.message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    # Table border
    corners = [
        (0.0, 0.0),
        (table.length_mm, 0.0),
        (table.length_mm, table.width_mm),
        (0.0, table.width_mm),
    ]
    poly = np.array([table_to_image(H_cam_inv, np.array(c)) for c in corners], dtype=np.int32)
    cv2.polylines(vis, [poly], True, (80, 80, 80), 1)

    return vis
