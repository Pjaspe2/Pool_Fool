from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pool_fool.desktop.physics.ghost_ball import GhostBallResult
from pool_fool.desktop.physics.pocket_shot import PocketShotResult
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

    def pocket_shot_to_guide(self, result: PocketShotResult) -> ShotGuide:
        from pool_fool.shared.schemas import shot_from_pocket_shot

        return shot_from_pocket_shot(result)

    def draw_on_camera(self, frame: np.ndarray, result: GhostBallResult) -> np.ndarray:
        return draw_debug_frame(frame, self.H_cam_inv, result, self.cfg, self.table)

    def render_projector_frame(
        self,
        result: GhostBallResult | PocketShotResult,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Black canvas with bright lines in projector pixel coordinates."""
        from pool_fool.shared.schemas import shot_from_pocket_shot

        if isinstance(result, PocketShotResult):
            guide = shot_from_pocket_shot(result)
        else:
            guide = self.shot_to_guide(result)
        return self.render_projector_guide(guide, width, height)

    def render_projector_guide(self, guide: ShotGuide, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if not guide.valid or self.H_proj_inv is None:
            return canvas

        line_color = tuple(self.cfg.get("line_color_bgr", [0, 255, 200]))
        pocket_color = tuple(self.cfg.get("pocket_line_color_bgr", [0, 200, 255]))
        ghost_color = tuple(self.cfg.get("ghost_ball_color_bgr", [0, 180, 255]))
        thickness = int(self.cfg.get("line_thickness_px", 3))
        gr = int(self.cfg.get("ghost_ball_radius_px", 12))

        cue = np.array(guide.cue_mm, dtype=np.float64)
        ghost = np.array(guide.ghost_mm, dtype=np.float64)
        obj = np.array(guide.object_mm, dtype=np.float64)
        pts_px = [
            self._table_to_projector(cue),
            self._table_to_projector(ghost),
            self._table_to_projector(obj),
        ]
        for i in range(len(pts_px) - 1):
            cv2.line(canvas, pts_px[i], pts_px[i + 1], line_color, thickness, cv2.LINE_AA)
        if len(guide.pocket_mm) >= 2:
            pocket = np.array(guide.pocket_mm, dtype=np.float64)
            px_pocket = self._table_to_projector(pocket)
            cv2.line(canvas, pts_px[2], px_pocket, pocket_color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, px_pocket, max(6, gr // 2), pocket_color, 2, cv2.LINE_AA)
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

    return vis


def draw_selected_pocket_marker(
    frame: np.ndarray,
    H_cam_inv: np.ndarray,
    pocket_center_mm: tuple[float, float],
    *,
    pocket_id: str = "",
    color_bgr: tuple[int, int, int] = (255, 180, 0),
) -> None:
    """Highlight the user-selected pocket on the debug frame (in-place)."""
    px = table_to_image(H_cam_inv, np.array(pocket_center_mm, dtype=np.float64))
    cv2.circle(frame, px, 22, color_bgr, 3, cv2.LINE_AA)
    if pocket_id:
        cv2.putText(
            frame,
            pocket_id,
            (px[0] + 24, px[1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_bgr,
            1,
            cv2.LINE_AA,
        )


def draw_pocket_shot_frame(
    frame: np.ndarray,
    H_cam_inv: np.ndarray,
    result: PocketShotResult,
    overlay_cfg: dict,
    *,
    session_label: str | None = None,
) -> np.ndarray:
    """Cue → ghost → object (cyan) and object → pocket (yellow)."""
    vis = frame.copy()
    line_color = tuple(overlay_cfg.get("line_color_bgr", [0, 255, 200]))
    pocket_color = tuple(overlay_cfg.get("pocket_line_color_bgr", [0, 200, 255]))
    ghost_color = tuple(overlay_cfg.get("ghost_ball_color_bgr", [0, 180, 255]))
    thickness = int(overlay_cfg.get("line_thickness_px", 3))
    gr = int(overlay_cfg.get("ghost_ball_radius_px", 12))

    if result.valid:
        pts = result.polyline()
        px_pts = [table_to_image(H_cam_inv, p) for p in pts]
        # Aim: cue → ghost → contact (midpoint) → object center, then object → pocket
        contact_mm = 0.5 * (result.ghost + result.object_ball)
        px_contact = table_to_image(H_cam_inv, contact_mm)
        cv2.line(vis, px_pts[0], px_pts[1], line_color, thickness, cv2.LINE_AA)
        cv2.line(vis, px_pts[1], px_contact, line_color, thickness, cv2.LINE_AA)
        cv2.line(vis, px_contact, px_pts[2], line_color, max(1, thickness - 1), cv2.LINE_AA)
        cv2.line(vis, px_pts[2], px_pts[3], pocket_color, thickness, cv2.LINE_AA)
        cv2.circle(vis, px_pts[1], gr, ghost_color, 2, cv2.LINE_AA)
        cv2.circle(vis, px_contact, 5, line_color, -1, cv2.LINE_AA)
        cv2.circle(vis, px_pts[2], gr // 2, line_color, 2, cv2.LINE_AA)
        cv2.circle(vis, px_pts[3], max(6, gr // 2), pocket_color, 2, cv2.LINE_AA)
        label = session_label or f"PoC → {result.pocket_id}  cut {result.cut_angle_deg:.0f}°"
        cv2.putText(
            vis,
            label,
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            pocket_color,
            2,
            cv2.LINE_AA,
        )
    elif result.message:
        cv2.putText(
            vis,
            result.message,
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return vis
