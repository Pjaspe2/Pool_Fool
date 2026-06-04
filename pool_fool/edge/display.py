from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pool_fool.shared.config import load_config, resolve_path, table_spec_from_config
from pool_fool.shared.homography import load_homography, table_to_image
from pool_fool.shared.schemas import OverlayMessage, ShotGuide
from pool_fool.shared.table import TableSpec


class ProjectorDisplay:
    """Fullscreen HDMI output with warped ghost-ball lines."""

    def __init__(
        self,
        table: TableSpec,
        overlay_cfg: dict,
        H_proj_inv: np.ndarray,
        width: int,
        height: int,
        window_name: str = "pool_fool_projector",
    ) -> None:
        self.table = table
        self.cfg = overlay_cfg
        self.H_proj_inv = H_proj_inv
        self.width = width
        self.height = height
        self.window_name = window_name
        self._initialized = False

    @classmethod
    def from_config(cls, config_path: Path) -> ProjectorDisplay:
        cfg = load_config(config_path)
        root = config_path.parent.parent
        table = table_spec_from_config(cfg)
        proj_path = resolve_path(cfg, "projector_homography", root)
        H_proj = load_homography(proj_path)
        H_proj_inv = np.linalg.inv(H_proj)
        pw = int(cfg["projector"]["display_width"])
        ph = int(cfg["projector"]["display_height"])
        return cls(table, cfg["overlay"], H_proj_inv, pw, ph)

    def _ensure_window(self) -> None:
        if self._initialized:
            return
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._initialized = True

    def render(self, msg: OverlayMessage | None) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if msg is None or not msg.stationary or not msg.shot.valid:
            return canvas

        shot = msg.shot
        if len(shot.cue_mm) < 2 or len(shot.ghost_mm) < 2 or len(shot.object_mm) < 2:
            return canvas

        cue = np.array(shot.cue_mm, dtype=np.float64)
        ghost = np.array(shot.ghost_mm, dtype=np.float64)
        obj = np.array(shot.object_mm, dtype=np.float64)

        line_color = tuple(self.cfg.get("line_color_bgr", [0, 255, 200]))
        ghost_color = tuple(self.cfg.get("ghost_ball_color_bgr", [0, 180, 255]))
        thickness = int(self.cfg.get("line_thickness_px", 3))
        gr = int(self.cfg.get("ghost_ball_radius_px", 12))

        pocket_color = tuple(self.cfg.get("pocket_line_color_bgr", [0, 200, 255]))
        pts = [
            table_to_image(self.H_proj_inv, cue),
            table_to_image(self.H_proj_inv, ghost),
            table_to_image(self.H_proj_inv, obj),
        ]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], line_color, thickness, cv2.LINE_AA)
        if len(shot.pocket_mm) >= 2:
            pocket = np.array(shot.pocket_mm, dtype=np.float64)
            px_pocket = table_to_image(self.H_proj_inv, pocket)
            cv2.line(canvas, pts[2], px_pocket, pocket_color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, px_pocket, max(6, gr // 2), pocket_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pts[1], gr, ghost_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pts[2], gr // 2, line_color, 2, cv2.LINE_AA)
        return canvas

    def show(self, msg: OverlayMessage | None) -> None:
        self._ensure_window()
        frame = self.render(msg)
        cv2.imshow(self.window_name, frame)

    def show_frame(self, frame: np.ndarray) -> None:
        self._ensure_window()
        cv2.imshow(self.window_name, frame)
