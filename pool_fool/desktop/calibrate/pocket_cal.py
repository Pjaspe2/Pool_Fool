from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pool_fool.shared.config import load_config, resolve_path
from pool_fool.shared.homography import image_to_table, load_homography
from pool_fool.shared.pocket_calibration import POCKET_CALIBRATION_ORDER, save_calibrated_pockets
from pool_fool.shared.play_region import PlayRegion
from pool_fool.shared.table_layout import PocketSpec


class PocketPointPicker:
    """Click pocket centers on a frozen camera frame (same workflow as play-region)."""

    def __init__(self, window: str = "pocket_calibrate") -> None:
        self.window = window
        self.points: list[tuple[int, int]] = []
        self._frame: np.ndarray | None = None
        self._hint = ""

    def _mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < len(POCKET_CALIBRATION_ORDER):
            self.points.append((x, y))

    def run(self, frame: np.ndarray, *, hint: str = "") -> list[tuple[int, int]]:
        self.points.clear()
        self._frame = frame.copy()
        self._hint = hint
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._mouse)
        n = len(POCKET_CALIBRATION_ORDER)
        print("Click each pocket center on the image (aim at where the ball drops).")
        print("Keys: u=undo last, s=save when all clicked, q=quit without save")

        while True:
            vis = self._frame.copy()
            y0 = 24
            for line in self._hint.split("\n")[:3]:
                cv2.putText(
                    vis,
                    line[:72],
                    (12, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 255, 200),
                    1,
                    cv2.LINE_AA,
                )
                y0 += 22
            if len(self.points) < n:
                _pid, _kind, label = POCKET_CALIBRATION_ORDER[len(self.points)]
                cv2.putText(
                    vis,
                    f"Next: {label} ({len(self.points) + 1}/{n})",
                    (12, y0 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            for i, (px, py) in enumerate(self.points):
                pid = POCKET_CALIBRATION_ORDER[i][0]
                cv2.circle(vis, (px, py), 8, (255, 220, 0), 2)
                cv2.putText(
                    vis,
                    pid.replace("corner_", "c").replace("side_", "s"),
                    (px + 10, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 220, 0),
                    1,
                )
            cv2.imshow(self.window, vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return []
            if key == ord("u") and self.points:
                self.points.pop()
            if key == ord("s") and len(self.points) == n:
                return list(self.points)
        return []


def calibrate_pockets(
    config_path: Path,
    *,
    image_path: Path | None = None,
    camera: str | int = 0,
) -> int:
    from pool_fool.desktop.calibrate.cli import _load_frame

    cfg = load_config(config_path)
    root = config_path.resolve().parent.parent
    H_path = resolve_path(cfg, "table_homography", root)
    if not H_path.exists():
        print("Run table calibration first: pool-fool-calibrate table")
        return 1
    H = load_homography(H_path)

    frame = _load_frame(camera, cfg, image_path, root=root)
    if frame is None:
        return 1

    play_path = resolve_path(cfg, "play_region", root)
    region = PlayRegion.load(play_path)
    if region is not None:
        H_inv = np.linalg.inv(H)
        region.draw(frame, H_inv)

    picker = PocketPointPicker()
    hint = (
        "Click pocket centers in order (cyan labels).\n"
        "Orange = play region. Match real pocket mouths."
    )
    clicks = picker.run(frame, hint=hint)
    cv2.destroyAllWindows()
    if len(clicks) != len(POCKET_CALIBRATION_ORDER):
        print("Cancelled")
        return 1

    pockets: list[PocketSpec] = []
    for i, (px, py) in enumerate(clicks):
        pid, kind, _label = POCKET_CALIBRATION_ORDER[i]
        mm = image_to_table(H, (px, py))
        pockets.append(PocketSpec(pid, (float(mm[0]), float(mm[1])), kind))
        print(f"  {pid}: ({mm[0]:.1f}, {mm[1]:.1f}) mm")

    border = region.corners_mm.copy() if region is not None else None
    out = resolve_path(cfg, "pockets", root)
    save_calibrated_pockets(out, tuple(pockets))
    print(f"Saved {len(pockets)} pockets to {out}")
    print("Set table.use_calibrated_pockets: true (default) and run pool-fool-app")
    return 0
