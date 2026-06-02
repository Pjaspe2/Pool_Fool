from __future__ import annotations

from pathlib import Path

import cv2

from pool_fool.shared.config import load_config, resolve_path
from pool_fool.shared.homography import load_homography
from pool_fool.shared.play_region import PlayRegion


def calibrate_play_region(
    config_path: Path,
    *,
    image_path: Path | None = None,
    camera: int = 0,
) -> int:
    from pool_fool.desktop.calibrate.cli import CornerPicker, _load_frame

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

    picker = CornerPicker("play_region")
    corners = picker.run(
        frame,
        "Play area mask (image -> table mm)",
        hint="Draw INSIDE the rails — exclude pockets and wood.\n"
        "TL → TR → BR → BL along the felt you want to keep.",
    )
    cv2.destroyAllWindows()
    if len(corners) != 4:
        print("Cancelled")
        return 1

    region = PlayRegion.from_image_clicks(corners, H)
    out = resolve_path(cfg, "play_region", root)
    region.save(out)
    print(f"Saved play region to {out}")
    print("Only balls and cue lines inside this quad are used.")
    print("Run: pool-fool-app --config", config_path)
    return 0
