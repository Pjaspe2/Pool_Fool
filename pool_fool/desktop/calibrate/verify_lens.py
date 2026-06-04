from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pool_fool.shared.config import load_config, resolve_path
from pool_fool.shared.frame_pipeline import build_lens_corrector, preprocess_frame
from pool_fool.shared.lens import load_lens_calibration, undistort_frame
from pool_fool.shared.live_camera import LiveCamera


def verify_lens(
    config_path: Path,
    camera: str | int,
    *,
    image_path: Path | None = None,
    output: Path | None = None,
) -> int:
    cfg = load_config(config_path)
    root = config_path.resolve().parent.parent
    lens_path = resolve_path(cfg, "lens_calibration", root)
    if not lens_path.exists():
        print(f"Missing {lens_path}")
        return 1

    loaded = load_lens_calibration(lens_path)
    if loaded is None:
        return 1
    K, dist, sz, reproj = loaded
    fx, fy = float(K[0, 0]), float(K[1, 1])
    alpha = float(cfg.get("cameras", {}).get("undistort_alpha", 0.0))
    print(f"Lens file: {lens_path}")
    print(f"  Resolution: {int(sz[0])}x{int(sz[1])}")
    print(f"  Focal length (px): fx={fx:.1f} fy={fy:.1f}")
    print(f"  Distortion: {np.array2string(dist.ravel(), precision=4, suppress_small=True)}")
    if reproj is not None:
        quality = "good" if reproj < 0.5 else ("ok" if reproj < 1.0 else "poor — recalibrate")
        print(f"  Reprojection error: {reproj:.3f} px ({quality})")
    print(f"  undistort_alpha: {alpha} (try 0.0–0.3 if edges still bent)")

    if image_path and image_path.exists():
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Cannot read {image_path}")
            return 1
    else:
        try:
            with LiveCamera(camera, cfg.get("cameras", {})) as cam:
                frame = cam.read()
        except Exception as e:
            print(e)
            return 1
        if frame is None:
            print("No camera frame")
            return 1

    h, w = frame.shape[:2]
    if (w, h) != (int(sz[0]), int(sz[1])):
        print(f"  Note: frame is {w}x{h}, calibrating for {int(sz[0])}x{int(sz[1])} — resizing for preview")
        frame = cv2.resize(frame, (int(sz[0]), int(sz[1])))

    corrector = build_lens_corrector(cfg, root)
    fixed = (
        preprocess_frame(frame, corrector)
        if corrector
        else undistort_frame(frame, K, dist, alpha=alpha)
    )

    out = output or (root / "config/calibration/lens_verify.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack([frame, fixed]))
    print(f"  Before/after saved: {out.resolve()}")
    print("  Left=raw, right=undistorted. Straight lines on the table should look straighter on the right.")
    return 0
