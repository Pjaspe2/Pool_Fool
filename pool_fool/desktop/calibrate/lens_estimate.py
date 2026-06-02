from __future__ import annotations

from pathlib import Path

import numpy as np

from pool_fool.shared.config import load_config, resolve_path, table_spec_from_config
from pool_fool.shared.homography import image_to_table, load_homography, table_to_image
from pool_fool.shared.lens import save_lens_calibration


def estimate_lens_from_table(
    config_path: Path,
    *,
    camera_height_mm: float,
) -> int:
    """
    Rough pinhole focal length from camera height + existing table homography.

    Does NOT estimate fisheye distortion (k1,k2,... stay zero). Good for mild
    wide-angle; use aruco or chessboard for strong fisheye.
    """
    cfg = load_config(config_path)
    root = config_path.resolve().parent.parent
    table = table_spec_from_config(cfg)
    cam_cfg = cfg.get("cameras", {})

    H_path = resolve_path(cfg, "table_homography", root)
    if not H_path.exists():
        print("Run table calibration first (pool-fool-calibrate table).")
        return 1
    H = load_homography(H_path)
    H_inv = np.linalg.inv(H)

    w_px = int(cam_cfg.get("width", 1280))
    h_px = int(cam_cfg.get("height", 720))

    center_mm = np.array([table.length_mm / 2, table.width_mm / 2])
    p0 = np.array(table_to_image(H_inv, center_mm), dtype=np.float64)
    p1 = np.array(table_to_image(H_inv, center_mm + np.array([100.0, 0.0])), dtype=np.float64)
    px_per_mm = float(np.linalg.norm(p1 - p0)) / 100.0
    if px_per_mm < 1e-3:
        print("Homography scale looks wrong; re-run table calibration.")
        return 1

    focal_px = camera_height_mm * px_per_mm
    cx, cy = w_px / 2.0, h_px / 2.0
    camera_matrix = np.array(
        [[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    out = resolve_path(cfg, "lens_calibration", root)
    save_lens_calibration(out, camera_matrix, dist_coeffs, (w_px, h_px))
    print(f"Estimated focal length ≈ {focal_px:.0f} px (height={camera_height_mm} mm, scale={px_per_mm:.2f} px/mm)")
    print(f"Saved to {out} (distortion coefficients = 0)")
    print("For fisheye bend, use: pool-fool-calibrate lens-aruco")
    return 0
