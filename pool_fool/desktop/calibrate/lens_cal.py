from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pool_fool.shared.camera import CameraOpenError
from pool_fool.shared.config import load_config, resolve_path
from pool_fool.shared.lens import save_lens_calibration
from pool_fool.shared.live_camera import LiveCamera


def calibrate_lens(
    config_path: Path,
    camera: int,
    *,
    pattern_cols: int = 9,
    pattern_rows: int = 6,
    square_mm: float = 25.0,
    min_images: int = 12,
) -> int:
    """
    Collect chessboard views and solve camera matrix + distortion (fisheye / wide lens).

    Print a 9x6 inner-corner chessboard (or adjust --cols/--rows).
    """
    cfg = load_config(config_path)
    root = config_path.resolve().parent.parent
    pattern_size = (pattern_cols, pattern_rows)
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2) * square_mm
    )

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    last_frame: np.ndarray | None = None

    print("Lens calibration (chessboard)")
    print("  Move the printed board around the frame (center, edges, angles).")
    print("  SPACE = capture sample   c = compute & save   q = quit")
    print(f"  Need at least {min_images} good detections (have 0).")

    try:
        cam = LiveCamera(camera, cfg.get("cameras", {}))
    except CameraOpenError as e:
        print(e)
        return 1

    while True:
        frame = cam.read()
        if frame is None:
            continue
        last_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
        )
        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        cv2.putText(
            vis,
            f"samples={len(obj_points)}  {'OK' if found else 'no board'}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if found else (0, 0, 255),
            2,
        )
        cv2.imshow("lens_calibrate", vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return 1
        if key == ord(" ") and found:
            corners_ref = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            obj_points.append(objp.copy())
            img_points.append(corners_ref)
            image_size = (gray.shape[1], gray.shape[0])
            print(f"  captured sample {len(obj_points)}")
        if key == ord("c"):
            break

    cam.release()
    cv2.destroyAllWindows()
    if len(obj_points) < min_images or image_size is None:
        print(f"Need at least {min_images} samples; only got {len(obj_points)}.")
        return 1

    ret, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    if not ret:
        print("calibrateCamera failed")
        return 1

    out = resolve_path(cfg, "lens_calibration", root)
    save_lens_calibration(out, camera_matrix, dist_coeffs, image_size)
    print(f"Saved lens calibration to {out}")
    print("Set cameras.undistort: true in config (default). Re-run table calibration on undistorted view.")

    if last_frame is not None:
        from pool_fool.shared.lens import undistort_frame

        fixed = undistort_frame(last_frame, camera_matrix, dist_coeffs)
        compare = np.hstack([last_frame, fixed])
        cv2.imwrite(str(root / "config/calibration/lens_before_after.jpg"), compare)
        print(f"Wrote before/after preview to config/calibration/lens_before_after.jpg")

    return 0
