from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

from pool_fool.shared.camera import CameraOpenError
from pool_fool.shared.config import load_config, resolve_path
from pool_fool.shared.lens import save_lens_calibration
from pool_fool.shared.live_camera import LiveCamera

from pool_fool.desktop.calibrate.screen_warp import ScreenWarp


def _make_charuco_board(square_mm: float, cols: int, rows: int) -> aruco.CharucoBoard:
    square_m = square_mm / 1000.0
    marker_m = square_m * 0.75
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    return aruco.CharucoBoard((cols, rows), square_m, marker_m, dictionary)


def _make_detector(board: aruco.CharucoBoard) -> aruco.CharucoDetector:
    charuco_params = aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    detector_params = aruco.DetectorParameters()
    detector_params.adaptiveThreshConstant = 7
    detector_params.minMarkerPerimeterRate = 0.02
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    refine_params = aruco.RefineParameters()
    return aruco.CharucoDetector(board, charuco_params, detector_params, refine_params)


def save_aruco_pattern(
    output: Path,
    *,
    square_mm: float = 40.0,
    cols: int = 5,
    rows: int = 7,
    width_px: int = 1200,
    height_px: int = 1600,
) -> Path:
    board = _make_charuco_board(square_mm, cols, rows)
    img = board.generateImage((width_px, height_px), marginSize=40)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), img)
    return output


def _detect_charuco(
    board: aruco.CharucoBoard,
    detector: aruco.CharucoDetector,
    gray: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    corners, ids, _mc, _mi = detector.detectBoard(gray)
    n = 0 if corners is None else len(corners)
    if corners is not None and n >= 4 and ids is not None:
        return corners, ids, n
    return None, None, n


def _append_sample(
    board: aruco.CharucoBoard,
    corners: np.ndarray,
    ids: np.ndarray,
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    warp: ScreenWarp | None,
) -> tuple[bool, str]:
    obj, imgp = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < 4:
        return False, "matchImagePoints failed"
    imgp = imgp.reshape(-1, 2).astype(np.float32)
    if warp is not None and warp.is_active:
        imgp = warp.map_points_to_original(imgp.reshape(-1, 1, 2)).reshape(-1, 2)
    obj_points.append(obj.reshape(-1, 3).astype(np.float32))
    img_points.append(imgp)
    return True, f"ok ({len(obj)} points)"


def _compose_view(camera_bgr: np.ndarray, warped_bgr: np.ndarray | None) -> np.ndarray:
    """Single window: camera + warped (avoids focus stuck on second window)."""
    if warped_bgr is None:
        return camera_bgr
    h = max(camera_bgr.shape[0], warped_bgr.shape[0])
    w1 = camera_bgr.shape[1]
    w2 = warped_bgr.shape[1]
    pad1 = np.zeros((h, w1, 3), dtype=np.uint8)
    pad2 = np.zeros((h, w2, 3), dtype=np.uint8)
    pad1[: camera_bgr.shape[0], :w1] = camera_bgr
    pad2[: warped_bgr.shape[0], :w2] = warped_bgr
    divider = np.full((h, 4, 3), 180, dtype=np.uint8)
    return np.hstack([pad1, divider, pad2])


def calibrate_lens_aruco(
    config_path: Path,
    camera: int,
    *,
    square_mm: float,
    cols: int = 5,
    rows: int = 7,
    min_images: int = 5,
    show_pattern: bool = False,
) -> int:
    cfg = load_config(config_path)
    root = config_path.resolve().parent.parent
    board = _make_charuco_board(square_mm, cols, rows)
    detector = _make_detector(board)

    pattern_path = root / "config/calibration/aruco_charuco.png"
    save_aruco_pattern(pattern_path, square_mm=square_mm, cols=cols, rows=rows)
    print(f"Pattern image: {pattern_path}")
    print(f"Measured square on screen: {square_mm} mm (must match what you display)")
    if show_pattern:
        print("Warning: built-in fullscreen pattern conflicts with camera view on one laptop.")

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    warp = ScreenWarp()
    status_msg = "Click this window, then w = outline phone/screen"
    status_until = time.monotonic() + 8.0
    picking_corners = False
    pick_points: list[tuple[int, int]] = []

    print("\n=== Controls (focus the ONE capture window) ===")
    print("  w     = click 4 corners of phone/screen (TL→TR→BR→BL)")
    print("  SPACE or s = SAVE a sample (required before finish)")
    print("  c     = FINISH calibration (needs 5+ samples)")
    print("  q     = quit")
    print("  u     = undo corner while picking")
    print("  NOTE: c does NOT capture — use SPACE\n")

    window = "aruco_lens_capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal pick_points
        if not picking_corners or event != cv2.EVENT_LBUTTONDOWN:
            return
        # Clicks only on left (camera) panel
        if x > camera_bgr.shape[1] if "camera_bgr" in dir() else False:
            return
        cam_w = last_cam_w[0]
        if x >= cam_w:
            return
        if len(pick_points) < 4:
            pick_points.append((x, y))

    last_cam_w = [1280]

    def on_mouse_safe(event: int, x: int, y: int, flags: int, param: object) -> None:
        cam_w = last_cam_w[0]
        if picking_corners and x < cam_w and event == cv2.EVENT_LBUTTONDOWN and len(pick_points) < 4:
            pick_points.append((x, y))

    cv2.setMouseCallback(window, on_mouse_safe)

    try:
        with LiveCamera(camera, cfg.get("cameras", {})) as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    status_msg = "No camera frame"
                    status_until = time.monotonic() + 2.0
                    continue

                last_cam_w[0] = frame.shape[1]
                vis_cam = frame.copy()
                warp.draw_quad(vis_cam)

                warped_bgr: np.ndarray | None = None
                detect_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if warp.is_active:
                    warped_bgr = warp.warp(frame)
                    if warped_bgr is not None:
                        detect_gray = (
                            warped_bgr
                            if len(warped_bgr.shape) == 2
                            else cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
                        )

                corners, ids, n_corners = _detect_charuco(board, detector, detect_gray)
                ready = corners is not None and n_corners >= 4

                if warped_bgr is not None:
                    panel = (
                        cv2.cvtColor(detect_gray, cv2.COLOR_GRAY2BGR)
                        if len(detect_gray.shape) == 2
                        else detect_gray.copy()
                    )
                    if corners is not None and ids is not None:
                        try:
                            aruco.drawDetectedCornersCharuco(panel, corners, ids)
                        except Exception:
                            pass
                    cv2.putText(
                        panel,
                        "warped (detection)",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )
                    warped_bgr = panel

                display = _compose_view(vis_cam, warped_bgr)
                color = (0, 255, 0) if ready else (0, 128, 255)
                cv2.putText(
                    display,
                    f"samples {len(obj_points)}/{min_images}   corners {n_corners}   warp {'ON' if warp.is_active else 'off'}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                if ready:
                    cv2.putText(
                        display,
                        ">>> Press SPACE or s to capture <<<",
                        (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                    )
                if time.monotonic() < status_until:
                    cv2.putText(
                        display,
                        status_msg[:72],
                        (12, 88),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 255, 255),
                        2,
                    )
                if picking_corners:
                    cv2.putText(
                        display,
                        f"Click corners on LEFT view {len(pick_points)}/4",
                        (12, 118),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )
                    for i, (px, py) in enumerate(pick_points):
                        cv2.circle(display, (px, py), 6, (0, 255, 255), -1)
                        cv2.putText(
                            display,
                            str(i + 1),
                            (px + 8, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )

                cv2.imshow(window, display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    return 1
                if key == ord("w"):
                    picking_corners = True
                    pick_points = []
                    status_msg = "Click 4 corners on LEFT (camera) view"
                    status_until = time.monotonic() + 8.0
                if key == ord("u") and pick_points:
                    pick_points.pop()
                if picking_corners and len(pick_points) == 4:
                    warp.set_corners(pick_points)
                    picking_corners = False
                    status_msg = "Warp ON — move phone slightly, press SPACE when corners green"
                    status_until = time.monotonic() + 6.0
                    print("Screen warp enabled.")

                capture_key = key in (ord(" "), ord("s"))
                if capture_key:
                    if not ready or corners is None or ids is None:
                        print(
                            f"  Capture skipped: only {n_corners} corners detected "
                            f"(need 4+). Adjust light / square-mm / warp."
                        )
                        status_msg = f"Not ready ({n_corners} corners)"
                        status_until = time.monotonic() + 3.0
                    else:
                        if image_size is None:
                            image_size = (frame.shape[1], frame.shape[0])
                        ok, msg = _append_sample(
                            board, corners, ids, obj_points, img_points, warp
                        )
                        if ok:
                            print(f"  >>> captured sample {len(obj_points)} <<<")
                            status_msg = f"Saved sample {len(obj_points)}"
                        else:
                            print(f"  Capture failed: {msg}")
                            status_msg = msg
                        status_until = time.monotonic() + 3.0

                if key == ord("c"):
                    if len(obj_points) < min_images:
                        print(
                            f"  Cannot finish yet: {len(obj_points)}/{min_images} samples. "
                            "Press SPACE (not c) while corners are green."
                        )
                        status_msg = f"Need {min_images - len(obj_points)} more SPACE captures"
                        status_until = time.monotonic() + 4.0
                        continue
                    break
    except CameraOpenError as e:
        print(e)
        for h in e.hints:
            print(f"  • {h}")
        return 1
    finally:
        cv2.destroyAllWindows()

    if len(obj_points) < min_images or image_size is None:
        print(f"Need at least {min_images} samples; got {len(obj_points)}.")
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
    return 0
