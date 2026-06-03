from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pool_fool.desktop.calibrate.aruco_lens import calibrate_lens_aruco, save_aruco_pattern
from pool_fool.desktop.calibrate.lens_cal import calibrate_lens
from pool_fool.desktop.calibrate.lens_estimate import estimate_lens_from_table
from pool_fool.desktop.calibrate.play_region_cal import calibrate_play_region
from pool_fool.desktop.calibrate.verify_lens import verify_lens
from pool_fool.desktop.vision.felt import build_felt_mask
from pool_fool.shared.calibration_regions import parse_dst_corners_mm, table_corners_for_region
from pool_fool.shared.frame_pipeline import build_lens_corrector, preprocess_frame
from pool_fool.shared.camera import (
    CameraOpenError,
    capture_frame,
    capture_stream_frame,
    is_stream_url,
    parse_camera_arg,
    print_camera_doctor,
    probe_modes,
)
from pool_fool.shared.config import load_config, resolve_path, table_spec_from_config
from pool_fool.shared.homography import (
    compute_table_homography,
    default_table_corners_mm,
    load_homography,
    save_homography,
)


class CornerPicker:
    def __init__(self, window: str = "calibrate") -> None:
        self.window = window
        self.points: list[tuple[int, int]] = []
        self._frame: np.ndarray | None = None
        self._hint: str = ""

    def _mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def run(self, frame: np.ndarray, title: str, *, hint: str = "") -> list[tuple[int, int]]:
        self.points.clear()
        self._frame = frame.copy()
        self._hint = hint
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._mouse)
        print(title)
        print("Click 4 corners in order: TL → TR → BR → BL (on the playing surface you see).")
        if hint:
            print(hint)
        print("Keys: u=undo last, s=save & quit, q=quit without save")

        while True:
            vis = self._frame.copy()
            if self._hint:
                y0 = 24
                for line in self._hint.split("\n")[:4]:
                    cv2.putText(
                        vis,
                        line[:70],
                        (12, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 255, 200),
                        1,
                        cv2.LINE_AA,
                    )
                    y0 += 22
            for i, (px, py) in enumerate(self.points):
                cv2.circle(vis, (px, py), 6, (0, 255, 255), -1)
                cv2.putText(
                    vis,
                    str(i + 1),
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            if len(self.points) == 4:
                pts = np.array(self.points, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.imshow(self.window, vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return []
            if key == ord("u") and self.points:
                self.points.pop()
            if key == ord("s") and len(self.points) == 4:
                return list(self.points)
        return []


def _resolve_dst_corners(
    table,
    region: str | None,
    dst_corners_mm: str | None,
) -> tuple[list[tuple[float, float]], str]:
    if dst_corners_mm:
        return parse_dst_corners_mm(dst_corners_mm), "custom"
    if region:
        return table_corners_for_region(table, region), region
    return default_table_corners_mm(table.length_mm, table.width_mm), "full"


def _print_camera_error(exc: CameraOpenError) -> None:
    print(exc)
    if exc.hints:
        print("\nTry:")
        for h in exc.hints:
            print(f"  • {h}")


def _load_frame(
    camera: str | int,
    cfg: dict,
    image_path: Path | None,
    *,
    wide: bool = False,
    root: Path | None = None,
) -> np.ndarray | None:
    if image_path is not None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Cannot read image: {image_path}")
            return None
        if root is not None:
            corrector = build_lens_corrector(cfg, root)
            if corrector:
                frame = preprocess_frame(frame, corrector)
        return frame
    try:
        if is_stream_url(camera):
            frame = capture_stream_frame(str(camera))
            print(f"Captured {frame.shape[1]}x{frame.shape[0]} from stream {camera}")
        else:
            idx = int(camera)
            frame, idx_used, backend = capture_frame(
                idx, cfg.get("cameras", {}), wide=wide, scan_indices=True
            )
            print(
                f"Captured {frame.shape[1]}x{frame.shape[0]} from camera index {idx_used} ({backend})"
            )
            if idx_used != idx:
                print(f"  (requested --camera {idx}; update config if this is your Logitech)")
        if root is not None:
            corrector = build_lens_corrector(cfg, root)
            if corrector:
                frame = preprocess_frame(frame, corrector)
        return frame
    except CameraOpenError as e:
        _print_camera_error(e)
        return None


def run_felt_sample(config_path: Path, camera: int, image_path: Path | None) -> int:
    """Live HSV tuner: move mouse over felt; adjust ranges in config/default.yaml."""
    cfg = load_config(config_path)
    root = config_path.resolve().parent.parent
    frame = _load_frame(camera, cfg, image_path, root=root)
    if frame is None:
        return 1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    window = "felt_sample"
    state = {"h": 0, "s": 0, "v": 0}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_MOUSEMOVE and 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]:
            state["h"], state["s"], state["v"] = [int(v) for v in hsv[y, x]]

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print("Move cursor over RED felt. Note H,S,V printed on screen.")
    print("Update config vision.felt_hsv_red ranges, then restart pool-fool-app.")
    print("Press q to quit.")

    while True:
        mask = build_felt_mask(hsv, cfg["vision"])
        overlay = frame.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 80, 0])).astype(np.uint8)
        txt = f"H={state['h']} S={state['s']} V={state['v']}  (felt_color={cfg['vision'].get('felt_color', 'red')})"
        cv2.putText(overlay, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window, overlay)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    return 0


def capture_snapshot(
    config_path: Path,
    camera: str | int,
    output: Path,
    *,
    wide: bool = False,
) -> int:
    cfg = load_config(config_path)
    frame = _load_frame(camera, cfg, None, wide=wide)
    if frame is None:
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    print(f"Saved snapshot to {output.resolve()}")
    print("Calibrate from it with:")
    print(f"  pool-fool-calibrate table --config {config_path} --image {output}")
    if not wide:
        print("Tip: if the table does not fit, retry with: pool-fool-calibrate capture --wide")
    return 0


def run_probe(camera: int) -> int:
    print(f"Probing camera {camera} (zoom=0, common resolutions)...")
    modes = probe_modes(camera)
    if not modes:
        print("No camera or no modes detected.")
        return 1
    print(f"{'Requested':<16} {'Actual':<16}")
    for m in modes:
        rw, rh = m["requested"]
        aw, ah = m["actual"]
        print(f"{rw}x{rh:<10} {aw}x{ah}")
    print("\nWidest FOV is often 640x480 on webcams. Set cameras.wide_width/height in config.")
    return 0


def calibrate_table(
    config_path: Path,
    camera: str | int,
    *,
    image_path: Path | None = None,
    wide: bool = False,
    region: str | None = None,
    dst_corners_mm: str | None = None,
) -> int:
    cfg = load_config(config_path)
    table = table_spec_from_config(cfg)
    cal = cfg.get("calibration", {})
    region = region or cal.get("region")
    root = config_path.resolve().parent.parent
    frame = _load_frame(camera, cfg, image_path, wide=wide, root=root)
    if frame is None:
        return 1

    dst, region_name = _resolve_dst_corners(table, region, dst_corners_mm)
    partial = region_name != "full"

    hint = ""
    if partial:
        hint = (
            "Partial table: click the 4 corners of the VISIBLE felt only.\n"
            f"Mapped to table region '{region_name}' in mm.\n"
            "Ghost-ball assist works best inside this quad."
        )

    picker = CornerPicker("table_calibrate")
    corners = picker.run(frame, "Table homography (image -> mm)", hint=hint)
    cv2.destroyAllWindows()
    if len(corners) != 4:
        print("Cancelled")
        return 1

    H = compute_table_homography(corners, dst)
    out = resolve_path(cfg, "table_homography", root)
    save_homography(
        out,
        H,
        kind="table",
        region=region_name,
        dst_corners_mm=np.array(dst, dtype=np.float64),
    )
    print(f"Saved table homography to {out} (region={region_name})")
    if partial:
        print("Only the visible region is accurately mapped; remount higher later for 'full'.")
    return 0


def calibrate_projector(config_path: Path, camera: str | int) -> int:
    cfg = load_config(config_path)
    root = config_path.resolve().parent.parent
    table_path = resolve_path(cfg, "table_homography", root)
    if not table_path.exists():
        print("Run table calibration first.")
        return 1
    H_cam = load_homography(table_path)

    pw = cfg["projector"]["display_width"]
    ph = cfg["projector"]["display_height"]

    proj_corners_px = [(0, 0), (pw - 1, 0), (pw - 1, ph - 1), (0, ph - 1)]
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    for i, (px, py) in enumerate(proj_corners_px):
        cv2.circle(canvas, (px, py), 30, (0, 255, 255), -1)
        cv2.putText(canvas, str(i + 1), (px + 40, py + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.namedWindow("projector_pattern", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("projector_pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("projector_pattern", canvas)

    frame = _load_frame(camera, cfg, None)
    cv2.destroyWindow("projector_pattern")

    if frame is None:
        return 1

    picker = CornerPicker("projector_calibrate")
    image_corners = picker.run(
        frame,
        "Click the 4 projected corner dots (same order as numbered on projector)",
    )
    cv2.destroyAllWindows()
    if len(image_corners) != 4:
        print("Cancelled")
        return 1

    table_corners = []
    for ic in image_corners:
        t = cv2.perspectiveTransform(
            np.array([[ic]], dtype=np.float32).reshape(1, 1, 2), H_cam
        ).reshape(2)
        table_corners.append((float(t[0]), float(t[1])))

    H_proj = compute_table_homography(
        [(float(p[0]), float(p[1])) for p in proj_corners_px],
        table_corners,
    )
    out = resolve_path(cfg, "projector_homography", root)
    save_homography(out, H_proj, kind="projector")
    print(f"Saved projector homography to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pool Fool calibration tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_table = sub.add_parser("table", help="Overhead camera -> table plane")
    p_table.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_table.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera index or MJPEG URL (e.g. http://pool.local:8080/stream.mjpg)",
    )
    p_table.add_argument("--image", type=Path, default=None)
    p_table.add_argument("--wide", action="store_true", help="Use wide FOV capture settings")
    p_table.add_argument(
        "--region",
        choices=["full", "half_near", "half_far", "center"],
        default=None,
        help="Map visible quad to a table sub-region (when full table not in frame)",
    )
    p_table.add_argument(
        "--dst-corners-mm",
        default=None,
        help="Custom TL;TR;BR;BL in mm, e.g. '0,0;1270,0;1270,635;0,635'",
    )

    p_cap = sub.add_parser("capture", help="Save one camera frame for offline calibration")
    p_cap.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_cap.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera index or MJPEG URL",
    )
    p_cap.add_argument("--output", type=Path, default=Path("config/calibration/snapshot.jpg"))
    p_cap.add_argument("--wide", action="store_true", help="640x480 + min zoom for widest view")

    p_probe = sub.add_parser("probe", help="List resolutions the camera actually provides")
    p_probe.add_argument("--camera", type=int, default=0)

    p_doc = sub.add_parser("doctor", help="Diagnose macOS/Terminal camera permission and indices")
    p_doc.add_argument("--max-index", type=int, default=3)

    p_lens = sub.add_parser("lens", help="Chessboard lens cal (needs printed board)")
    p_lens.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_lens.add_argument("--camera", type=int, default=0)
    p_lens.add_argument("--cols", type=int, default=9, help="Inner corners per row")
    p_lens.add_argument("--rows", type=int, default=6, help="Inner corners per column")
    p_lens.add_argument("--square-mm", type=float, default=25.0)

    p_aruco = sub.add_parser(
        "lens-aruco",
        help="Lens cal via ChArUco on phone/tablet (no printer)",
    )
    p_aruco.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_aruco.add_argument("--camera", type=int, default=0)
    p_aruco.add_argument(
        "--square-mm",
        type=float,
        required=True,
        help="Measure one square on screen with a ruler",
    )
    p_aruco.add_argument("--no-fullscreen", action="store_true")

    p_gen = sub.add_parser("aruco-pattern", help="Save ChArUco PNG for phone/tablet")
    p_gen.add_argument("--output", type=Path, default=Path("config/calibration/aruco_charuco.png"))
    p_gen.add_argument("--square-mm", type=float, default=40.0)

    p_est = sub.add_parser(
        "lens-estimate",
        help="Rough focal length from camera height + table homography",
    )
    p_est.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_est.add_argument(
        "--camera-height-mm",
        type=float,
        required=True,
        help="Vertical distance from camera lens to felt (mm)",
    )

    p_ver = sub.add_parser("verify-lens", help="Show lens.npz stats + before/after image")
    p_ver.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_ver.add_argument("--camera", type=int, default=0)
    p_ver.add_argument("--image", type=Path, default=None)
    p_ver.add_argument("--output", type=Path, default=None)

    p_play = sub.add_parser(
        "play-region",
        help="Click 4 corners: detection only inside this quad (excludes pockets)",
    )
    p_play.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_play.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera index or MJPEG URL (live frame; no --image needed)",
    )
    p_play.add_argument("--image", type=Path, default=None)

    p_felt = sub.add_parser("felt-sample", help="Hover mouse to read HSV on red felt")
    p_felt.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_felt.add_argument("--camera", type=int, default=0)
    p_felt.add_argument("--image", type=Path, default=None)

    p_proj = sub.add_parser("projector", help="Projector pixels -> table plane")
    p_proj.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p_proj.add_argument("--camera", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "table":
        return calibrate_table(
            args.config,
            parse_camera_arg(args.camera),
            image_path=args.image,
            wide=args.wide,
            region=args.region,
            dst_corners_mm=args.dst_corners_mm,
        )
    if args.command == "capture":
        return capture_snapshot(
            args.config, parse_camera_arg(args.camera), args.output, wide=args.wide
        )
    if args.command == "probe":
        return run_probe(args.camera)
    if args.command == "doctor":
        return print_camera_doctor(args.max_index)
    if args.command == "lens":
        return calibrate_lens(
            args.config,
            args.camera,
            pattern_cols=args.cols,
            pattern_rows=args.rows,
            square_mm=args.square_mm,
        )
    if args.command == "lens-aruco":
        return calibrate_lens_aruco(
            args.config,
            args.camera,
            square_mm=args.square_mm,
            show_pattern=not args.no_fullscreen,
        )
    if args.command == "aruco-pattern":
        save_aruco_pattern(args.output, square_mm=args.square_mm)
        print(f"Saved {args.output.resolve()}")
        print("Display on a phone; measure one square; run lens-aruco with that --square-mm")
        return 0
    if args.command == "lens-estimate":
        return estimate_lens_from_table(args.config, camera_height_mm=args.camera_height_mm)
    if args.command == "verify-lens":
        return verify_lens(
            args.config,
            args.camera,
            image_path=args.image,
            output=args.output,
        )
    if args.command == "play-region":
        cam = parse_camera_arg(args.camera)
        if args.image is not None:
            cam = 0  # unused when --image set
        return calibrate_play_region(args.config, image_path=args.image, camera=cam)
    if args.command == "felt-sample":
        return run_felt_sample(args.config, args.camera, args.image)
    if args.command == "projector":
        return calibrate_projector(args.config, args.camera)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
