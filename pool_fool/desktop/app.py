from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from pool_fool.desktop.compose.overlay import OverlayRenderer, draw_debug_frame
from pool_fool.desktop.latency import FrameTimer, LatencyStats
from pool_fool.desktop.network.overlay_udp import OverlaySender
from pool_fool.desktop.network.stream_client import MjpegStreamClient
from pool_fool.desktop.physics.ghost_ball import solve_shot
from pool_fool.desktop.vision.detector_factory import create_ball_detector
from pool_fool.desktop.vision.cue import CueDetector
from pool_fool.desktop.vision.fusion import fuse_cue_direction
from pool_fool.desktop.vision.tracking import BallTracker, StationaryGate
from pool_fool.shared.camera import CameraOpenError, capture_frame, open_camera
from pool_fool.shared.config import load_config, resolve_path, table_spec_from_config
from pool_fool.shared.frame_pipeline import build_lens_corrector, preprocess_frame
from pool_fool.shared.homography import load_homography
from pool_fool.shared.play_region import PlayRegion
from pool_fool.shared.schemas import OverlayMessage


def _is_stream_url(source: str | int) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://", "rtsp://"))


def _side_camera_index(cam_cfg: dict) -> int | None:
    """Return side camera index if configured; None when disabled."""
    raw = cam_cfg.get("side_index")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _open_capture(source: str | int, cfg: dict) -> cv2.VideoCapture:
    cam_cfg = cfg.get("cameras", {})
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if _is_stream_url(source):
        return cv2.VideoCapture(source)
    return open_camera(int(source), cam_cfg)


def run_loop(
    config_path: Path,
    camera: str | int,
    *,
    send_overlay: bool = False,
    show_projector_preview: bool = False,
) -> int:
    cfg = load_config(config_path)
    root = config_path.parent.parent
    table = table_spec_from_config(cfg)
    vision = cfg["vision"]

    H_path = resolve_path(cfg, "table_homography", root)
    if not H_path.exists():
        print(f"Missing table homography: {H_path}")
        print("Run: pool-fool-calibrate table --config", config_path)
        return 1
    H = load_homography(H_path)
    H_inv = np.linalg.inv(H)
    play_path = resolve_path(cfg, "play_region", root)
    play_region = PlayRegion.load(play_path)
    if play_region is None:
        print("No play region mask — pockets/rails may cause false detections.")
        print("  Run: pool-fool-calibrate play-region --config", config_path)
    lens = build_lens_corrector(cfg, root)
    if cfg.get("cameras", {}).get("undistort") and lens is None:
        lens_path = resolve_path(cfg, "lens_calibration", root)
        print(f"Note: cameras.undistort is on but no file at {lens_path}")
        print("  Run: pool-fool-calibrate lens   (or set undistort: false)")

    use_mjpeg = _is_stream_url(camera) if isinstance(camera, str) else False

    try:
        ball_detector = create_ball_detector(vision, table, H, play_region)
    except ImportError as e:
        print(e)
        print("Falling back to classical detector.")
        vision = {**vision, "detector": "classical"}
        ball_detector = create_ball_detector(vision, table, H, play_region)
    print(f"Ball detector: {vision.get('detector', 'classical')}")
    cue_detector = CueDetector(vision, H, play_region)
    side_index = _side_camera_index(cfg.get("cameras", {}))
    side_cap: cv2.VideoCapture | None = None
    side_cue_detector: CueDetector | None = None
    if side_index is not None and not use_mjpeg:
        side_cap = _open_capture(side_index, cfg)
        if side_cap.isOpened():
            side_cue_detector = CueDetector(vision, H, play_region)
        else:
            side_cap = None
    tracker = BallTracker(
        alpha=float(vision.get("ball_tracker_alpha", 0.35)),
    )
    gate = StationaryGate(
        float(vision.get("stationary_velocity_mm_s", 25.0)),
        still_frames_required=int(vision.get("stationary_frames_required", 5)),
    )

    overlay_renderer: OverlayRenderer | None = None
    try:
        overlay_renderer = OverlayRenderer.from_config(cfg, root)
    except FileNotFoundError:
        overlay_renderer = OverlayRenderer(table, cfg["overlay"], H, None)

    sender: OverlaySender | None = None
    if send_overlay:
        net = cfg["network"]
        sender = OverlaySender(net["overlay_udp_host"], int(net["overlay_udp_port"]))

    latest_frame: list[np.ndarray | None] = [None]
    stream_client: MjpegStreamClient | None = None

    if use_mjpeg:
        def on_frame(f: np.ndarray) -> None:
            latest_frame[0] = f

        stream_client = MjpegStreamClient(str(camera), on_frame)
        stream_client.start()
        cap = None
    else:
        try:
            if isinstance(camera, int):
                _, idx_used, _ = capture_frame(camera, cfg.get("cameras", {}))
                print(f"Using camera index {idx_used}")
                cap = open_camera(idx_used, cfg.get("cameras", {}))
            else:
                cap = _open_capture(camera, cfg)
                if not cap.isOpened():
                    print(f"Cannot open camera/source: {camera}")
                    return 1
        except CameraOpenError as e:
            print(e)
            for h in e.hints:
                print(f"  • {h}")
            print("Run: pool-fool-calibrate doctor")
            return 1

    print("Keys: q=quit  r=lock cue ball  (orange outline = play-area mask)")

    last_result = None
    fps_t0 = time.monotonic()
    frames = 0
    frame_timer = FrameTimer()
    latency_stats = LatencyStats()

    while True:
        if use_mjpeg:
            frame = latest_frame[0]
            if frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                break

        latency_stats.record(frame_timer.tick())
        frame = preprocess_frame(frame, lens)

        balls = ball_detector.detect(frame)
        balls = tracker.update(balls)
        stationary = gate.update(balls)

        cue_ball, objects = ball_detector.split_cue_and_objects(balls)
        result = None
        if cue_ball is not None:
            cue_line = cue_detector.detect(frame, cue_ball.center_mm)
            if side_cap is not None and side_cue_detector is not None:
                ret_side, side_frame = side_cap.read()
                if ret_side:
                    side_line = side_cue_detector.detect(side_frame, cue_ball.center_mm)
                    cue_line = fuse_cue_direction(cue_line, side_line)
            if cue_line is not None:
                obj_centers = [o.center_mm for o in objects]
                result = solve_shot(
                    cue_ball.center_mm,
                    cue_line.direction_mm,
                    obj_centers,
                    table,
                    angle_threshold_deg=float(vision.get("aim_angle_threshold_deg", 12.0)),
                )
                last_result = result
        elif last_result is not None and cue_detector._last_direction is not None:
            pass

        if result is None and last_result is not None:
            result = last_result

        vis = draw_debug_frame(frame, H_inv, result, cfg["overlay"], table) if result is not None else frame.copy()
        if play_region is not None:
            play_region.draw(vis, H_inv)

        for b in balls:
            px = (int(b.center_px[0]), int(b.center_px[1]))
            color = (255, 255, 255) if b.is_cue else (200, 120, 50)
            cv2.circle(vis, px, int(b.radius_px), color, 2)

        status = "STATIONARY" if stationary else "MOVING"
        cv2.putText(vis, status, (20, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(
            vis,
            f"vmax {gate.max_velocity_mm_s:.0f} mm/s (need <{gate.threshold:.0f})",
            (20, vis.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            vis,
            latency_stats.format(),
            (20, vis.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )

        frames += 1
        if time.monotonic() - fps_t0 >= 1.0:
            cv2.putText(
                vis,
                f"{frames} fps",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            frames = 0
            fps_t0 = time.monotonic()

        cv2.imshow("pool_fool_debug", vis)

        if show_projector_preview and overlay_renderer and result is not None:
            pw = cfg["projector"]["display_width"]
            ph = cfg["projector"]["display_height"]
            proj = overlay_renderer.render_projector_frame(result, pw, ph)
            cv2.imshow("pool_fool_projector_preview", proj)

        if sender and result is not None and overlay_renderer:
            guide = overlay_renderer.shot_to_guide(result)
            msg = OverlayMessage(
                timestamp_ms=int(time.time() * 1000),
                stationary=stationary,
                shot=guide,
                balls=[
                    {"x_mm": float(b.center_mm[0]), "y_mm": float(b.center_mm[1]), "is_cue": b.is_cue}
                    for b in balls
                ],
            )
            if stationary:
                sender.send(msg)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            if cue_ball is not None:
                ball_detector.set_cue_hint(cue_ball.center_mm.copy())
                print("Cue ball locked to current white circle.")
            else:
                print("No ball detected — put cue ball in view, then press r.")
        if key == ord("c"):
            print("Run pool-fool-calibrate table separately, then restart app.")

    if cap is not None:
        cap.release()
    if side_cap is not None:
        side_cap.release()
    if stream_client:
        stream_client.stop()
    cv2.destroyAllWindows()
    if sender:
        sender.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pool Fool desktop vision loop")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--camera", default="0", help="Camera index or video path")
    parser.add_argument("--send-overlay", action="store_true", help="UDP overlay to Pi")
    parser.add_argument("--projector-preview", action="store_true")
    args = parser.parse_args(argv)

    cam: str | int = args.camera
    if isinstance(cam, str) and cam.isdigit():
        cam = int(cam)

    return run_loop(
        args.config,
        cam,
        send_overlay=args.send_overlay,
        show_projector_preview=args.projector_preview,
    )


if __name__ == "__main__":
    raise SystemExit(main())
