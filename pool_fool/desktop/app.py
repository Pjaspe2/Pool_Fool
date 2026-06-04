from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from pool_fool.desktop.compose.overlay import (
    OverlayRenderer,
    draw_debug_frame,
    draw_pocket_shot_frame,
    draw_selected_pocket_marker,
)
from pool_fool.desktop.latency import FrameTimer, LatencyStats
from pool_fool.desktop.network.overlay_udp import OverlaySender
from pool_fool.desktop.network.stream_client import MjpegStreamClient
from pool_fool.desktop.physics.ghost_ball import solve_shot
from pool_fool.desktop.physics.pocket_shot import (
    PocketShotResult,
    solve_poc_pocket_shot,
    solve_target_pocket_shot,
)
from pool_fool.desktop.session import SessionLogger
from pool_fool.desktop.vision.detector_factory import create_ball_detector, detector_mode_label
from pool_fool.desktop.vision.cue import CueDetector
from pool_fool.desktop.vision.fusion import fuse_cue_direction
from pool_fool.desktop.vision.tracking import BallTracker, StationaryGate
from pool_fool.shared.camera import CameraOpenError, capture_frame, open_camera
from pool_fool.shared.config import load_config, resolve_path, table_spec_from_config
from pool_fool.shared.table_draw import draw_table_layout
from pool_fool.shared.table_layout import TableLayout
from pool_fool.shared.frame_pipeline import build_lens_corrector, preprocess_frame
from pool_fool.shared.homography import load_homography
from pool_fool.shared.play_region import PlayRegion
from pool_fool.shared.schemas import OverlayMessage


def _is_stream_url(source: str | int) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://", "rtsp://"))


POCKET_KEY_TO_INDEX = {ord(str(d)): d - 1 for d in range(1, 7)}


def _session_cfg(cfg: dict, root: Path) -> dict:
    raw = cfg.get("session")
    if not isinstance(raw, dict):
        return {"enabled": False}
    out = dict(raw)
    log_dir = out.get("log_dir", "logs")
    out["log_dir"] = (root / log_dir).resolve() if not Path(str(log_dir)).is_absolute() else Path(log_dir)
    out.setdefault("enabled", False)
    out.setdefault("track_interval_s", 0.5)
    out.setdefault("track_when_moving", False)
    return out


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
    table_cfg = cfg.get("table", {})
    show_pockets = bool(table_cfg.get("show_pockets_debug", True))
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
    expand_scale = float(vision.get("play_region_expand_scale", 1.0))
    if play_region is not None and expand_scale > 1.0:
        play_region = play_region.expanded(expand_scale, table)
        print(f"Play region expanded {expand_scale:.0%} from saved calibration")

    from pool_fool.shared.config import resolve_path as resolve_cfg_path
    from pool_fool.shared.pocket_calibration import load_calibrated_pockets

    layout = TableLayout.resolve(cfg, root, play_region)
    if load_calibrated_pockets(resolve_cfg_path(cfg, "pockets", root)):
        print("Pocket model: calibrated clicks (pool-fool-calibrate pockets).")
    elif play_region is not None and bool(table_cfg.get("pockets_from_play_region", True)):
        print("Pocket model: estimated from play-region quad.")
    else:
        print("Pocket model: config table rectangle.")
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
        return 1
    detector_label = detector_mode_label(vision)
    print(f"Ball detector: {detector_label}")
    print("  Magenta/green boxes = YOLO. Orange quad = play-region. Cyan = pocket model.")
    session_cfg = _session_cfg(cfg, root)
    session_enabled = bool(session_cfg.get("enabled", False))
    aim_mode = str(vision.get("aim_mode", "none")).lower()
    poc_pocket = aim_mode in ("poc_pocket", "poc", "easiest_pocket", "session_pocket")
    poc_require_stationary = bool(vision.get("poc_shot_require_stationary", True))
    use_cue_line = bool(vision.get("use_cue_line", False)) and not poc_pocket
    if session_enabled:
        poc_pocket = True
        print("  Aim: session mode — pick pocket (1–6), trajectory for nearest object.")
    elif poc_pocket:
        print("  Aim: PoC pocket shot (nearest object → easiest pocket) when balls still.")
    cue_detector: CueDetector | None = CueDetector(vision, H, play_region) if use_cue_line else None
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
        match_gate_mm=float(vision.get("ball_tracker_match_gate_mm", 80.0)),
    )
    gate = StationaryGate(
        float(vision.get("stationary_velocity_mm_s", 45.0)),
        still_frames_required=int(vision.get("stationary_frames_required", 8)),
        velocity_ema_alpha=float(vision.get("stationary_velocity_ema_alpha", 0.2)),
    )

    overlay_renderer: OverlayRenderer | None = None
    try:
        overlay_renderer = OverlayRenderer.from_config(cfg, root)
    except FileNotFoundError:
        overlay_renderer = OverlayRenderer(table, cfg["overlay"], H, None)

    sender: OverlaySender | None = None
    if send_overlay:
        net = cfg["network"]
        host = net["overlay_udp_host"]
        port = int(net["overlay_udp_port"])
        sender = OverlaySender(host, port)
        print(f"Overlay UDP → {host}:{port} (Pi must run edge --mode combined; needs projector cal)")

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

    if use_mjpeg:
        print(
            "Browser URL is raw Pi video only — no overlay in Safari/Chrome.\n"
            "This app opens a separate window: pool_fool_debug (YOLO circles + aim lines).\n"
            "If the stream fails: restart pool-fool-edge on the Pi (git pull for threaded server),\n"
            "  or close the browser tab — old Pi server allowed only one viewer."
        )
    session_logger: SessionLogger | None = None
    selected_pocket_idx = 0
    session_flash = ""
    session_flash_until = 0.0
    if session_enabled:
        log_root = Path(session_cfg["log_dir"])
        session_logger = SessionLogger(
            log_root,
            meta={
                "config": str(config_path.resolve()),
                "table_length_mm": table.length_mm,
                "table_width_mm": table.width_mm,
            },
        )
        print(f"Session log: {session_logger.paths.session_dir}")
        for i, p in enumerate(layout.pockets):
            print(f"  {i + 1} = {p.id}")
        print("Keys: 1–6 pocket  [ ] prev/next  m=made  x=miss  q=quit  r=lock cue")
    else:
        print("Keys: q=quit  r=lock cue ball  (orange outline = play-area mask)")

    cv2.namedWindow("pool_fool_debug", cv2.WINDOW_NORMAL)
    waiting_logged = False

    last_result = None
    last_poc_shot: PocketShotResult | None = None
    fps_t0 = time.monotonic()
    frames = 0
    frame_timer = FrameTimer()
    latency_stats = LatencyStats()

    while True:
        if use_mjpeg:
            frame = latest_frame[0]
            if frame is None:
                if not waiting_logged:
                    err = stream_client.last_error if stream_client else None
                    hint = f" ({err})" if err else ""
                    print(f"Waiting for frames from stream…{hint}")
                    waiting_logged = True
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Waiting for Pi stream...",
                    (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 200, 200),
                    2,
                )
                cv2.imshow("pool_fool_debug", placeholder)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            waiting_logged = False
        else:
            ret, frame = cap.read()
            if not ret:
                break

        latency_stats.record(frame_timer.tick())
        frame = preprocess_frame(frame, lens)

        raw_balls = ball_detector.detect(frame)
        raw_count = len(raw_balls)
        balls = tracker.update(raw_balls)
        stationary = gate.update(balls)

        cue_ball, objects = ball_detector.split_cue_and_objects(balls)
        result = None
        poc_shot: PocketShotResult | None = None
        can_aim = stationary or not poc_require_stationary
        max_cut = float(vision.get("poc_max_cut_angle_deg", 48.0))
        if poc_pocket and cue_ball is not None and objects and can_aim:
            obj_centers = [o.center_mm for o in objects]
            if session_enabled:
                target = layout.pocket_at_index(selected_pocket_idx)
                poc_shot = solve_target_pocket_shot(
                    cue_ball.center_mm,
                    obj_centers,
                    layout,
                    table,
                    target,
                    max_cut_angle_deg=max_cut,
                )
            else:
                poc_shot = solve_poc_pocket_shot(
                    cue_ball.center_mm,
                    obj_centers,
                    layout,
                    table,
                    max_cut_angle_deg=max_cut,
                )
            if poc_shot.valid:
                last_poc_shot = poc_shot
        elif (
            poc_pocket
            and not can_aim
            and last_poc_shot is not None
            and last_poc_shot.valid
        ):
            # Freeze last good line only while balls are moving (not stale on stationary)
            poc_shot = last_poc_shot

        if session_logger is not None:
            session_logger.maybe_log_tracks(
                balls,
                stationary=stationary,
                interval_s=float(session_cfg.get("track_interval_s", 0.5)),
                when_moving=bool(session_cfg.get("track_when_moving", False)),
            )

        if cue_ball is not None and cue_detector is not None:
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
        elif last_result is not None and cue_detector is not None and cue_detector._last_direction is not None:
            pass

        if result is None and last_result is not None and use_cue_line:
            result = last_result

        if poc_shot is not None:
            session_label = None
            if session_enabled:
                pid = layout.pocket_at_index(selected_pocket_idx).id
                cut = f"{poc_shot.cut_angle_deg:.0f}°" if poc_shot.valid else "—"
                session_label = f"→ {pid}  cut {cut}  |  m made  x miss"
            vis = draw_pocket_shot_frame(
                frame, H_inv, poc_shot, cfg["overlay"], session_label=session_label
            )
            if session_enabled:
                sel = layout.pocket_at_index(selected_pocket_idx)
                draw_selected_pocket_marker(vis, H_inv, sel.center_mm, pocket_id=sel.id)
        elif result is not None:
            vis = draw_debug_frame(frame, H_inv, result, cfg["overlay"], table)
        else:
            vis = frame.copy()
        if show_pockets:
            # Orange play-region already draws the quad; skip duplicate gray border.
            draw_border = layout.border_corners_mm is None
            draw_table_layout(
                vis,
                H_inv,
                layout,
                label_pockets=bool(table_cfg.get("label_pockets", False)),
                draw_border=draw_border,
            )
        if play_region is not None:
            play_region.draw(vis, H_inv)

        for b in balls:
            if b.bbox_px is not None:
                x1, y1, x2, y2 = b.bbox_px
                color = (0, 255, 120) if b.is_cue else (255, 0, 255)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(vis, (int(b.center_px[0]), int(b.center_px[1])), 3, color, -1)
            else:
                px = (int(b.center_px[0]), int(b.center_px[1]))
                color = (255, 255, 255) if b.is_cue else (200, 120, 50)
                cv2.circle(vis, px, int(b.radius_px), color, 2)

        status = "STATIONARY" if stationary else "MOVING"
        cv2.putText(
            vis,
            f"Balls: {detector_label}",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            vis,
            f"tracked {len(balls)}  raw {raw_count}",
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
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
            (20, vis.shape[0] - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )
        if session_logger is not None:
            sel_id = layout.pocket_at_index(selected_pocket_idx).id
            cv2.putText(
                vis,
                f"LOG {session_logger.paths.session_dir.name}  "
                f"events {session_logger.event_count}  pocket [{selected_pocket_idx + 1}] {sel_id}",
                (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 200, 120),
                1,
                cv2.LINE_AA,
            )
            if time.monotonic() < session_flash_until and session_flash:
                cv2.putText(
                    vis,
                    session_flash,
                    (20, 156),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
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

        if show_projector_preview and overlay_renderer:
            pw = cfg["projector"]["display_width"]
            ph = cfg["projector"]["display_height"]
            if poc_shot is not None and poc_shot.valid:
                proj = overlay_renderer.render_projector_frame(poc_shot, pw, ph)
            elif result is not None and result.valid:
                proj = overlay_renderer.render_projector_frame(result, pw, ph)
            else:
                proj = None
            if proj is not None:
                cv2.imshow("pool_fool_projector_preview", proj)

        if sender and overlay_renderer:
            guide = None
            if poc_shot is not None and poc_shot.valid:
                guide = overlay_renderer.pocket_shot_to_guide(poc_shot)
            elif result is not None and result.valid:
                guide = overlay_renderer.shot_to_guide(result)
            if guide is None:
                pass
            else:
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
        if session_enabled and key in POCKET_KEY_TO_INDEX:
            idx = POCKET_KEY_TO_INDEX[key]
            if idx < len(layout.pockets):
                selected_pocket_idx = idx
                sel = layout.pocket_at_index(selected_pocket_idx)
                print(f"Target pocket: [{idx + 1}] {sel.id}")
                last_poc_shot = None
        if session_enabled and key == ord("["):
            selected_pocket_idx = (selected_pocket_idx - 1) % len(layout.pockets)
            print(f"Target pocket: [{selected_pocket_idx + 1}] {layout.pocket_at_index(selected_pocket_idx).id}")
            last_poc_shot = None
        if session_enabled and key == ord("]"):
            selected_pocket_idx = (selected_pocket_idx + 1) % len(layout.pockets)
            print(f"Target pocket: [{selected_pocket_idx + 1}] {layout.pocket_at_index(selected_pocket_idx).id}")
            last_poc_shot = None
        if session_enabled and session_logger and key in (ord("m"), ord("x")):
            outcome = "made" if key == ord("m") else "missed"
            shot = poc_shot if poc_shot is not None else last_poc_shot
            target_id = layout.pocket_at_index(selected_pocket_idx).id
            if shot is None or cue_ball is None:
                print("Nothing to log — need cue + object balls in view.")
            else:
                session_logger.log_shot_outcome(
                    outcome,
                    target_pocket_id=target_id,
                    shot=shot,
                    balls=balls,
                )
                session_flash = f"Logged {outcome.upper()} → {target_id}"
                session_flash_until = time.monotonic() + 2.5
                print(
                    f"{session_flash}  "
                    f"(event #{session_logger.event_count}, "
                    f"valid={shot.valid}, cut={shot.cut_angle_deg:.0f}°)"
                )

    if session_logger is not None:
        session_logger.close()
        print(
            f"Session saved: {session_logger.paths.session_dir} "
            f"({session_logger.event_count} events, {session_logger.track_count} track rows)"
        )
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
