from __future__ import annotations

import argparse
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2

from pool_fool.edge.capture import CameraCapture, DualCameraCapture
from pool_fool.edge.display import ProjectorDisplay
from pool_fool.edge.overlay_receiver import EdgeOverlayReceiver
from pool_fool.edge.stream_server import FfmpegRtspPublisher, MjpegHttpServer
from pool_fool.shared.config import load_config
from pool_fool.shared.schemas import OverlayMessage


def _lan_ip() -> str | None:
    """Best-effort IPv4 on the default route (for printing Mac-facing URLs)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def _print_stream_urls(port: int) -> None:
    hostname = socket.gethostname().split(".")[0] or "raspberrypi"
    print(f"Streaming MJPEG on port {port} (bound to all interfaces)")
    print(f"  On the Pi:       http://127.0.0.1:{port}/stream.mjpg")
    print(f"  From your Mac:   http://{hostname}.local:{port}/stream.mjpg")
    ip = _lan_ip()
    if ip:
        print(f"  Or by IP:        http://{ip}:{port}/stream.mjpg")
    print("  Do not use http://0.0.0.0 on the Mac — that only works on the Pi.")


def _run_mjpeg_http(server: MjpegHttpServer) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path not in ("/", "/stream.mjpg"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                jpeg = server.get_latest()
                if jpeg is None:
                    time.sleep(0.03)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1.0 / 30.0)
                except BrokenPipeError:
                    break

        def log_message(self, format, *args):  # noqa: A003
            return

    httpd = HTTPServer((server.host, server.port), Handler)
    httpd.serve_forever()


def run_stream_mode(config_path: Path, camera: int, use_rtsp: bool) -> int:
    cfg = load_config(config_path)
    cam_cfg = cfg["cameras"]
    cap = CameraCapture(
        camera,
        width=int(cam_cfg.get("width", 1280)),
        height=int(cam_cfg.get("height", 720)),
        fps=int(cam_cfg.get("fps", 30)),
        manual_exposure=bool(cam_cfg.get("manual_exposure", True)),
        exposure=float(cam_cfg.get("exposure", -6)),
    )
    if not cap.open():
        print(f"Cannot open camera {camera}")
        return 1

    w, h = int(cam_cfg.get("width", 1280)), int(cam_cfg.get("height", 720))
    fps = int(cam_cfg.get("fps", 30))
    mjpeg = MjpegHttpServer(host="0.0.0.0", port=8080)
    threading.Thread(target=_run_mjpeg_http, args=(mjpeg,), daemon=True).start()

    rtsp: FfmpegRtspPublisher | None = None
    if use_rtsp:
        net = cfg["network"]
        url = f"rtsp://{net.get('stream_host', '0.0.0.0')}:{net.get('stream_port', 8554)}{net.get('stream_path', '/pool')}"
        rtsp = FfmpegRtspPublisher(w, h, fps, rtsp_url=url)
        try:
            rtsp.start()
            print(f"RTSP publishing to {url}")
        except FileNotFoundError:
            print("ffmpeg not found; MJPEG only on :8080/stream.mjpg")
            rtsp = None

    _print_stream_urls(8080)
    try:
        while True:
            pkt = cap.read()
            if pkt is None:
                continue
            _, jpeg = cv2.imencode(".jpg", pkt.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            mjpeg.update_jpeg(jpeg.tobytes())
            if rtsp:
                rtsp.write_frame(pkt.frame)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if rtsp:
            rtsp.stop()
    return 0


def run_display_mode(config_path: Path, overlay_port: int) -> int:
    cfg = load_config(config_path)
    latest: list[OverlayMessage | None] = [None]

    def on_msg(msg: OverlayMessage) -> None:
        latest[0] = msg

    receiver = EdgeOverlayReceiver(overlay_port, on_msg)
    receiver.start()

    try:
        display = ProjectorDisplay.from_config(config_path)
    except FileNotFoundError as e:
        print(f"Projector calibration required: {e}")
        return 1

    print(f"Listening for overlay UDP on port {overlay_port}")
    try:
        while True:
            display.show(latest[0])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        cv2.destroyAllWindows()
    return 0


def run_combined(config_path: Path, camera: int, overlay_port: int, use_rtsp: bool) -> int:
    """Stream video + show projector overlay (typical Pi deployment)."""
    stream_exit = threading.Event()

    def stream_worker() -> None:
        run_stream_mode(config_path, camera, use_rtsp)
        stream_exit.set()

    t = threading.Thread(target=stream_worker, daemon=True)
    t.start()
    time.sleep(0.5)
    return run_display_mode(config_path, overlay_port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pool Fool edge (Raspberry Pi)")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument(
        "--mode",
        choices=["stream", "display", "combined"],
        default="combined",
    )
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--overlay-port", type=int, default=None)
    parser.add_argument("--rtsp", action="store_true", help="Use ffmpeg RTSP (else MJPEG HTTP only)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    camera = args.camera if args.camera is not None else int(cfg["cameras"].get("overhead_index", 0))
    overlay_port = args.overlay_port or int(cfg["network"].get("desktop_overlay_port", 8765))

    if args.mode == "stream":
        return run_stream_mode(args.config, camera, args.rtsp)
    if args.mode == "display":
        return run_display_mode(args.config, overlay_port)
    return run_combined(args.config, camera, overlay_port, args.rtsp)


if __name__ == "__main__":
    raise SystemExit(main())
