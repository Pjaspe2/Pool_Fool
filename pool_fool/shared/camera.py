from __future__ import annotations

import platform
import sys
import time
from typing import Any

import cv2


WIDE_MODE_CANDIDATES: list[tuple[int, int]] = [
    (640, 480),
    (800, 600),
    (1280, 720),
    (1920, 1080),
]


def is_stream_url(source: str | int) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://", "rtsp://"))


def parse_camera_arg(raw: str) -> str | int:
    """CLI value: integer index or MJPEG/RTSP URL."""
    if raw.isdigit():
        return int(raw)
    return raw


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _backend_for_macos() -> int | None:
    if not _is_macos():
        return None
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.CAP_AVFOUNDATION
    return None


def _make_capture(index: int, backend: int | None = None) -> cv2.VideoCapture:
    if backend is not None:
        return cv2.VideoCapture(index, backend)
    return cv2.VideoCapture(index)


def configure_capture(cap: cv2.VideoCapture, cam_cfg: dict[str, Any]) -> None:
    width = int(cam_cfg.get("width", 1280))
    height = int(cam_cfg.get("height", 720))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = cam_cfg.get("fps")
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))

    if cam_cfg.get("manual_exposure"):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(cam_cfg.get("exposure", -6)))

    zoom = cam_cfg.get("zoom")
    if zoom is not None:
        cap.set(cv2.CAP_PROP_ZOOM, float(zoom))


def apply_wide_capture(cap: cv2.VideoCapture, cam_cfg: dict[str, Any]) -> tuple[int, int]:
    wide = dict(cam_cfg)
    wide["zoom"] = cam_cfg.get("wide_zoom", 0)
    w, h = cam_cfg.get("wide_width"), cam_cfg.get("wide_height")
    if w and h:
        wide["width"], wide["height"] = int(w), int(h)
    else:
        wide["width"], wide["height"] = 640, 480
    configure_capture(cap, wide)
    return actual_frame_size(cap)


def actual_frame_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h


def read_frame_with_warmup(
    cap: cv2.VideoCapture,
    *,
    attempts: int = 30,
    delay_s: float = 0.05,
) -> tuple[bool, Any]:
    """Many webcams return empty frames until warmed up (especially on macOS)."""
    import numpy as np

    frame: np.ndarray | None = None
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(delay_s)
    return False, frame


def open_camera_raw(
    index: int,
    *,
    backend: int | None = None,
    try_macos_avfoundation: bool = True,
) -> cv2.VideoCapture:
    """Open device; on macOS prefer AVFoundation."""
    if try_macos_avfoundation and backend is None:
        av = _backend_for_macos()
        if av is not None:
            cap = _make_capture(index, av)
            if cap.isOpened():
                return cap
            cap.release()
    return _make_capture(index, backend)


class CameraOpenError(RuntimeError):
    def __init__(self, message: str, *, index: int, hints: list[str] | None = None) -> None:
        super().__init__(message)
        self.index = index
        self.hints = hints or []


def macos_camera_hints() -> list[str]:
    return [
        "System Settings → Privacy & Security → Camera → enable Terminal (not only Cursor).",
        "Fully quit Terminal (Cmd+Q) and reopen after granting permission.",
        "Close Zoom, FaceTime, Photo Booth, or any app using the webcam.",
        "If you use iPhone Continuity Camera, try index 1: --camera 1",
        "Reset permission: tccutil reset Camera com.apple.Terminal  (then reopen Terminal)",
        "No camera needed: take a photo, then: pool-fool-calibrate table --image your.jpg",
    ]


def open_camera(
    index: int,
    cam_cfg: dict[str, Any],
    *,
    wide: bool = False,
) -> cv2.VideoCapture:
    cap = open_camera_raw(index)
    if not cap.isOpened():
        cap.release()
        raise CameraOpenError(
            f"Cannot open camera index {index}.",
            index=index,
            hints=macos_camera_hints() if _is_macos() else [],
        )
    if wide:
        apply_wide_capture(cap, cam_cfg)
    else:
        configure_capture(cap, cam_cfg)
    return cap


def capture_frame(
    index: int,
    cam_cfg: dict[str, Any],
    *,
    wide: bool = False,
    scan_indices: bool = True,
) -> tuple[Any, int, str]:
    """
    Open camera, warmup read, return (frame, index_used, backend_label).
    Tries requested index first, then 0..2 on failure if scan_indices.
    """
    indices = [index]
    if scan_indices:
        indices.extend(i for i in range(3) if i != index)

    last_err: str | None = None
    for idx in indices:
        for label, use_av in (("AVFoundation", True), ("default", False)):
            cap = open_camera_raw(idx, try_macos_avfoundation=use_av)
            if not cap.isOpened():
                cap.release()
                last_err = f"index {idx} ({label}): not opened"
                continue
            try:
                if wide:
                    apply_wide_capture(cap, cam_cfg)
                else:
                    configure_capture(cap, cam_cfg)
                ret, frame = read_frame_with_warmup(cap)
                if ret and frame is not None:
                    return frame, idx, label
                last_err = f"index {idx} ({label}): opened but no frame after warmup"
            finally:
                cap.release()

    hints = macos_camera_hints() if _is_macos() else [
        "Check USB connection and close other apps using the camera.",
        "Use --image path/to/photo.jpg to calibrate without live capture.",
    ]
    msg = "Cannot read a frame from the camera."
    if last_err:
        msg += f" Last attempt: {last_err}."
    raise CameraOpenError(msg, index=index, hints=hints)


def capture_stream_frame(url: str, *, attempts: int = 60) -> Any:
    """Read one JPEG frame from an MJPEG/RTSP URL."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise CameraOpenError(
            f"Cannot open stream: {url}",
            index=-1,
            hints=[
                "Close browser tabs on /stream.mjpg (Pi serves one client on older builds).",
                "Check pool-fool-edge is running on the Pi.",
            ],
        )
    try:
        ret, frame = read_frame_with_warmup(cap, attempts=attempts, delay_s=0.1)
        if not ret or frame is None:
            raise CameraOpenError(
                f"Stream opened but no frame: {url}",
                index=-1,
                hints=["Retry in a few seconds; ensure the Pi camera is connected."],
            )
        return frame
    finally:
        cap.release()


def probe_modes(index: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    cap = open_camera_raw(index)
    if not cap.isOpened():
        return results
    for w, h in WIDE_MODE_CANDIDATES:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_ZOOM, 0)
        aw, ah = actual_frame_size(cap)
        ret, frame = read_frame_with_warmup(cap, attempts=5)
        got = (int(frame.shape[1]), int(frame.shape[0])) if ret and frame is not None else (aw, ah)
        results.append({"requested": (w, h), "actual": got, "readable": ret})
    cap.release()
    return results


def print_camera_doctor(max_index: int = 3) -> int:
    """Print diagnostics; return 0 if any index yields a frame."""
    print(f"Python {sys.version.split()[0]}  OpenCV {cv2.__version__}  OS {platform.system()}")
    if _is_macos():
        print("\nmacOS cameras (system_profiler):")
        try:
            import subprocess

            out = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            text = out.stdout.strip() or out.stderr.strip() or "(no output)"
            for line in text.splitlines()[:25]:
                print(f"  {line}")
            if len(text.splitlines()) > 25:
                print("  ...")
        except Exception as e:
            print(f"  (could not run system_profiler: {e})")

    print("\nTrying OpenCV capture:")
    any_ok = False
    for idx in range(max_index + 1):
        for label, use_av in (("AVFoundation", True), ("default", False)):
            cap = open_camera_raw(idx, try_macos_avfoundation=use_av)
            opened = cap.isOpened()
            ret, frame = (False, None)
            if opened:
                ret, frame = read_frame_with_warmup(cap, attempts=15)
            cap.release()
            status = "OK" if ret and frame is not None else ("open, no frame" if opened else "failed")
            shape = f"{frame.shape[1]}x{frame.shape[0]}" if ret and frame is not None else "-"
            print(f"  index {idx} {label:14} {status:16} {shape}")
            if ret:
                any_ok = True

    if not any_ok:
        print("\n--- Fix checklist ---")
        for h in macos_camera_hints():
            print(f"  • {h}")
        return 1

    print("\nUse the index that showed OK, e.g.: --camera 0")
    return 0
