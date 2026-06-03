from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from pool_fool.desktop.annotate.yolo_format import (
    YoloBox,
    load_labels,
    pixel_box_to_yolo,
    save_labels,
    yolo_to_pixel,
)
from pool_fool.desktop.network.stream_client import MjpegStreamClient


@dataclass
class AnnotatorState:
    class_names: list[str]
    class_id: int = 1
    boxes: list[YoloBox] = field(default_factory=list)
    drag_start: tuple[int, int] | None = None
    drag_end: tuple[int, int] | None = None
    dirty: bool = False
    mouse_xy: tuple[int, int] = (0, 0)
    status_msg: str = ""


CLASS_COLORS = [
    (0, 255, 120),
    (255, 0, 255),
    (255, 200, 0),
    (0, 180, 255),
    (180, 100, 255),
    (100, 255, 255),
]


def _class_color(class_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def _draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """Black text with light outline so class names read on red felt / white balls."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
        cv2.putText(
            img,
            text,
            (x + dx, y + dy),
            font,
            scale,
            (255, 255, 255),
            thickness + 1,
            cv2.LINE_AA,
        )
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if (folder / "images").is_dir():
        folder = folder / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def _write_dataset_yaml(out_dir: Path, class_names: list[str]) -> None:
    yaml_path = out_dir / "data.yaml"
    if yaml_path.exists():
        return
    content = {
        "train": "train/images",
        "val": "train/images",
        "test": "train/images",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path.write_text(yaml.dump(content, default_flow_style=False), encoding="utf-8")


def _box_index_at(
    boxes: list[YoloBox], x: int, y: int, img_w: int, img_h: int
) -> int | None:
    """Index of box under (x,y), else nearest box within ~60px of center."""
    inside: list[int] = []
    nearest: tuple[int, float] | None = None
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = yolo_to_pixel(box, img_w, img_h)
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside.append(i)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        if nearest is None or d2 < nearest[1]:
            nearest = (i, d2)
    if inside:
        return inside[-1]
    if nearest is not None and nearest[1] <= 60**2:
        return nearest[0]
    return None


def grab_mjpeg_frame(url: str, timeout_s: float = 10.0) -> np.ndarray | None:
    latest: list[np.ndarray | None] = [None]

    def on_frame(f: np.ndarray) -> None:
        latest[0] = f

    client = MjpegStreamClient(url, on_frame, connect_timeout_s=timeout_s)
    client.start()
    t0 = time.monotonic()
    while latest[0] is None and time.monotonic() - t0 < timeout_s:
        time.sleep(0.05)
    client.stop()
    return latest[0]


def run_annotator(
    *,
    output_dir: Path,
    class_names: list[str],
    images_dir: Path | None = None,
    stream_url: str | None = None,
    start_index: int = 0,
) -> int:
    """
    Fast local YOLO box labeling.

    Keys:
      drag LMB     draw box (current class)
      Tab / , .    prev / next class
      0-9          class 0-9 (if defined)
      u / Backspace  undo last box
      d / x          delete box under mouse cursor
      r            clear all boxes on this image
      s            save labels + next image
      n / p        next / prev image (folder mode)
      g            grab frame from --stream (saved to output)
      q            save & quit
    """
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    _write_dataset_yaml(output_dir, class_names)

    image_paths: list[Path] = []
    if images_dir is not None:
        image_paths = _list_images(images_dir)
    else:
        image_paths = _list_images(train_images)

    idx = max(0, min(start_index, max(0, len(image_paths) - 1)))
    capture_counter = len(image_paths)
    state = AnnotatorState(class_names=class_names, class_id=min(1, len(class_names) - 1))
    window = "pool_fool_annotate"
    current_path: Path | None = None
    frame: np.ndarray | None = None
    def load_image(path: Path) -> None:
        nonlocal frame, current_path, state
        img = cv2.imread(str(path))
        if img is None:
            print(f"Cannot read {path}")
            return
        frame = img
        current_path = path
        label_path = train_labels / f"{path.stem}.txt"
        state.boxes = load_labels(label_path)
        state.dirty = False
        state.drag_start = None
        state.drag_end = None

    def save_current() -> None:
        nonlocal state, current_path
        if frame is None or current_path is None:
            return
        try:
            current_path.resolve().relative_to(train_images.resolve())
        except ValueError:
            dest = train_images / current_path.name
            if not dest.exists() or dest.resolve() != current_path.resolve():
                cv2.imwrite(str(dest), frame)
                print(f"Copied image -> {dest}")
                current_path = dest
        label_path = train_labels / f"{current_path.stem}.txt"
        save_labels(label_path, state.boxes)
        state.dirty = False
        print(f"Saved {len(state.boxes)} boxes -> {label_path}")

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        state.mouse_xy = (x, y)
        if frame is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drag_start = (x, y)
            state.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.drag_start is not None:
            state.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state.drag_start is not None:
            state.drag_end = (x, y)
            x1, y1 = state.drag_start
            x2, y2 = state.drag_end
            if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                h, w = frame.shape[:2]
                cname = class_names[state.class_id]
                state.boxes.append(pixel_box_to_yolo(x1, y1, x2, y2, state.class_id, w, h))
                state.dirty = True
                state.status_msg = f"Added {cname} ({len(state.boxes)} boxes)"
            state.drag_start = None
            state.drag_end = None
        elif event == cv2.EVENT_MOUSEMOVE:
            state.mouse_xy = (x, y)

    def undo_last_box() -> None:
        if not state.boxes:
            state.status_msg = "No boxes to undo"
            return
        removed = state.boxes.pop()
        state.dirty = True
        name = class_names[removed.class_id] if removed.class_id < len(class_names) else "?"
        state.status_msg = f"Undid {name} ({len(state.boxes)} boxes left)"

    def delete_box_at_cursor() -> None:
        if frame is None or not state.boxes:
            state.status_msg = "No boxes to delete"
            return
        h, w = frame.shape[:2]
        bi = _box_index_at(state.boxes, state.mouse_xy[0], state.mouse_xy[1], w, h)
        if bi is None:
            state.status_msg = "No box under cursor — use u to undo last"
            return
        removed = state.boxes.pop(bi)
        state.dirty = True
        name = class_names[removed.class_id] if removed.class_id < len(class_names) else "?"
        state.status_msg = f"Deleted {name} ({len(state.boxes)} boxes left)"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    print("Pool Fool YOLO annotator")
    print(f"  Output: {output_dir.resolve()}")
    print(f"  Classes ({len(class_names)}):")
    for i, name in enumerate(class_names):
        key_hint = str(i) if i <= 9 else "Tab"
        print(f"    [{key_hint}] {i}: {name}")
    for line in [
        "Drag: box   Tab/,: class   0-9: class id",
        "g: grab   s: save   p: prev image (fix mistakes)   n: next",
        "u / Backspace: undo last box   d / x: delete box under mouse   r: clear all",
        "q: quit",
    ]:
        print(f"  {line}")

    if image_paths:
        load_image(image_paths[idx])
    elif stream_url:
        print("Press g to grab a frame from the Pi stream.")
    else:
        print("No images yet — press g (with --stream) or add files to train/images/")

    while True:
        if frame is None:
            placeholder = np.zeros((480, 800, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "No image — g=grab stream, or use --images-dir",
                (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            cv2.imshow(window, placeholder)
        else:
            vis = frame.copy()
            h, w = vis.shape[:2]
            for box in state.boxes:
                x1, y1, x2, y2 = yolo_to_pixel(box, w, h)
                color = _class_color(box.class_id)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                name = class_names[box.class_id] if box.class_id < len(class_names) else str(box.class_id)
                _draw_text(vis, name, (x1, max(14, y1 - 4)), scale=0.45, thickness=1)
            if state.drag_start and state.drag_end:
                cv2.rectangle(vis, state.drag_start, state.drag_end, _class_color(state.class_id), 2)
            cname = class_names[state.class_id] if state.class_id < len(class_names) else "?"
            title = f"class [{state.class_id}] {cname}  |  boxes {len(state.boxes)}"
            if current_path:
                title += f"  |  {current_path.name}"
            if state.dirty:
                title += " *"
            _draw_text(vis, title, (10, 28), scale=0.65, thickness=2)
            if state.status_msg:
                _draw_text(vis, state.status_msg[:70], (10, 56), scale=0.5, thickness=1)
            help_y = vis.shape[0] - 14
            _draw_text(
                vis,
                "u=undo  d/x=delete under mouse  r=clear  p=prev  s=save",
                (10, help_y),
                scale=0.45,
                thickness=1,
            )
            if image_paths:
                _draw_text(
                    vis,
                    f"image {idx + 1}/{len(image_paths)}",
                    (10, 82),
                    scale=0.5,
                    thickness=1,
                )
            cv2.imshow(window, vis)

        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key == ord("q"):
            if state.dirty:
                save_current()
            break
        if key == ord("s"):
            save_current()
            if image_paths and idx < len(image_paths) - 1:
                idx += 1
                load_image(image_paths[idx])
            continue
        if key == ord("n") and image_paths:
            if state.dirty:
                save_current()
            idx = min(idx + 1, len(image_paths) - 1)
            load_image(image_paths[idx])
            continue
        if key == ord("p") and image_paths:
            if state.dirty:
                save_current()
            idx = max(idx - 1, 0)
            load_image(image_paths[idx])
            continue
        if key in (ord("u"), 8, 127):  # u, Backspace, Delete
            undo_last_box()
            print(state.status_msg)
            continue
        if key in (ord("d"), ord("x")):
            delete_box_at_cursor()
            print(state.status_msg)
            continue
        if key == ord("r"):
            n = len(state.boxes)
            state.boxes.clear()
            state.dirty = n > 0
            state.status_msg = f"Cleared {n} boxes" if n else "No boxes"
            print(state.status_msg)
            continue
        if key == ord("\t") or key == ord("."):
            state.class_id = (state.class_id + 1) % len(class_names)
            continue
        if key == ord(","):
            state.class_id = (state.class_id - 1) % len(class_names)
            continue
        if ord("0") <= key <= ord("9"):
            cid = key - ord("0")
            if cid < len(class_names):
                state.class_id = cid
            continue
        if key == ord("g"):
            url = stream_url
            if url is None:
                print("Set --stream http://pool.local:8080/stream.mjpg")
                continue
            print(f"Grabbing frame from {url} ...")
            grabbed = grab_mjpeg_frame(url)
            if grabbed is None:
                print("Grab failed — is pool-fool-edge running?")
                continue
            capture_counter += 1
            out_name = f"capture_{capture_counter:04d}.jpg"
            out_path = train_images / out_name
            cv2.imwrite(str(out_path), grabbed)
            image_paths.append(out_path)
            idx = len(image_paths) - 1
            load_image(out_path)
            print(f"Saved {out_path}")
            continue

    cv2.destroyAllWindows()
    print(f"Dataset folder: {output_dir.resolve()}")
    print("Add more images to train/images/ or grab with g, then re-train with data=<output>/data.yaml")
    return 0
