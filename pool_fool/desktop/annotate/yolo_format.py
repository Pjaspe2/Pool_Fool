from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class YoloBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def pixel_box_to_yolo(
    x1: float, y1: float, x2: float, y2: float, class_id: int, img_w: int, img_h: int
) -> YoloBox:
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return YoloBox(
        class_id=class_id,
        cx=cx / img_w,
        cy=cy / img_h,
        w=bw / img_w,
        h=bh / img_h,
    )


def yolo_to_pixel(box: YoloBox, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    bw = box.w * img_w
    bh = box.h * img_h
    cx = box.cx * img_w
    cy = box.cy * img_h
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return x1, y1, x2, y2


def load_labels(path: Path) -> list[YoloBox]:
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        boxes.append(
            YoloBox(
                class_id=int(parts[0]),
                cx=float(parts[1]),
                cy=float(parts[2]),
                w=float(parts[3]),
                h=float(parts[4]),
            )
        )
    return boxes


def save_labels(path: Path, boxes: list[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(b.to_line() for b in boxes)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def load_class_names(data_yaml: Path | None) -> list[str]:
    if data_yaml is None or not data_yaml.exists():
        return ["Cue_Ball", "Object_Ball"]
    import yaml

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)
