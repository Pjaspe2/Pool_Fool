#!/usr/bin/env python3
"""
Merge Roboflow Pool Billiard (12 classes) + your my_red_felt (--simple 2 classes).

Remaps your labels:
  Cue_Ball (0) -> 1
  Object_Ball (1) -> 6

Use --custom-repeat 4 to oversample your table (~4x more gradient on red felt).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

# pool-fool-annotate --simple class ids -> Pool Billiard.yolov8 names
SIMPLE_TO_FULL: dict[int, int] = {
    0: 1,  # Cue_Ball
    1: 6,  # Object_Ball
}

FULL_NAMES = [
    "Break",
    "Cue_Ball",
    "Eight",
    "Five",
    "Four",
    "Nine",
    "Object_Ball",
    "One",
    "Seven",
    "Six",
    "Three",
    "Two",
]


def remap_label(src: Path, dst: Path) -> None:
    lines_out: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        if cid not in SIMPLE_TO_FULL:
            print(f"  skip unknown class {cid} in {src.name}")
            continue
        parts[0] = str(SIMPLE_TO_FULL[cid])
        lines_out.append(" ".join(parts))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def copy_split(
    src_root: Path,
    split: str,
    dst_images: Path,
    dst_labels: Path,
    *,
    prefix: str = "",
    repeat: int = 1,
    remap: bool = False,
) -> int:
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    if not src_images.is_dir():
        return 0
    n = 0
    for img in sorted(src_images.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label = src_labels / f"{img.stem}.txt"
        for r in range(repeat):
            suffix = f"_{r}" if r > 0 else ""
            out_stem = f"{prefix}{img.stem}{suffix}"
            out_img = dst_images / f"{out_stem}{img.suffix}"
            out_lbl = dst_labels / f"{out_stem}.txt"
            shutil.copy2(img, out_img)
            if label.exists():
                if remap:
                    remap_label(label, out_lbl)
                else:
                    shutil.copy2(label, out_lbl)
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge pool YOLO datasets for training")
    parser.add_argument(
        "--roboflow",
        type=Path,
        default=Path("Datasets/Pool Billiard.yolov8"),
    )
    parser.add_argument(
        "--custom",
        type=Path,
        default=Path("Datasets/my_red_felt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Datasets/pool_combined"),
    )
    parser.add_argument(
        "--custom-repeat",
        type=int,
        default=4,
        help="How many times to copy each custom image (weight red felt higher)",
    )
    args = parser.parse_args()

    out = args.output
    train_img = out / "train" / "images"
    train_lbl = out / "train" / "labels"
    valid_img = out / "valid" / "images"
    valid_lbl = out / "valid" / "labels"
    for d in (train_img, train_lbl, valid_img, valid_lbl):
        d.mkdir(parents=True, exist_ok=True)

    n_rf = copy_split(args.roboflow, "train", train_img, train_lbl, prefix="rf_")
    n_val = copy_split(args.roboflow, "valid", valid_img, valid_lbl, prefix="rf_")
    n_custom = copy_split(
        args.custom,
        "train",
        train_img,
        train_lbl,
        prefix="red_",
        repeat=max(1, args.custom_repeat),
        remap=True,
    )

    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "valid/images",
        "nc": len(FULL_NAMES),
        "names": FULL_NAMES,
    }
    (out / "data.yaml").write_text(yaml.dump(data_yaml, default_flow_style=False), encoding="utf-8")

    print(f"Wrote {out.resolve()}")
    print(f"  Roboflow train images: {n_rf}")
    print(f"  Roboflow valid images: {n_val}")
    print(f"  Your table (x{args.custom_repeat}): {n_custom} file copies")
    print(f"  Train total images: {len(list(train_img.iterdir()))}")
    print("Next:")
    print(
        '  yolo detect train model=yolov8n.pt '
        f'data="{out / "data.yaml"}" epochs=80 imgsz=640 batch=8 name=pool_combined project=.'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
