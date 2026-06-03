from __future__ import annotations

import argparse
from pathlib import Path

from pool_fool.desktop.annotate.yolo_annotator import run_annotator
from pool_fool.desktop.annotate.yolo_format import load_class_names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw YOLO boxes on pool table images (for custom training)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Datasets/my_table"),
        help="Dataset root (creates train/images + train/labels + data.yaml)",
    )
    parser.add_argument(
        "--classes-yaml",
        type=Path,
        default=Path("config/annotation_classes_v2.yaml"),
        help="Class names (v2: balls + Pocket + Cue_Stick)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Only two classes: Cue_Ball, Object_Ball (faster labeling)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Folder of images to label (or train/images inside a dataset)",
    )
    parser.add_argument(
        "--stream",
        type=str,
        default=None,
        help="MJPEG URL to grab frames (e.g. http://pool.local:8080/stream.mjpg)",
    )
    parser.add_argument("--index", type=int, default=0, help="Start image index")
    args = parser.parse_args(argv)

    if args.simple:
        class_names = ["Cue_Ball", "Object_Ball"]
    else:
        class_names = load_class_names(args.classes_yaml)

    return run_annotator(
        output_dir=args.output,
        class_names=class_names,
        images_dir=args.images_dir,
        stream_url=args.stream,
        start_index=args.index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
