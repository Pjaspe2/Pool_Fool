from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pool_fool.shared.table import TableSpec


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def table_spec_from_config(cfg: dict[str, Any]) -> TableSpec:
    t = cfg["table"]
    return TableSpec(
        width_mm=float(t["width_mm"]),
        length_mm=float(t["length_mm"]),
        ball_radius_mm=float(t["ball_radius_mm"]),
    )


def resolve_path(cfg: dict[str, Any], key: str, base: Path | None = None) -> Path:
    rel = cfg["paths"][key]
    root = base or Path.cwd()
    return (root / rel).resolve()
