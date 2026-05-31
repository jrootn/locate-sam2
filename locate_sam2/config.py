from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or ROOT / "configs" / "default.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    return cfg


def resolve_path(cfg: dict[str, Any], *parts: str) -> Path:
    return (cfg["_root"].joinpath(*parts)).resolve()
