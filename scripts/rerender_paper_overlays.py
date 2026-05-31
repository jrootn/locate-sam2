#!/usr/bin/env python3
"""Re-export paper figure overlays without query text burned into PNGs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.adapter import AdapterConfig
from locate_sam2.config import load_config, resolve_path
from locate_sam2.pipeline import LocateSam2Pipeline
from locate_sam2.visualize import save_overlay

# README + qual_figures.tex cases
DEFAULT_DIRS = [
    "win_ref5466",
    "win_ref2764",
    "fail_ref2885",
    "both_ref3281",
    "fail_wrong_instance_ref20398",
    "fail_wrong_instance_ref13650",
    "fail_spatial_ref24664",
    "fail_spatial_ref5750",
    "fail_attribute_ref4832",
    "fail_attribute_ref18360",
    "fail_rare_or_long_ref36776",
    "fail_rare_or_long_ref32628",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "research_paper" / "figures")
    parser.add_argument("--dirs", nargs="*", default=DEFAULT_DIRS)
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])

    ours_pipe = LocateSam2Pipeline(
        grounder="locateanything",
        adapter_config=AdapterConfig(crop_mode="crop", rerank="best_score"),
        generation_mode="hybrid",
    )
    dino_pipe = LocateSam2Pipeline(
        grounder="grounding_dino_tiny",
        adapter_config=AdapterConfig(crop_mode="crop", rerank="best_score"),
    )

    for name in args.dirs:
        out_dir = args.paper_dir / name
        meta_path = out_dir / "metadata.json"
        query_path = out_dir / "query.txt"
        if not meta_path.exists() and not query_path.exists():
            print(f"SKIP {name} (no metadata)")
            continue

        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        query = query_path.read_text().strip() if query_path.exists() else meta.get("query", "")
        image_id = meta.get("image_id")
        if image_id is None:
            print(f"SKIP {name} (no image_id)")
            continue

        image_path = data_dir / "train2014" / f"COCO_train2014_{image_id:012d}.jpg"
        if not image_path.exists():
            print(f"SKIP {name} (missing {image_path.name})")
            continue

        image = Image.open(image_path).convert("RGB")
        image.save(out_dir / "image_raw.jpg")

        ores = ours_pipe.run_path(image_path, query)
        dres = dino_pipe.run_path(image_path, query)

        if ores.masks:
            save_overlay(image, ores.boxes, ores.masks, out_dir / "ours_overlay.png")
        if dres.masks:
            save_overlay(image, dres.boxes, dres.masks, out_dir / "dino_overlay.png")

        print(f"OK {name}")

    print("Done.")


if __name__ == "__main__":
    main()
