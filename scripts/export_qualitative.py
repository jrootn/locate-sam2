#!/usr/bin/env python3
"""Export structured qualitative comparison figures for the paper."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.adapter import AdapterConfig
from locate_sam2.config import load_config, resolve_path
from locate_sam2.pipeline import LocateSam2Pipeline
from locate_sam2.visualize import save_overlay


def _load(path: Path) -> dict[int, dict]:
    return {r["ref_id"]: r for r in json.loads(path.read_text())}


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-records", type=Path, required=True)
    parser.add_argument("--dino-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full_val/figures"))
    parser.add_argument("--max-per-group", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    ours = _load(args.ours_records)
    dino = _load(args.dino_records)
    common = sorted(set(ours) & set(dino))

    ranked_wins = sorted(common, key=lambda r: ours[r]["iou"] - dino[r]["iou"], reverse=True)
    ranked_fails = sorted(common, key=lambda r: ours[r]["iou"])
    both_ok = sorted(common, key=lambda r: min(ours[r]["iou"], dino[r]["iou"]), reverse=True)

    groups = {
        "ours_wins_over_dino": ranked_wins[: args.max_per_group],
        "ours_failures": ranked_fails[: args.max_per_group],
        "both_success": both_ok[: args.max_per_group],
    }

    ours_pipe = LocateSam2Pipeline(
        grounder="locateanything",
        adapter_config=AdapterConfig(crop_mode="crop", rerank="best_score"),
        generation_mode="hybrid",
    )
    dino_pipe = LocateSam2Pipeline(
        grounder="grounding_dino_tiny",
        adapter_config=AdapterConfig(crop_mode="crop", rerank="best_score"),
    )

    manifest = []
    for group, ref_ids in groups.items():
        for ref_id in ref_ids:
            orec = ours[ref_id]
            drec = dino[ref_id]
            image_path = data_dir / "train2014" / f"COCO_train2014_{orec['image_id']:012d}.jpg"
            if not image_path.exists():
                continue

            out_dir = args.output_dir / group / f"ref{ref_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "query.txt").write_text(orec["sentence"])

            image = Image.open(image_path).convert("RGB")
            image.save(out_dir / "image_raw.jpg")

            ores = ours_pipe.run_path(image_path, orec["sentence"])
            dres = dino_pipe.run_path(image_path, drec["sentence"])

            meta = {
                "dataset": "refcoco_val",
                "ref_id": ref_id,
                "image_id": orec["image_id"],
                "query": orec["sentence"],
                "group": group,
                "ours_miou": orec["iou"],
                "dino_miou": drec["iou"],
                "delta_miou": orec["iou"] - drec["iou"],
                "ours_box_iou": orec.get("box_iou"),
                "dino_box_iou": drec.get("box_iou"),
            }
            (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

            if ores.masks:
                _save_mask(ores.masks[0], out_dir / "ours_mask.png")
                save_overlay(image, ores.boxes, ores.masks, out_dir / "ours_overlay.png", orec["sentence"])
            if dres.masks:
                _save_mask(dres.masks[0], out_dir / "dino_mask.png")
                save_overlay(image, dres.boxes, dres.masks, out_dir / "dino_overlay.png", orec["sentence"])

            manifest.append({"group": group, "ref_id": ref_id, "dir": str(out_dir.relative_to(args.output_dir))})
            print(f"exported {out_dir}")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
