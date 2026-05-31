#!/usr/bin/env python3
"""Export labeled failure/success cases into research_paper/figures/."""
from __future__ import annotations

import argparse
import json
import re
import shutil
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

SPATIAL = re.compile(
    r"\b(left|right|behind|front|near|next to|above|below|top|bottom|middle|"
    r"upper|lower|far|closest|farthest|between)\b",
    re.I,
)
ATTRIBUTE = re.compile(
    r"\b(red|blue|white|black|green|yellow|dark|bright|small|large|big|"
    r"tiny|striped|wooden|metal|plastic|clear|transparent|empty|full)\b",
    re.I,
)
RARE = re.compile(r"\b[A-Z]{2,}\b|[0-9]+|['\u2019]s\b|logo|text|sign|label|brand|ocr", re.I)


def _load(path: Path) -> dict[int, dict]:
    return {r["ref_id"]: r for r in json.loads(path.read_text())}


def classify_failure(rec: dict) -> str | None:
    if rec["iou"] >= 0.5:
        return None
    sent = rec["sentence"]
    if rec.get("box_iou", 0) < 0.3 or rec.get("detected_boxes", 1) == 0:
        return "wrong_instance"
    if SPATIAL.search(sent):
        return "spatial"
    if ATTRIBUTE.search(sent):
        return "attribute"
    if RARE.search(sent) or len(sent.split()) >= 8:
        return "rare_or_long"
    return "wrong_instance"


def pick_cases(
    ours: dict[int, dict],
    dino: dict[int, dict],
    *,
    per_mode: int,
    existing: set[str],
) -> list[tuple[str, int]]:
    common = set(ours) & set(dino)
    buckets: dict[str, list[tuple[int, float]]] = {
        "wrong_instance": [],
        "spatial": [],
        "attribute": [],
        "rare_or_long": [],
    }
    for ref_id in common:
        orec = ours[ref_id]
        mode = classify_failure(orec)
        if mode is None:
            continue
        # prefer cases where DINO did OK but ours failed (Locate-SAM2-specific failure)
        dino_iou = dino[ref_id]["iou"]
        score = (dino_iou - orec["iou"], -orec["iou"])
        buckets[mode].append((ref_id, score))

    chosen: list[tuple[str, int]] = []
    for mode, items in buckets.items():
        items.sort(key=lambda x: x[1], reverse=True)
        n = 0
        for ref_id, _ in items:
            folder = f"fail_{mode}_ref{ref_id}"
            if folder in existing:
                continue
            chosen.append((mode, ref_id))
            n += 1
            if n >= per_mode:
                break
    return chosen


def export_case(
    mode: str,
    ref_id: int,
    ours: dict,
    dino: dict,
    *,
    data_dir: Path,
    paper_dir: Path,
    ours_pipe: LocateSam2Pipeline,
    dino_pipe: LocateSam2Pipeline,
) -> Path | None:
    orec = ours[ref_id]
    drec = dino[ref_id]
    image_path = data_dir / "train2014" / f"COCO_train2014_{orec['image_id']:012d}.jpg"
    if not image_path.exists():
        return None

    out_dir = paper_dir / f"fail_{mode}_ref{ref_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image.save(out_dir / "image_raw.jpg")
    (out_dir / "query.txt").write_text(orec["sentence"])

    ores = ours_pipe.run_path(image_path, orec["sentence"])
    dres = dino_pipe.run_path(image_path, orec["sentence"])

    if ores.masks:
        save_overlay(image, ores.boxes, ores.masks, out_dir / "ours_overlay.png", orec["sentence"])
    if dres.masks:
        save_overlay(image, dres.boxes, dres.masks, out_dir / "dino_overlay.png", orec["sentence"])

    meta = {
        "failure_mode": mode,
        "ref_id": ref_id,
        "image_id": orec["image_id"],
        "query": orec["sentence"],
        "ours_miou": orec["iou"],
        "dino_miou": drec["iou"],
        "ours_box_iou": orec.get("box_iou"),
        "dino_box_iou": drec.get("box_iou"),
        "label_for_paper": {
            "wrong_instance": "Failure (i): wrong object instance",
            "spatial": "Failure (ii): spatial / ordinal language",
            "attribute": "Failure (iii): attribute ambiguity",
            "rare_or_long": "Failure (iv): rare or long expression",
        }.get(mode, mode),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-records", type=Path, required=True)
    parser.add_argument("--dino-records", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "research_paper" / "figures")
    parser.add_argument("--per-mode", type=int, default=2)
    parser.add_argument("--copy-existing-wins", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    ours = _load(args.ours_records)
    dino = _load(args.dino_records)

    args.paper_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in args.paper_dir.iterdir() if p.is_dir()}
    picks = pick_cases(ours, dino, per_mode=args.per_mode, existing=existing)

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
    for mode, ref_id in picks:
        out = export_case(
            mode, ref_id, ours, dino,
            data_dir=data_dir,
            paper_dir=args.paper_dir,
            ours_pipe=ours_pipe,
            dino_pipe=dino_pipe,
        )
        if out:
            manifest.append({"mode": mode, "ref_id": ref_id, "dir": out.name})
            print(f"exported {out.name}: {ours[ref_id]['sentence'][:70]}")

    (args.paper_dir / "failure_export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nExported {len(manifest)} failure cases to {args.paper_dir}")


if __name__ == "__main__":
    main()
