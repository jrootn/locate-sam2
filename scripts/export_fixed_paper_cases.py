#!/usr/bin/env python3
"""Re-export fixed paper/README case folders (DINO-Tiny + hybrid overlays)."""
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

# LaTeX + README anchor cases (dir name, ref_id, group label)
FIXED_CASES: list[tuple[str, int, str]] = [
    ("win_ref5466", 5466, "ours_wins_over_dino"),
    ("win_ref2764", 2764, "ours_wins_over_dino"),
    ("fail_ref2885", 2885, "ours_failures"),
    ("both_ref3281", 3281, "both_success"),
]


def _load(path: Path) -> dict[int, dict]:
    return {r["ref_id"]: r for r in json.loads(path.read_text())}


def export_dir(
    dir_name: str,
    ref_id: int,
    group: str,
    *,
    ours: dict,
    dino: dict,
    data_dir: Path,
    paper_dir: Path,
    ours_pipe: LocateSam2Pipeline,
    dino_pipe: LocateSam2Pipeline,
) -> bool:
    if ref_id not in ours or ref_id not in dino:
        print(f"SKIP {dir_name}: ref {ref_id} missing from records")
        return False

    orec, drec = ours[ref_id], dino[ref_id]
    image_path = data_dir / "train2014" / f"COCO_train2014_{orec['image_id']:012d}.jpg"
    if not image_path.exists():
        print(f"SKIP {dir_name}: missing {image_path.name}")
        return False

    out_dir = paper_dir / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image.save(out_dir / "image_raw.jpg")
    (out_dir / "query.txt").write_text(orec["sentence"])

    ores = ours_pipe.run_path(image_path, orec["sentence"])
    dres = dino_pipe.run_path(image_path, orec["sentence"])

    if ores.masks:
        save_overlay(image, ores.boxes, ores.masks, out_dir / "ours_overlay.png")
    if dres.masks:
        save_overlay(image, dres.boxes, dres.masks, out_dir / "dino_overlay.png")

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
        "grounder_baseline": "grounding_dino_tiny",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"OK {dir_name} ref={ref_id}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-records", type=Path, required=True)
    parser.add_argument("--dino-records", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "research_paper" / "figures")
    parser.add_argument(
        "--rerender-all",
        action="store_true",
        help="Re-export every case dir that already has metadata.json",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    ours = _load(args.ours_records)
    dino = _load(args.dino_records)

    adapter = AdapterConfig(crop_mode="crop", rerank="best_score")
    ours_pipe = LocateSam2Pipeline(
        grounder="locateanything",
        adapter_config=adapter,
        generation_mode="hybrid",
    )
    dino_pipe = LocateSam2Pipeline(
        grounder="grounding_dino_tiny",
        adapter_config=adapter,
    )

    cases = list(FIXED_CASES)
    if args.rerender_all:
        for meta_path in sorted(args.paper_dir.glob("*/metadata.json")):
            meta = json.loads(meta_path.read_text())
            ref_id = meta.get("ref_id")
            if ref_id is None:
                continue
            dir_name = meta_path.parent.name
            group = meta.get("group", meta.get("failure_mode", "unknown"))
            if (dir_name, ref_id, group) not in cases:
                cases.append((dir_name, ref_id, group))

    ok = 0
    for dir_name, ref_id, group in cases:
        if export_dir(
            dir_name, ref_id, group,
            ours=ours, dino=dino,
            data_dir=data_dir,
            paper_dir=args.paper_dir,
            ours_pipe=ours_pipe,
            dino_pipe=dino_pipe,
        ):
            ok += 1
    print(f"Exported {ok}/{len(cases)} cases.")


if __name__ == "__main__":
    main()
