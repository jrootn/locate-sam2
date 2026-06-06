#!/usr/bin/env python3
"""Measure mask mIoU on RefCOCO refs where fast vs hybrid LocateAnything disagree."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.adapter import AdapterConfig
from locate_sam2.config import load_config, resolve_path
from locate_sam2.eval.metrics import mask_iou
from locate_sam2.locate import LocateAnythingGrounder
from locate_sam2.pipeline import LocateSam2Pipeline


def _sentence(ref: dict) -> str:
    sent = ref["sentences"][0]["sent"]
    if isinstance(sent, str):
        return sent
    if isinstance(sent, dict):
        return sent.get("raw", sent.get("sent", ""))
    return str(sent)


def boxes_differ(fast_boxes, hybrid_boxes, tol: float = 2.0) -> bool:
    if len(fast_boxes) != len(hybrid_boxes):
        return True
    return any(
        abs(a.x1 - b.x1) > tol
        or abs(a.y1 - b.y1) > tol
        or abs(a.x2 - b.x2) > tol
        or abs(a.y2 - b.y2) > tol
        for a, b in zip(fast_boxes, hybrid_boxes)
    )


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}

    def _mean(key: str, subset: list[dict]) -> float | None:
        vals = [r[key] for r in subset if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    disagree = [r for r in rows if r["box_diff"]]
    agree = [r for r in rows if not r["box_diff"]]
    hybrid_wins = [r for r in disagree if r["hybrid_mask_iou"] > r["fast_mask_iou"] + 1e-9]
    fast_wins = [r for r in disagree if r["fast_mask_iou"] > r["hybrid_mask_iou"] + 1e-9]
    ties = [r for r in disagree if abs(r["hybrid_mask_iou"] - r["fast_mask_iou"]) <= 1e-9]

    return {
        "n": len(rows),
        "box_diff": len(disagree),
        "box_diff_rate": len(disagree) / len(rows),
        "agree_mean_fast_mask_iou": _mean("fast_mask_iou", agree),
        "agree_mean_hybrid_mask_iou": _mean("hybrid_mask_iou", agree),
        "disagree_mean_fast_mask_iou": _mean("fast_mask_iou", disagree),
        "disagree_mean_hybrid_mask_iou": _mean("hybrid_mask_iou", disagree),
        "disagree_hybrid_wins": len(hybrid_wins),
        "disagree_fast_wins": len(fast_wins),
        "disagree_ties": len(ties),
        "disagree_hybrid_win_rate": len(hybrid_wins) / max(len(disagree), 1),
        "disagree_mean_delta_hybrid_minus_fast": _mean(
            "delta_hybrid_minus_fast",
            [r for r in disagree if r.get("delta_hybrid_minus_fast") is not None],
        ),
        "interpretation": (
            "Hybrid fallback is sparse but targeted: compare disagree-subset mIoU to see "
            "whether AR fallback fixes hard parallel-decode cases."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast vs hybrid disagreement mIoU study")
    parser.add_argument("--dataset", default="refcoco")
    parser.add_argument("--split", default="val")
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/analysis/hybrid_disagreement_miou.json",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    model_path = resolve_path(cfg, cfg["models"]["locateanything"]["local_dir"])

    ref_file = data_dir / args.dataset / "refs(unc).p"
    if not ref_file.exists():
        ref_file = data_dir / args.dataset / "refs(google).p"
    with ref_file.open("rb") as f:
        refs = [r for r in pickle.load(f) if r.get("split") == args.split]
    rng = random.Random(args.seed)
    refs = rng.sample(refs, min(args.subset_size, len(refs)))

    instances_path = data_dir / args.dataset / "instances.json"
    if not instances_path.exists() and args.dataset != "refcoco":
        instances_path = data_dir / "refcoco" / "instances.json"
    coco = COCO(str(instances_path))

    adapter = AdapterConfig(crop_mode="crop", rerank="best_score")
    fast_pipe = LocateSam2Pipeline(
        grounder="locateanything",
        adapter_config=adapter,
        generation_mode="fast",
    )
    hybrid_pipe = LocateSam2Pipeline(
        grounder="locateanything",
        adapter_config=adapter,
        generation_mode="hybrid",
    )
    fast_grounder = LocateAnythingGrounder(str(model_path), generation_mode="fast")
    hybrid_grounder = LocateAnythingGrounder(str(model_path), generation_mode="hybrid")

    rows: list[dict] = []
    examples: list[dict] = []

    for ref in tqdm(refs, desc="hybrid disagreement mIoU"):
        image_path = data_dir / "train2014" / f"COCO_train2014_{ref['image_id']:012d}.jpg"
        if not image_path.exists() or ref["ann_id"] not in coco.anns:
            continue

        gt_mask = coco.annToMask(coco.loadAnns([ref["ann_id"]])[0])
        sentence = _sentence(ref)
        image = Image.open(image_path).convert("RGB")

        _, fast_boxes = fast_grounder.ground(image, sentence)
        _, hybrid_boxes = hybrid_grounder.ground(image, sentence)
        box_diff = boxes_differ(fast_boxes, hybrid_boxes)

        fast_result = fast_pipe.run(image, sentence)
        hybrid_result = hybrid_pipe.run(image, sentence)

        fast_miou = mask_iou(fast_result.masks[0], gt_mask)[0] if fast_result.masks else 0.0
        hybrid_miou = mask_iou(hybrid_result.masks[0], gt_mask)[0] if hybrid_result.masks else 0.0

        row = {
            "ref_id": ref["ref_id"],
            "sentence": sentence,
            "box_diff": box_diff,
            "fast_boxes": len(fast_boxes),
            "hybrid_boxes": len(hybrid_boxes),
            "fast_mask_iou": fast_miou,
            "hybrid_mask_iou": hybrid_miou,
            "delta_hybrid_minus_fast": hybrid_miou - fast_miou,
            "fast_ground_ms": fast_result.ground_ms,
            "hybrid_ground_ms": hybrid_result.ground_ms,
        }
        rows.append(row)
        if box_diff and len(examples) < 8:
            examples.append(row)

    summary = summarize(rows)
    out = {
        "dataset": args.dataset,
        "split": args.split,
        "subset_size": args.subset_size,
        "seed": args.seed,
        **summary,
        "examples": examples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
