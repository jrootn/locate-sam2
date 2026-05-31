#!/usr/bin/env python3
"""Negative-prompt probe: does LocateAnything emit boxes for unrelated phrases?"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.adapter import AdapterConfig
from locate_sam2.config import load_config, resolve_path
from locate_sam2.pipeline import LocateSam2Pipeline
from locate_sam2.visualize import save_overlay

NEGATIVE_PROMPTS = [
    "purple elephant flying over the moon",
    "quantum computer keyboard made of glass",
    "the number seven wearing a hat",
    "nonexistent object xyz123",
    "empty void nothing here",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("research_paper/figures/hallucination_probe"))
    parser.add_argument("--analysis-json", type=Path, default=Path("outputs/analysis/hallucination_probe.json"))
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])

    with (data_dir / "refcoco" / "refs(unc).p").open("rb") as f:
        refs = [r for r in pickle.load(f) if r.get("split") == "val"]
    rng = random.Random(args.seed)
    refs = rng.sample(refs, min(args.n_images, len(refs)))

    pipe = LocateSam2Pipeline(
        grounder="locateanything",
        adapter_config=AdapterConfig(crop_mode="crop", rerank="best_score"),
        generation_mode="hybrid",
    )

    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ref in tqdm(refs, desc="hallucination probe"):
        image_path = data_dir / "train2014" / f"COCO_train2014_{ref['image_id']:012d}.jpg"
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        gt_sent = ref["sentences"][0]["sent"]
        if isinstance(gt_sent, dict):
            gt_sent = gt_sent.get("raw", gt_sent.get("sent", ""))

        for prompt in NEGATIVE_PROMPTS[:2]:  # 2 neg prompts per image for speed
            result = pipe.run(image, prompt)
            emitted_box = len(result.boxes) > 0
            emitted_mask = len(result.masks) > 0
            sam_score = float(result.mask_scores[0]) if result.mask_scores else 0.0
            case_id = f"img{ref['image_id']}_neg{abs(hash(prompt)) % 10000}"
            case_dir = args.output_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            image.save(case_dir / "image_raw.jpg")
            (case_dir / "query.txt").write_text(prompt)
            (case_dir / "gt_query.txt").write_text(gt_sent)
            if result.masks:
                save_overlay(image, result.boxes, result.masks, case_dir / "ours_overlay.png", prompt[:80])

            rows.append({
                "case_id": case_id,
                "image_id": ref["image_id"],
                "negative_prompt": prompt,
                "gt_prompt": gt_sent,
                "emitted_box": emitted_box,
                "emitted_mask": emitted_mask,
                "sam_mask_score": sam_score,
                "answer_snippet": (result.answer or "")[:300],
            })

    n = len(rows)
    summary = {
        "n_probes": n,
        "box_emission_rate": sum(r["emitted_box"] for r in rows) / max(n, 1),
        "mask_emission_rate": sum(r["emitted_mask"] for r in rows) / max(n, 1),
        "mean_sam_score_when_mask": float(
            sum(r["sam_mask_score"] for r in rows if r["emitted_mask"])
            / max(sum(1 for r in rows if r["emitted_mask"]), 1)
        ),
        "note": (
            "LocateAnything has no native detection threshold (unlike DINO). "
            "Mitigation: treat empty parse as no-detection; optionally gate on SAM iou_scores."
        ),
        "rows": rows,
    }
    args.analysis_json.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_json.write_text(json.dumps(summary, indent=2))
    (args.output_dir / "metadata.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
