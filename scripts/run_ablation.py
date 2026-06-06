#!/usr/bin/env python3
"""Run Locate-SAM2 ablations on RefCOCO val."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.eval.refcoco_eval import run_refcoco_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate-SAM2 ablation suite")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablations"))
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    ablations = [
        {"tag": "abl_fast", "generation_mode": "fast"},
        {"tag": "abl_hybrid", "generation_mode": "hybrid"},
        {"tag": "abl_slow", "generation_mode": "slow"},
        {"tag": "abl_prompt_box", "prompt_mode": "box"},
        {"tag": "abl_prompt_box_point", "prompt_mode": "box_point"},
        {"tag": "abl_prompt_point", "prompt_mode": "point"},
        {"tag": "abl_crop_full", "crop_mode": "full"},
        {"tag": "abl_crop_crop", "crop_mode": "crop"},
        {"tag": "abl_rerank_top1", "rerank": "top1"},
        {"tag": "abl_rerank_best_score", "rerank": "best_score"},
    ]

    table = []
    for abl in ablations:
        summary = run_refcoco_eval(
            subset_size=args.subset_size,
            seed=args.seed,
            output_dir=out,
            tag=abl["tag"],
            generation_mode=abl.get("generation_mode", "hybrid"),
            prompt_mode=abl.get("prompt_mode", "box"),
            crop_mode=abl.get("crop_mode", "crop"),
            rerank=abl.get("rerank", "best_score"),
        )
        table.append(summary)

    (out / "ablation_table.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
