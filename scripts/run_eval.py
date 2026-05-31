#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.eval.refcoco_eval import run_refcoco_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate-SAM2 RefCOCO evaluation")
    parser.add_argument("--subset-size", type=int, default=200, help="0 = full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--dataset", choices=["refcoco", "refcoco+", "refcocog"], default="refcoco")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--grounder",
        choices=["locateanything", "grounding_dino_tiny", "grounding_dino_swint", "gt_oracle"],
        default="locateanything",
    )
    parser.add_argument("--generation-mode", choices=["fast", "hybrid", "slow"], default="hybrid")
    parser.add_argument("--prompt-mode", choices=["box", "box_point", "point"], default="box")
    parser.add_argument("--crop-mode", choices=["full", "crop"], default="crop")
    parser.add_argument("--rerank", choices=["top1", "best_score", "largest_box"], default="best_score")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    summary = run_refcoco_eval(
        subset_size=args.subset_size,
        seed=args.seed,
        output_dir=args.output_dir,
        dataset=args.dataset,
        split=args.split,
        grounder=args.grounder,
        generation_mode=args.generation_mode,
        prompt_mode=args.prompt_mode,
        crop_mode=args.crop_mode,
        rerank=args.rerank,
        tag=args.tag,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
