#!/usr/bin/env python3
"""Sanity-check DINO-Tiny thresholds on the 200-ref subset (seed=42)."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.eval.refcoco_eval import run_refcoco_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="DINO-Tiny threshold sweep on 200-ref subset")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dino_sweep"))
    parser.add_argument("--box-thresholds", type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.35])
    parser.add_argument("--text-thresholds", type=float, nargs="+", default=[0.20, 0.25, 0.30])
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    table = []
    for box_t, text_t in itertools.product(args.box_thresholds, args.text_thresholds):
        tag = f"dino_tiny_b{box_t:.2f}_t{text_t:.2f}"
        summary = run_refcoco_eval(
            subset_size=args.subset_size,
            seed=args.seed,
            output_dir=out,
            grounder="grounding_dino_tiny",
            tag=tag,
            box_threshold=box_t,
            text_threshold=text_t,
        )
        summary["box_threshold"] = box_t
        summary["text_threshold"] = text_t
        table.append(summary)
        print(f"{tag}: mIoU={summary['mean_mask_iou']:.3f} P@0.5={summary.get('precision_at_0.5', 0):.1%}")

    table.sort(key=lambda r: r["mean_mask_iou"], reverse=True)
    (out / "dino_threshold_sweep.json").write_text(json.dumps(table, indent=2))
    best = table[0]
    print(f"\nBest: box={best['box_threshold']} text={best['text_threshold']} mIoU={best['mean_mask_iou']:.3f}")


if __name__ == "__main__":
    main()
