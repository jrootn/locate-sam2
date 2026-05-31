#!/usr/bin/env python3
"""Run the main Locate-SAM2 vs GroundingDINO-Tiny benchmark on RefCOCO val."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.eval.refcoco_eval import run_refcoco_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate-SAM2 main benchmark")
    parser.add_argument("--subset-size", type=int, default=200, help="0 = full RefCOCO val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    runs = [
        {
            "tag": "locate_sam2_hybrid",
            "grounder": "locateanything",
            "generation_mode": "hybrid",
        },
        {
            "tag": "locate_sam2_fast",
            "grounder": "locateanything",
            "generation_mode": "fast",
        },
        {
            "tag": "grounded_sam2_style_dino_tiny",
            "grounder": "grounding_dino_tiny",
            "generation_mode": "hybrid",
        },
    ]

    table = []
    for run in runs:
        summary = run_refcoco_eval(
            subset_size=args.subset_size,
            seed=args.seed,
            output_dir=out,
            tag=run["tag"],
            grounder=run["grounder"],  # type: ignore[arg-type]
            generation_mode=run["generation_mode"],
        )
        table.append(summary)
        print(json.dumps(summary, indent=2))

    (out / "benchmark_table.json").write_text(json.dumps(table, indent=2))
    print(f"\nSaved comparison table: {out / 'benchmark_table.json'}")


if __name__ == "__main__":
    main()
