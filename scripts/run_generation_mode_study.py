#!/usr/bin/env python3
"""Compare LocateAnything fast / hybrid / slow on a fixed RefCOCO subset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.eval.refcoco_eval import run_refcoco_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="LocateAnything generation-mode study")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/analysis/generation_mode_study.json",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    modes = ["fast", "hybrid", "slow"]
    table = []
    for mode in modes:
        summary = run_refcoco_eval(
            subset_size=args.subset_size,
            seed=args.seed,
            output_dir=out,
            tag=f"genmode_{mode}",
            generation_mode=mode,
        )
        table.append(summary)

    payload = {
        "subset_size": args.subset_size,
        "seed": args.seed,
        "modes": table,
        "note": "PBD decoding modes under the same SAM2 adapter (RefCOCO val subset).",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
