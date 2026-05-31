#!/usr/bin/env python3
"""Validate RefCOCO eval setup and compare run records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.config import load_config, resolve_path
from locate_sam2.eval.experiment_log import subset_ref_ids, validate_eval_setup


def compare_records(a_path: Path, b_path: Path) -> dict:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    a_map = {r["ref_id"]: r["iou"] for r in a}
    b_map = {r["ref_id"]: r["iou"] for r in b}
    common = sorted(set(a_map) & set(b_map))
    deltas = [b_map[r] - a_map[r] for r in common]
    return {
        "a": str(a_path),
        "b": str(b_path),
        "n_a": len(a),
        "n_b": len(b),
        "same_ref_id_set": set(a_map) == set(b_map),
        "same_order": [r["ref_id"] for r in a] == [r["ref_id"] for r in b],
        "mean_iou_a": sum(a_map[r] for r in common) / len(common) if common else 0,
        "mean_iou_b": sum(b_map[r] for r in common) / len(common) if common else 0,
        "mean_delta_b_minus_a": sum(deltas) / len(deltas) if deltas else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Locate-SAM2 eval setup")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", nargs=2, metavar=("RECORDS_A", "RECORDS_B"))
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])

    report = validate_eval_setup(
        data_dir,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    ref_ids = subset_ref_ids(data_dir, subset_size=args.subset_size, seed=args.seed)
    report["first_10_ref_ids"] = ref_ids[:10]
    print(json.dumps(report, indent=2))

    if args.compare:
        cmp = compare_records(Path(args.compare[0]), Path(args.compare[1]))
        print("\nCOMPARE:")
        print(json.dumps(cmp, indent=2))

    if not report["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
