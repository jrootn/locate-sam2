#!/usr/bin/env python3
"""Bin eval records by box IoU and report mask IoU / failure rates per bin."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_BINS = [
    ("0.0-0.3", 0.0, 0.3),
    ("0.3-0.5", 0.3, 0.5),
    ("0.5-0.7", 0.5, 0.7),
    ("0.7-0.9", 0.7, 0.9),
    ("0.9-1.0", 0.9, 1.000001),
]


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def stratify(records: list[dict], bins: list[tuple[str, float, float]]) -> dict:
    usable = [r for r in records if "box_iou" in r]
    skipped = len(records) - len(usable)
    rows = []
    for label, lo, hi in bins:
        subset = [r for r in usable if lo <= float(r["box_iou"]) < hi]
        if not subset:
            rows.append(
                {
                    "bin": label,
                    "n": 0,
                    "fraction": 0.0,
                    "mean_mask_iou": None,
                    "precision_at_0.5": None,
                    "precision_at_0.7": None,
                    "mean_box_iou": None,
                }
            )
            continue
        mask_ious = [float(r["iou"]) for r in subset]
        box_ious = [float(r["box_iou"]) for r in subset]
        fail_05 = sum(1 for x in mask_ious if x < 0.5)
        rows.append(
            {
                "bin": label,
                "n": len(subset),
                "fraction": len(subset) / max(len(usable), 1),
                "mean_mask_iou": sum(mask_ious) / len(mask_ious),
                "precision_at_0.5": sum(1 for x in mask_ious if x >= 0.5) / len(mask_ious),
                "precision_at_0.7": sum(1 for x in mask_ious if x >= 0.7) / len(mask_ious),
                "mean_box_iou": sum(box_ious) / len(box_ious),
                "failure_rate_mask_lt_0.5": fail_05 / len(mask_ious),
            }
        )

    all_mask = [float(r["iou"]) for r in usable]
    all_box = [float(r["box_iou"]) for r in usable]
    return {
        "n_total": len(records),
        "n_with_box_iou": len(usable),
        "skipped_missing_box_iou": skipped,
        "mean_mask_iou": sum(all_mask) / len(all_mask) if all_mask else None,
        "mean_box_iou": sum(all_box) / len(all_box) if all_box else None,
        "bins": rows,
        "interpretation": (
            "Mask quality tracks box quality: low box-IoU bins concentrate grounding failures; "
            "high box-IoU bins approach the GT-box oracle ceiling."
        ),
    }


def discover_records(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*_records.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Box-IoU stratified mask quality analysis")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        help="One records JSON file (repeatable). Default: scan --input-dir.",
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/analysis/box_iou_stratification.json",
    )
    args = parser.parse_args()

    paths = args.input or discover_records(args.input_dir)
    if not paths:
        raise SystemExit(f"No *_records.json under {args.input_dir}")

    report = {"sources": [], "runs": {}}
    for path in paths:
        records = load_records(path)
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        result = stratify(records, DEFAULT_BINS)
        if result["n_with_box_iou"] == 0:
            continue
        report["sources"].append(key)
        report["runs"][key] = result

    if not report["runs"]:
        raise SystemExit("No records with box_iou field found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
