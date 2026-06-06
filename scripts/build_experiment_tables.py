#!/usr/bin/env python3
"""Build consolidated JSON tables from eval output directories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _collect_summaries(directory: Path, pattern: str = "*_summary.json") -> list[dict]:
    if not directory.exists():
        return []
    rows = []
    for path in sorted(directory.glob(pattern)):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (no runs yet)")
        return
    for r in rows:
        name = r.get("run_name", "?")
        miou = r.get("mean_mask_iou", 0.0)
        p05 = r.get("precision_at_0.5", r.get("success_at_0.5", 0.0))
        lat = r.get("mean_latency_ms", 0.0)
        n = r.get("subset_size", r.get("evaluated_n", "?"))
        print(f"  {name:35} n={n:>5}  mIoU={miou:.3f}  P@0.5={p05:.1%}  {lat:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    out = args.outputs.resolve()

    tables = {
        "refcoco_val": out / "full_val" / "full_val_table.json",
        "refcoco_plus": out / "refcoco_plus" / "refcoco_plus_table.json",
        "refcocog": out / "refcocog" / "refcocog_table.json",
    }

    full_val_dir = out / "full_val"
    full_rows = _collect_summaries(full_val_dir, "*_full_summary.json")
    if full_rows:
        tables["refcoco_val"].parent.mkdir(parents=True, exist_ok=True)
        tables["refcoco_val"].write_text(json.dumps(full_rows, indent=2))
    _print_table("RefCOCO val", full_rows)

    plus_dir = out / "refcoco_plus"
    plus_rows = _collect_summaries(plus_dir)
    if plus_rows:
        tables["refcoco_plus"].parent.mkdir(parents=True, exist_ok=True)
        tables["refcoco_plus"].write_text(json.dumps(plus_rows, indent=2))
    _print_table("RefCOCO+", plus_rows)

    g_dir = out / "refcocog"
    g_rows = _collect_summaries(g_dir)
    if g_rows:
        tables["refcocog"].parent.mkdir(parents=True, exist_ok=True)
        tables["refcocog"].write_text(json.dumps(g_rows, indent=2))
    _print_table("RefCOCO-g", g_rows)

    print("\nTables written:")
    for name, path in tables.items():
        status = "ok" if path.exists() else "pending"
        print(f"  [{status}] {name}: {path}")


if __name__ == "__main__":
    main()
