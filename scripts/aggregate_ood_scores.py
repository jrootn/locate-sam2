#!/usr/bin/env python3
"""Aggregate human OOD scores from results.csv into summary JSON."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _yes_rate(values: list[str]) -> float | None:
    scored = [v.strip().upper() for v in values if v.strip()]
    if not scored:
        return None
    yes = sum(1 for v in scored if v in ("Y", "YES", "1", "TRUE"))
    return yes / len(scored)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "outputs" / "ood" / "results.csv",
        help="Filled human score sheet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "ood" / "ood_summary.json",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(
            f"Missing {args.results}. Copy results_template.csv to results.csv and fill scores."
        )

    rows: list[dict[str, str]] = []
    with args.results.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"No rows in {args.results}")

    criteria = ["prompt_ok", "instance_ok", "box_ok", "mask_usable"]
    by_domain_method: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("domain", ""), row.get("method", ""))
        by_domain_method[key].append(row)

    summary_rows = []
    for (domain, method), group in sorted(by_domain_method.items()):
        entry = {
            "domain": domain,
            "method": method,
            "n_scored": len(group),
        }
        for crit in criteria:
            entry[crit] = _yes_rate([r.get(crit, "") for r in group])
        summary_rows.append(entry)

    overall: dict[str, dict[str, float | None]] = defaultdict(dict)
    for method in sorted({m for _, m in by_domain_method}):
        method_rows = [r for (d, m), rs in by_domain_method.items() if m == method for r in rs]
        for crit in criteria:
            overall[method][crit] = _yes_rate([r.get(crit, "") for r in method_rows])

    failure_tags: dict[str, int] = defaultdict(int)
    for row in rows:
        tag = (row.get("failure_tag") or "").strip()
        if tag:
            failure_tags[tag] += 1

    out = {
        "source": str(args.results),
        "n_rows": len(rows),
        "by_domain_method": summary_rows,
        "overall_by_method": dict(overall),
        "failure_tag_counts": dict(sorted(failure_tags.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
