#!/usr/bin/env python3
"""Generate qualitative figure grid from eval records (best / worst / failures)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from locate_sam2.config import load_config, resolve_path
from locate_sam2.pipeline import LocateSam2Pipeline
from locate_sam2.visualize import save_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative segmentation figures")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--bottom", type=int, default=8)
    parser.add_argument("--failures", type=int, default=4)
    parser.add_argument("--grounder", choices=["locateanything", "grounding_dino_tiny"], default="locateanything")
    parser.add_argument("--generation-mode", default="hybrid")
    parser.add_argument("--crop-mode", choices=["full", "crop"], default="crop")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    records = json.loads(args.records.read_text())
    records = sorted(records, key=lambda r: r["iou"])

    picks: list[dict] = []
    failures = [r for r in records if r["iou"] <= 0.01][: args.failures]
    nonzero = [r for r in records if r["iou"] > 0.01]
    bottom = nonzero[: args.bottom]
    top = list(reversed(nonzero[-args.top :]))

    for group_name, group in [("failures", failures), ("bottom", bottom), ("top", top)]:
        for r in group:
            r = dict(r)
            r["_group"] = group_name
            picks.append(r)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    from locate_sam2.adapter import AdapterConfig

    pipeline = LocateSam2Pipeline(
        grounder=args.grounder,
        adapter_config=AdapterConfig(prompt_mode="box", crop_mode=args.crop_mode, rerank="best_score"),
        generation_mode=args.generation_mode if args.grounder == "locateanything" else None,
    )

    manifest = []
    for r in picks:
        image_path = data_dir / "train2014" / f"COCO_train2014_{r['image_id']:012d}.jpg"
        if not image_path.exists():
            continue
        result = pipeline.run_path(image_path, r["sentence"])
        fname = f"{r['_group']}_ref{r['ref_id']}_iou{r['iou']:.2f}.png"
        fpath = out / fname
        save_overlay(
            Image.open(image_path).convert("RGB"),
            result.boxes,
            result.masks,
            fpath,
            title=f"{r['sentence'][:80]} | IoU={r['iou']:.2f}",
        )
        manifest.append({"file": fname, "ref_id": r["ref_id"], "iou": r["iou"], "sentence": r["sentence"], "group": r["_group"]})
        print(f"saved {fpath}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {out / 'manifest.json'} ({len(manifest)} figures)")


if __name__ == "__main__":
    main()
