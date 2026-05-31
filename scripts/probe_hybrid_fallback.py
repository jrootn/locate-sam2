#!/usr/bin/env python3
"""Compare fast vs hybrid LocateAnything outputs on a RefCOCO subset."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.config import load_config, resolve_path
from locate_sam2.locate import LocateAnythingGrounder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/hybrid_fallback_stats.json"))
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    model_path = resolve_path(cfg, cfg["models"]["locateanything"]["local_dir"])

    with (data_dir / "refcoco" / "refs(unc).p").open("rb") as f:
        refs = [r for r in pickle.load(f) if r.get("split") == "val"]
    rng = random.Random(args.seed)
    refs = rng.sample(refs, min(args.subset_size, len(refs)))

    fast = LocateAnythingGrounder(str(model_path), generation_mode="fast")
    hybrid = LocateAnythingGrounder(str(model_path), generation_mode="hybrid")

    stats = {
        "n": 0,
        "answer_diff": 0,
        "box_diff": 0,
        "fast_no_box": 0,
        "hybrid_no_box": 0,
        "examples": [],
    }

    for ref in tqdm(refs, desc="fast vs hybrid"):
        image_path = data_dir / "train2014" / f"COCO_train2014_{ref['image_id']:012d}.jpg"
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        sent = ref["sentences"][0]["sent"]
        if isinstance(sent, dict):
            sent = sent.get("raw", sent.get("sent", ""))

        fa, fb = fast.ground(image, sent)
        ha, hb = hybrid.ground(image, sent)
        stats["n"] += 1
        if fa.strip() != ha.strip():
            stats["answer_diff"] += 1
        if len(fb) != len(hb) or any(
            abs(a.x1 - b.x1) > 2 or abs(a.y1 - b.y1) > 2 for a, b in zip(fb, hb)
        ):
            stats["box_diff"] += 1
        if not fb:
            stats["fast_no_box"] += 1
        if not hb:
            stats["hybrid_no_box"] += 1
        if len(stats["examples"]) < 5 and fa.strip() != ha.strip():
            stats["examples"].append({"ref_id": ref["ref_id"], "sentence": sent, "fast": fa[:200], "hybrid": ha[:200]})

    for key in ("answer_diff", "box_diff", "fast_no_box", "hybrid_no_box"):
        stats[f"{key}_rate"] = stats[key] / max(stats["n"], 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in stats if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
