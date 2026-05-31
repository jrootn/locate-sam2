#!/usr/bin/env python3
"""Run Locate-SAM2 hybrid + DINO-Tiny on OOD prompts.csv and export overlays."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.adapter import AdapterConfig
from locate_sam2.pipeline import LocateSam2Pipeline
from locate_sam2.visualize import save_overlay


def _case_id(domain: str, image_file: str, image_id: str | None) -> str:
    if image_id:
        return f"{domain}_{image_id}"
    return f"{domain}_{Path(image_file).stem}"


def _load_prompts(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            domain = (row.get("domain") or "").strip()
            image_file = (row.get("image_file") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if not domain or not image_file or not prompt:
                continue
            rows.append(
                {
                    "domain": domain,
                    "image_id": (row.get("image_id") or "").strip(),
                    "image_file": image_file,
                    "prompt": prompt,
                    "notes": (row.get("notes") or "").strip(),
                }
            )
    return rows


def _write_results_template(prompts: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "domain",
                "image_id",
                "image_file",
                "method",
                "prompt",
                "prompt_ok",
                "instance_ok",
                "box_ok",
                "mask_usable",
                "failure_tag",
                "reviewer",
                "notes",
            ],
        )
        writer.writeheader()
        for row in prompts:
            for method in ("locate_sam2_hybrid", "dino_tiny"):
                writer.writerow(
                    {
                        "domain": row["domain"],
                        "image_id": row["image_id"],
                        "image_file": row["image_file"],
                        "method": method,
                        "prompt": row["prompt"],
                        "prompt_ok": "",
                        "instance_ok": "",
                        "box_ok": "",
                        "mask_usable": "",
                        "failure_tag": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch OOD inference for Locate-SAM2 eval")
    parser.add_argument(
        "--ood-dir",
        type=Path,
        default=ROOT / "experiments" / "ood",
        help="OOD experiment root (contains prompts.csv and images/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "ood",
        help="Where to write runs/ and updated results template",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Override prompts.csv path (default: <ood-dir>/prompts.csv)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["locate_sam2_hybrid", "dino_tiny"],
        default=["locate_sam2_hybrid", "dino_tiny"],
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases that already have metadata.json for all requested methods",
    )
    args = parser.parse_args()

    ood_dir = args.ood_dir.resolve()
    prompts_path = args.prompts or (ood_dir / "prompts.csv")
    images_root = ood_dir / "images"
    runs_root = args.output_dir.resolve() / "runs"

    if not prompts_path.exists():
        raise SystemExit(f"Missing prompts file: {prompts_path}")

    prompts = _load_prompts(prompts_path)
    if not prompts:
        raise SystemExit(f"No valid rows in {prompts_path}")

    adapter = AdapterConfig(prompt_mode="box", crop_mode="crop", rerank="best_score")
    pipelines: dict[str, LocateSam2Pipeline] = {}
    if "locate_sam2_hybrid" in args.methods:
        pipelines["locate_sam2_hybrid"] = LocateSam2Pipeline(
            grounder="locateanything",
            adapter_config=adapter,
            generation_mode="hybrid",
        )
    if "dino_tiny" in args.methods:
        pipelines["dino_tiny"] = LocateSam2Pipeline(
            grounder="grounding_dino_tiny",
            adapter_config=adapter,
        )

    manifest: list[dict] = []
    skipped_missing = 0
    skipped_existing = 0
    processed = 0

    for row in prompts:
        image_path = images_root / row["image_file"]
        case = _case_id(row["domain"], row["image_file"], row["image_id"] or None)
        case_dir = runs_root / row["domain"] / case

        if not image_path.exists():
            print(f"SKIP missing image: {image_path}")
            skipped_missing += 1
            continue

        if args.skip_existing:
            existing = all((case_dir / f"{method}_overlay.png").exists() for method in args.methods)
            if existing and (case_dir / "metadata.json").exists():
                skipped_existing += 1
                continue

        case_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        image.save(case_dir / "image_raw.jpg")

        case_meta = {
            "case_id": case,
            "domain": row["domain"],
            "image_id": row["image_id"],
            "image_file": row["image_file"],
            "prompt": row["prompt"],
            "notes": row["notes"],
            "methods": {},
        }

        for method, pipeline in pipelines.items():
            result = pipeline.run(image, row["prompt"], multi=False)
            overlay_name = f"{method}_overlay.png"
            save_overlay(
                image,
                result.boxes,
                result.masks,
                case_dir / overlay_name,
                title=f"{method}: {row['prompt'][:80]}",
            )
            case_meta["methods"][method] = {
                "grounder": result.grounder,
                "ground_ms": result.ground_ms,
                "segment_ms": result.segment_ms,
                "latency_ms": result.latency_ms,
                "num_boxes": len(result.boxes),
                "num_masks": len(result.masks),
                "overlay": overlay_name,
            }

        (case_dir / "metadata.json").write_text(json.dumps(case_meta, indent=2))
        manifest.append(case_meta)
        processed += 1
        print(f"OK {case} ({row['prompt'][:60]})")

    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_results_template(prompts, args.output_dir / "results_template.csv")

    print(
        f"\nDone: processed={processed}, missing_images={skipped_missing}, "
        f"skipped_existing={skipped_existing}"
    )
    print(f"Runs: {runs_root}")
    print(f"Score sheet: {args.output_dir / 'results_template.csv'}")
    if skipped_missing:
        print("Add images under experiments/ood/images/ and re-run.")


if __name__ == "__main__":
    main()
