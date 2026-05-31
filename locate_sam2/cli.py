from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from locate_sam2.adapter import AdapterConfig
from locate_sam2.pipeline import LocateSam2Pipeline
from locate_sam2.visualize import save_overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="locate-sam2",
        description="Locate-SAM2: zero-shot text-to-mask segmentation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    seg = sub.add_parser("segment", help="Segment objects from a text prompt")
    seg.add_argument("image", type=Path)
    seg.add_argument("prompt", type=str)
    seg.add_argument("-o", "--output", type=Path, default=Path("outputs/demo.png"))
    seg.add_argument("--multi", action="store_true")
    seg.add_argument("--grounder", choices=["locateanything", "grounding_dino_tiny"], default="locateanything")
    seg.add_argument("--generation-mode", choices=["fast", "hybrid", "slow"], default="hybrid")
    seg.add_argument("--prompt-mode", choices=["box", "box_point", "point"], default="box")
    seg.add_argument("--crop-mode", choices=["full", "crop"], default="crop")
    seg.add_argument("--rerank", choices=["top1", "best_score", "largest_box"], default="best_score")

    args = parser.parse_args(argv)

    if args.command == "segment":
        adapter = AdapterConfig(
            prompt_mode=args.prompt_mode,
            crop_mode=args.crop_mode,
            rerank=args.rerank,
        )
        pipeline = LocateSam2Pipeline(
            grounder=args.grounder,
            adapter_config=adapter,
        )
        if args.grounder == "locateanything":
            pipeline.adapter.grounder.generation_mode = args.generation_mode

        result = pipeline.run_path(args.image, args.prompt, multi=args.multi)
        print(
            f"grounder={result.grounder} boxes={len(result.boxes)} "
            f"ground_ms={result.ground_ms:.1f} segment_ms={result.segment_ms:.1f}"
        )
        if result.boxes:
            save_overlay(
                Image.open(args.image).convert("RGB"),
                result.boxes,
                result.masks,
                args.output,
                title=args.prompt,
            )
            print(f"saved: {args.output}")
        else:
            print("No boxes detected.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
