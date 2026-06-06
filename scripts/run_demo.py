#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.pipeline import LocateSam2Pipeline
from locate_sam2.visualize import save_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate-SAM2 demo")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("prompt", type=str, help="Text prompt / referring expression")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/demo.png"))
    parser.add_argument("--multi", action="store_true", help="Allow multiple boxes")
    parser.add_argument("--grounder", choices=["locateanything", "grounding_dino_tiny"], default="locateanything")
    args = parser.parse_args()

    pipeline = LocateSam2Pipeline(grounder=args.grounder)
    result = pipeline.run_path(args.image, args.prompt, multi=args.multi)

    print(
        f"grounder={result.grounder} boxes={len(result.boxes)} "
        f"ground_ms={result.ground_ms:.1f} segment_ms={result.segment_ms:.1f}"
    )
    print(f"answer snippet: {result.answer[:200]}...")

    if result.boxes:
        from PIL import Image

        save_overlay(
            Image.open(args.image).convert("RGB"),
            result.boxes,
            result.masks,
            args.output,
            title=args.prompt,
        )
        print(f"saved: {args.output}")
    else:
        print("No boxes detected; nothing to segment.")


if __name__ == "__main__":
    main()
