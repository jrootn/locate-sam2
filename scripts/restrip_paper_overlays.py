#!/usr/bin/env python3
"""Crop matplotlib title padding from overlay PNGs by aligning to image_raw.jpg."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OVERLAY_NAMES = ("dino_overlay.png", "ours_overlay.png")


def _photo_top_row(arr: np.ndarray, threshold: float = 0.25) -> int:
    """First row that looks like image content (not white title margin)."""
    for y in range(arr.shape[0]):
        non_white = float(np.mean(np.any(arr[y] < 230, axis=1)))
        if non_white > threshold:
            return y
    return 0


def align_crop_overlay(overlay: Image.Image, raw: Image.Image) -> Image.Image:
    """Return only the photo (+ overlays), without matplotlib title band."""
    ow, oh = overlay.size
    rw, rh = raw.size
    arr = np.asarray(overlay.convert("RGB"))
    top = _photo_top_row(arr)
    target_h = int(round(ow * rh / rw))
    bottom = min(oh, top + target_h)
    # If already stripped, fall back to bottom-aligned crop.
    if top == 0 and oh > target_h:
        top = oh - target_h
        bottom = oh
    return overlay.crop((0, top, ow, bottom))


def restrip_case(case_dir: Path) -> bool:
    raw_path = case_dir / "image_raw.jpg"
    if not raw_path.exists():
        return False

    raw = Image.open(raw_path).convert("RGB")
    changed = False
    for name in OVERLAY_NAMES:
        overlay_path = case_dir / name
        if not overlay_path.exists():
            continue
        cleaned = align_crop_overlay(Image.open(overlay_path).convert("RGB"), raw)
        cleaned.save(overlay_path)
        changed = True
    return changed


def iter_case_dirs(paper_dir: Path):
    for path in sorted(paper_dir.iterdir()):
        if not path.is_dir():
            continue
        if (path / "image_raw.jpg").exists() or any((path / n).exists() for n in OVERLAY_NAMES):
            yield path
    probe = paper_dir / "hallucination_probe"
    if probe.is_dir():
        for path in sorted(probe.iterdir()):
            if path.is_dir() and (path / "image_raw.jpg").exists():
                yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "research_paper" / "figures")
    args = parser.parse_args()

    ok = 0
    for case_dir in iter_case_dirs(args.paper_dir):
        if restrip_case(case_dir):
            print(f"OK {case_dir.relative_to(args.paper_dir)}")
            ok += 1
    print(f"Restripped {ok} case(s).")


if __name__ == "__main__":
    main()
