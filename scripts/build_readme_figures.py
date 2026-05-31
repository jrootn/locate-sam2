#!/usr/bin/env python3
"""Build full-width README qualitative figures with readable labels (no GPU)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROWS = [
    ("win_ref5466", "Ref 5466", "right white spoon"),
    ("fail_ref2885", "Ref 2885", "man turned around"),
    ("fail_spatial_ref5750", "Ref 5750", "zebra on left"),
]

COL_TITLES = ("Input", "DINO-Tiny + SAM2", "Locate-SAM2 hybrid")


def _load_meta(fig_dir: Path) -> dict:
    meta_path = fig_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _caption_for(meta: dict, case_dir: str) -> str:
    ours = meta.get("ours_miou")
    dino = meta.get("dino_miou")
    if ours is None or dino is None:
        return ""
    if case_dir.startswith("fail_ref2885"):
        return f"Hybrid selects the wrong person (mIoU {ours:.2f}); DINO-Tiny mIoU {dino:.2f}."
    return f"Hybrid mIoU {ours:.2f}; DINO-Tiny mIoU {dino:.2f}."


def build_grid(rows: list[tuple[str, str, str]], figures_dir: Path, output_path: Path) -> None:
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(14.5, 4.6 * n_rows),
        gridspec_kw={"wspace": 0.05, "hspace": 0.55},
    )
    if n_rows == 1:
        axes = [axes]

    for row_idx, (case_name, ref_label, query) in enumerate(rows):
        case_dir = figures_dir / case_name
        meta = _load_meta(case_dir)
        caption = _caption_for(meta, case_name)
        row_axes = axes[row_idx]

        for col_idx, (ax, col_title) in enumerate(zip(row_axes, COL_TITLES)):
            img_path = [
                case_dir / "image_raw.jpg",
                case_dir / "dino_overlay.png",
                case_dir / "ours_overlay.png",
            ][col_idx]
            if img_path.exists():
                ax.imshow(Image.open(img_path).convert("RGB"))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_title, fontsize=17, fontweight="bold", pad=12)

        bbox = row_axes[1].get_position()
        fig.text(
            0.5,
            bbox.y1 + 0.025,
            f'{ref_label}: "{query}"',
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )
        if caption:
            fig.text(0.5, bbox.y0 - 0.012, caption, ha="center", va="top", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "research_paper" / "figures")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets" / "readme_qualitative.png")
    args = parser.parse_args()
    build_grid(DEFAULT_ROWS, args.figures_dir, args.output)


if __name__ == "__main__":
    main()
