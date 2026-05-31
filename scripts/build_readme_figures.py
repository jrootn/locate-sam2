#!/usr/bin/env python3
"""Build README qualitative figures with legends and readable labels (no GPU)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

COL_TITLES = (
    "Input",
    "DINO-Tiny + SAM2",
    "Locate-SAM2 hybrid",
)

LEGEND = (
    "Green box = grounder prediction  |  Red mask = SAM 2.1  |  "
    "mIoU vs RefCOCO GT mask  |  Same adapter; only the grounder changes"
)


@dataclass(frozen=True)
class QualRow:
    case_dir: str
    category: str
    ref_label: str
    query: str


PANELS: list[tuple[str, str, list[QualRow], float]] = [
    (
        "readme_qualitative_wins.png",
        "RefCOCO val — hybrid wins vs DINO-Tiny (same adapter)",
        [
            QualRow("win_ref5466", "Hybrid win", "Ref 5466", "right white spoon"),
            QualRow("win_ref2764", "Hybrid win (spatial)", "Ref 2764", "right"),
        ],
        3.6,
    ),
    (
        "readme_qualitative_failures.png",
        "RefCOCO val — failure taxonomy (hybrid wrong region)",
        [
            QualRow("fail_ref2885", "Wrong instance", "Ref 2885", "man turned around"),
            QualRow("fail_spatial_ref5750", "Spatial / ordinal", "Ref 5750", "zebra on left"),
            QualRow("fail_wrong_instance_ref20398", "Wrong instance", "Ref 20398", "second wedge grapefruit from left to right"),
            QualRow("fail_attribute_ref18360", "Attribute ambiguity", "Ref 18360", "red bike"),
        ],
        3.4,
    ),
]


def _load_meta(fig_dir: Path) -> dict:
    meta_path = fig_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _caption(row: QualRow, meta: dict) -> str:
    ours = meta.get("ours_miou")
    dino = meta.get("dino_miou")
    if ours is None or dino is None:
        return row.category
    o, d = float(ours), float(dino)
    if row.case_dir.startswith("both_"):
        return f"{row.category} — hybrid {o:.2f}, DINO-Tiny {d:.2f}"
    if o >= d + 0.3:
        return f"{row.category} — hybrid {o:.2f}, DINO-Tiny {d:.2f}"
    if d >= o + 0.3:
        return f"{row.category} — DINO-Tiny {d:.2f}, hybrid {o:.2f}"
    return f"{row.category} — hybrid {o:.2f}, DINO-Tiny {d:.2f}"


def build_panel(
    rows: list[QualRow],
    figures_dir: Path,
    output_path: Path,
    panel_title: str,
    row_h: float,
) -> None:
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(13.5, row_h * n_rows + 1.4),
        gridspec_kw={"wspace": 0.05, "hspace": 0.72},
    )
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(panel_title, fontsize=15, fontweight="bold", y=0.998)
    fig.text(0.5, 0.962, LEGEND, ha="center", va="top", fontsize=10, color="#333333")

    for row_idx, row in enumerate(rows):
        case_dir = figures_dir / row.case_dir
        meta = _load_meta(case_dir)
        caption = _caption(row, meta)
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
                ax.set_title(col_title, fontsize=12, fontweight="bold", pad=8)

        bbox = row_axes[1].get_position()
        fig.text(
            0.5,
            bbox.y1 + 0.032,
            f'{row.ref_label}: "{row.query}"',
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
        fig.text(0.5, bbox.y0 - 0.018, caption, ha="center", va="top", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.90, bottom=0.04, left=0.02, right=0.98)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "research_paper" / "figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets")
    args = parser.parse_args()

    for filename, title, rows, row_h in PANELS:
        build_panel(rows, args.figures_dir, args.output_dir / filename, title, row_h)


if __name__ == "__main__":
    main()
