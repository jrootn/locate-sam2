from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from locate_sam2.types import Box


def save_overlay(
    image: Image.Image,
    boxes: list[Box],
    masks: list[np.ndarray],
    output_path: Path,
    title: str = "",
    *,
    dpi: int = 200,
    box_linewidth: float = 3.0,
) -> None:
    """Save box + mask overlay. Captions belong in the paper/README, not on the image."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")
    ax.set_axis_off()
    # Do not draw long query strings on the image; they become unreadable when scaled.
    if title:
        ax.set_title(title, fontsize=14, pad=8)

    for idx, box in enumerate(boxes):
        rect = Rectangle(
            (box.x1, box.y1),
            box.x2 - box.x1,
            box.y2 - box.y1,
            fill=False,
            edgecolor="lime",
            linewidth=box_linewidth,
        )
        ax.add_patch(rect)
        if idx < len(masks):
            mask = masks[idx]
            color = np.array([1.0, 0.0, 0.0, 0.45])
            overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
            overlay[mask] = color
            ax.imshow(overlay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
