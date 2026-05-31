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
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    for idx, box in enumerate(boxes):
        rect = Rectangle(
            (box.x1, box.y1),
            box.x2 - box.x1,
            box.y2 - box.y1,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
        ax.add_patch(rect)
        if idx < len(masks):
            mask = masks[idx]
            color = np.array([1.0, 0.0, 0.0, 0.45])
            overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
            overlay[mask] = color
            ax.imshow(overlay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
