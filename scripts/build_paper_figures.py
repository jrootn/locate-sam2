#!/usr/bin/env python3
"""Build paper-ready comparison figure grids from full-val records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locate_sam2.adapter import AdapterConfig
from locate_sam2.config import load_config, resolve_path
from locate_sam2.eval.metrics import coco_ann_to_box
from locate_sam2.pipeline import LocateSam2Pipeline


def _load_records(path: Path) -> dict[int, dict]:
    return {r["ref_id"]: r for r in json.loads(path.read_text())}


def _mask_overlay(image: np.ndarray, mask: np.ndarray, color=(1.0, 0.0, 0.0, 0.45)) -> np.ndarray:
    out = np.zeros((*mask.shape, 4))
    out[mask.astype(bool)] = color
    return out


def _box_on_ax(ax, box, color="lime"):
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (box.x1, box.y1),
            box.x2 - box.x1,
            box.y2 - box.y1,
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
    )


def _render_row(
    ref_id: int,
    ours_rec: dict,
    dino_rec: dict,
    oracle_rec: dict | None,
    data_dir: Path,
    coco: COCO,
    ann_id_map: dict[int, int],
    ours_pipe: LocateSam2Pipeline,
    dino_pipe: LocateSam2Pipeline,
    oracle_pipe: LocateSam2Pipeline,
) -> np.ndarray:
    image_id = ours_rec["image_id"]
    image_path = data_dir / "train2014" / f"COCO_train2014_{image_id:012d}.jpg"
    image = Image.open(image_path).convert("RGB")
    img = np.array(image)
    sentence = ours_rec["sentence"]

    ann_id = ann_id_map.get(ref_id)
    gt_mask = np.zeros(img.shape[:2], dtype=bool)
    if ann_id and ann_id in coco.anns:
        gt = coco.loadAnns([ann_id])[0]
        gt_mask = coco.annToMask(gt).astype(bool)
        oracle_pipe.set_oracle_box(coco_ann_to_box(gt))

    ores = ours_pipe.run(image, sentence)
    dres = dino_pipe.run(image, sentence)
    oracle_res = oracle_pipe.run(image, sentence)

    cols = ["Input", "GT mask", "DINO-Tiny", "Locate-SAM2", "Oracle"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, title in zip(axes, cols):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    axes[0].set_title(f'"{sentence[:50]}"', fontsize=8)

    if gt_mask.any():
        axes[1].imshow(_mask_overlay(img, gt_mask, (0, 1, 0, 0.45)))

    if dres.masks:
        axes[2].imshow(_mask_overlay(img, dres.masks[0]))
        if dres.boxes:
            _box_on_ax(axes[2], dres.boxes[0], "yellow")
    axes[2].set_title(f"DINO-Tiny\nIoU={dino_rec['iou']:.2f}", fontsize=8)

    if ores.masks:
        axes[3].imshow(_mask_overlay(img, ores.masks[0]))
        if ores.boxes:
            _box_on_ax(axes[3], ores.boxes[0], "lime")
    axes[3].set_title(f"Locate-SAM2\nIoU={ours_rec['iou']:.2f}", fontsize=8)

    if oracle_res.masks:
        axes[4].imshow(_mask_overlay(img, oracle_res.masks[0], (0, 0.5, 1, 0.45)))
    if oracle_rec:
        axes[4].set_title(f"GT-box oracle\nIoU={oracle_rec['iou']:.2f}", fontsize=8)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    arr = buf[..., :3].copy()
    plt.close(fig)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-records", type=Path, default=Path("outputs/full_val/locate_sam2_hybrid_full_records.json"))
    parser.add_argument("--dino-records", type=Path, default=Path("outputs/full_val/dino_tiny_full_records.json"))
    parser.add_argument("--oracle-records", type=Path, default=Path("outputs/full_val/gt_oracle_full_records.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full_val/paper_figures"))
    parser.add_argument("--n-wins", type=int, default=4)
    parser.add_argument("--n-fails", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    coco = COCO(str(data_dir / "refcoco" / "instances.json"))

    import pickle

    with (data_dir / "refcoco" / "refs(unc).p").open("rb") as f:
        refs = pickle.load(f)
    ann_id_map = {r["ref_id"]: r["ann_id"] for r in refs if r.get("split") == "val"}

    ours = _load_records(args.ours_records)
    dino = _load_records(args.dino_records)
    oracle = _load_records(args.oracle_records) if args.oracle_records.exists() else {}
    common = sorted(set(ours) & set(dino))

    adapter = AdapterConfig(crop_mode="crop", rerank="best_score")
    ours_pipe = LocateSam2Pipeline(grounder="locateanything", adapter_config=adapter, generation_mode="hybrid")
    dino_pipe = LocateSam2Pipeline(grounder="grounding_dino_tiny", adapter_config=adapter)
    oracle_pipe = LocateSam2Pipeline(grounder="gt_oracle", adapter_config=adapter)

    wins = sorted(common, key=lambda r: ours[r]["iou"] - dino[r]["iou"], reverse=True)[: args.n_wins]
    fails = sorted(common, key=lambda r: ours[r]["iou"])[: args.n_fails]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for group, ref_ids in [("comparison_wins", wins), ("comparison_failures", fails)]:
        rows = []
        for ref_id in ref_ids:
            row = _render_row(
                ref_id,
                ours[ref_id],
                dino[ref_id],
                oracle.get(ref_id),
                data_dir,
                coco,
                ann_id_map,
                ours_pipe,
                dino_pipe,
                oracle_pipe,
            )
            rows.append(row)

        if not rows:
            continue

        grid = np.vstack(rows)
        path = out / f"{group}.png"
        Image.fromarray(grid).save(path, dpi=(150, 150))
        manifest.append({"group": group, "refs": ref_ids, "file": path.name})
        print(f"saved {path}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
