from __future__ import annotations

import json
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from pycocotools.coco import COCO
from tqdm import tqdm

from locate_sam2.adapter import AdapterConfig
from locate_sam2.config import load_config, resolve_path
from locate_sam2.eval.experiment_log import (
    build_run_manifest,
    load_config_snapshot,
    refs_pickle_for_dataset,
    save_run_manifest,
    subset_hash,
    subset_ref_ids,
)
from locate_sam2.eval.metrics import box_iou_xyxy, coco_ann_to_box, mask_iou, summarize_records
from locate_sam2.pipeline import GrounderName, LocateSam2Pipeline
from locate_sam2.types import Box

DatasetName = Literal["refcoco", "refcoco+", "refcocog"]


@dataclass
class EvalRecord:
    ref_id: int
    image_id: int
    sentence: str
    iou: float
    inter_pixels: int
    union_pixels: int
    box_iou: float
    detected_boxes: int
    ground_ms: float
    segment_ms: float
    grounder: str
    prompt_mode: str
    crop_mode: str
    rerank: str
    generation_mode: str | None = None


def _sentence_text(ref: dict) -> str:
    sent = ref["sentences"][0]["sent"]
    if isinstance(sent, str):
        return sent
    if isinstance(sent, dict):
        return sent.get("raw", sent.get("sent", ""))
    return str(sent)


def load_split_refs(
    data_dir: Path,
    dataset: DatasetName = "refcoco",
    split: str = "val",
) -> list[dict]:
    ref_file = data_dir / dataset / refs_pickle_for_dataset(dataset)
    with ref_file.open("rb") as f:
        refs = pickle.load(f)
    return [r for r in refs if r.get("split") == split]


def run_refcoco_eval(
    *,
    subset_size: int = 200,
    seed: int = 42,
    output_dir: Path | None = None,
    dataset: DatasetName = "refcoco",
    split: str = "val",
    grounder: GrounderName = "locateanything",
    generation_mode: str = "hybrid",
    prompt_mode: str = "box",
    crop_mode: str = "crop",
    rerank: str = "best_score",
    tag: str | None = None,
    box_threshold: float | None = None,
    text_threshold: float | None = None,
) -> dict:
    cfg = load_config()
    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    out_dir = output_dir or (resolve_path(cfg, cfg["paths"]["output_dir"]) / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = load_split_refs(data_dir, dataset=dataset, split=split)
    expected_ref_ids = subset_ref_ids(data_dir, dataset, split, subset_size, seed)
    rng = random.Random(seed)
    if subset_size > 0 and subset_size < len(refs):
        refs = rng.sample(refs, subset_size)

    instances_path = data_dir / dataset / "instances.json"
    if not instances_path.exists() and dataset != "refcoco":
        instances_path = data_dir / "refcoco" / "instances.json"
    coco = COCO(str(instances_path))

    adapter = AdapterConfig(
        prompt_mode=prompt_mode,  # type: ignore[arg-type]
        crop_mode=crop_mode,  # type: ignore[arg-type]
        rerank=rerank,  # type: ignore[arg-type]
    )
    dino_overrides = {}
    if box_threshold is not None:
        dino_overrides["box_threshold"] = box_threshold
    if text_threshold is not None:
        dino_overrides["text_threshold"] = text_threshold

    pipeline = LocateSam2Pipeline(
        grounder=grounder,
        adapter_config=adapter,
        generation_mode=generation_mode if grounder == "locateanything" else None,
        dino_overrides=dino_overrides or None,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records: list[EvalRecord] = []
    t_start = time.perf_counter()

    for ref in tqdm(refs, desc=f"{dataset} {split} eval"):
        image_id = ref["image_id"]
        ann_id = ref["ann_id"]
        image_path = data_dir / "train2014" / f"COCO_train2014_{image_id:012d}.jpg"
        if not image_path.exists() or ann_id not in coco.anns:
            continue

        gt = coco.loadAnns([ann_id])[0]
        gt_mask = coco.annToMask(gt)
        gt_box = coco_ann_to_box(gt)
        sentence = _sentence_text(ref)

        if grounder == "gt_oracle":
            pipeline.set_oracle_box(gt_box)

        result = pipeline.run_path(image_path, sentence, multi=False)

        if result.masks:
            iou, inter, union = mask_iou(result.masks[0], gt_mask)
        else:
            iou, inter, union = 0.0, 0, int(gt_mask.astype(bool).sum())

        pred_box = result.boxes[0] if result.boxes else Box(0, 0, 0, 0)
        b_iou = box_iou_xyxy(pred_box, gt_box) if result.boxes else 0.0

        records.append(
            EvalRecord(
                ref_id=ref["ref_id"],
                image_id=image_id,
                sentence=sentence,
                iou=iou,
                inter_pixels=inter,
                union_pixels=union,
                box_iou=b_iou,
                detected_boxes=len(result.boxes),
                ground_ms=result.ground_ms,
                segment_ms=result.segment_ms,
                grounder=grounder,
                prompt_mode=prompt_mode,
                crop_mode=crop_mode,
                rerank=rerank,
                generation_mode=generation_mode if grounder == "locateanything" else None,
            )
        )

    elapsed = time.perf_counter() - t_start
    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    metrics = summarize_records([asdict(r) for r in records], elapsed, peak_vram_gb=peak_vram)

    run_name = tag or f"{grounder}_{generation_mode if grounder == 'locateanything' else 'dino'}_{prompt_mode}_{crop_mode}_{rerank}"
    ref_ids = [r.ref_id for r in records]
    metrics_dict = metrics.to_dict()
    summary = {
        "run_name": run_name,
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "requested_subset_size": subset_size if subset_size > 0 else len(load_split_refs(data_dir, dataset, split)),
        "subset_size": len(records),
        "subset_hash": subset_hash(ref_ids) if subset_size > 0 and subset_size < 5000 else None,
        "grounder": grounder,
        "generation_mode": generation_mode if grounder == "locateanything" else None,
        "prompt_mode": prompt_mode,
        "crop_mode": crop_mode,
        "rerank": rerank,
        **metrics_dict,
    }

    manifest = build_run_manifest(
        run_name=run_name,
        dataset=dataset,
        split=split,
        seed=seed,
        subset_size=subset_size,
        ref_ids=ref_ids,
        config=load_config_snapshot(),
        extra={
            "metrics": metrics_dict,
            "grounder": grounder,
            "generation_mode": generation_mode if grounder == "locateanything" else None,
            "prompt_mode": prompt_mode,
            "crop_mode": crop_mode,
            "rerank": rerank,
            "expected_subset_ref_ids_match": ref_ids == expected_ref_ids[: len(ref_ids)] if subset_size > 0 else None,
        },
    )
    save_run_manifest(manifest, out_dir)

    (out_dir / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{run_name}_records.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2)
    )
    return summary
