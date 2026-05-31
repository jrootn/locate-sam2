from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from locate_sam2.types import Box

IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, int, int]:
    """Return (IoU, intersection_pixels, union_pixels)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 0.0, inter, union
    return float(inter / union), inter, union


def box_iou_xyxy(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = a.area()
    area_b = b.area()
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def coco_ann_to_box(ann: dict) -> Box:
    x, y, w, h = ann["bbox"]
    return Box(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h))


def precision_at(ious: list[float], threshold: float) -> float:
    if not ious:
        return 0.0
    return float(np.mean([1.0 if x >= threshold else 0.0 for x in ious]))


@dataclass
class EvalMetrics:
    n: int
    mean_mask_iou: float
    overall_iou: float
    box_detection_rate: float
    mean_box_iou: float
    box_success_at_0_5: float
    mean_ground_ms: float
    mean_segment_ms: float
    mean_total_ms: float
    elapsed_sec: float
    peak_vram_gb: float | None = None
    precision: dict[str, float] | None = None

    def to_dict(self) -> dict:
        out = {
            "n": self.n,
            "mean_mask_iou": self.mean_mask_iou,
            "overall_iou": self.overall_iou,
            "box_detection_rate": self.box_detection_rate,
            "mean_box_iou": self.mean_box_iou,
            "box_success_at_0.5": self.box_success_at_0_5,
            "mean_ground_ms": self.mean_ground_ms,
            "mean_segment_ms": self.mean_segment_ms,
            "mean_total_ms": self.mean_total_ms,
            "elapsed_sec": self.elapsed_sec,
        }
        if self.peak_vram_gb is not None:
            out["peak_vram_gb"] = self.peak_vram_gb
        if self.precision:
            out.update(self.precision)
        return out


def summarize_records(records: list[dict], elapsed_sec: float, peak_vram_gb: float | None = None) -> EvalMetrics:
    if not records:
        return EvalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, elapsed_sec, peak_vram_gb, {})

    ious = [float(r["iou"]) for r in records]
    box_ious = [float(r.get("box_iou", 0.0)) for r in records]
    ground_ms = [float(r["ground_ms"]) for r in records]
    segment_ms = [float(r["segment_ms"]) for r in records]
    hits = [1.0 if int(r.get("detected_boxes", 0)) > 0 else 0.0 for r in records]

    total_inter = sum(int(r.get("inter_pixels", 0)) for r in records)
    total_union = sum(int(r.get("union_pixels", 0)) for r in records)
    overall = float(total_inter / total_union) if total_union > 0 else 0.0

    precision = {f"precision_at_{t}": precision_at(ious, t) for t in IOU_THRESHOLDS}
    # backward-compatible aliases
    precision["success_at_0.5"] = precision["precision_at_0.5"]
    precision["success_at_0.7"] = precision["precision_at_0.7"]

    return EvalMetrics(
        n=len(records),
        mean_mask_iou=float(np.mean(ious)),
        overall_iou=overall,
        box_detection_rate=float(np.mean(hits)),
        mean_box_iou=float(np.mean(box_ious)),
        box_success_at_0_5=precision_at(box_ious, 0.5),
        mean_ground_ms=float(np.mean(ground_ms)),
        mean_segment_ms=float(np.mean(segment_ms)),
        mean_total_ms=float(np.mean(ground_ms) + np.mean(segment_ms)),
        elapsed_sec=elapsed_sec,
        peak_vram_gb=peak_vram_gb,
        precision=precision,
    )
