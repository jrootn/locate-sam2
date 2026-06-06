from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from locate_sam2.types import Box

PromptMode = Literal["box", "box_point", "point"]
CropMode = Literal["full", "crop"]


@dataclass
class MaskCandidate:
    mask: np.ndarray
    score: float
    box: Box


class Sam2Segmenter:
    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        device: str = "cuda",
        multimask_output: bool = True,
    ) -> None:
        self.device = device
        self.multimask_output = multimask_output
        self.model = Sam2Model.from_pretrained(model_id).to(device)
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model.eval()

    @staticmethod
    def _box_center_point(box: Box) -> list[list[float]]:
        cx, cy = box.center()
        return [[cx, cy]]

    @torch.no_grad()
    def segment_box(
        self,
        image: Image.Image,
        box: Box,
        *,
        prompt_mode: PromptMode = "box",
        crop_box: Box | None = None,
    ) -> MaskCandidate | None:
        if not box.is_valid():
            return None

        work_image = image
        offset_x = 0.0
        offset_y = 0.0
        local_box = box

        if crop_box is not None and crop_box.is_valid():
            x1, y1, x2, y2 = (int(v) for v in crop_box.as_xyxy())
            if x2 <= x1 or y2 <= y1:
                return None
            work_image = image.crop((x1, y1, x2, y2))
            if work_image.size[0] < 1 or work_image.size[1] < 1:
                return None
            offset_x = float(x1)
            offset_y = float(y1)
            local_box = Box(
                box.x1 - offset_x,
                box.y1 - offset_y,
                box.x2 - offset_x,
                box.y2 - offset_y,
            ).clamp(work_image.size[0], work_image.size[1])
            if not local_box.is_valid():
                return None

        if work_image.size[0] < 1 or work_image.size[1] < 1:
            return None

        processor_kwargs: dict = {"images": work_image, "return_tensors": "pt"}

        if prompt_mode == "point":
            processor_kwargs["input_points"] = [[self._box_center_point(local_box)]]
            processor_kwargs["input_labels"] = [[[1]]]
        elif prompt_mode == "box_point":
            processor_kwargs["input_boxes"] = [[local_box.as_xyxy()]]
            processor_kwargs["input_points"] = [[self._box_center_point(local_box)]]
            processor_kwargs["input_labels"] = [[[1]]]
        else:
            processor_kwargs["input_boxes"] = [[local_box.as_xyxy()]]

        try:
            inputs = self.processor(**processor_kwargs).to(self.device)
            outputs = self.model(**inputs, multimask_output=self.multimask_output)
        except (RuntimeError, ValueError, OSError):
            return None

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]

        scores_flat = outputs.iou_scores.cpu().reshape(-1)
        if masks.ndim == 4:
            num_masks = masks.shape[1]
        elif masks.ndim == 3:
            num_masks = masks.shape[0]
        else:
            num_masks = 1

        num_masks = min(num_masks, max(1, scores_flat.numel()))
        if scores_flat.numel() >= num_masks:
            pick_scores = scores_flat[:num_masks]
        else:
            pick_scores = scores_flat
            num_masks = pick_scores.numel() or 1

        best_idx = int(torch.argmax(pick_scores).item()) if pick_scores.numel() else 0
        best_idx = min(best_idx, num_masks - 1)

        if masks.ndim == 4:
            mask = masks[0, best_idx].numpy().astype(bool)
        elif masks.ndim == 3:
            mask = masks[best_idx].numpy().astype(bool)
        else:
            mask = masks.numpy().astype(bool)

        score = float(pick_scores[best_idx].item()) if pick_scores.numel() else 0.0

        if crop_box is not None and crop_box.is_valid():
            full = np.zeros((image.size[1], image.size[0]), dtype=bool)
            cx1, cy1, cx2, cy2 = (int(v) for v in crop_box.as_xyxy())
            h, w = mask.shape
            full[cy1 : cy1 + h, cx1 : cx1 + w] = mask[: cy2 - cy1, : cx2 - cx1]
            mask = full

        return MaskCandidate(mask=mask, score=score, box=box)
