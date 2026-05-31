from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from PIL import Image

from locate_sam2.segment import CropMode, PromptMode, Sam2Segmenter
from locate_sam2.types import Box

RerankMode = Literal["top1", "best_score", "largest_box"]


class Grounder(Protocol):
    name: str

    def ground(self, image: Image.Image, phrase: str, multi: bool = False) -> tuple[str, list[Box]]: ...


@dataclass
class AdapterConfig:
    prompt_mode: PromptMode = "box"
    crop_mode: CropMode = "crop"
    crop_padding: float = 0.05
    rerank: RerankMode = "best_score"
    multi: bool = False


@dataclass
class AdapterResult:
    phrase: str
    answer: str
    boxes: list[Box] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    mask_scores: list[float] = field(default_factory=list)
    selected_box_index: int = -1
    ground_ms: float = 0.0
    segment_ms: float = 0.0

    @property
    def latency_ms(self) -> float:
        return self.ground_ms + self.segment_ms

    @property
    def primary_mask(self) -> np.ndarray | None:
        return self.masks[0] if self.masks else None


class PromptToMaskAdapter:
    """Convert grounder outputs into ranked SAM2 mask predictions."""

    def __init__(
        self,
        grounder: Grounder,
        segmenter: Sam2Segmenter,
        config: AdapterConfig | None = None,
    ) -> None:
        self.grounder = grounder
        self.segmenter = segmenter
        self.config = config or AdapterConfig()

    def _select_boxes(self, boxes: list[Box], width: int, height: int) -> list[Box]:
        valid = [b.clamp(width, height) for b in boxes if b.is_valid()]
        if not valid:
            return []

        if self.config.rerank == "largest_box":
            valid.sort(key=lambda b: b.area(), reverse=True)
            return valid[:1] if not self.config.multi else valid

        if self.config.rerank == "top1" or len(valid) == 1:
            return valid[:1] if not self.config.multi else valid

        return valid if self.config.multi else valid

    def _crop_for_box(self, box: Box, width: int, height: int) -> Box | None:
        if self.config.crop_mode == "full":
            return None
        return box.pad(width, height, self.config.crop_padding)

    def run(self, image: Image.Image, phrase: str) -> AdapterResult:
        import time

        image = image.convert("RGB")
        width, height = image.size

        t0 = time.perf_counter()
        answer, boxes = self.grounder.ground(image, phrase, multi=self.config.multi)
        ground_ms = (time.perf_counter() - t0) * 1000

        candidate_boxes = self._select_boxes(boxes, width, height)
        if self.config.rerank == "top1" and candidate_boxes and not self.config.multi:
            candidate_boxes = candidate_boxes[:1]

        t1 = time.perf_counter()
        mask_candidates = []
        for box in candidate_boxes:
            crop = self._crop_for_box(box, width, height)
            candidate = self.segmenter.segment_box(
                image,
                box,
                prompt_mode=self.config.prompt_mode,
                crop_box=crop,
            )
            if candidate is not None:
                mask_candidates.append(candidate)

        if self.config.rerank == "best_score" and len(mask_candidates) > 1:
            mask_candidates.sort(key=lambda c: c.score, reverse=True)
            if not self.config.multi:
                mask_candidates = mask_candidates[:1]

        segment_ms = (time.perf_counter() - t1) * 1000

        masks = [c.mask for c in mask_candidates]
        scores = [c.score for c in mask_candidates]
        selected = 0 if mask_candidates else -1

        return AdapterResult(
            phrase=phrase,
            answer=answer,
            boxes=[c.box for c in mask_candidates] or candidate_boxes,
            masks=masks,
            mask_scores=scores,
            selected_box_index=selected,
            ground_ms=ground_ms,
            segment_ms=segment_ms,
        )
