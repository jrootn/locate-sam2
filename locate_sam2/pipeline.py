from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from locate_sam2.adapter import AdapterConfig, AdapterResult, PromptToMaskAdapter
from locate_sam2.config import load_config, resolve_path
from locate_sam2.images import load_rgb_image
from locate_sam2.dino import GroundingDinoGrounder
from locate_sam2.locate import LocateAnythingGrounder
from locate_sam2.oracle import OracleBoxGrounder
from locate_sam2.segment import Sam2Segmenter
from locate_sam2.types import Box

GrounderName = Literal[
    "locateanything",
    "grounding_dino_tiny",
    "grounding_dino_swint",
    "gt_oracle",
]


@dataclass
class PipelineResult:
    phrase: str
    answer: str
    boxes: list[Box] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    mask_scores: list[float] = field(default_factory=list)
    ground_ms: float = 0.0
    segment_ms: float = 0.0
    grounder: str = "locateanything"

    @property
    def locate_ms(self) -> float:
        return self.ground_ms

    @property
    def latency_ms(self) -> float:
        return self.ground_ms + self.segment_ms


def _build_grounder(cfg: dict, name: GrounderName, dino_overrides: dict | None = None):
    if name == "gt_oracle":
        return OracleBoxGrounder()

    if name in ("grounding_dino_tiny", "grounding_dino_swint"):
        key = "grounding_dino" if name == "grounding_dino_tiny" else "grounding_dino_swint"
        dino_cfg = cfg["models"]["baseline"][key]
        overrides = dino_overrides or {}
        path = resolve_path(cfg, dino_cfg["local_dir"])
        g = GroundingDinoGrounder(
            model_path=str(path),
            box_threshold=float(overrides.get("box_threshold", dino_cfg.get("box_threshold", 0.25))),
            text_threshold=float(overrides.get("text_threshold", dino_cfg.get("text_threshold", 0.25))),
        )
        g.name = name
        return g

    locate_cfg = cfg["models"]["locateanything"]
    path = resolve_path(cfg, locate_cfg["local_dir"])
    return LocateAnythingGrounder(
        model_path=str(path),
        generation_mode=locate_cfg.get("generation_mode", "hybrid"),
        max_new_tokens=int(locate_cfg.get("max_new_tokens", 2048)),
    )


class LocateSam2Pipeline:
    """Locate-SAM2: modular zero-shot text-to-mask pipeline."""

    def __init__(
        self,
        config_path: Path | None = None,
        grounder: GrounderName = "locateanything",
        adapter_config: AdapterConfig | None = None,
        generation_mode: str | None = None,
        dino_overrides: dict | None = None,
    ) -> None:
        self.cfg = load_config(config_path)
        pipe_cfg = self.cfg["pipeline"]
        adapter_cfg = adapter_config or AdapterConfig(
            prompt_mode=pipe_cfg.get("prompt_mode", "box"),
            crop_mode=pipe_cfg.get("crop_mode", "crop"),
            crop_padding=float(pipe_cfg.get("crop_padding", 0.05)),
            rerank=pipe_cfg.get("rerank", "best_score"),
        )

        segment_cfg = self.cfg["models"]["segment"]
        segmenter = Sam2Segmenter(
            model_id=segment_cfg.get("repo_id", "facebook/sam2.1-hiera-large"),
            multimask_output=bool(segment_cfg.get("multimask_output", True)),
        )
        self.grounder_name = grounder
        grounder_model = _build_grounder(self.cfg, grounder, dino_overrides)
        if generation_mode and grounder == "locateanything":
            grounder_model.generation_mode = generation_mode
        self.adapter = PromptToMaskAdapter(grounder_model, segmenter, adapter_cfg)

    def set_oracle_box(self, box: Box) -> None:
        if isinstance(self.adapter.grounder, OracleBoxGrounder):
            self.adapter.grounder.set_box(box)

    def _to_result(self, adapter_result: AdapterResult) -> PipelineResult:
        return PipelineResult(
            phrase=adapter_result.phrase,
            answer=adapter_result.answer,
            boxes=adapter_result.boxes,
            masks=adapter_result.masks,
            mask_scores=adapter_result.mask_scores,
            ground_ms=adapter_result.ground_ms,
            segment_ms=adapter_result.segment_ms,
            grounder=self.grounder_name,
        )

    def run(self, image: Image.Image, phrase: str, multi: bool = False) -> PipelineResult:
        self.adapter.config.multi = multi
        return self._to_result(self.adapter.run(image, phrase))

    def run_path(self, image_path: Path, phrase: str, multi: bool = False) -> PipelineResult:
        return self.run(load_rgb_image(image_path), phrase, multi=multi)


LocateGroundedSamPipeline = LocateSam2Pipeline
