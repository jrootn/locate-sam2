from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from locate_sam2.adapter import AdapterConfig
from locate_sam2.pipeline import LocateSam2Pipeline, PipelineResult

_PIPELINE: LocateSam2Pipeline | None = None


def _get_pipeline() -> LocateSam2Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = LocateSam2Pipeline()
    return _PIPELINE


def segment(
    image: Union[str, Path, Image.Image],
    prompt: str,
    *,
    multi: bool = False,
    return_details: bool = False,
) -> list[np.ndarray] | PipelineResult:
    """Zero-shot text-guided segmentation.

    Example::

        from locate_sam2 import segment
        masks = segment("image.jpg", "red car on the left")
    """
    pipeline = _get_pipeline()
    if isinstance(image, (str, Path)):
        result = pipeline.run_path(Path(image), prompt, multi=multi)
    else:
        result = pipeline.run(image.convert("RGB"), prompt, multi=multi)

    if return_details:
        return result
    return result.masks


def configure(
    *,
    grounder: str = "locateanything",
    generation_mode: str | None = None,
    prompt_mode: str = "box",
    crop_mode: str = "crop",
    rerank: str = "best_score",
) -> None:
    """Reset the cached pipeline with new settings."""
    global _PIPELINE
    cfg = None
    adapter = AdapterConfig(
        prompt_mode=prompt_mode,  # type: ignore[arg-type]
        crop_mode=crop_mode,  # type: ignore[arg-type]
        rerank=rerank,  # type: ignore[arg-type]
    )
    _PIPELINE = LocateSam2Pipeline(
        grounder=grounder,  # type: ignore[arg-type]
        adapter_config=adapter,
        generation_mode=generation_mode,
    )
