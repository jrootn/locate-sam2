"""Locate-SAM2: fast zero-shot text-guided segmentation with parallel box grounding."""

from locate_sam2.api import configure, segment
from locate_sam2.pipeline import LocateGroundedSamPipeline, LocateSam2Pipeline, PipelineResult

__all__ = [
    "segment",
    "configure",
    "LocateSam2Pipeline",
    "LocateGroundedSamPipeline",
    "PipelineResult",
]
__version__ = "0.1.0"
