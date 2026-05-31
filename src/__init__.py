"""Backward-compatible import path. Prefer `locate_sam2`."""

from locate_sam2.pipeline import LocateGroundedSamPipeline, LocateSam2Pipeline, PipelineResult
from locate_sam2.api import segment

__all__ = ["LocateSam2Pipeline", "LocateGroundedSamPipeline", "PipelineResult", "segment"]
