"""Locate-SAM2: fast zero-shot text-guided segmentation with parallel box grounding."""

__version__ = "0.1.0"

__all__ = [
    "segment",
    "configure",
    "LocateSam2Pipeline",
    "LocateGroundedSamPipeline",
    "PipelineResult",
]


def __getattr__(name: str):
    if name == "segment":
        from locate_sam2.api import segment

        return segment
    if name == "configure":
        from locate_sam2.api import configure

        return configure
    if name == "LocateSam2Pipeline":
        from locate_sam2.pipeline import LocateSam2Pipeline

        return LocateSam2Pipeline
    if name == "LocateGroundedSamPipeline":
        from locate_sam2.pipeline import LocateGroundedSamPipeline

        return LocateGroundedSamPipeline
    if name == "PipelineResult":
        from locate_sam2.pipeline import PipelineResult

        return PipelineResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
