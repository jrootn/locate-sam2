from __future__ import annotations

from PIL import Image

from locate_sam2.types import Box


class OracleBoxGrounder:
    """Oracle upper bound: segment from ground-truth COCO box (no text grounding)."""

    name = "gt_oracle"

    def __init__(self) -> None:
        self._box: Box | None = None

    def set_box(self, box: Box) -> None:
        self._box = box

    def ground(self, image: Image.Image, phrase: str, multi: bool = False) -> tuple[str, list[Box]]:
        del image, phrase, multi
        if self._box is None or not self._box.is_valid():
            return "oracle: no box", []
        return "oracle gt box", [self._box]
