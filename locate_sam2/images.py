"""Safe RGB image loading for pipeline and eval."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

MIN_IMAGE_BYTES = 128
MIN_IMAGE_SIDE = 1


class ImageLoadError(ValueError):
    """Raised when an image path cannot be loaded for inference."""


def load_rgb_image(path: Path | str) -> Image.Image:
    """Load a PIL RGB image, validating file size and decoded dimensions."""
    p = Path(path)
    if not p.is_file():
        raise ImageLoadError(f"Missing image: {p}")
    if p.stat().st_size < MIN_IMAGE_BYTES:
        raise ImageLoadError(f"Image too small ({p.stat().st_size} bytes): {p}")

    try:
        with Image.open(p) as im:
            im.load()
            rgb = im.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Cannot decode image {p}: {exc}") from exc

    w, h = rgb.size
    if w < MIN_IMAGE_SIDE or h < MIN_IMAGE_SIDE:
        raise ImageLoadError(f"Invalid image dimensions {w}x{h}: {p}")
    return rgb


def is_loadable_image(path: Path | str) -> bool:
    try:
        load_rgb_image(path)
        return True
    except ImageLoadError:
        return False
