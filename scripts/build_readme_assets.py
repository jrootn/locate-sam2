from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "research_paper" / "figures"
OUT = ROOT / "docs" / "assets"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT = FONT_DIR / "DejaVuSans.ttf"
BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

INK = (28, 31, 36)
MUTED = (82, 89, 99)
GRID = (219, 224, 232)
PAPER = (248, 250, 252)
WHITE = (255, 255, 255)
GREEN = (25, 135, 84)
RED = (185, 28, 28)
BLUE = (37, 99, 235)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(text: str, max_chars: int) -> str:
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=False))


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    width_chars: int,
    line_gap: int = 6,
) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars, break_long_words=False):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += text_size(draw, line, fnt)[1] + line_gap
    return y


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return ImageOps.contain(img, size, Image.Resampling.LANCZOS)


def paste_framed(
    canvas: Image.Image,
    img: Image.Image,
    xy: tuple[int, int],
    frame_size: tuple[int, int],
    bg: tuple[int, int, int] = WHITE,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    w, h = frame_size
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=bg, outline=GRID, width=2)
    px = x + (w - img.width) // 2
    py = y + (h - img.height) // 2
    canvas.paste(img, (px, py))


def meta(case: str) -> dict:
    with (FIGURES / case / "metadata.json").open() as f:
        return json.load(f)


def build_main_examples() -> None:
    cases = [
        ("win_ref5466", "Hybrid corrects a missed object"),
        ("win_ref2764", "Hybrid resolves a terse expression"),
        ("both_ref3281", "Both methods succeed on a named object"),
    ]
    columns = [
        ("Input image", "image_raw.jpg"),
        ("DINO-Tiny + SAM2", "dino_overlay.png"),
        ("Locate-SAM2 hybrid", "ours_overlay.png"),
    ]
    col_w, frame_h = 420, 292
    margin, gap = 54, 0
    header_h, row_h = 210, 465
    width = margin * 2 + len(columns) * col_w
    height = header_h + len(cases) * row_h + 54
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 38), "RefCOCO qualitative examples", font=font(36, True), fill=INK)
    draw.text(
        (margin, 92),
        "Each row is one validation expression. Green boxes are grounder predictions; red masks are SAM 2.1 outputs.",
        font=font(20),
        fill=MUTED,
    )

    table_x0, table_y0 = margin, 132
    table_x1 = width - margin
    draw.rounded_rectangle((table_x0, table_y0, table_x1, height - 38), radius=8, fill=WHITE, outline=GRID, width=2)
    draw.rectangle((table_x0, table_y0, table_x1, table_y0 + 58), fill=(241, 245, 249))

    for idx, (label, _) in enumerate(columns):
        x = margin + idx * col_w
        tw, _ = text_size(draw, label, font(22, True))
        draw.text((x + (col_w - tw) // 2, table_y0 + 17), label, font=font(22, True), fill=INK)
        if idx > 0:
            draw.line((x, table_y0, x, height - 38), fill=GRID, width=2)

    y = header_h
    for case, row_title in cases:
        m = meta(case)
        draw.line((table_x0, y - 20, table_x1, y - 20), fill=GRID, width=2)
        title_end = draw_wrapped(draw, (margin + 22, y), row_title, font(22, True), INK, 31, line_gap=4)
        query = f"Expression: \"{m['query']}\""
        draw_wrapped(draw, (margin + 22, title_end + 8), query, font(18), MUTED, 34)

        image_y = y + 112
        for idx, (_, filename) in enumerate(columns):
            x = margin + idx * col_w
            img = fit_image(FIGURES / case / filename, (col_w - 54, frame_h - 18))
            paste_framed(canvas, img, (x + 24, image_y), (col_w - 48, frame_h))

        dino = f"Tiny mIoU {m['dino_miou']:.3f}"
        ours = f"Hybrid mIoU {m['ours_miou']:.3f}"
        metric_y = image_y + frame_h + 16
        dino_w, _ = text_size(draw, dino, font(18, True))
        ours_w, _ = text_size(draw, ours, font(18, True))
        dino_x = margin + col_w + (col_w - dino_w) // 2
        ours_x = margin + 2 * col_w + (col_w - ours_w) // 2
        draw.text((dino_x, metric_y), dino, font=font(18, True), fill=RED)
        draw.text((ours_x, metric_y), ours, font=font(18, True), fill=GREEN)
        y += row_h

    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / "readme_refcoco_examples.png", quality=95)


def build_failure_taxonomy() -> None:
    cases = [
        ("fail_wrong_instance_ref20398", "Wrong instance"),
        ("fail_spatial_ref5750", "Spatial language"),
        ("fail_attribute_ref18360", "Attribute ambiguity"),
        ("fail_rare_or_long_ref36776", "Rare or unusual phrase"),
    ]
    card_w, card_h = 630, 500
    margin, gap = 54, 34
    width = margin * 2 + 2 * card_w + gap
    height = 150 + 2 * card_h + gap + 44
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 38), "Observed failure modes", font=font(34, True), fill=INK)
    draw.text(
        (margin, 88),
        "These are selected RefCOCO validation cases where the hybrid grounder chooses the wrong region.",
        font=font(20),
        fill=MUTED,
    )

    for i, (case, title) in enumerate(cases):
        row, col = divmod(i, 2)
        x = margin + col * (card_w + gap)
        y = 150 + row * (card_h + gap)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=8, fill=WHITE, outline=GRID, width=2)

        m = meta(case)
        draw.text((x + 24, y + 22), title, font=font(24, True), fill=INK)
        draw_wrapped(draw, (x + 24, y + 60), f"Expression: \"{m['query']}\"", font(17), MUTED, 58)

        img_w, img_h = 276, 250
        y_img = y + 126
        dino = fit_image(FIGURES / case / "dino_overlay.png", (img_w - 12, img_h - 12))
        ours = fit_image(FIGURES / case / "ours_overlay.png", (img_w - 12, img_h - 12))
        paste_framed(canvas, dino, (x + 24, y_img), (img_w, img_h))
        paste_framed(canvas, ours, (x + 330, y_img), (img_w, img_h))
        draw.text((x + 24, y_img + img_h + 14), f"DINO mIoU {m['dino_miou']:.3f}", font=font(18, True), fill=BLUE)
        draw.text((x + 330, y_img + img_h + 14), f"Hybrid mIoU {m['ours_miou']:.3f}", font=font(18, True), fill=RED)

    canvas.save(OUT / "readme_failure_taxonomy.png", quality=95)


def build_hallucination_probe() -> None:
    case = "img510591_neg8763"
    base = FIGURES / "hallucination_probe" / case
    prompt = (base / "query.txt").read_text().strip()
    gt_prompt = (base / "gt_query.txt").read_text().strip()
    with (FIGURES / "hallucination_probe" / "metadata.json").open() as f:
        m = json.load(f)
    row = next(r for r in m["rows"] if r["case_id"] == case)

    width, height = 1320, 660
    margin, gap = 54, 36
    frame_w, frame_h = 560, 360
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 38), "Negative prompt sanity check", font=font(34, True), fill=INK)
    draw_wrapped(
        draw,
        (margin, 88),
        "LocateAnything sometimes emits confident boxes for impossible expressions. We report this as a limitation, not as a benchmark score.",
        font(20),
        MUTED,
        105,
    )

    raw = fit_image(base / "image_raw.jpg", (frame_w - 18, frame_h - 18))
    overlay = fit_image(base / "ours_overlay.png", (frame_w - 18, frame_h - 18))
    y_img = 165
    paste_framed(canvas, raw, (margin, y_img), (frame_w, frame_h))
    paste_framed(canvas, overlay, (margin + frame_w + gap, y_img), (frame_w, frame_h))
    draw_wrapped(
        draw,
        (margin, y_img + frame_h + 18),
        f"Original RefCOCO image; GT expression: \"{gt_prompt}\"",
        font(18),
        MUTED,
        58,
    )
    y_caption = draw_wrapped(
        draw,
        (margin + frame_w + gap, y_img + frame_h + 18),
        f"Negative expression: \"{prompt}\"",
        font(18),
        MUTED,
        58,
    )
    draw.text(
        (margin + frame_w + gap, y_caption + 10),
        f"Mask emitted; SAM score {row['sam_mask_score']:.3f}. Probe emission rate: {m['mask_emission_rate'] * 100:.1f}%.",
        font=font(18, True),
        fill=RED,
    )

    canvas.save(OUT / "readme_hallucination_probe.png", quality=95)


def main() -> None:
    build_main_examples()
    build_failure_taxonomy()
    build_hallucination_probe()


if __name__ == "__main__":
    main()
