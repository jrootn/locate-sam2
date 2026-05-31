#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit("Install demo extras: pip install 'locate-sam2[demo]'") from exc

from PIL import Image

from locate_sam2.adapter import AdapterConfig
from locate_sam2.pipeline import LocateSam2Pipeline
from locate_sam2.visualize import save_overlay


def _run(image, prompt, grounder, generation_mode, prompt_mode, crop_mode):
    if image is None:
        return None, "Upload an image."
    adapter = AdapterConfig(
        prompt_mode=prompt_mode,
        crop_mode=crop_mode,
        rerank="best_score",
    )
    pipeline = LocateSam2Pipeline(
        grounder=grounder,
        adapter_config=adapter,
        generation_mode=generation_mode if grounder == "locateanything" else None,
    )
    pil = Image.fromarray(image).convert("RGB")
    result = pipeline.run(pil, prompt)
    out_path = Path("outputs/gradio_demo.png")
    if result.boxes:
        save_overlay(pil, result.boxes, result.masks, out_path, title=prompt)
        msg = (
            f"boxes={len(result.boxes)} "
            f"ground_ms={result.ground_ms:.1f} segment_ms={result.segment_ms:.1f}"
        )
        return str(out_path), msg
    return None, "No boxes detected."


def main() -> None:
    demo = gr.Interface(
        fn=_run,
        inputs=[
            gr.Image(type="numpy", label="Image"),
            gr.Textbox(label="Prompt", placeholder="red car on the left"),
            gr.Dropdown(["locateanything", "grounding_dino_tiny"], value="locateanything", label="Grounder"),
            gr.Dropdown(["hybrid", "fast", "slow"], value="hybrid", label="LocateAnything mode"),
            gr.Dropdown(["box", "box_point", "point"], value="box", label="SAM prompt mode"),
            gr.Dropdown(["crop", "full"], value="crop", label="SAM crop mode"),
        ],
        outputs=[gr.Image(label="Overlay"), gr.Textbox(label="Stats")],
        title="Locate-SAM2 Demo",
        description="Zero-shot text-to-mask segmentation with LocateAnything + SAM2.",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
