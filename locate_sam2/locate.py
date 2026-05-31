from __future__ import annotations

import re

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from locate_sam2.types import Box


class LocateAnythingGrounder:
    BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self.name = "locateanything"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    @staticmethod
    def parse_boxes(answer: str, width: int, height: int) -> list[Box]:
        boxes: list[Box] = []
        for match in LocateAnythingGrounder.BOX_RE.finditer(answer):
            x1, y1, x2, y2 = (int(g) for g in match.groups())
            boxes.append(
                Box(
                    x1=x1 / 1000 * width,
                    y1=y1 / 1000 * height,
                    x2=x2 / 1000 * width,
                    y2=y2 / 1000 * height,
                )
            )
        return boxes

    @torch.no_grad()
    def ground(self, image: Image.Image, phrase: str, multi: bool = False) -> tuple[str, list[Box]]:
        if multi:
            prompt = f"Locate all the instances that match the following description: {phrase}."
        else:
            prompt = f"Locate a single instance that matches the following description: {phrase}."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            generation_mode=self.generation_mode,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )

        answer = response[0] if isinstance(response, tuple) else response
        width, height = image.size
        return answer, self.parse_boxes(answer, width, height)
