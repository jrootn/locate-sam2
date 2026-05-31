from __future__ import annotations

import torch
from PIL import Image
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

from locate_sam2.types import Box


class GroundingDinoGrounder:
    """Grounding DINO baseline grounder (Grounded-SAM-style front-end)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.name = "grounding_dino"

        self.processor = GroundingDinoProcessor.from_pretrained(model_path)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_path).to(device)
        self.model.eval()

    @staticmethod
    def _format_text(phrase: str) -> str:
        text = phrase.lower().strip()
        if not text.endswith("."):
            text += "."
        return text

    @torch.no_grad()
    def ground(self, image: Image.Image, phrase: str, multi: bool = False) -> tuple[str, list[Box]]:
        text = self._format_text(phrase)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_size = (image.size[1], image.size[0])
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[target_size],
        )[0]

        boxes: list[Box] = []
        scores = results.get("scores")
        for idx, box_tensor in enumerate(results["boxes"]):
            x1, y1, x2, y2 = box_tensor.tolist()
            boxes.append(Box(x1=x1, y1=y1, x2=x2, y2=y2))

        if not multi and boxes and scores is not None and len(scores) > 1:
            best_idx = int(torch.argmax(scores).item())
            boxes = [boxes[best_idx]]

        answer = f"detected {len(boxes)} box(es) for: {text}"
        return answer, boxes
