from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def clamp(self, width: int, height: int) -> Box:
        return Box(
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
        )

    def pad(self, width: int, height: int, ratio: float) -> Box:
        bw = self.x2 - self.x1
        bh = self.y2 - self.y1
        px = bw * ratio
        py = bh * ratio
        return self.__class__(
            x1=self.x1 - px,
            y1=self.y1 - py,
            x2=self.x2 + px,
            y2=self.y2 + py,
        ).clamp(width, height)

    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1
