"""Shared messages and geometry utilities (pixel coordinates, x1 y1 x2 y2)."""
from __future__ import annotations
from dataclasses import dataclass, asdict

BBox = tuple[int, int, int, int]


def iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    denominator = area_a + area_b - overlap
    return overlap / denominator if denominator else 0.0


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: BBox

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Selection:
    trigger: str
    roi: BBox | None


@dataclass
class FrameResult:
    frame_id: int
    timestamp_s: float
    strategy: str
    triggered: bool
    trigger_reason: str
    roi: BBox | None
    bytes_sent: int
    detections: list[Detection]
    decision: str
    latency_ms: float
    inference_ms: float
    event_ms: float

    def to_dict(self) -> dict:
        return asdict(self)
