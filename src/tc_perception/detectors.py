"""Interchangeable edge-side detectors: deterministic synthetic demo / real YOLO."""
from __future__ import annotations
from typing import Protocol
import cv2
import numpy as np
from .types import Detection


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class ColorShapeDetector:
    """ONLY for generated red obstacles / blue packages; NOT real-world AI."""
    def __init__(self, min_area: int = 130):
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Both red hue ranges, and blue package range; JPEG tolerant.
        red1 = cv2.inRange(hsv, (0, 100, 65), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 65), (179, 255, 255))
        blue = cv2.inRange(hsv, (95, 100, 65), (135, 255, 255))
        outputs = []
        for label, mask in (("obstacle", red1 | red2), ("package", blue)):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                outputs.append(Detection(label, 1.0, (x, y, x+w, y+h)))
        return outputs


class YOLODetector:
    """Real inference via Ultralytics (model download may require internet)."""
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install -e '.[yolo]'") from exc
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self.model.predict(image, conf=self.confidence, verbose=False)[0]
        outputs = []
        for box in result.boxes:
            label = self.model.names[int(box.cls.item())]
            xyxy = tuple(int(round(v)) for v in box.xyxy[0].tolist())
            outputs.append(Detection(label, float(box.conf.item()), xyxy))
        return outputs


def make_detector(name: str, model: str = "yolov8n.pt") -> Detector:
    if name == "color":
        return ColorShapeDetector()
    if name == "yolo":
        return YOLODetector(model)
    raise ValueError(f"Unsupported detector: {name}")
