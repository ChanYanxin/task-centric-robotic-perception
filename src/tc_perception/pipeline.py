"""Frame-level perception policy → JPEG edge inference → task decision."""
from __future__ import annotations
import time
from typing import Protocol
import numpy as np
from .sensing import SensingPolicy
from .types import BBox, Detection, FrameResult, iou


class Edge(Protocol):
    def infer(self, crop: np.ndarray, frame_id: int) -> tuple[list[Detection], int, float]: ...


class PerceptionPipeline:
    def __init__(self, policy: SensingPolicy, edge: Edge, hazard_labels: tuple[str,...] =
                 ("obstacle", "person", "car", "truck", "bicycle", "motorcycle"),
                 stale_frames: int = 20):
        self.policy, self.edge = policy, edge
        self.hazard_labels = set(hazard_labels)
        self.stale_frames = stale_frames
        self.last_hazard_frame: int | None = None

    def process(self, frame: np.ndarray, frame_id: int, timestamp_s: float) -> FrameResult:
        start = time.perf_counter()
        selection = self.policy.select(frame, frame_id)
        event_ms = (time.perf_counter() - start) * 1000
        detections: list[Detection] = []
        bytes_sent, inference_ms = 0, 0.0
        if selection.roi is not None:
            x1, y1, x2, y2 = selection.roi
            crop = frame[y1:y2, x1:x2]
            found, bytes_sent, inference_ms = self.edge.infer(crop, frame_id)
            for det in found:
                bx1, by1, bx2, by2 = det.box
                detections.append(Detection(det.label, det.confidence,
                                            (bx1+x1,by1+y1,bx2+x1,by2+y1)))
            if any(det.label in self.hazard_labels for det in detections):
                self.last_hazard_frame = frame_id
            # A full task-zone refresh that finds no hazard can clear stale alerts.
            elif selection.trigger in ("continuous", "fixed-task-zone", "periodic-refresh"):
                self.last_hazard_frame = None
        stop = self.last_hazard_frame is not None and frame_id - self.last_hazard_frame <= self.stale_frames
        decision = "STOP" if stop else "GO"
        return FrameResult(frame_id, timestamp_s, self.policy.mode,
                           selection.roi is not None, selection.trigger, selection.roi,
                           bytes_sent, detections, decision,
                           (time.perf_counter()-start)*1000, inference_ms, event_ms)


def assess(results: list[FrameResult], truth: dict[int, list[Detection]],
           hazard_label: str = "obstacle") -> dict:
    """Frame-level safety decision metrics; separate from COCO mAP."""
    tp = fp = tn = fn = 0
    detection_tp = detection_fp = detection_fn = 0
    for result in results:
        gt = [d for d in truth.get(result.frame_id, []) if d.label == hazard_label]
        pred = result.decision == "STOP"
        actual = bool(gt)
        if pred and actual: tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else: tn += 1
        # Detection IoU evaluated on all frames: skipped inference = no fresh detection.
        guess = [d for d in result.detections if d.label == hazard_label]
        matched = set()
        for prediction in guess:
            match = next((i for i,d in enumerate(gt) if i not in matched and
                          iou(prediction.box,d.box) >= 0.5), None)
            if match is None: detection_fp += 1
            else:
                detection_tp += 1
                matched.add(match)
        detection_fn += len(gt) - len(matched)
    return {
        "decision_precision":tp/max(1,tp+fp),
        "decision_recall":tp/max(1,tp+fn),
        "decision_false_positive_frames":fp,
        "decision_false_negative_frames":fn,
        "fresh_detection_precision_iou50":detection_tp/max(1,detection_tp+detection_fp),
        "fresh_detection_recall_iou50":detection_tp/max(1,detection_tp+detection_fn),
        "decision_tp":tp,"decision_tn":tn,
    }
