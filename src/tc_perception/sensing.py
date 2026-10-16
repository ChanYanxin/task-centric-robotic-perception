"""Three comparable sensing policies.

Continuous and fixed ROI always offload a frame. Event policy performs cheap
local motion analysis plus a periodic full task-zone refresh, mitigating the
risk that stationary objects never trigger motion.
"""
from __future__ import annotations
import cv2
import numpy as np
from .types import BBox, Selection


def task_zone(shape: tuple, fraction: float = 0.7) -> BBox:
    h, w = shape[:2]
    margin = int((1.0 - fraction) * w / 2)
    return (margin, 0, w - margin, h)


def clip_roi(roi: BBox, shape: tuple) -> BBox:
    h, w = shape[:2]
    x1, y1, x2, y2 = roi
    return (max(0, min(x1, w-1)), max(0, min(y1, h-1)),
            max(1, min(x2, w)), max(1, min(y2, h)))


class SensingPolicy:
    def __init__(self, mode: str, refresh_frames: int = 15,
                 motion_threshold: float = 0.0015, pixel_threshold: int = 18,
                 min_area: int = 90, padding: int = 30, zone_fraction: float = 0.7):
        if mode not in {"continuous", "roi", "event"}:
            raise ValueError(f"Unknown mode: {mode}")
        if refresh_frames < 1:
            raise ValueError("refresh_frames must be >= 1")
        self.mode, self.refresh_frames = mode, refresh_frames
        self.motion_threshold, self.pixel_threshold = motion_threshold, pixel_threshold
        self.min_area, self.padding, self.zone_fraction = min_area, padding, zone_fraction
        self.previous_gray: np.ndarray | None = None

    def select(self, frame: np.ndarray, frame_id: int) -> Selection:
        h, w = frame.shape[:2]
        zone = task_zone(frame.shape, self.zone_fraction)
        if self.mode == "continuous":
            return Selection("continuous", (0, 0, w, h))
        if self.mode == "roi":
            return Selection("fixed-task-zone", zone)

        # Motion is cheap but NOT free. The benchmark includes its runtime.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        prior = self.previous_gray
        self.previous_gray = gray
        if prior is None or frame_id % self.refresh_frames == 0:
            return Selection("periodic-refresh", zone)
        delta = cv2.absdiff(prior, gray)
        mask = cv2.threshold(delta, self.pixel_threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, None, iterations=2)
        x1z, y1z, x2z, y2z = zone
        zone_mask = mask[y1z:y2z, x1z:x2z]
        changed_fraction = cv2.countNonZero(zone_mask) / max(zone_mask.size, 1)
        if changed_fraction < self.motion_threshold:
            return Selection("no-motion", None)
        contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, rw, rh = cv2.boundingRect(contour)
            regions.append((x + x1z, y + y1z, x + x1z + rw, y + y1z + rh))
        if not regions:
            return Selection("motion-too-small", None)
        x1 = min(r[0] for r in regions) - self.padding
        y1 = min(r[1] for r in regions) - self.padding
        x2 = max(r[2] for r in regions) + self.padding
        y2 = max(r[3] for r in regions) + self.padding
        clipped = clip_roi((x1, y1, x2, y2), frame.shape)
        # Restrict dynamic crop to the task zone, not unrelated background.
        clipped = (max(clipped[0], x1z), clipped[1], min(clipped[2], x2z), clipped[3])
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return Selection("invalid-roi", None)
        return Selection("motion", clipped)
