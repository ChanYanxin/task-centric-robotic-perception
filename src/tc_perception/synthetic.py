"""Reproducible illustrative frames and per-frame ground truth.

Generated colors are intentionally easy for a color-threshold detector.
Synthetic metrics are NOT representative of real-world robotic perception.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from .types import Detection


def synthetic_frame(index: int, width: int = 640, height: int = 360) -> tuple[np.ndarray, list[Detection]]:
    if not 0 <= index < 180:
        raise ValueError("Demo frame index must be 0..179")
    frame = np.full((height, width, 3), (55, 52, 50), dtype=np.uint8)
    # Static, grayscale warehouse aisle: not confused with red/blue targets.
    for x in range(0, width, 80):
        cv2.line(frame, (x, 0), (x, height), (72, 72, 72), 1)
    for y in range(0, height, 60):
        cv2.line(frame, (0, y), (width, y), (66, 66, 66), 1)
    cv2.rectangle(frame, (96, 30), (544, 330), (105, 95, 83), 2)
    cv2.putText(frame, "ROBOT AISLE / SYNTHETIC", (15, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (230, 230, 230), 1, cv2.LINE_AA)
    objects = []
    # Moving package for short intervals, then mostly stationary.
    package_x = 160 + min(index, 25) * 4 + max(0, min(index-95, 15)) * 3
    package_box = (package_x, 224, package_x+56, 270)
    cv2.rectangle(frame, package_box[:2], package_box[2:], (245, 110, 22), -1)
    objects.append(Detection("package", 1.0, package_box))

    # A dangerous obstacle enters, then stays stationary, then disappears.
    if 65 <= index < 138:
        obstacle_x = 455 - min(index - 65, 22) * 4
        obstacle_box = (obstacle_x, 160, obstacle_x+55, 215)
        cv2.rectangle(frame, obstacle_box[:2], obstacle_box[2:], (18, 18, 240), -1)
        objects.append(Detection("obstacle", 1.0, obstacle_box))

    cv2.putText(frame, f"frame {index:03d}", (500, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (225, 225, 225), 1, cv2.LINE_AA)
    return frame, objects


def generate(directory: str | Path, fps: int = 15) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "warehouse_synthetic.mp4"
    truth = directory / "warehouse_synthetic.jsonl"
    first, _ = synthetic_frame(0)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize MP4 writer; check codec support")
    with truth.open("w", encoding="utf-8") as f:
        for index in range(180):
            frame, annotations = synthetic_frame(index)
            writer.write(frame)
            f.write(json.dumps({"frame_id":index, "timestamp_s":index/fps,
                                "detections":[d.to_dict() for d in annotations]}) + "\n")
    writer.release()
    cv2.imwrite(str(directory / "demo_frame.png"), synthetic_frame(80)[0])
    return video, truth


def load_truth(path: str | Path) -> dict[int, list[Detection]]:
    with Path(path).open(encoding="utf-8") as f:
        return {int(row["frame_id"]):
                [Detection(str(d["label"]),float(d["confidence"]),tuple(d["box"]))
                 for d in row.get("detections",[])]
                for line in f if (row := json.loads(line))}
