"""Same-video, equal JPEG-quality comparisons with CSV and JSON export."""
from __future__ import annotations
import csv
import json
import platform
import time
from pathlib import Path
import cv2
import numpy as np
from .detectors import make_detector
from .pipeline import PerceptionPipeline, assess
from .sensing import SensingPolicy
from .synthetic import load_truth
from .transport import LocalEdge, TCPEdgeClient
from .types import FrameResult


def read_video(video: str | Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("Video contains zero readable frames")
    return frames, fps


def evaluate_mode(frames: list[np.ndarray], fps: float, strategy: str,
                  detector_name: str = "color", model: str = "yolov8n.pt",
                  refresh: int = 15, edge=None) -> list[FrameResult]:
    edge = edge or LocalEdge(make_detector(detector_name, model))
    policy = SensingPolicy(strategy, refresh_frames=refresh)
    pipeline = PerceptionPipeline(policy, edge)
    return [pipeline.process(frame, i, i/fps) for i,frame in enumerate(frames)]


def summarize(records: list[FrameResult], truth=None) -> dict:
    latency = [r.latency_ms for r in records]
    out = {
        "strategy":records[0].strategy,
        "frames":len(records),
        "frames_offloaded":sum(r.triggered for r in records),
        "offload_fraction":round(sum(r.triggered for r in records)/len(records),4),
        "sent_bytes":sum(r.bytes_sent for r in records),
        "sent_megabytes":round(sum(r.bytes_sent for r in records)/1e6,4),
        "latency_p50_ms":round(float(np.percentile(latency,50)),3),
        "latency_p95_ms":round(float(np.percentile(latency,95)),3),
        "mean_inference_ms_per_frame":round(sum(r.inference_ms for r in records)/len(records),3),
        "mean_trigger_ms_per_frame":round(sum(r.event_ms for r in records)/len(records),3),
        "stop_frames":sum(r.decision == "STOP" for r in records),
    }
    if truth is not None:
        out.update(assess(records, truth))
    return out


def benchmark(video: str, output: str, gt: str | None = None,
              detector: str = "color", model: str = "yolov8n.pt",
              refresh: int = 15, max_frames: int | None = None):
    frames, fps = read_video(video,max_frames)
    truth = load_truth(gt) if gt else None
    report = []
    raw_records = []
    for mode in ("continuous", "roi", "event"):
        records = evaluate_mode(frames,fps,mode,detector,model,refresh)
        report.append(summarize(records,truth))
        raw_records.extend(r.to_dict() for r in records)
    baseline_bytes = report[0]["sent_bytes"]
    for row in report:
        row["traffic_reduction_vs_continuous_pct"] = round(100*(1-row["sent_bytes"]/baseline_bytes),2) if baseline_bytes else 0
    folder = Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"benchmark_summary.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]))
        writer.writeheader(); writer.writerows(report)
    with (folder/"per_frame.jsonl").open("w",encoding="utf8") as f:
        for record in raw_records:
            f.write(json.dumps(record)+"\n")
    payload = {
        "methodology":"Same decoded video frames, JPEG quality=80, local process encode/decode, detector per crop. No real wireless latency.",
        "warning":"Color threshold + synthetic inputs are for engineering validation only; not a real-world AI benchmark.",
        "source_video":str(video),"gt":str(gt) if gt else None,
        "fps":fps,"detector":detector,"model":model if detector=="yolo" else None,
        "refresh_frames":refresh,"system":{"python":platform.python_version(),"opencv":cv2.__version__,"platform":platform.platform()},
        "results":report,
    }
    (folder/"benchmark_summary.json").write_text(json.dumps(payload,indent=2),encoding="utf8")
    return payload
