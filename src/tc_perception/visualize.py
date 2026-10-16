"""Annotated video to inspect strategy behavior frame by frame."""
from pathlib import Path
import cv2
from .types import FrameResult
from .benchmark import read_video, evaluate_mode


def annotate(frame, result: FrameResult):
    output = frame.copy()
    if result.roi:
        x1,y1,x2,y2 = result.roi
        cv2.rectangle(output,(x1,y1),(x2,y2),(0,220,220),2)
    for det in result.detections:
        x1,y1,x2,y2=det.box
        cv2.rectangle(output,(x1,y1),(x2,y2),(0,230,0),2)
        cv2.putText(output,det.label,(x1,max(0,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    cv2.rectangle(output,(0,0),(output.shape[1],38),(20,20,20),-1)
    cv2.putText(output,f"{result.strategy.upper()} | {result.trigger_reason} | {result.decision} | bytes={result.bytes_sent}",
                (7,25),cv2.FONT_HERSHEY_SIMPLEX,.55,
                (0,80,255) if result.decision=="STOP" else (150,255,150),2)
    return output


def render(video: str, output: str, mode: str, detector: str = "color",
           model: str = "yolov8n.pt", refresh: int = 15):
    frames,fps=read_video(video)
    results=evaluate_mode(frames,fps,mode,detector,model,refresh)
    h,w=frames[0].shape[:2]
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not writer.isOpened():
        raise RuntimeError("Could not open annotated video writer")
    for frame,result in zip(frames,results):
        writer.write(annotate(frame,result))
    writer.release()
