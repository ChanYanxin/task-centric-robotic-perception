import numpy as np
from tc_perception.detectors import ColorShapeDetector
from tc_perception.sensing import SensingPolicy
from tc_perception.pipeline import PerceptionPipeline, assess
from tc_perception.synthetic import synthetic_frame
from tc_perception.transport import LocalEdge
from tc_perception.types import Detection, FrameResult, iou


def test_coordinate_translation_and_stop():
    image,gt=synthetic_frame(75)
    pipeline=PerceptionPipeline(SensingPolicy("roi"),LocalEdge(ColorShapeDetector()))
    result=pipeline.process(image,75,5.0)
    assert result.decision=="STOP"
    actual=next(d for d in gt if d.label=="obstacle")
    pred=next(d for d in result.detections if d.label=="obstacle")
    assert iou(actual.box,pred.box)>.8 # JPEG compression expands color mask a few pixels


def test_idle_skips_offload():
    frame=np.zeros((100,200,3),dtype=np.uint8)
    pipeline=PerceptionPipeline(SensingPolicy("event"),LocalEdge(ColorShapeDetector()))
    assert pipeline.process(frame,0,0).triggered
    later=pipeline.process(frame,1,0.1)
    assert later.bytes_sent==0
    assert not later.triggered


def test_assess_penalizes_skipped_detection():
    r=FrameResult(0,0,"event",False,"none",None,0,[],"GO",0,0,0)
    measured=assess([r],{0:[Detection("obstacle",1,(0,0,40,40))]})
    assert measured["decision_recall"]==0
    assert measured["fresh_detection_recall_iou50"]==0
