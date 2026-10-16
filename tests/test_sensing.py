import numpy as np
from tc_perception.sensing import SensingPolicy, task_zone, clip_roi
from tc_perception.synthetic import synthetic_frame


def test_event_periodic_and_idle():
    policy=SensingPolicy("event",refresh_frames=10)
    blank=np.zeros((100,150,3),dtype=np.uint8)
    assert policy.select(blank,0).trigger == "periodic-refresh"
    assert policy.select(blank,1).roi is None
    assert policy.select(blank,10).trigger == "periodic-refresh"


def test_motion_inside_task_zone():
    policy=SensingPolicy("event",motion_threshold=0.0001,min_area=10)
    blank=np.zeros((100,200,3),dtype=np.uint8)
    policy.select(blank,0)
    changed=blank.copy()
    changed[35:65,75:110,:]=200
    chosen=policy.select(changed,1)
    assert chosen.trigger=="motion"
    assert chosen.roi is not None
    assert chosen.roi[0] <= 75 and chosen.roi[2] >= 110


def test_full_fixed_roi():
    frame=np.zeros((100,200,3),dtype=np.uint8)
    assert SensingPolicy("continuous").select(frame,1).roi == (0,0,200,100)
    assert SensingPolicy("roi").select(frame,1).roi == task_zone(frame.shape)


def test_synthetic_objects():
    _, first=synthetic_frame(0)
    _, hazard=synthetic_frame(70)
    assert not any(d.label=="obstacle" for d in first)
    assert any(d.label=="obstacle" for d in hazard)
