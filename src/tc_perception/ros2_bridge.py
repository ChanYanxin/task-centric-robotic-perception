"""Optional ROS 2 result publisher.

Use a ROS 2 Python environment with rclpy and std_msgs already installed.
Publishes observations only; never commands a real actuator.
"""
from __future__ import annotations
import json
from .benchmark import read_video
from .detectors import make_detector
from .pipeline import PerceptionPipeline
from .sensing import SensingPolicy
from .transport import LocalEdge


def publish_video(video: str, mode: str = "event", detector: str = "color",
                  model: str = "yolov8n.pt", topic: str = "/perception/task_status"):
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as exc:
        raise RuntimeError("ROS 2 rclpy/std_msgs unavailable. Source your ROS 2 environment first.") from exc

    frames,fps = read_video(video)
    pipeline = PerceptionPipeline(SensingPolicy(mode),LocalEdge(make_detector(detector,model)))
    rclpy.init()
    node = Node("tc_perception_demo")
    publisher = node.create_publisher(String,topic,10)
    try:
        for i,frame in enumerate(frames):
            result = pipeline.process(frame,i,i/fps)
            payload = {"frame_id":i,"timestamp_s":i/fps,"decision":result.decision,
                       "trigger":result.trigger_reason,"objects":[d.to_dict() for d in result.detections]}
            message = String()
            message.data = json.dumps(payload)
            publisher.publish(message)
            rclpy.spin_once(node,timeout_sec=0)
        print(f"Published {len(frames)} perception messages on {topic} (non-realtime playback)")
    finally:
        node.destroy_node()
        rclpy.shutdown()
