"""CLI commands. See README for the exact reproducible walkthrough."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .benchmark import benchmark, read_video
from .detectors import make_detector
from .pipeline import PerceptionPipeline
from .sensing import SensingPolicy
from .synthetic import generate
from .transport import TCPEdgeClient, serve
from .visualize import render


def main(argv=None):
    parser=argparse.ArgumentParser(prog="tc-perception",description="Task-centric robotic perception prototype")
    sub=parser.add_subparsers(dest="command",required=True)
    synth=sub.add_parser("generate",help="Generate reproducible synthetic warehouse video + ground truth")
    synth.add_argument("--output",default="examples/generated")
    bench=sub.add_parser("benchmark",help="Compare three sensing strategies on the same video")
    bench.add_argument("--video",default="examples/generated/warehouse_synthetic.mp4")
    bench.add_argument("--gt",default="examples/generated/warehouse_synthetic.jsonl")
    bench.add_argument("--output",default="results")
    bench.add_argument("--detector",choices=["color","yolo"],default="color")
    bench.add_argument("--model",default="yolov8n.pt")
    bench.add_argument("--refresh",type=int,default=15)
    bench.add_argument("--max-frames",type=int)
    render_cmd=sub.add_parser("render",help="Render an annotated diagnostic MP4")
    render_cmd.add_argument("--video",default="examples/generated/warehouse_synthetic.mp4")
    render_cmd.add_argument("--output",default="results/event_annotated.mp4")
    render_cmd.add_argument("--mode",choices=["continuous","roi","event"],default="event")
    render_cmd.add_argument("--detector",choices=["color","yolo"],default="color")
    render_cmd.add_argument("--model",default="yolov8n.pt")
    server=sub.add_parser("serve",help="Serve edge inference over framed TCP")
    server.add_argument("--host",default="127.0.0.1")
    server.add_argument("--port",type=int,default=8765)
    server.add_argument("--detector",choices=["color","yolo"],default="color")
    server.add_argument("--model",default="yolov8n.pt")
    client=sub.add_parser("stream",help="Offload video frames to a running TCP edge server")
    client.add_argument("--video",default="examples/generated/warehouse_synthetic.mp4")
    client.add_argument("--host",default="127.0.0.1")
    client.add_argument("--port",type=int,default=8765)
    client.add_argument("--mode",choices=["continuous","roi","event"],default="event")
    client.add_argument("--refresh",type=int,default=15)
    client.add_argument("--max-frames",type=int)
    ros=sub.add_parser("ros2-publish",help="Optional ROS 2 topic publisher (requires sourced ROS 2)")
    ros.add_argument("--video",default="examples/generated/warehouse_synthetic.mp4")
    ros.add_argument("--mode",choices=["continuous","roi","event"],default="event")
    ros.add_argument("--detector",choices=["color","yolo"],default="color")
    ros.add_argument("--model",default="yolov8n.pt")
    ros.add_argument("--topic",default="/perception/task_status")
    args=parser.parse_args(argv)
    if args.command=="generate":
        video,ground_truth=generate(args.output)
        print(f"Generated {video} and {ground_truth}")
    elif args.command=="benchmark":
        gt=args.gt if Path(args.gt).exists() else None
        report=benchmark(args.video,args.output,gt,args.detector,args.model,args.refresh,args.max_frames)
        for row in report["results"]:
            print(f'{row["strategy"]:10s} | offloads={row["frames_offloaded"]:3d}/{row["frames"]} | '
                  f'traffic={row["sent_megabytes"]:.3f} MB | p95={row["latency_p95_ms"]:.2f} ms | '
                  f'reduction={row["traffic_reduction_vs_continuous_pct"]:.1f}%'
                  + (f' | decision recall={row["decision_recall"]:.3f}' if gt else ''))
        print(f'Wrote {args.output}/benchmark_summary.json')
    elif args.command=="render":
        render(args.video,args.output,args.mode,args.detector,args.model)
        print(f"Annotated video: {args.output}")
    elif args.command=="serve":
        serve(args.host,args.port,make_detector(args.detector,args.model))
    elif args.command=="ros2-publish":
        from .ros2_bridge import publish_video
        publish_video(args.video,args.mode,args.detector,args.model,args.topic)
    elif args.command=="stream":
        frames,fps=read_video(args.video,args.max_frames)
        with TCPEdgeClient(args.host,args.port) as edge:
            pipeline=PerceptionPipeline(SensingPolicy(args.mode,refresh_frames=args.refresh),edge)
            total=0
            for index,frame in enumerate(frames):
                result=pipeline.process(frame,index,index/fps)
                total+=result.bytes_sent
                if index%20==0 or result.decision=="STOP" and index%10==0:
                    print(f'frame={index:03d} trigger={result.trigger_reason:20s} '
                          f'decision={result.decision:4s} bytes={result.bytes_sent:5d} '
                          f'round_trip_ms={result.latency_ms:.2f}')
            print(json.dumps({"frames":len(frames),"bytes_sent":total,"mode":args.mode}))

if __name__ == "__main__":
    main()
