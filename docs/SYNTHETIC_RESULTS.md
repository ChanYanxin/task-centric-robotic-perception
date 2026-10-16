# One-run synthetic benchmark snapshot

**Scope:** deterministic toy warehouse video (180 frames at 15 FPS) with blue package and red moving→stationary→disappearing obstacle. OpenCV color segmentation; JPEG quality 80; local edge process simulation. This is **not** performance on a real robotic platform, YOLO model, or wireless network. Timing and byte counts should be rerun in the target environment.

| Strategy | Offloaded frames | Sent MB | Traffic reduction vs continuous | STOP recall | Fresh detection recall (IoU50) | Excess STOP frames |
|---|---:|---:|---:|---:|---:|---:|
| continuous | 180/180 | 2.7525 | 0.00% | 1.000 | 1.000 | 0 |
| roi | 180/180 | 1.9118 | 30.54% | 1.000 | 1.000 | 0 |
| event | 73/180 | 0.2185 | 92.06% | 1.000 | 0.370 | 12 |

The event-triggered policy retains STOP decisions between observations and
refreshes the task zone every 15 frames. Consequently high frame-level STOP
recall can coexist with low **fresh** detection recall and false positive STOP
frames after an obstacle disappears. These are critical limitations, not
proof of safety.

**Protocol caution:** per-frame latency includes JPEG/edge simulation but not
hardware camera acquisition, Wi-Fi RTT, full TCP/IP overhead, or energy use.
The synthetic color-detector results cannot be presented as state-of-the-art
computer vision performance. For experimental protocol and extensions, see
[EXPERIMENTS.md](EXPERIMENTS.md).
