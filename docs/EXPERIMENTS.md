# Experimental protocol

## Hypotheses

- H1: event-triggering reduces edge inference calls and transmission volume versus a continuous full-frame baseline, on a video with idle periods.
- H2: task-zone cropping reduces transmission volume even without event triggering.
- H3: aggressive triggering can lower fresh object-detection recall or delay reacting to hazards; decision-level and detection-level metrics must be examined separately.

## Comparisons

| Name | Selection | Detection | Transmission |
|---|---|---|---|
| continuous | all frames, full frame | every frame | JPEG full frame |
| roi | all frames, fixed aisle ROI | every frame | JPEG fixed ROI |
| event | motion ROI + task-zone refresh every N frames | only on triggers | JPEG of trigger-specific ROI |

For the controlled offline benchmark all three receive **identical decoded frames** and use the **same detector** and JPEG quality. The local edge simulates image encoding and edge-side JPEG decoding/inference in one process; it does **not** measure real wireless RTT or energy use. Local latency includes CPU event detection (if any), JPEG encoding/decoding, and detector wall time; Python cold start / model initialization are not included. Input video decode is outside the timing window for all modes. Synthetic red/blue labels are not representative of general object detection.

### Metrics

- Sent bytes: length of JPEG and framing header in one direction (client to server); **not** full bidirectional traffic, actual radio bandwidth, TCP/IP headers or retries.
- Offload fraction: calls to edge detector divided by all decoded frames.
- p50/p95 frame-processing latency: elapsed time from start of policy step to completed decision; skipped frames have only trigger processing, so p95 of all frames is not equivalent to hazard-response delay.
- Decision precision/recall: frame-by-frame STOP versus obstacle-in-ground-truth. Stale hazard state persists briefly between detections; full-zone negative refresh clears it.
- Fresh detection IoU50 precision/recall: obstacle boxes received in the current frame matched to ground truth at IoU >= 0.5. Skipped frames count as missed fresh detections, even when STOP persists.

## Avoid false claims

1. This is a frame-difference trigger on conventional RGB frames, **not** a DVS event camera.
2. This prototype's motion detection still reads and processes every frame. It only **skips costly inference and network offload** for many frames.
3. YOLO is optional and generally recognizes COCO classes, not necessarily custom industrial hazards. A meaningful real-world experiment must define hazard classes and annotation protocol.
4. A webcam video is not equivalent to deployment on a robot. Raspberry Pi / Jetson / wireless 6G claims require separate device experiments.
5. Synthetic label quality does not establish real-world reliability or safety. Avoid deploying this STOP/GO example on real hardware without an independent safety stack.
6. One run is not a statistical performance claim. For research, benchmark multiple runs, seeds, event rates and hardware/network profiles; report confidence intervals and failures.

## Suggested next experiments

1. MOT or warehouse camera data with person, pallet and static obstacle annotations; compare COCO mAP or class-specific PR across fixed ROI and event modes.
2. Multiple trigger thresholds / refresh intervals and Pareto curve of bandwidth versus missed hazard events.
3. Pi / Jetson CPU-GPU profiling (per-stage CPU, GPU utilization, energy and memory).
4. Real two-device Wi-Fi link for RTT, packet loss and interference stress test.
5. ROS 2 control integration with a physical robot only after appropriate safety review.
