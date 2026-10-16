# Optional ROS 2 observer bridge (not an autonomous robot controller)

From a Python environment with a working ROS 2 installation, sourced ROS 2 setup and
`std_msgs` / `rclpy` available:

```bash
tc-perception generate
tc-perception ros2-publish --mode event
```

In another ROS 2 terminal, while the publisher is running:

```bash
ros2 topic echo /perception/task_status
```

Message schema: `std_msgs/String` containing JSON with `frame_id`, synthetic-video
`timestamp_s`, `decision` (`STOP`/`GO`), `trigger` and a list of `{label,
confidence, box}`. This uses video playback as fast as inference permits, not
wall-clock synchronized camera frames. This bridge **does not command motors or
implement any physical safety response**. This optional integration cannot be
unit tested unless ROS 2 is installed; core CI intentionally runs without it.

Research extension: replace JSON/String with stamped structured ROS 2 interfaces,
subscribe to `/camera/image_raw` using `cv_bridge`, and measure image transport
and QoS behavior on actual robotic hardware.
