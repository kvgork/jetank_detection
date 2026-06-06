#!/usr/bin/env python3
"""
Headless image-capture node for building sock-detection training datasets.

This is the sim-track counterpart of the web UI ``/capture`` button (which
serves the real-robot track). It subscribes to a camera image topic and writes
frames to disk at a fixed interval, so a Gazebo ``sock_arena`` session can
produce a dataset without a human clicking a button.

Captured frames land in ``<output_dir>`` as ``sock_<domain>_NNNNNN.jpg`` and are
ready to annotate (single class ``sock``) and train per the sim track of
``plans/sock-detection-plan.md`` (§4) and ``plans/sock-sim-autotrain-plan.md``.

Parameters (ROS):
  input_image_topic (str)  image topic to capture (default sim left camera)
  output_dir (str)         directory to write JPEGs into (created if missing)
  domain (str)             filename tag, ``sim`` or ``real`` (default ``sim``)
  interval_sec (float)     seconds between saved frames (default 1.0)
  max_frames (int)         stop after this many frames; 0 = unlimited (default 0)
  jpeg_quality (int)       JPEG quality 1-100 (default 95)

Example::

    # With sim_demo running (world:=sock_arena):
    ros2 run jetank_detection capture_frames \
        --ros-args -p output_dir:=$HOME/datasets/detection/sim \
                   -p interval_sec:=0.5 -p max_frames:=600
"""

import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class FrameCaptureNode(Node):
    """Subscribe to an image topic and persist frames at a fixed interval."""

    def __init__(self):
        super().__init__('frame_capture')

        self.declare_parameter(
            'input_image_topic', '/stereo_camera/left/image_raw')
        self.declare_parameter(
            'output_dir', os.path.expanduser('~/datasets/detection/sim'))
        self.declare_parameter('domain', 'sim')
        self.declare_parameter('interval_sec', 1.0)
        self.declare_parameter('max_frames', 0)
        self.declare_parameter('jpeg_quality', 95)

        self._topic = self.get_parameter(
            'input_image_topic').get_parameter_value().string_value
        self._output_dir = os.path.expanduser(
            self.get_parameter('output_dir').get_parameter_value().string_value)
        self._domain = self.get_parameter(
            'domain').get_parameter_value().string_value
        self._interval = self.get_parameter(
            'interval_sec').get_parameter_value().double_value
        self._max_frames = self.get_parameter(
            'max_frames').get_parameter_value().integer_value
        self._jpeg_quality = self.get_parameter(
            'jpeg_quality').get_parameter_value().integer_value

        os.makedirs(self._output_dir, exist_ok=True)

        self._bridge = CvBridge()
        self._saved = 0
        self._last_save = 0.0
        self._start_index = self._next_start_index()

        self._sub = self.create_subscription(
            Image, self._topic, self._on_image, 10)

        self.get_logger().info(
            f"Capturing '{self._topic}' -> {self._output_dir} "
            f"(domain={self._domain}, interval={self._interval}s, "
            f"max_frames={self._max_frames or 'unlimited'})")

    def _next_start_index(self):
        """Resume numbering after any existing frames for this domain."""
        prefix = f'sock_{self._domain}_'
        existing = [
            f for f in os.listdir(self._output_dir)
            if f.startswith(prefix) and f.endswith('.jpg')
        ]
        if not existing:
            return 0
        indices = []
        for name in existing:
            stem = name[len(prefix):-len('.jpg')]
            if stem.isdigit():
                indices.append(int(stem))
        return (max(indices) + 1) if indices else 0

    def _on_image(self, msg):
        """Save a frame if the capture interval has elapsed."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_save and (now - self._last_save) < self._interval:
            return
        self._last_save = now

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001 - log and skip a bad frame
            self.get_logger().warn(f'cv_bridge conversion failed: {exc}')
            return

        index = self._start_index + self._saved
        path = os.path.join(
            self._output_dir, f'sock_{self._domain}_{index:06d}.jpg')
        cv2.imwrite(
            path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        self._saved += 1
        self.get_logger().info(f'saved {path} ({self._saved} this run)')

        if self._max_frames and self._saved >= self._max_frames:
            self.get_logger().info(
                f'reached max_frames={self._max_frames}; shutting down')
            raise SystemExit


def main(args=None):
    """Run the frame-capture node until max_frames or Ctrl-C."""
    rclpy.init(args=args)
    node = None
    try:
        node = FrameCaptureNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
