#!/usr/bin/env python3
"""
Sock detector lifecycle node for the JeTank robot.

This node wraps the YOLO-based sock detector as a ROS 2 lifecycle node with a
DetectSocks action server.  The model engine is loaded once at on_configure
and stays GPU-resident; the heavy inference only runs while the node is active.

Lifecycle transitions::

    ros2 lifecycle set /sock_detector configure
    ros2 lifecycle set /sock_detector activate
    # call the action or observe /detections/socks
    ros2 lifecycle set /sock_detector deactivate
    ros2 lifecycle set /sock_detector cleanup

DetectSocks action (on-demand mode)::

    ros2 action send_goal /detect_socks jetank_detection/action/DetectSocks \
        '{timeout: 5.0, min_confidence: 0.5, n_frames: 10}'

Continuous mode (live publisher)::

    ros2 launch jetank_detection detect.launch.py continuous:=true
"""

import threading
import time

import rclpy
from cv_bridge import CvBridge
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from jetank_detection.backends import make_backend


class SockDetectorNode(LifecycleNode):
    """
    Lifecycle node that exposes a DetectSocks action server.

    Parameters (declared in on_configure)
    --------------------------------------
    sim                 : bool  if true, load the sim model; else the real one
    model_path          : str   explicit model path; overrides sim/real selection
    model_path_sim      : str   model to load when sim=true (Gazebo imagery)
    model_path_real     : str   model to load when sim=false (real camera)
    input_image_topic   : str   left camera topic
    confidence          : float detection confidence threshold (default 0.5)
    n_frames            : int   frames to process in on-demand action (default 10)
    continuous          : bool  if true, subscribe and publish every frame when active
    debug               : bool  publish annotated debug image (default true)
    detections_topic    : str   topic for Detection2DArray output
    debug_image_topic   : str   topic for debug annotated image
    """

    def __init__(self) -> None:
        """Initialise the lifecycle node."""
        super().__init__("sock_detector")
        self._bridge = CvBridge()
        self._backend = None
        self._action_server = None
        self._continuous_sub = None
        self._det_pub = None
        self._debug_pub = None
        self._latest_image = None
        self._image_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def on_configure(self, state):
        """Configure the node: create backend, publishers and action server."""
        self.get_logger().info("Configuring SockDetectorNode...")

        # Declare parameters
        self.declare_parameter("sim", False)
        self.declare_parameter("model_path", "")
        self.declare_parameter("model_path_sim", "")
        self.declare_parameter("model_path_real", "")
        self.declare_parameter("input_image_topic", "/stereo_camera/left/image_raw")
        self.declare_parameter("confidence", 0.5)
        self.declare_parameter("n_frames", 10)
        self.declare_parameter("continuous", False)
        self.declare_parameter("debug", True)
        self.declare_parameter("detections_topic", "/detections/socks")
        self.declare_parameter("debug_image_topic", "/detections/socks/debug")

        # Resolve which model to load. Sim and real need *different* models:
        # the synthetic Gazebo imagery (perfect rectification, synthetic
        # textures/lighting) differs enough from real camera frames that one
        # model does not serve both well (see plan §2a). An explicit
        # `model_path` always wins (override / back-compat); otherwise the
        # `sim` flag selects the sim or real model.
        sim = self.get_parameter("sim").get_parameter_value().bool_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        model_path_sim = (
            self.get_parameter("model_path_sim").get_parameter_value().string_value
        )
        model_path_real = (
            self.get_parameter("model_path_real").get_parameter_value().string_value
        )
        if model_path:
            resolved_model, model_source = model_path, "model_path (explicit override)"
        elif sim:
            resolved_model, model_source = model_path_sim, "model_path_sim (sim)"
        else:
            resolved_model, model_source = model_path_real, "model_path_real (real)"
        self.get_logger().info(
            f"Environment: {'SIM' if sim else 'REAL'} — model from {model_source}"
        )

        detections_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value
        )
        debug_image_topic = (
            self.get_parameter("debug_image_topic").get_parameter_value().string_value
        )
        debug = self.get_parameter("debug").get_parameter_value().bool_value

        # Create backend
        self._backend = make_backend("ultralytics")

        if resolved_model:
            try:
                self._backend.load(resolved_model)
                self.get_logger().info(f"Model loaded from {resolved_model}")
            except RuntimeError as exc:
                self.get_logger().error(f"Failed to load model: {exc}")
                # Don't crash — node starts without inference capability
        else:
            self.get_logger().warn(
                f"no model resolved for {'SIM' if sim else 'REAL'} environment "
                "(set model_path, or model_path_sim/model_path_real) — node will "
                "start but cannot infer until configured with a model"
            )

        # Create lifecycle publishers
        self._det_pub = self.create_lifecycle_publisher(
            Detection2DArray, detections_topic, 10
        )
        if debug:
            self._debug_pub = self.create_lifecycle_publisher(
                Image, debug_image_topic, 10
            )

        # Create action server with ReentrantCallbackGroup so its callback
        # can run concurrently with subscriptions on the MultiThreadedExecutor.
        from jetank_detection.action import DetectSocks  # noqa: PLC0415

        _action_cbg = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            DetectSocks,
            "detect_socks",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=_action_cbg,
        )

        self.get_logger().info("SockDetectorNode configured.")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Activate publishers and optionally start the continuous subscriber."""
        self.get_logger().info("Activating SockDetectorNode...")

        # Activate all managed lifecycle publishers (rclpy handles them via the
        # base LifecycleNode; do NOT call publisher.on_activate manually — there
        # is no Node.get_current_state() in rclpy).
        super().on_activate(state)

        continuous = self.get_parameter("continuous").get_parameter_value().bool_value
        if continuous:
            topic = (
                self.get_parameter("input_image_topic").get_parameter_value().string_value
            )
            self._continuous_sub = self.create_subscription(
                Image, topic, self._continuous_image_callback, 1
            )
            self.get_logger().info(f"Continuous mode enabled: subscribing to {topic}")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Deactivate publishers and destroy the continuous subscriber."""
        self.get_logger().info("Deactivating SockDetectorNode...")

        if self._continuous_sub is not None:
            self.destroy_subscription(self._continuous_sub)
            self._continuous_sub = None

        # Deactivate managed lifecycle publishers via the base node.
        super().on_deactivate(state)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """Clean up the backend and action server."""
        self.get_logger().info("Cleaning up SockDetectorNode...")
        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None
        # Guard against cleanup reached without a preceding deactivate
        if self._continuous_sub is not None:
            self.destroy_subscription(self._continuous_sub)
            self._continuous_sub = None
        self._backend = None
        self._latest_image = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        """Shut down the node and destroy the action server."""
        self.get_logger().info("Shutting down SockDetectorNode...")
        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None
        if self._continuous_sub is not None:
            self.destroy_subscription(self._continuous_sub)
            self._continuous_sub = None
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Continuous mode
    # ------------------------------------------------------------------

    def _store_latest(self, msg: Image) -> None:
        """Store the most recent image message under the image lock."""
        with self._image_lock:
            self._latest_image = msg

    def _continuous_image_callback(self, msg: Image) -> None:
        """Process and publish detections for every incoming frame."""
        self._store_latest(msg)

        if self._backend is None:
            return

        try:
            image_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        conf = self.get_parameter("confidence").get_parameter_value().double_value
        try:
            detections = self._backend.infer(image_bgr, conf_threshold=conf)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Inference skipped: {exc}")
            return

        det_array = self._build_detection_array(detections, msg.header)
        if self._det_pub is not None and self._det_pub.is_activated:
            self._det_pub.publish(det_array)

        if self._debug_pub is not None and self._debug_pub.is_activated:
            try:
                debug_img = self._draw_detections(image_bgr, detections)
                debug_msg = self._bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
                debug_msg.header = msg.header
                self._debug_pub.publish(debug_msg)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Debug image publish failed: {exc}")

    # ------------------------------------------------------------------
    # Action server
    # ------------------------------------------------------------------

    def _goal_callback(self, goal_request):
        """Accept all incoming DetectSocks goals."""
        self.get_logger().info("DetectSocks goal received.")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """Accept cancel requests for DetectSocks goals."""
        self.get_logger().info("DetectSocks goal cancel requested.")
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        """Collect up to n_frames and return the best detection."""
        from jetank_detection.action import DetectSocks  # noqa: PLC0415

        goal = goal_handle.request
        n_frames: int = goal.n_frames if goal.n_frames > 0 else 10
        timeout: float = goal.timeout if goal.timeout > 0 else 5.0
        min_conf: float = goal.min_confidence if goal.min_confidence > 0 else 0.5

        topic = self.get_parameter("input_image_topic").get_parameter_value().string_value

        self.get_logger().info(
            f"DetectSocks: collecting {n_frames} frames on {topic} "
            f"(timeout={timeout}s, min_conf={min_conf})"
        )

        continuous = self.get_parameter("continuous").get_parameter_value().bool_value

        # In continuous mode the persistent subscription already feeds
        # _latest_image via _store_latest; in on-demand mode we create a
        # temporary subscription in a ReentrantCallbackGroup so that it fires
        # concurrently with this action callback on the MultiThreadedExecutor.
        tmp_sub = None
        if not continuous:
            cbg = ReentrantCallbackGroup()
            tmp_sub = self.create_subscription(
                Image, topic, self._store_latest, 1, callback_group=cbg
            )

        deadline = time.monotonic() + timeout
        frames_processed = 0
        feedback_msg = DetectSocks.Feedback()
        best_detections = []
        best_score = 0.0
        best_header = None
        last_stamp = None  # deduplicate consecutive identical frames

        try:
            while frames_processed < n_frames and time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return DetectSocks.Result()

                with self._image_lock:
                    current_msg = self._latest_image

                if current_msg is None:
                    time.sleep(0.01)
                    continue

                # Deduplicate: skip if we already processed this stamp
                stamp = (current_msg.header.stamp.sec, current_msg.header.stamp.nanosec)
                if stamp == last_stamp:
                    time.sleep(0.01)
                    continue

                last_stamp = stamp

                if self._backend is None:
                    time.sleep(0.01)
                    continue

                try:
                    image_bgr = self._bridge.imgmsg_to_cv2(
                        current_msg, desired_encoding="bgr8"
                    )
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"Frame conversion failed: {exc}")
                    time.sleep(0.01)
                    continue

                try:
                    dets = self._backend.infer(image_bgr, conf_threshold=min_conf)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"Inference skipped: {exc}")
                    time.sleep(0.01)
                    continue

                frames_processed += 1
                feedback_msg.frames_processed = frames_processed
                goal_handle.publish_feedback(feedback_msg)

                if dets:
                    frame_best = max(dets, key=lambda d: d.score)
                    if frame_best.score > best_score:
                        best_score = frame_best.score
                        best_detections = dets
                        best_header = current_msg.header

        finally:
            if tmp_sub is not None:
                self.destroy_subscription(tmp_sub)

        # Build result
        result = DetectSocks.Result()
        if best_detections and best_header is not None:
            det_array = self._build_detection_array(best_detections, best_header)
            result.best = det_array
            result.confidence = best_score
            result.found = True
        else:
            result.best = Detection2DArray()
            result.confidence = 0.0
            result.found = False

        self.get_logger().info(
            f"DetectSocks done: found={result.found}, confidence={result.confidence:.3f}, "
            f"frames_processed={frames_processed}"
        )
        goal_handle.succeed()
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_detection_array(self, detections, header) -> Detection2DArray:
        """Convert a list of :class:`Detection` to a Detection2DArray message."""
        array = Detection2DArray()
        array.header = header

        for det in detections:
            d2d = Detection2D()
            d2d.header = header
            d2d.bbox.center.position.x = det.cx
            d2d.bbox.center.position.y = det.cy
            d2d.bbox.size_x = det.w
            d2d.bbox.size_y = det.h

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det.label
            hyp.hypothesis.score = det.score
            d2d.results.append(hyp)

            array.detections.append(d2d)

        return array

    def _draw_detections(self, image_bgr, detections):
        """Draw bounding boxes on *image_bgr* and return the annotated image."""
        import cv2  # noqa: PLC0415

        img = image_bgr.copy()
        for det in detections:
            x1 = int(det.cx - det.w / 2)
            y1 = int(det.cy - det.h / 2)
            x2 = int(det.cx + det.w / 2)
            y2 = int(det.cy + det.h / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.label} {det.score:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        return img


def main(args=None):
    """Run the sock_detector_node as a lifecycle node."""
    rclpy.init(args=args)
    node = SockDetectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
