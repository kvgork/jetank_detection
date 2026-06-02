#!/usr/bin/env python3
r"""
Launch file for the sock detector node.

Usage::

    # Basic (must configure/activate lifecycle manually)
    ros2 launch jetank_detection detect.launch.py model_path:=/path/to/sock.pt

    # Continuous live detection (no action needed)
    ros2 launch jetank_detection detect.launch.py model_path:=/path/to/sock.pt continuous:=true

    # With custom confidence threshold
    ros2 launch jetank_detection detect.launch.py model_path:=/path/to/sock.pt confidence:=0.6

Lifecycle management after launch::

    ros2 lifecycle set /sock_detector configure
    ros2 lifecycle set /sock_detector activate

    # Then send a goal (on-demand mode):
    ros2 action send_goal /detect_socks jetank_detection/action/DetectSocks \
        '{timeout: 5.0, min_confidence: 0.5, n_frames: 10}'
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the sock detector launch description."""
    declare_model_path = DeclareLaunchArgument(
        "model_path",
        default_value="",
        description="Path to the YOLO model file (.pt or .engine)",
    )
    declare_continuous = DeclareLaunchArgument(
        "continuous",
        default_value="false",
        description="If true, run inference on every incoming frame (live publisher mode)",
    )
    declare_confidence = DeclareLaunchArgument(
        "confidence",
        default_value="0.5",
        description="Detection confidence threshold",
    )
    declare_debug = DeclareLaunchArgument(
        "debug",
        default_value="true",
        description="Publish annotated debug image on /detections/socks/debug",
    )
    declare_input_topic = DeclareLaunchArgument(
        "input_image_topic",
        default_value="/stereo_camera/left/image_raw",
        description="Input image topic (left camera)",
    )
    declare_n_frames = DeclareLaunchArgument(
        "n_frames",
        default_value="10",
        description="Number of frames to process per DetectSocks action goal",
    )

    sock_detector_node = Node(
        package="jetank_detection",
        executable="sock_detector_node",
        name="sock_detector",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "continuous": LaunchConfiguration("continuous"),
                "confidence": LaunchConfiguration("confidence"),
                "debug": LaunchConfiguration("debug"),
                "input_image_topic": LaunchConfiguration("input_image_topic"),
                "n_frames": LaunchConfiguration("n_frames"),
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            declare_model_path,
            declare_continuous,
            declare_confidence,
            declare_debug,
            declare_input_topic,
            declare_n_frames,
            sock_detector_node,
        ]
    )
