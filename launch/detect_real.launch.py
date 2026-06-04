#!/usr/bin/env python3
r"""Real-robot sock-detector entry point.

Thin wrapper over ``detect.launch.py`` that pins ``sim:=false`` so the node
loads the **real** model (``model_path_real``), trained on real camera frames.
Use this on the physical JeTank. Defaults to on-demand mode (continuous=false),
matching the discrete drive -> detect -> grasp pick task.

Usage::

    ros2 launch jetank_detection detect_real.launch.py model_path_real:=/path/to/sock_real.pt

    ros2 lifecycle set /sock_detector configure
    ros2 lifecycle set /sock_detector activate
    ros2 action send_goal /detect_socks jetank_detection/action/DetectSocks \
        '{timeout: 5.0, min_confidence: 0.5, n_frames: 10}'

For the simulator use ``detect_sim.launch.py`` instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate the real-robot sock-detector launch description."""
    declare_model_path_real = DeclareLaunchArgument(
        "model_path_real",
        default_value="",
        description="Real model path (.pt/.engine), trained on real camera frames",
    )
    declare_continuous = DeclareLaunchArgument(
        "continuous",
        default_value="false",
        description="On-demand mode (default false for the discrete pick task)",
    )
    declare_confidence = DeclareLaunchArgument(
        "confidence", default_value="0.5", description="Detection confidence threshold"
    )
    declare_debug = DeclareLaunchArgument(
        "debug", default_value="true", description="Publish annotated debug image"
    )
    declare_input_topic = DeclareLaunchArgument(
        "input_image_topic",
        default_value="/stereo_camera/left/image_raw",
        description="Input image topic (real left camera)",
    )
    declare_n_frames = DeclareLaunchArgument(
        "n_frames",
        default_value="10",
        description="Frames to process per DetectSocks action goal",
    )

    detect = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("jetank_detection"), "launch", "detect.launch.py"]
            )
        ),
        launch_arguments={
            "sim": "false",
            "model_path_real": LaunchConfiguration("model_path_real"),
            "continuous": LaunchConfiguration("continuous"),
            "confidence": LaunchConfiguration("confidence"),
            "debug": LaunchConfiguration("debug"),
            "input_image_topic": LaunchConfiguration("input_image_topic"),
            "n_frames": LaunchConfiguration("n_frames"),
        }.items(),
    )

    return LaunchDescription(
        [
            declare_model_path_real,
            declare_continuous,
            declare_confidence,
            declare_debug,
            declare_input_topic,
            declare_n_frames,
            detect,
        ]
    )
