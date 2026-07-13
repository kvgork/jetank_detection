#!/usr/bin/env python3
r"""
Real-robot sock-detector entry point.

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

The remaining knobs (``confidence``, ``debug``, ``input_image_topic``,
``n_frames``, ...) are declared by the included ``detect.launch.py`` and pass
straight through — in Humble, launch configurations are not scoped by
IncludeLaunchDescription, so e.g. ``confidence:=0.6`` on this wrapper reaches
the node without re-declaration here (they just don't show in ``--show-args``).

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

    # Only the args this wrapper pins (sim) or re-defaults (continuous) are
    # declared/forwarded; everything else passes through to detect.launch.py
    # unscoped (see module docstring).
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
        }.items(),
    )

    return LaunchDescription(
        [
            declare_model_path_real,
            declare_continuous,
            detect,
        ]
    )
