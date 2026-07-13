#!/usr/bin/env python3
r"""
Sim sock-detector entry point.

Thin wrapper over ``detect.launch.py`` that pins ``sim:=true`` so the node
loads the **sim** model (``model_path_sim``), trained on Gazebo imagery. Use
this when running against the simulator (``sim_demo.launch.py``). Defaults to
continuous (live publisher) mode, which is what the sim demo expects.

Usage::

    ros2 launch jetank_detection detect_sim.launch.py model_path_sim:=/path/to/sock_sim.pt

    ros2 lifecycle set /sock_detector configure
    ros2 lifecycle set /sock_detector activate
    ros2 topic echo /detections/socks

The remaining knobs (``confidence``, ``debug``, ``input_image_topic``, ...)
are declared by the included ``detect.launch.py`` and pass straight through —
in Humble, launch configurations are not scoped by IncludeLaunchDescription,
so e.g. ``confidence:=0.6`` on this wrapper reaches the node without
re-declaration here (they just don't show in ``--show-args``).

For the real robot use ``detect_real.launch.py`` instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate the sim sock-detector launch description."""
    declare_model_path_sim = DeclareLaunchArgument(
        "model_path_sim",
        default_value="",
        description="Sim model path (.pt/.engine), trained on Gazebo imagery",
    )
    declare_continuous = DeclareLaunchArgument(
        "continuous",
        default_value="true",
        description="Live publisher mode (default true for the sim demo)",
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
            "sim": "true",
            "model_path_sim": LaunchConfiguration("model_path_sim"),
            "continuous": LaunchConfiguration("continuous"),
        }.items(),
    )

    return LaunchDescription(
        [
            declare_model_path_sim,
            declare_continuous,
            detect,
        ]
    )
