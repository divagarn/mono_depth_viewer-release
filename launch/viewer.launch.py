"""Launch the monocular depth node.

Live (headless, default):  ros2 launch mono_depth_viewer viewer.launch.py auto_level:=true hfov:=100
Debug GUI with a bag:      ros2 launch mono_depth_viewer viewer.launch.py debug:=true bag:=/path/to/bag pointcloud:=true play:=true

`python:=/path/to/venv/bin/python` runs the node with that interpreter (for torch/transformers installed in a
virtualenv); it can also be set once with the MONO_DEPTH_PYTHON environment variable.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# launch argument -> viewer flag; an empty value means "use the viewer's own default"
VALUE_ARGS = {
    "image_topic": "--image-topic", "model": "--model", "input_size": "--input-size",
    "pc_topic": "--pc-topic", "pc_frame": "--pc-frame", "pc_axes": "--pc-axes", "pc_stamp": "--pc-stamp",
    "pc_stride": "--pc-stride", "hfov": "--hfov", "near": "--near", "far": "--far",
    "pitch": "--pitch", "cam_height": "--cam-height",
}
FLAG_ARGS = {"debug": "--debug", "play": "--play", "pointcloud": "--pointcloud", "auto_level": "--auto-level"}
DEFAULTS = {
    "python": os.environ.get("MONO_DEPTH_PYTHON", ""), "debug": "false", "bag": "", "play": "false", "pointcloud": "false",
    "image_topic": "/camera/image_raw", "model": "small", "input_size": "364",
    "pc_topic": "/mono_depth/points", "pc_frame": "", "pc_axes": "ros", "pc_stamp": "", "pc_stride": "4",
    "hfov": "90.0", "near": "0.5", "far": "20.0", "pitch": "0.0", "cam_height": "0.0", "auto_level": "false",
}


def _launch_viewer(context):
    cfg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    args = [cfg("bag")] if cfg("bag") else []
    for name, flag in VALUE_ARGS.items():
        if cfg(name) != "":
            args += [flag, cfg(name)]
    args += [flag for name, flag in FLAG_ARGS.items() if cfg(name).lower() in ("true", "1", "yes")]
    if cfg("python"):
        return [ExecuteProcess(cmd=[cfg("python"), "-m", "mono_depth_viewer.main"] + args,
                               output="screen", emulate_tty=True, name="mono_depth_viewer")]
    return [Node(package="mono_depth_viewer", executable="viewer", output="screen", emulate_tty=True, arguments=args)]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in DEFAULTS.items()]
                             + [OpaqueFunction(function=_launch_viewer)])
