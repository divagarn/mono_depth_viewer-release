from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("bag", default_value="", description="rosbag2 folder or .db3 file"),
        Node(package="mono_depth_viewer", executable="viewer", output="screen",
             arguments=[LaunchConfiguration("bag")]),
    ])
