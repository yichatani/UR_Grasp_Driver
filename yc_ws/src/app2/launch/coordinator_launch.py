# launch/coordinator_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Path to the model checkpoint
    checkpoint_path = '/home/artc/abdo_ws/src/my_pkg/scripts/anygrasp_bridging/anygrasp_sdk/checkpoint_detection.tar'  # Replace with actual path

    return LaunchDescription([
        # Coordinator Node
        Node(
            package='app2',
            executable='coordinator2',
            name='coordinator2',
            output='screen',
            parameters=[
                {'grasp_detector_checkpoint': checkpoint_path}
            ]
        ),
    ])

