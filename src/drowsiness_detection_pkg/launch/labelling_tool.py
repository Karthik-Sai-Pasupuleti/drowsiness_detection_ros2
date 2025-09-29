from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='drowsiness_detection_pkg',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),
        Node(
            package='drowsiness_detection_pkg',
            executable='mediapipe_node',
            name='mediapipe_node',
            output='screen'
        ),
        Node(
            package='drowsiness_detection_pkg',
            executable='carla_node',
            name='carla_node',
            output='screen'
        ),
        Node(
            package='drowsiness_detection_pkg',
            executable='driver_assistance_node',
            name='driver_assistance_node',
            output='screen'
        ),
        Node(
            package='drowsiness_detection_pkg',
            executable='drowsiness_gui',
            name='drowsiness_gui',
            output='screen'
        ),

    ])
