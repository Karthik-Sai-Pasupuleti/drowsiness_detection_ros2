from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    driver_id_arg = DeclareLaunchArgument(
        "driver_id",
        default_value="test_driver",
        description="Driver identifier for saving session data."
    )

    driver_id = LaunchConfiguration("driver_id")
    
    return LaunchDescription(
        [
            Node(
                package="drowsiness_detection_pkg",
                executable="mediapipe_node",
                name="mediapipe_node",
                output="screen",
                parameters=[{"driver_id": driver_id}],  
            ),
            Node(
                package="drowsiness_detection_pkg",
                executable="carla_node",
                name="carla_node",
                output="screen",
            ),
            Node(
                package="drowsiness_detection_pkg",
                executable="driver_assistance_node",
                name="driver_assistance_node",
                output="screen",
                parameters=[{"driver_id": driver_id}],  
            ),
            Node(
            package='drowsiness_detection_pkg',
            executable='drowsiness_gui',
            name='drowsiness_gui',
            output='screen'
        ),

        ]
    )