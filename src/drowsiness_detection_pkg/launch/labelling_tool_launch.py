from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    driver_id_arg = DeclareLaunchArgument(
        "driver_id",
        default_value="test_driver",
        description="Driver identifier for saving session data."
    )

    driver_id = LaunchConfiguration("driver_id")

    return LaunchDescription([
        driver_id_arg,

        # 1. Mediapipe Node (Camera processing)
        Node(
            package="drowsiness_detection_pkg",
            executable="mediapipe_node",
            name="mediapipe_node",
            output="screen",
            parameters=[{"driver_id": driver_id}],
        ),

        # 2. Carla Node (Simulator control)
        Node(
            package="drowsiness_detection_pkg",
            executable="carla_node",
            name="carla_node",
            output="screen",
        ),

        # 3. Driver Assistance Node (Logic core)
        Node(
            package="drowsiness_detection_pkg",
            executable="driver_assistance_node",
            name="driver_assistance_node",
            output="screen",
            parameters=[{"driver_id": driver_id}],
        ),
        
        # 4. NEW: Heart Rate Node (SiFi Sensor)
        Node(
            package="drowsiness_detection_pkg",
            executable="heartratenode",
            name="heart_rate_node",
            output="screen",
        ),

        # 5. Drowsiness GUI (Flask Web App)
        Node(
            package='drowsiness_detection_pkg',
            executable='drowsiness_gui',
            name='drowsiness_gui',
            output='screen'
        ),

        # --- OPTIONAL HARDWARE NODES (Uncomment if connected) ---
        # Node(
        #     package="drowsiness_detection_pkg",
        #     executable="steering_wheel_ffb_node",
        #     name="steering_wheel_ffb_node",
        #     output="screen",
        # ),
        # Node(
        #     package="drowsiness_detection_pkg",
        #     executable="steering_vibration_node",
        #     name="steering_vibration_node",
        #     output="screen",
        # ),
        # Node(
        #     package="drowsiness_detection_pkg",
        #     executable="speaker_node",
        #     name="speaker_node",
        #     output="screen",
        # ),
        # Node(
        #     package="drowsiness_detection_pkg",
        #     executable="fan_node",
        #     name="fan_node",
        #     output="screen",
        # ),
    ])
