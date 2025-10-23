from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # === Launch argument: driver ID for consistent session naming ===
    driver_id_arg = DeclareLaunchArgument(
        "driver_id",
        default_value="test_driver",
        description="Driver identifier for saving session data."
    )

    driver_id = LaunchConfiguration("driver_id")

    # === Launch CARLA simulator (runs immediately) ===
    carla_process = ExecuteProcess(
        cmd=[
            "/home/user/Downloads/CARLA_0.9.16/CarlaUE4.sh",
            "-quality-level=Low",
            "--ros2",
        ],
        output="screen",
        shell=True,  # allows environment variables / proper execution
    )

    # === Define ROS2 nodes ===
    mediapipe_node = Node(
        package="drowsiness_detection_pkg",
        executable="mediapipe_node",
        name="camera_mediapipe_node",
        output="screen",
        parameters=[{"driver_id": driver_id}],  
    )

    carla_node = Node(
        package="drowsiness_detection_pkg",
        executable="carla_node",
        name="carla_node",
        output="screen",
    )

    driver_assistance_node = Node(
        package="drowsiness_detection_pkg",
        executable="driver_assistance_node",
        name="driver_assistance_node",
        output="screen",
        parameters=[{"driver_id": driver_id}],  
    )

    steering_wheel_ffb_node = Node(
        package="drowsiness_detection_pkg",
        executable="steering_wheel_ffb_node",
        name="steering_wheel_ffb_node",
        output="screen",
    )

    steering_vibration_node = Node(
        package="drowsiness_detection_pkg",
        executable="steering_vibration_node",
        name="steering_vibration_node",
        output="screen",
    )

    speaker_node = Node(
        package="drowsiness_detection_pkg",
        executable="speaker_node",
        name="speaker_node",
        output="screen",
    )

    fan_node = Node(
        package="drowsiness_detection_pkg",
        executable="fan_node",
        name="fan_node",
        output="screen",
    )

    drowsiness_gui = Node(
        package="drowsiness_detection_pkg",
        executable="drowsiness_gui",
        name="drowsiness_gui",
        output="screen",
    )

    # === Delay ROS2 nodes until CARLA is fully ready ===
    delayed_nodes = TimerAction(
        period=10.0,  # seconds
        actions=[
            mediapipe_node,
            carla_node,
            driver_assistance_node,
            steering_wheel_ffb_node,
            steering_vibration_node,
            speaker_node,
            fan_node,
            drowsiness_gui,
        ],
    )

    # === Launch description ===
    return LaunchDescription([
        driver_id_arg,
        carla_process,
        delayed_nodes,
    ])
