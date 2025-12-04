from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='spinnaker_camera',
            executable='spinnaker_cam',
            name='spinnaker_camera',
            output='screen',
            parameters=[]
        )
    ])
