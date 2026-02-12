from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spinnaker_camera',
            executable='spinnaker_cam',
            name='spinnaker_camera',
            output='screen',
            parameters=[{
                'device_id': '24364301',
                
                # --- NEW: Exact Resolution Scaling ---
                'TargetWidth': 640,          # Desired Width
                'TargetHeight': 480,         # Desired Height
                # -------------------------------------

                'Brightness': 1.0,
                'ExposureAuto': 'Off',
                'GainAuto': 'Off',
                'GammaEnable': 0.5
            }]
        )
    ])