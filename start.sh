#!/bin/bash
set -e
# Source the workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
# Launch the camera driver in the background (example)
# Launch the camera driver (and Flask app) in the FOREGROUND
exec ros2 launch drowsiness_detection_pkg labelling_tool_launch.py driver_id:=${DRIVER_ID}