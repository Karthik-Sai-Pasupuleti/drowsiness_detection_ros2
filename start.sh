#!/bin/bash
set -e
# Source the workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# === START VLM VISION PIPELINE IN BACKGROUND ===
# This monitors the camera stream and saves occlusion JSONs
echo "Starting Hybrid VLM Vision Node..."
ros2 run drowsiness_detection_pkg hybrid_vlm_node --ros-args \
  -p driver_id:=${DRIVER_ID} \
  -p output_dir:=/root/ws/drowsiness_data/vlm_events &
# ===============================================

# Launch the camera driver (and Flask app) in the FOREGROUND
exec ros2 launch drowsiness_detection_pkg labelling_tool_launch.py driver_id:=${DRIVER_ID}