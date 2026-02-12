#!/bin/bash
set -e

# Source ROS 2 installation
source /opt/ros/humble/setup.bash
source /root/ws/install/setup.bash

# Launch the labelling tool (GUI) in the background? 
# The user wants TWO launches. One is the GUI (Flask) and one is the Camera.
# We can use `&` to run in background.

echo "Launching Spinnaker Camera (Standard Driver)..."
# Using standard driver launch as local package is broken/missing SDK.
# Adjust params as needed. Default launch is usually driver_node.launch.py or camera.launch.py
ros2 launch spinnaker_camera_driver driver_node.launch.py serial:="'24364301'" --ros-args -r image_raw:=/camera/image_raw &
SPINNAKER_PID=$!

echo "Launching Labelling Tool (Flask GUI)..."
# This one likely blocks or runs the Flask server.
ros2 launch drowsiness_detection_pkg labelling_tool_launch.py

# Wait for background processes to ensure container doesn't exit immediately if the main one does
wait $SPINNAKER_PID
