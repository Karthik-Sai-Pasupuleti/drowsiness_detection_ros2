# Base image with ROS 2 Humble and Ubuntu 22.04
FROM osrf/ros:humble-desktop

# Set environment variables
ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    usbutils \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-message-filters \
    ros-humble-spinnaker-camera-driver \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /root/ws/

# Copy the source code
COPY src ./src

# Mark the broken/missing-dependency spinnaker_camera package to be ignored by colcon
RUN touch src/spinnaker_camera/COLCON_IGNORE

# Install Python requirements
# The user specified exact path: src\drowsiness_detection_pkg\drowsiness_detection\requirements.txt
# In Docker it will be src/drowsiness_detection_pkg/drowsiness_detection/requirements.txt
RUN pip3 install -r src/drowsiness_detection_pkg/drowsiness_detection/requirements.txt

# Install rosdep dependencies
RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Copy and setup entrypoint script
COPY start.sh /start.sh
RUN dos2unix /start.sh && chmod +x /start.sh

# Source the setup.bash in .bashrc so manual execs work nicely
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /root/ws/install/setup.bash" >> /root/.bashrc

# Expose the Flask port
EXPOSE 5000

# Set the entrypoint
ENTRYPOINT ["/start.sh"]