# Base image with ROS 2 Humble and Ubuntu 22.04
FROM osrf/ros:humble-desktop

# Set environment variables
ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# 2. Set working directory
WORKDIR /root/ws

# 4. Copy your local package source code
# 4. Copy your local package source code
COPY src ./src
COPY requirements.txt /root/ws/requirements.txt
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Install pip
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# 5. Install Python and ROS dependencies
# Note: We run rosdep update before installing
RUN pip install --no-cache-dir -r /root/ws/requirements.txt


# Set the entrypoint
ENTRYPOINT ["/start.sh"]