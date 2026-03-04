# Base image with ROS 2 Humble and Ubuntu 22.04
FROM osrf/ros:humble-desktop

# Set environment variables
ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# 2. Set working directory
WORKDIR /root/ws

# 4. Copy your local package source code
COPY src ./src
COPY requirements.txt /root/ws/requirements.txt
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Install pip and dependencies
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "pip<25" "setuptools<58"
RUN pip install --no-cache-dir -r /root/ws/requirements.txt


# Install ROS dependencies via rosdep (Crucial for building the new VLM_LLM nodes)
RUN apt-get update && rosdep init || true
RUN rosdep update && rosdep install --from-paths src --ignore-src -r -y && rm -rf /var/lib/apt/lists/*



# Set the entrypoint
ENTRYPOINT ["/start.sh"]