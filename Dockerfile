# Base ROS 2 image (Humble)
FROM ros:humble

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building ROS 2 workspace
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    build-essential \
    git \
    lsb-release \
    curl \
    gnupg2 \
    # The clean-up is better here to keep the layer small
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true
RUN rosdep update

# Create ROS 2 workspace
WORKDIR /ros2_ws

RUN apt-get update && apt-get install -y \
    libusb-1.0-0 \
    libusb-1.0-0-dev \
    libavcodec-extra58 \
    libavformat58 \
    libswscale5 \
    libavutil56

# Copy Spinnaker .deb installers from temp folder
COPY temp /tmp/spinnaker

RUN export DEBIAN_FRONTEND=noninteractive && \
    echo "flir-eula flir-eula/accepted boolean true" | debconf-set-selections && \
    cd /tmp/spinnaker/spinnaker-4.2.0.88-amd64-22.04-pkg/spinnaker-4.2.0.88-amd64 && \
    dpkg -i libgentl_*.deb \
    libspinnaker_*.deb \
    libspinnaker-dev_*.deb \
    libspinnaker-c_*.deb \
    libspinnaker-c-dev_*.deb \
    libspinvideo_*.deb \
    libspinvideo-dev_*.deb \
    libspinvideo-c_*.deb \
    libspinvideo-c-dev_*.deb \
    spinupdate_*.deb \
    spinupdate-dev_*.deb \
    spinnaker_*.deb \
    spinnaker-doc_*.deb || true && \
    apt-get install -fy && \
    sh configure_usbfs.sh && \
    sh configure_spinnaker.sh && \
    sh configure_spinnaker_paths.sh && \
    sh configure_gentl_paths.sh $BITS
# Copy source code
COPY src ./src

# Copy requirements.txt and install Python dependencies
COPY src/drowsiness_detection_pkg/drowsiness_detection/requirements.txt ./requirements.txt
RUN sudo pip3 install --no-cache-dir -r requirements.txt

# Install ROS dependencies for all packages
# --- MODIFIED: Added apt-get update before rosdep install ---
RUN apt-get update && rosdep install --from-paths src --ignore-src -r -y

RUN pip3 install --no-cache-dir "setuptools<65"

# Build the workspace
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

# Set bash as default shell and source ROS 2 + workspace
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
    && echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Default command
CMD ["bash"]