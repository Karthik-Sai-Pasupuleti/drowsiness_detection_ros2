# Driver Drowsiness Detection System (ROS2)

Driver drowsiness detection and assistance system implemented using ROS2. The system monitors driver drowsiness through facial features and vehicle data while providing automated interventions when drowsiness is detected.

## Project Goals

- Real-time monitoring of driver drowsiness using multiple indicators:
  - Eye closure patterns (PERCLOS)
  - Blink frequency
  - Yawning detection
  - Steering behavior
  - Lane position

- Automated intervention system providing:
  - Steering wheel haptic feedback
  - Voice commands/alerts
  - Cooling fan activation

## ROS2 Nodes

### Core Nodes

- [`camera_mediapipe_node`](src/drowsiness_detection_pkg/drowsiness_detection/camera/camera_mediapipe_node.py): Captures and processes camera feed using MediaPipe face detection
  - **Publishes:**
    - `/camera/image_raw`: Video feed with facial landmarks overlay
    - `/ear_mar`: Eye Aspect Ratio and Mouth Aspect Ratio values

- [`carla_node`](src/drowsiness_detection_pkg/drowsiness_detection/carla/carla_manual_control.py): Interface with CARLA simulator
  - **Subscribes:**
    - `/carla/hero/vehicle_control_cmd`: Vehicle control commands
  - **Publishes:**
    - `/carla/lane_offset`: Vehicle's lane position data

- [`driver_assistance_node`](src/drowsiness_detection_pkg/drowsiness_detection/main.py): Main drowsiness detection and coordination node
  - **Subscribes:**
    - `/ear_mar`: Facial measurements
    - `/camera/image_raw`: Processed video feed
    - `/carla/hero/vehicle_control_cmd`: Vehicle control data
    - `/carla/lane_offset`: Lane position data
  - **Publishes:**
    - `/driver_assistance/window_data`: Drowsiness metrics calculated from 1-minute data window
  - **Functions:**
    - stores 1-minute window of sensor data
    - Computes drowsiness metrics from aggregated data

### Intervention Nodes

- [`steering_vibration_node`](src/drowsiness_detection_pkg/drowsiness_detection/bot/wheel_voice_node.py): Controls steering wheel haptic feedback
- [`speaker_node`](src/drowsiness_detection_pkg/drowsiness_detection/bot/controls/speaker_node.py): Handles audio alerts
- [`fan_node`](src/drowsiness_detection_pkg/drowsiness_detection/bot/controls/fan_node.py): Controls cooling system

### Visualization/Labeling Interface

- [`labelling_tool`](src/drowsiness_detection_pkg/drowsiness_detection/labelling/app_v2.py): Web interface for:
  - Real-time monitoring
  - Data collection
  - Manual intervention control
  - Data labeling

## Launch Files

### build
```bash
colcon build --symlink-install
source install/setup.bash
```

### Prerequisites
Before running the ROS2 launch files, you need to start the CARLA simulator:

1. Install CARLA version 0.9.16 from [official website](https://carla.org/2025/09/16/release-0.9.16/)
2. [Installation guide](https://medium.com/@pasupuletikarthiksai/a-step-by-step-guide-to-installing-carla-on-windows-c092a469e6f6) 
2. Launch CARLA with ROS2 support:
```bash
./CarlaUE4.sh --ros2 -RenderOffScreen
```

### Driver Assistance System
```bash
ros2 launch drowsiness_detection_pkg driver_assistance_launch.py
```

This launch file starts:
- Camera and facial processing pipeline
  - Camera capture and MediaPipe processing
  - Facial landmark detection
- Driver assistance core system
  - Drowsiness detection algorithms
  - Intervention management
- CARLA simulator interface
  - Vehicle control and telemetry
  - Lane position tracking
- Intervention nodes
  - Steering wheel haptic feedback
  - Audio alert system
  - Cooling system control

### Data Collection & Labeling Tool
```bash
ros2 launch drowsiness_detection_pkg labelling_tool_launch.py
```

Launches the Flask-based web interface for:
- Real-time system monitoring
- Data collection and visualization
- Manual intervention controls
- Data labeling and annotation
- All core system nodes (camera, processing, CARLA interface)

## Custom ROS2 Messages

### Drowsiness Detection Messages (`drowsiness_detection_msg`)
Located in `src/drowsiness_detection_msg/msg/`:

| Message Type | Description | Key Fields |
|-------------|-------------|------------|
| `DrowsinessMetricsData.msg` | Complete drowsiness metrics | PERCLOS, blink rate, yawn frequency |
| `EarMarValue.msg` | Facial measurements | Eye aspect ratio (EAR), mouth aspect ratio (MAR) |
| `LanePosition.msg` | Vehicle positioning | Lane deviation, heading angle |
| `Vibration.msg` | Haptic feedback control | Intensity, duration, pattern |

### CARLA Interface Messages (`ros-carla-msgs`)
Located in `src/ros-carla-msgs/msg/`:

| Message Type | Description |
|-------------|-------------|
| `CarlaEgoVehicleControl.msg` | Vehicle control interface for CARLA simulator |


## Steering Wheel Vibration 

Setup the rules by following steps:
- sudo nano /etc/udev/rules.d/99-logitech-g29.rules
- paste: KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c24f", MODE="0666", SYMLINK+="logitech_g29"

- Reload udev rules and trigger:
  - sudo udevadm control --reload-rules
  - sudo udevadm trigger


## Ethernet Connection Setup to Access Flask Interface from Windows

This section describes how to connect your Linux workstation hosting the Flask-based web interface with a Windows laptop over an Ethernet cable, enabling you to access the interface remotely.

### Hardware Requirements
- Ethernet cable connecting Linux and Windows machines directly or via a switch.

### Step 1: Configure Static IP Addresses

#### On Linux
1. Identify your Ethernet interface name (e.g., `enp130s0`):
```bash
ip link show
```

2. Bring the interface up and assign a static IP:

```bash
sudo ip link set enp130s0 up
sudo ip addr add 192.168.1.10/24 dev enp130s0
```


#### On Windows
1. Open **Control Panel** → **Network and Internet** → **Network and Sharing Center**.
2. Click **Change adapter settings** on the left panel.
3. Right-click the Ethernet adapter and select **Properties**.
4. Select **Internet Protocol Version 4 (TCP/IPv4)** and click **Properties**.
5. Choose **Use the following IP address** and enter:
- **IP address:** `192.168.1.20`
- **Subnet mask:** `255.255.255.0`
- Leave **Default gateway** blank.
6. Click **OK** and close the dialogs.

### Step 2: Adjust Ethernet Speed and Duplex in Windows (Optional but recommended)

1. Open **Task Manager** (Ctrl + Shift + Esc).
2. Go to the **Performance** tab and click **Ethernet**.
3. Click on **Open Network & Internet settings** at the bottom.
4. Under **Advanced network settings**, click **Change adapter options**.
5. Right-click your Ethernet adapter → **Properties** → **Configure**.
6. Go to the **Advanced** tab, find **Speed & Duplex** or similar property.
7. Select an appropriate speed and duplex setting (e.g., `1.0 Gbps Full Duplex` or `100 Mbps Full Duplex`), or choose `Auto Negotiation`.
8. Click **OK** to apply.

### Step 3: Run Flask App on Linux Listening on All Interfaces
Launch the labelling gui

### Step 4: Allow Incoming Traffic on Linux Firewall
Allow port 5000 through the firewall:

```bash
sudo ip addr add 192.168.0.10/24 dev enp97s0 
sudo ip link set enp97s0 up
```
Check the ip in linux workstation
```bash
ip addr show
ping 192.168.0.10
```

Same goes for windows too
```bash
ping 192.168.0.10
```
if it failed then update the ip address in windows network settings.


### Step 5: Access Flask Interface from Windows
1. Open a web browser on Windows.
2. Enter the Linux IP and Flask port in the address bar:

```bash
http://192.168.1.10:5000
```

### Launch 
ros2 launch drowsiness_detection_pkg complete_system_launch.py driver_id:=test_drive


## Vehicle info

id: 283
type: vehicle.mercedes.coupe_2020
rolename: hero
wheels:
- tire_friction: 3.5
  damping_rate: 0.25
  max_steer_angle: 69.99999237060547
  radius: 34.25
  max_brake_torque: 800.0
  max_handbrake_torque: 0.0
  position:
    x: -794.5993041992188
    y: 30230.58984375
    z: 62.566890716552734
- tire_friction: 3.5
  damping_rate: 0.25
  max_steer_angle: 69.99999237060547
  radius: 34.25
  max_brake_torque: 800.0
  max_handbrake_torque: 0.0
  position:
    x: -648.115234375
    y: 30294.5703125
    z: 62.566890716552734
- tire_friction: 3.5
  damping_rate: 0.25
  max_steer_angle: 0.0
  radius: 34.25
  max_brake_torque: 800.0
  max_handbrake_torque: 1600.0
  position:
    x: -907.9749145507812
    y: 30490.1640625
    z: 62.566890716552734
- tire_friction: 3.5
  damping_rate: 0.25
  max_steer_angle: 0.0
  radius: 34.25
  max_brake_torque: 800.0
  max_handbrake_torque: 1600.0
  position:
    x: -761.4908447265625
    y: 30554.14453125
    z: 62.566890716552734
max_rpm: 0.0
moi: 0.0
damping_rate_full_throttle: 0.0
damping_rate_zero_throttle_clutch_engaged: 0.0
damping_rate_zero_throttle_clutch_disengaged: 0.0
use_gear_autobox: false
gear_switch_time: 0.0
clutch_strength: 0.0
mass: 0.0
drag_coefficient: 0.0
center_of_mass:
  x: 0.0
  y: 0.0
  z: 0.0
---

. ~/ros2_humble/install/local_setup.bash
camera node launch: ros2 launch spinnaker_camera spinnaker_cam.launch.py
camera serial number: 24364301
# install cv_bridge and vision_opencv
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-vision-opencv

deactivate  # if in venv
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch drowsiness_detection_pkg drowsiness.launch.py


# start docker
xhost +local:docker


