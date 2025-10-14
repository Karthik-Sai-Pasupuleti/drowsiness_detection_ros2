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

### Prerequisites
Before running the ROS2 launch files, you need to start the CARLA simulator:

1. Install CARLA version 0.9.16 from [official website](https://carla.org/2025/09/16/release-0.9.16/)
2. [Installation guide](https://medium.com/@pasupuletikarthiksai/a-step-by-step-guide-to-installing-carla-on-windows-c092a469e6f6) 
2. Launch CARLA with ROS2 support:
```bash
./CarlaUE4.sh --ros2
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