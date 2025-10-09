#!/usr/bin/env python3

import math
from collections import deque
import time
from pathlib import Path
import threading
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from std_msgs.msg import Float32MultiArray, Float32
from carla_msgs.msg import CarlaEgoVehicleControl

from ament_index_python.packages import get_package_share_directory


from drowsiness_detection_pkg.camera.utils import (
    calculate_perclos,
    calculate_blink_frequency,
    calculate_yawn_frequency,
    vehicle_feature_extraction,
)

import cv2
from cv_bridge import CvBridge

# --- Shared state ---
data_lock = threading.Lock()
latest_image = None


class DriverAssistanceNode(Node):
    def __init__(self):
        super().__init__("driver_assistance_node")

        # Parameters
        self.declare_parameter("EAR_Threshold", 0.2)
        self.declare_parameter("MAR_Threshold", 0.4)
        self.declare_parameter("blink_threshold", 3)
        self.declare_parameter("Yawning_threshold", 4)

        self.ear_threshold = self.get_parameter("EAR_Threshold").value
        self.mar_threshold = self.get_parameter("MAR_Threshold").value
        self.ear_consec_frames = self.get_parameter("blink_threshold").value
        self.mar_consec_time = self.get_parameter("Yawning_threshold").value

        # Buffers: Use long-term deques to store continuous data
        self.ear_buffer = deque(maxlen=2000)
        self.mar_buffer = deque(maxlen=2000)
        self.steering_buffer = deque(maxlen=2000)
        self.lane_offset_buffer = deque(maxlen=2000)
        self.image_buffer = deque(maxlen=1000)
        self.buffer_lock = threading.Lock()

        # Window timing
        self.window_duration = 60.0  # Single 60-second window
        self.label_collection_time = 10.0  # Last 10 seconds for labeling
        self.current_window_id = 0
        self.last_window_end_time = time.time()

        # New: Store data for the current window to be processed later
        self.window_data_to_process = None

        # Image bridge
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscription(
            Float32MultiArray,
            "/ear_mar",
            self.ear_mar_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CarlaEgoVehicleControl,
            "/carla/hero/vehicle_control_cmd",
            self.steering_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32,
            "/carla/lane_offset",
            self.lane_offset_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, "/camera/image_raw", self.cb_camera, qos_profile_sensor_data
        )

        # We no longer need this callback, as the web server will store labels directly in a buffer
        # self.create_subscription(Float32, '/driver_assistance/label_submitted', self.label_submitted_callback, 10)

        # Publishers
        self.metrics_pub = self.create_publisher(
            Float32MultiArray, "/driver_assistance/metrics", 10
        )
        self.ear_pub = self.create_publisher(
            Float32MultiArray, "/driver_assistance/ear_array", 10
        )
        self.mar_pub = self.create_publisher(
            Float32MultiArray, "/driver_assistance/mar_array", 10
        )
        self.steering_pub = self.create_publisher(
            Float32MultiArray, "/driver_assistance/steering_array", 10
        )
        self.lane_pub = self.create_publisher(
            Float32MultiArray, "/driver_assistance/lane_position_array", 10
        )

        # Publisher for phase, ID, and remaining time
        self.window_phase_pub = self.create_publisher(
            Float32MultiArray, "/driver_assistance/window_phase", 10
        )

        # [NEW] Publisher to send the window data to the web node for saving
        self.window_data_pub = self.create_publisher(
            Float32MultiArray, "/driver_assistance/window_data", 10
        )

        # Timer for periodic state update (e.g., every 0.1 seconds)
        self.create_timer(0.1, self.update_window_phase)

        self.get_logger().info("Driver Assistance Node started.")

    # ----------------- Callbacks -----------------
    def cb_camera(self, msg):
        global latest_image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.buffer_lock:
                latest_image = cv_img.copy()
                self.image_buffer.append((time.time(), cv_img))
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def ear_mar_callback(self, msg):
        ts = time.time()
        if len(msg.data) >= 2:
            with self.buffer_lock:
                self.ear_buffer.append((ts, msg.data[0]))
                self.mar_buffer.append((ts, msg.data[1]))

    def steering_callback(self, msg):
        ts = time.time()
        with self.buffer_lock:
            self.steering_buffer.append((ts, msg.steer))

    def lane_offset_callback(self, msg):
        ts = time.time()
        with self.buffer_lock:
            self.lane_offset_buffer.append((ts, msg.data))

    # ----------------- Window Management -----------------
    def update_window_phase(self):
        """
        Periodically checks window state, publishes status, and triggers processing
        at the end of the 60-second window.
        """
        now = time.time()

        time_in_current_window = now - self.last_window_end_time
        remaining_time = max(0.0, self.window_duration - time_in_current_window)

        phase = 0 if remaining_time > self.label_collection_time else 1

        msg = Float32MultiArray()
        msg.data = [
            float(phase),
            float(self.current_window_id),
            remaining_time,
        ]
        self.window_phase_pub.publish(msg)

        if remaining_time == 0.0:
            self.process_completed_window()

            self.current_window_id += 1
            self.last_window_end_time = now

    def process_completed_window(self):
        self.get_logger().info(
            f"Window {self.current_window_id} complete. Publishing data for saving."
        )

        window_start_time = self.last_window_end_time
        self.window_data_to_process = self.compute_metrics(window_start_time)

        if self.window_data_to_process:
            # Publish the entire window data as a serialized JSON string
            msg = Float32MultiArray()
            # Convert JSON string to a list of ASCII values to send over ROS
            serialized_data = json.dumps(self.window_data_to_process, default=str)
            msg.data = [float(ord(c)) for c in serialized_data]
            self.window_data_pub.publish(msg)

        # Clear data from buffers for the new window
        with self.buffer_lock:
            # A more robust way would be to only clear data older than the new window's start time.
            # For simplicity, we clear all buffers here.
            self.ear_buffer.clear()
            self.mar_buffer.clear()
            self.steering_buffer.clear()
            self.lane_offset_buffer.clear()
            self.image_buffer.clear()

        self.window_data_to_process = None

    # ----------------- Compute Metrics -----------------
    def compute_metrics(self, window_start_time):
        end_time = window_start_time + self.window_duration

        with self.buffer_lock:
            ear_values = [
                v for t, v in self.ear_buffer if window_start_time <= t <= end_time
            ]
            mar_values = [
                v for t, v in self.mar_buffer if window_start_time <= t <= end_time
            ]
            steering_values = [
                v for t, v in self.steering_buffer if window_start_time <= t <= end_time
            ]
            lane_values = [
                v
                for t, v in self.lane_offset_buffer
                if window_start_time <= t <= end_time
            ]
            images = [
                img
                for t, img in self.image_buffer
                if window_start_time <= t <= end_time
            ]

        if not ear_values or not mar_values or not steering_values or not lane_values:
            self.get_logger().warn("Not enough data for this window. Skipping metrics.")
            return None

        fps = len(ear_values) / self.window_duration
        perclos = calculate_perclos(
            ear_values,
            ear_threshold=self.ear_threshold,
            min_consec_frames=self.ear_consec_frames,
        )
        blink_rate = calculate_blink_frequency(
            ear_values, ear_threshold=self.ear_threshold, fps=fps
        )
        yawn_rate = calculate_yawn_frequency(
            mar_values,
            mar_threshold=self.mar_threshold,
            min_consec_frames=self.ear_consec_frames,
            fps=fps,
        )
        entropy, steering_rate, sdlp = vehicle_feature_extraction(
            steering_values, lane_values, self.window_duration
        )

        metrics = [
            float(x)
            for x in [perclos, blink_rate, yawn_rate, entropy, steering_rate, sdlp]
        ]

        sampled_images = []
        if images:
            step = max(1, len(images) // 5)
            # Store images as a list of bytes
            sampled_images = [
                cv2.imencode(".jpg", img)[1].tobytes()
                for img in [images[i] for i in range(0, len(images), step)][:5]
            ]

        msg = Float32MultiArray()
        msg.data = metrics
        self.metrics_pub.publish(msg)

        self._publish_array(self.ear_pub, ear_values)
        self._publish_array(self.mar_pub, mar_values)
        self._publish_array(self.steering_pub, steering_values)
        self._publish_array(self.lane_pub, lane_values)

        self.get_logger().info(
            f"Window {self.current_window_id} metrics published with {len(sampled_images)} images sampled."
        )

        return {
            "metrics": {
                "perclos": perclos,
                "blink_rate": blink_rate,
                "yawn_rate": yawn_rate,
                "steering_entropy": entropy,
                "steering_rate": steering_rate,
                "sdlp": sdlp,
            },
            "raw_data": {
                "ear_array": ear_values,
                "mar_array": mar_values,
                "steering_array": steering_values,
                "lane_position_array": lane_values,
            },
            "images": [
                img.hex() for img in sampled_images
            ],  # Convert bytes to hex string for JSON serialization
            "window_id": self.current_window_id,
        }

    # ----------------- Helper -----------------
    def _publish_array(self, pub, values):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in values]
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DriverAssistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
