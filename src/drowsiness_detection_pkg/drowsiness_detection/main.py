#!/usr/bin/env python3

import math
from collections import deque
from pathlib import Path
import threading
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage

from carla_msgs.msg import CarlaEgoVehicleControl
from drowsiness_detection_msg.msg import (
    EarMarValue,
    LanePosition,
    DrowsinessMetricsData,
)

from ament_index_python.packages import get_package_share_directory

from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from drowsiness_detection.camera.utils import (
    calculate_perclos,
    calculate_blink_frequency,
    calculate_yawn_frequency,
    vehicle_feature_extraction,
)

import cv2
import numpy as np
from cv_bridge import CvBridge


class DriverAssistanceNode(Node):
    def __init__(self):
        super().__init__("driver_assistance_node")

        # ---------- Parameters ----------
        self.declare_parameter("EAR_Threshold", 0.2)
        self.declare_parameter("MAR_Threshold", 0.4)
        self.declare_parameter(
            "blink_threshold", 3
        )  # EAR consecutive frames for blink/PERCLOS
        self.declare_parameter(
            "Yawning_threshold", 4
        )  # MAR consecutive time (seconds) for yawn

        self.ear_threshold = self.get_parameter("EAR_Threshold").value
        self.mar_threshold = self.get_parameter("MAR_Threshold").value
        self.ear_consec_frames = self.get_parameter("blink_threshold").value
        self.mar_consec_time = self.get_parameter("Yawning_threshold").value  # seconds

        # ---------- Buffers ----------
        # Use sufficiently large maxlen; pruning is time-based after each window.

        self.ear_buffer = deque()
        self.mar_buffer = deque()
        self.steering_buffer = deque()
        self.lane_offset_buffer = deque()
        self.image_buffer = deque()

        self.buffer_lock = threading.Lock()

        # ---------- Window timing ----------
        self.window_duration = 60.0  # seconds
        self.label_collection_time = 10.0  # last 10s phase indicator
        self.current_window_id = 0

        # Initialize to ROS time (seconds)
        self.last_window_end_time = self.get_clock().now().nanoseconds / 1e9

        # ---------- Image bridge ----------
        self.bridge = CvBridge()

        # ---------- Subscribers ----------
        self.create_subscription(
            EarMarValue, "/ear_mar", self.ear_mar_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            CarlaEgoVehicleControl,
            "/carla/hero/vehicle_control_cmd",
            self.steering_callback,
            10,
        )

        self.create_subscription(
            LanePosition,
            "/carla/lane_offset",
            self.lane_offset_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, "/camera/image_raw", self.cb_camera, qos_profile_sensor_data
        )

        # ---------- Publishers ----------

        # Phase/ID/remaining time
        self.window_phase_pub = self.create_publisher(
            Float32MultiArray, "/driver_assistance/window_phase", 10
        )

        # publishes (metrics + raw arrays + images)
        self.window_data_pub = self.create_publisher(
            DrowsinessMetricsData, "/driver_assistance/window_data", 10
        )
        # ---------- Timer ----------
        self.create_timer(0.1, self.update_window_phase)
        self.get_logger().info("Driver Assistance Node started.")

    def cb_camera(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.buffer_lock:
                self.image_buffer.append((ts, cv_img))
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def ear_mar_callback(self, msg: EarMarValue):
        try:
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.buffer_lock:
                self.ear_buffer.append((ts, float(msg.ear_value)))
                self.mar_buffer.append((ts, float(msg.mar_value)))
        except Exception as e:
            self.get_logger().error(f"Error in ear_mar_callback: {e}")

    def steering_callback(self, msg: CarlaEgoVehicleControl):
        try:
            # Use ROS time if header stamp is zero or missing
            if hasattr(msg, "header") and (
                msg.header.stamp.sec > 0 or msg.header.stamp.nanosec > 0
            ):
                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            else:
                ts = self.get_clock().now().nanoseconds / 1e9

            with self.buffer_lock:
                self.steering_buffer.append((ts, float(msg.steer)))

        except Exception as e:
            self.get_logger().error(f"Error in steering_callback: {e}")

    def lane_offset_callback(self, msg: LanePosition):
        try:
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.buffer_lock:
                self.lane_offset_buffer.append((ts, float(msg.lane_offset)))
        except Exception as e:
            self.get_logger().error(f"Error in lane_offset_callback: {e}")

    def update_window_phase(self):
        """
        Periodically checks window state, publishes status, and triggers processing
        at the end of the 60-second window. Uses ROS time consistently.
        """
        now_ros = self.get_clock().now().nanoseconds / 1e9
        time_in_current_window = now_ros - self.last_window_end_time
        remaining_time = max(0.0, self.window_duration - time_in_current_window)
        phase = 0 if remaining_time > self.label_collection_time else 1

        msg = Float32MultiArray()
        msg.data = [float(phase), float(self.current_window_id), remaining_time]
        self.window_phase_pub.publish(msg)

        if remaining_time <= 0.0:
            self.process_completed_window()

    def process_completed_window(self):
        """
        Process the just-finished window.
        Defines window as [start, end) in ROS time; prunes buffers by time (keep t >= end).
        """
        window_start_time = self.last_window_end_time
        end_time = window_start_time + self.window_duration

        self.get_logger().info(
            f"Window {self.current_window_id} complete. Publishing data for saving."
        )

        self.compute_metrics(window_start_time)

        # ---- Time-based pruning: keep samples for next window (t >= end_time) ----
        with self.buffer_lock:
            self.ear_buffer = deque(
                [(t, v) for t, v in self.ear_buffer if t >= end_time],
                maxlen=self.ear_buffer.maxlen,
            )
            self.mar_buffer = deque(
                [(t, v) for t, v in self.mar_buffer if t >= end_time],
                maxlen=self.mar_buffer.maxlen,
            )
            self.steering_buffer = deque(
                [(t, v) for t, v in self.steering_buffer if t >= end_time],
                maxlen=self.steering_buffer.maxlen,
            )
            self.lane_offset_buffer = deque(
                [(t, v) for t, v in self.lane_offset_buffer if t >= end_time],
                maxlen=self.lane_offset_buffer.maxlen,
            )
            self.image_buffer = deque(
                [(t, img) for t, img in self.image_buffer if t >= end_time],
                maxlen=self.image_buffer.maxlen,
            )

        # Advance exact window boundaries (prevents drift)
        self.current_window_id += 1
        self.last_window_end_time = end_time

    def sample_images_as_ros(
        self, image_samples, window_start_time, end_time, num_samples=5
    ):
        """
        Sample images evenly across the window and convert them to ROS Image messages.
        - image_samples: list of (timestamp, cv2_image)
        - window_start_time, end_time: float
        - num_samples: number of representative images to return
        """
        if not image_samples:
            return []

        targets = [
            window_start_time + frac * (end_time - window_start_time)
            for frac in np.linspace(0.0, 1.0, num_samples)
        ]

        sampled_ros_images = []
        idx = 0
        for tt in targets:
            # advance until next frame time >= target
            while idx + 1 < len(image_samples) and image_samples[idx + 1][0] < tt:
                idx += 1
            candidates = [idx] + ([idx + 1] if idx + 1 < len(image_samples) else [])
            best_i = min(candidates, key=lambda k: abs(image_samples[k][0] - tt))
            img = image_samples[best_i][1]
            ros_img = cv2.imencode(".jpg", img)[1].tobytes()
            sampled_ros_images.append(ros_img)

        return sampled_ros_images

    def compute_metrics(self, window_start_time):
        """
        Compute metrics for a single 60s window using variable-frequency inputs.
        - Window is sliced as [start, end) to avoid double-counting boundary samples.
        - Each metric is computed independently per source (no cross-stream alignment).
        - Uses robust FPS from timestamps for EAR/MAR.
        - Yawn detection uses MAR consecutive time converted to frames via MAR FPS.
        - Images are sampled by time percentiles for consistent temporal coverage.
        """
        end_time = window_start_time + self.window_duration

        # --- Helper: FPS from timestamps (median Δt) ---
        def robust_fps(ts):
            if len(ts) < 2:
                return 0.0
            dts = [ts[i + 1] - ts[i] for i in range(len(ts) - 1) if ts[i + 1] > ts[i]]
            if not dts:
                return 0.0
            dts.sort()
            med_dt = dts[len(dts) // 2]
            return 1.0 / med_dt if med_dt > 0 else 0.0

        # ---- Slice buffers with [start, end) and keep timestamps ----
        with self.buffer_lock:
            ear_samples = [
                (t, v) for t, v in self.ear_buffer if window_start_time <= t < end_time
            ]
            mar_samples = [
                (t, v) for t, v in self.mar_buffer if window_start_time <= t < end_time
            ]
            steering_samples = [
                (t, v)
                for t, v in self.steering_buffer
                if window_start_time <= t < end_time
            ]
            lane_samples = [
                (t, v)
                for t, v in self.lane_offset_buffer
                if window_start_time <= t < end_time
            ]
            image_samples = [
                (t, img)
                for t, img in self.image_buffer
                if window_start_time <= t < end_time
            ]

        # sorting by timestamp
        ear_samples.sort(key=lambda x: x[0])
        mar_samples.sort(key=lambda x: x[0])
        steering_samples.sort(key=lambda x: x[0])
        lane_samples.sort(key=lambda x: x[0])
        image_samples.sort(key=lambda x: x[0])

        # checking for empty samples
        if (
            not ear_samples
            or not mar_samples
            or not steering_samples
            or not lane_samples
        ):
            self.get_logger().warn(
                f"Not enough data for this window "
                f"(EAR={len(ear_samples)}, MAR={len(mar_samples)}, "
                f"Steering={len(steering_samples)}, Lane={len(lane_samples)}). Skipping metrics."
            )
            return None

        # Split into ts / values
        ear_ts, ear_vals = zip(*ear_samples)
        mar_ts, mar_vals = zip(*mar_samples)
        _, steering_vals = zip(*steering_samples)
        _, lane_vals = zip(*lane_samples)

        ear_vals = list(map(float, ear_vals))
        mar_vals = list(map(float, mar_vals))
        steering_vals = list(map(float, steering_vals))
        lane_vals = list(map(float, lane_vals))

        # Robust fps for face streams
        fps_ear = robust_fps(list(ear_ts))
        fps_mar = robust_fps(list(mar_ts))
        if fps_ear <= 0.0 or fps_mar <= 0.0:
            self.get_logger().warn(
                f"Invalid fps (EAR={fps_ear:.3f}, MAR={fps_mar:.3f}) for window; skipping metrics."
            )
            return None

        # ---- Face metrics (independent) ----
        perclos = calculate_perclos(
            ear_vals,
            ear_threshold=self.ear_threshold,
            min_consec_frames=self.ear_consec_frames,  # frame-based threshold for EAR
        )
        blink_rate = calculate_blink_frequency(
            ear_vals, ear_threshold=self.ear_threshold, fps=fps_ear
        )

        # Convert MAR consecutive time (seconds) -> frames using MAR fps
        min_consec_mar_frames = max(1, int(round(self.mar_consec_time * fps_mar)))
        yawn_rate = calculate_yawn_frequency(
            mar_vals,
            mar_threshold=self.mar_threshold,
            min_consec_frames=min_consec_mar_frames,  # <-- MAR-derived frames
            fps=fps_mar,
        )

        # ---- Vehicle metrics (independent; as in your original API) ----
        entropy, steering_rate, sdlp = vehicle_feature_extraction(
            steering_vals, lane_vals, self.window_duration
        )

        sampled_ros_images = self.sample_images_as_ros(
            image_samples, window_start_time, end_time
        )

        # ---- Build custom message ----
        out_msg = DrowsinessMetricsData()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.metrics = [
            float(perclos or 0.0),
            float(blink_rate or 0.0),
            float(yawn_rate or 0.0),
            float(entropy or 0.0),
            float(steering_rate or 0.0),
            float(sdlp or 0.0),
        ]

        out_msg.ear_array = ear_vals
        out_msg.mar_array = mar_vals
        out_msg.steering_array = steering_vals
        out_msg.lane_position_array = lane_vals
        out_msg.window_id = self.current_window_id
        # adding images as compressed jpg
        for jpg_bytes in sampled_ros_images:
            ros_img = CompressedImage()
            ros_img.format = "jpeg"
            ros_img.data = jpg_bytes
            out_msg.images.append(ros_img)

        # ---- Publish ----
        self.window_data_pub.publish(out_msg)

        self.get_logger().info(
            f"Window {self.current_window_id} metrics published with {len(sampled_ros_images)} images."
        )


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
