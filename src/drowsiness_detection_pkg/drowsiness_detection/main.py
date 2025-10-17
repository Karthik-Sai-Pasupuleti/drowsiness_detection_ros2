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
    AnnotatorLabels, 
    CombinedAnnotations
)

from drowsiness_detection_msg.srv import StoreLabels

from ament_index_python.packages import get_package_share_directory

from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from drowsiness_detection.camera.utils import (
    calculate_perclos,
    calculate_blink_frequency,
    calculate_yawn_frequency,
    vehicle_feature_extraction,
)

import time
import cv2
import numpy as np
from cv_bridge import CvBridge


import os
import csv
import cv2
import numpy as np
import json
import csv
import os


def save_to_csv(window_id, window_data, labels_dict, driver_id="driver_1"):
    """
    Append metrics and labels for each window to a CSV.
    Dynamically adds new annotator columns when new annotators appear.
    Saves results in: drowsiness_data/<driver_id>/session_metrics.csv
    """
    base_folder = "drowsiness_data"
    driver_folder = os.path.join(base_folder, driver_id)
    os.makedirs(driver_folder, exist_ok=True)

    csv_path = os.path.join(driver_folder, "session_metrics.csv")

    # --- Prepare base metric data ---
    row = {
        "window_id": window_id,
        "video": f"window_{window_id}.mp4",
        "metric_PERCLOS": window_data["metrics"]["PERCLOS"],
        "metric_BlinkRate": window_data["metrics"]["BlinkRate"],
        "metric_YawnRate": window_data["metrics"]["YawnRate"],
        "metric_Entropy": window_data["metrics"]["Entropy"],
        "metric_SteeringRate": window_data["metrics"]["SteeringRate"],
        "metric_SDLP": window_data["metrics"]["SDLP"],
        "raw_ear": str(window_data["raw_data"]["ear"]),
        "raw_mar": str(window_data["raw_data"]["mar"]),
        "raw_steering": str(window_data["raw_data"]["steering"]),
        "raw_lane": str(window_data["raw_data"]["lane"]),
    }

    # --- Add labels dynamically (per annotator, with separate action flags) ---
    for annotator, lbl in labels_dict.items():
        prefix = annotator.replace(" ", "_")  # safe column prefix
        row[f"{prefix}_drowsiness_level"] = lbl.get("drowsiness_level", "")
        row[f"{prefix}_notes"] = lbl.get("notes", "")
        row[f"{prefix}_voice_feedback"] = lbl.get("voice_feedback", "")
        row[f"{prefix}_submission_type"] = lbl.get("submission_type", "")
        row[f"{prefix}_auto_submitted"] = lbl.get("auto_submitted", "")
        row[f"{prefix}_is_flagged"] = lbl.get("is_flagged", "")

        # Separate action flags (booleans/ints)
        row[f"{prefix}_action_fan"] = int(bool(lbl.get("action_fan", False)))
        row[f"{prefix}_action_voice_command"] = int(bool(lbl.get("action_voice_command", False)))
        row[f"{prefix}_action_steering_vibration"] = int(bool(lbl.get("action_steering_vibration", False)))
        row[f"{prefix}_action_save_video"] = int(bool(lbl.get("action_save_video", False)))

    # --- Dynamic column management ---
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"[CSV] Created new session file: {csv_path}")
        print(f"[CSV] Added annotators: {list(labels_dict.keys())}")
    else:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)

        new_fields = [f for f in row.keys() if f not in existing_fields]
        if new_fields:
            print(f"[CSV] New annotator columns detected: {new_fields}")
            all_fields = existing_fields + new_fields
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields)
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(row)
        else:
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=existing_fields)
                writer.writerow(row)

        print(f"[CSV] Saved window {window_id} (annotators={list(labels_dict.keys())})")


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
        self.create_subscription(
            CombinedAnnotations,
            "/driver_assistance/combined_annotations",
            self.combined_annotations_callback,
            10,
        )
        
        
        self.store_labels_srv = self.create_service(
            StoreLabels,
            "store_labels",
            self.handle_store_labels,
        )
        self.get_logger().info("Service 'store_labels' ready to receive labels.")
        
        self.combined_annotations = {}
        
        
        self.video_writer = None
        self.current_video_path = None
        self.video_base_dir = os.path.join("drowsiness_data", "driver_1", "videos")
        os.makedirs(self.video_base_dir, exist_ok=True)

     
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
        
        
    def combined_annotations_callback(self, msg: CombinedAnnotations):
        """Store CombinedAnnotations received from the Flask backend."""
        with self.buffer_lock:
            self.combined_annotations[msg.window_id] = msg
        self.get_logger().info(
            f"Received CombinedAnnotations for window {msg.window_id} "
            f"({len(msg.annotator_labels)} annotators, flagged={msg.is_flagged})"
        )
    
    def handle_store_labels(self, request, response):
        """Store labels received via service and merge with metrics if ready."""
        window_id = request.window_id
        self.get_logger().info(
            f"Service received {len(request.annotator_labels)} labels for window {window_id}"
        )

        combined = CombinedAnnotations()
        combined.window_id = window_id
        combined.annotator_labels = list(request.annotator_labels)
        combined.is_flagged = False

        with self.buffer_lock:
            self.combined_annotations[window_id] = combined

        # Try merge immediately if metrics already ready (conflict/video decision happens there)
        self.try_merge_and_save(window_id)

        response.success = True
        response.message = f"Stored labels for window {window_id}"
        return response

        
    def save_video_segment(self, window_id, output_path):
        """
        Save frames corresponding to the given window from the image_buffer to a video file.
        """
        try:
            # Determine window time range
            window_end_time = self.last_window_end_time
            window_start_time = window_end_time - self.window_duration

            # Extract frames belonging to that window
            with self.buffer_lock:
                frames = [img for (ts, img) in self.image_buffer if window_start_time <= ts < window_end_time]

            if not frames:
                self.get_logger().warn(f"[VIDEO] No frames found for window {window_id}, skipping save.")
                return

            height, width, _ = frames[0].shape
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()

            self.get_logger().info(f"[VIDEO] Segment saved for window {window_id}: {output_path}")

        except Exception as e:
            self.get_logger().error(f"[VIDEO ERROR] Failed to save video for window {window_id}: {e}")



    def try_merge_and_save(self, window_id: int):
        """If both metrics and labels for a window exist, merge and write to CSV; conditionally save video."""
        with self.buffer_lock:
            if not hasattr(self, "pending_metrics"):
                self.pending_metrics = {}
            if (
                window_id not in self.pending_metrics
                or window_id not in self.combined_annotations
            ):
                return  # not ready yet

            window_data = self.pending_metrics.pop(window_id)
            combined = self.combined_annotations.pop(window_id)

        # --- Build per-annotator label dictionary ---
        labels_dict = {}
        drowsiness_levels = []
        save_video_requested = False

        for ann in combined.annotator_labels:
            drowsiness_levels.append(ann.drowsiness_level or "")
            if ann.action_save_video:
                save_video_requested = True

            labels_dict[ann.annotator_name] = {
                "drowsiness_level": ann.drowsiness_level,
                "notes": ann.notes,
                "voice_feedback": ann.voice_feedback,
                "submission_type": ann.submission_type,
                "auto_submitted": ann.auto_submitted,
                "is_flagged": ann.is_flagged,
                # Separate action flags
                "action_fan": ann.action_fan,
                "action_voice_command": ann.action_voice_command,
                "action_steering_vibration": ann.action_steering_vibration,
                "action_save_video": ann.action_save_video,
            }

        # --- Conflict detection ---
        conflict = len({lvl for lvl in drowsiness_levels if lvl}) > 1

        # --- Always save CSV ---
        save_to_csv(window_id=window_id, window_data=window_data, labels_dict=labels_dict, driver_id="driver_1")

        # --- Conditional video saving (conflict OR any annotator clicked Save Video) ---
        if conflict or save_video_requested:
            # Save video inside same driver folder as CSV
            base_folder = "drowsiness_data"
            driver_folder = os.path.join(base_folder, "driver_1", "videos")
            os.makedirs(driver_folder, exist_ok=True)
            video_filename = os.path.join(driver_folder, f"window_{window_id}.mp4")

            try:
                self.save_video_segment(window_id, video_filename)
                reason = "conflict" if conflict else "manual request"
                self.get_logger().warn(f"[VIDEO] Saved video for window {window_id} (reason: {reason})")
            except Exception as e:
                self.get_logger().error(f"[VIDEO ERROR] Failed to save video for window {window_id}: {e}")

        self.get_logger().info(
            f"[MERGE] Saved window {window_id} with metrics + {len(labels_dict)} annotator labels."
        )


    def cb_camera(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
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
        """Triggered when a 60s window ends — compute metrics and store for later merge."""
        window_start_time = self.last_window_end_time
        end_time = window_start_time + self.window_duration
        window_id = self.current_window_id

        self.get_logger().info(f"Window {window_id} complete. Computing metrics...")

        window_data = self.compute_metrics(window_start_time)
        if window_data is None:
            self.get_logger().warn(f"Skipping save for window {window_id}: no valid data.")
        else:
            # Store metrics until matching labels arrive
            with self.buffer_lock:
                if not hasattr(self, "pending_metrics"):
                    self.pending_metrics = {}
                self.pending_metrics[window_id] = window_data

            # Try to save if labels already arrived
            self.try_merge_and_save(window_id)

        # ---- Prune old buffers ----
        with self.buffer_lock:
            self.ear_buffer = deque([(t, v) for t, v in self.ear_buffer if t >= end_time])
            self.mar_buffer = deque([(t, v) for t, v in self.mar_buffer if t >= end_time])
            self.steering_buffer = deque([(t, v) for t, v in self.steering_buffer if t >= end_time])
            self.lane_offset_buffer = deque([(t, v) for t, v in self.lane_offset_buffer if t >= end_time])
            self.image_buffer = deque([(t, img) for t, img in self.image_buffer if t >= end_time])

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
        steering_vals_fps = len(steering_vals)
        lane_vals_fps = len(lane_vals)

        self.get_logger().info(f"Window {self.current_window_id} - EAR FPS: {fps_ear:.2f}, MAR FPS: {fps_mar:.2f}, Steering FPS: {steering_vals_fps:.2f}, Lane FPS: {lane_vals_fps:.2f}")
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


        self.get_logger().info(
            f"Window {self.current_window_id} metrics published with {len(sampled_ros_images)} images."
        )
        
        metrics = {
            "PERCLOS": float(perclos or 0.0),
            "BlinkRate": float(blink_rate or 0.0),
            "YawnRate": float(yawn_rate or 0.0),
            "Entropy": float(entropy or 0.0),
            "SteeringRate": float(steering_rate or 0.0),
            "SDLP": float(sdlp or 0.0),
        }

        raw_data = {
            "ear": ear_vals,
            "mar": mar_vals,
            "steering": steering_vals,
            "lane": lane_vals,
        }

        window_data = {
            "metrics": metrics,
            "raw_data": raw_data,
            "images": sampled_ros_images,
        }

        return window_data



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