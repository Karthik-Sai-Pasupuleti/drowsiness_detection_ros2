#!/usr/bin/env python3
import os
import csv
import cv2
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from carla_msgs.msg import CarlaEgoVehicleControl
from drowsiness_detection_msg.msg import (
    EarMarValue,
    LanePosition,
    CombinedAnnotations,
)
from drowsiness_detection_msg.srv import StoreLabels
from drowsiness_detection.camera.utils import (
    calculate_perclos,
    calculate_blink_frequency,
    calculate_yawn_frequency,
    vehicle_feature_extraction,
)


# -------------------------------------------------------------------------
# CSV save utility
# -------------------------------------------------------------------------
def save_to_csv(window_id, window_data, labels_dict, driver_id="driver_1"):
    """Append metrics and labels for each window to a CSV file."""
    base_folder = "drowsiness_data"
    driver_folder = os.path.join(base_folder, driver_id)
    os.makedirs(driver_folder, exist_ok=True)
    csv_path = os.path.join(driver_folder, "session_metrics.csv")

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

    for annotator, lbl in labels_dict.items():
        prefix = annotator.replace(" ", "_")
        row[f"{prefix}_drowsiness_level"] = lbl.get("drowsiness_level", "")
        row[f"{prefix}_notes"] = lbl.get("notes", "")
        row[f"{prefix}_voice_feedback"] = lbl.get("voice_feedback", "")
        row[f"{prefix}_submission_type"] = lbl.get("submission_type", "")
        row[f"{prefix}_action_fan"] = int(bool(lbl.get("action_fan", False)))
        row[f"{prefix}_action_voice_command"] = int(bool(lbl.get("action_voice_command", False)))
        row[f"{prefix}_action_steering_vibration"] = int(bool(lbl.get("action_steering_vibration", False)))
        row[f"{prefix}_action_save_video"] = int(bool(lbl.get("action_save_video", False)))

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    new_fields = [f for f in row.keys() if f not in existing_fields]
    if new_fields:
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


# -------------------------------------------------------------------------
# Main node
# -------------------------------------------------------------------------
class DriverAssistanceNode(Node):
    def __init__(self):
        super().__init__("driver_assistance_node")

        # declare driver_id
        self.declare_parameter("driver_id", "maria")
        self.driver_id = self.get_parameter("driver_id").value
        
        # Parameters
        self.declare_parameter("EAR_Threshold", 0.2)
        self.declare_parameter("MAR_Threshold", 0.4)
        self.declare_parameter("blink_threshold", 3)
        self.declare_parameter("Yawning_threshold", 4)
        self.ear_threshold = self.get_parameter("EAR_Threshold").value
        self.mar_threshold = self.get_parameter("MAR_Threshold").value
        self.ear_consec_frames = self.get_parameter("blink_threshold").value
        self.mar_consec_time = self.get_parameter("Yawning_threshold").value

        # Data buffers
        self.ear_buffer = deque(maxlen=2000)
        self.mar_buffer = deque(maxlen=2000)
        self.steering_buffer = deque(maxlen=2000)
        self.lane_offset_buffer = deque(maxlen=2000)
        self.buffer_lock = threading.Lock()

        # Window timing
        self.window_duration = 60.0
        self.label_collection_time = 10.0
        self.current_window_id = 0
        self.last_window_end_time = self.get_clock().now().nanoseconds / 1e9

        # Video handling
        self.bridge = CvBridge()
        self.video_writer = None
        self.current_video_path = None
        self.finished_video_paths = {}  # window_id -> video path
        self.video_base_dir = os.path.join("drowsiness_data", self.driver_id, "videos")
        os.makedirs(self.video_base_dir, exist_ok=True)

        # ROS I/O
        self.create_subscription(EarMarValue, "/ear_mar", self.ear_mar_callback, qos_profile_sensor_data)
        self.create_subscription(CarlaEgoVehicleControl, "/carla/hero/vehicle_control_cmd", self.steering_callback, 10)
        self.create_subscription(LanePosition, "/carla/lane_offset", self.lane_offset_callback, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/image_raw", self.cb_camera, qos_profile_sensor_data)
        self.create_subscription(CombinedAnnotations, "/driver_assistance/combined_annotations", self.combined_annotations_callback, 10)
        self.store_labels_srv = self.create_service(StoreLabels, "store_labels", self.handle_store_labels)
        self.window_phase_pub = self.create_publisher(Float32MultiArray, "/driver_assistance/window_phase", 10)
        self.create_timer(0.1, self.update_window_phase)

        self.combined_annotations = {}  # window_id -> CombinedAnnotations
        self.pending_metrics = {}       # window_id -> computed window data

        self.get_logger().info("Driver Assistance Node started.")

        # Prepare first window recording (writer starts on first frame)
        self.start_new_video_writer()

    # ---------------------------------------------------------------------
    # Video handling
    # ---------------------------------------------------------------------
    def start_new_video_writer(self):
        """Prepare a new video file for the current window. The writer will be created on the first frame."""
        filename = f"window_{self.current_window_id}.mp4"
        self.current_video_path = os.path.join(self.video_base_dir, filename)
        self.video_writer = None
        self.get_logger().info(f"[VIDEO] Prepared recording for {self.current_video_path}")

    def _release_writer_only(self):
        """Release the current writer without deleting any file and without clearing path."""
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        except Exception as e:
            self.get_logger().error(f"Video release error: {e}")

    # ---------------------------------------------------------------------
    # Subscribers and service
    # ---------------------------------------------------------------------
    def cb_camera(self, msg: Image):
        """Start the writer on the first frame (to match resolution), then write frames continuously."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            if self.video_writer is None and self.current_video_path:
                h, w = cv_img.shape[:2]
                self.video_writer = cv2.VideoWriter(
                    self.current_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (w, h),
                )
                self.get_logger().info(f"[VIDEO] Recording started: {self.current_video_path} ({w}x{h})")
            if self.video_writer:
                self.video_writer.write(cv_img)
        except Exception as e:
            self.get_logger().error(f"Camera callback error: {e}")

    def ear_mar_callback(self, msg: EarMarValue):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.buffer_lock:
            self.ear_buffer.append((ts, float(msg.ear_value)))
            self.mar_buffer.append((ts, float(msg.mar_value)))

    def steering_callback(self, msg: CarlaEgoVehicleControl):
        ts = self.get_clock().now().nanoseconds / 1e9
        with self.buffer_lock:
            self.steering_buffer.append((ts, float(msg.steer)))

    def lane_offset_callback(self, msg: LanePosition):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.buffer_lock:
            self.lane_offset_buffer.append((ts, float(msg.lane_offset)))

    def combined_annotations_callback(self, msg: CombinedAnnotations):
        with self.buffer_lock:
            self.combined_annotations[msg.window_id] = msg

    def handle_store_labels(self, request, response):
        window_id = request.window_id
        combined = CombinedAnnotations()
        combined.window_id = window_id
        combined.annotator_labels = list(request.annotator_labels)
        combined.is_flagged = False
        with self.buffer_lock:
            self.combined_annotations[window_id] = combined
        self.try_merge_and_save(window_id)
        response.success = True
        response.message = f"Stored labels for window {window_id}"
        return response

    # ---------------------------------------------------------------------
    # Window lifecycle
    # ---------------------------------------------------------------------
    def update_window_phase(self):
        """Publish phase and remaining time; trigger window rollover at 0."""
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
        On window end:
          1) Finalize the current window's video file (release writer).
          2) Advance window bookkeeping and immediately prepare next window recording.
          3) Launch metrics computation in a background thread to keep UI responsive.
        """
        window_start_time = self.last_window_end_time
        window_id = self.current_window_id

        self.get_logger().info(f"Window {window_id} complete. Computing metrics (async)...")

        # Finalize current window video
        finished_path = self.current_video_path
        self._release_writer_only()
        if finished_path:
            self.finished_video_paths[window_id] = finished_path
            self.get_logger().info(f"[VIDEO] Finalized file for window {window_id}: {finished_path}")

        # Advance window and start next recording immediately
        self.current_window_id += 1
        self.last_window_end_time += self.window_duration
        self.start_new_video_writer()

        # Compute metrics asynchronously to avoid blocking the timer/UI
        threading.Thread(
            target=self._async_compute_and_merge,
            args=(window_id, window_start_time),
            daemon=True,
        ).start()

    def _async_compute_and_merge(self, window_id, window_start_time):
        """Compute metrics in a background thread and trigger merge/save when ready."""
        try:
            window_data = self.compute_metrics(window_start_time)
            if window_data:
                with self.buffer_lock:
                    self.pending_metrics[window_id] = window_data
                self.try_merge_and_save(window_id)
            else:
                self.get_logger().warn(f"[ASYNC] No valid metrics computed for window {window_id}.")
        except Exception as e:
            self.get_logger().error(f"[ASYNC] Metric computation failed for window {window_id}: {e}")

    # ---------------------------------------------------------------------
    # Metrics and merge
    # ---------------------------------------------------------------------
    def compute_metrics(self, window_start_time):
        end_time = window_start_time + self.window_duration

        def robust_fps(ts):
            if len(ts) < 2:
                return 0.0
            dts = [ts[i + 1] - ts[i] for i in range(len(ts) - 1) if ts[i + 1] > ts[i]] # calculate deltas
            if not dts:
                return 0.0
            dts.sort()
            med_dt = dts[len(dts) // 2] # calculate median
            return 1.0 / med_dt if med_dt > 0 else 0.0

        with self.buffer_lock:
            ear_samples = [(t, v) for t, v in self.ear_buffer if window_start_time <= t < end_time]
            mar_samples = [(t, v) for t, v in self.mar_buffer if window_start_time <= t < end_time]
            steering_samples = [(t, v) for t, v in self.steering_buffer if window_start_time <= t < end_time]
            lane_samples = [(t, v) for t, v in self.lane_offset_buffer if window_start_time <= t < end_time]

        if not ear_samples or not mar_samples or not steering_samples or not lane_samples:
            self.get_logger().warn(f"Insufficient data for window {self.current_window_id}")
            return None

        ear_ts, ear_vals = zip(*ear_samples)
        mar_ts, mar_vals = zip(*mar_samples)
        _, steering_vals = zip(*steering_samples)
        _, lane_vals = zip(*lane_samples)

        fps_ear = robust_fps(list(ear_ts))
        fps_mar = robust_fps(list(mar_ts))
        if fps_ear <= 0 or fps_mar <= 0:
            return None

        perclos = calculate_perclos(
            list(ear_vals),
            ear_threshold=self.ear_threshold,
            min_consec_frames=self.ear_consec_frames,
        )
        blink_rate = calculate_blink_frequency(
            list(ear_vals),
            ear_threshold=self.ear_threshold,
            fps=fps_ear,
        )
        min_consec_mar_frames = max(1, int(round(self.mar_consec_time * fps_mar)))
        yawn_rate = calculate_yawn_frequency(
            list(mar_vals),
            mar_threshold=self.mar_threshold,
            min_consec_frames=min_consec_mar_frames,
            fps=fps_mar,
        )
        entropy, steering_rate, sdlp = vehicle_feature_extraction(
            list(steering_vals), list(lane_vals), self.window_duration
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
            "ear": list(map(float, ear_vals)),
            "mar": list(map(float, mar_vals)),
            "steering": list(map(float, steering_vals)),
            "lane": list(map(float, lane_vals)),
        }
        return {"metrics": metrics, "raw_data": raw_data}

    def try_merge_and_save(self, window_id):
        with self.buffer_lock:
            if window_id not in self.pending_metrics or window_id not in self.combined_annotations:
                return
            window_data = self.pending_metrics.pop(window_id)
            combined = self.combined_annotations.pop(window_id)

        labels_dict, drowsiness_levels, save_video_requested = {}, [], False
        for ann in combined.annotator_labels:
            drowsiness_levels.append(ann.drowsiness_level or "")
            if ann.action_save_video:
                save_video_requested = True
            labels_dict[ann.annotator_name] = {
                "drowsiness_level": ann.drowsiness_level,
                "notes": ann.notes,
                "voice_feedback": ann.voice_feedback,
                "submission_type": ann.submission_type,
                # "auto_submitted": ann.auto_submitted,
                # "is_flagged": ann.is_flagged,
                "action_fan": ann.action_fan,
                "action_voice_command": ann.action_voice_command,
                "action_steering_vibration": ann.action_steering_vibration,
                "action_save_video": ann.action_save_video,
            }

        # Conflict if annotators chose different non-empty levels
        conflict = len({lvl for lvl in drowsiness_levels if lvl}) > 1

        # Persist metrics and labels
        save_to_csv(window_id, window_data, labels_dict, driver_id=self.driver_id)

        # Decide video retention and explain why
        keep_video = conflict or save_video_requested
        if keep_video:
            if conflict and save_video_requested:
                reason = "conflict between annotators and annotator save request"
            elif conflict:
                reason = "conflict between annotators"
            else:
                reason = "annotator save request"
            self.get_logger().info(f"[VIDEO] Keeping video for window {window_id} due to {reason}.")
        else:
            reason = "no conflict and no save request"
            self.get_logger().info(f"[VIDEO] Deleting video for window {window_id} ({reason}).")

        # Finalize the correct file for this window
        path = self.finished_video_paths.pop(window_id, None)
        if path:
            try:
                if keep_video:
                    self.get_logger().info(f"Kept video: {path}")
                else:
                    if os.path.exists(path):
                        os.remove(path)
                        self.get_logger().info(f"Deleted video: {path}")
            except Exception as e:
                self.get_logger().error(f"Video finalize error for window {window_id}: {e}")
        else:
            self.get_logger().warn(f"[VIDEO] No recorded file path found for window {window_id}. Nothing to keep/delete.")

        self.get_logger().info(f"[MERGE] Window {window_id} processed. Video kept: {keep_video} ({reason}).")


# -------------------------------------------------------------------------
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
