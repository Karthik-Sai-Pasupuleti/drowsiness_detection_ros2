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
# CSV save utility (UPSERT by window_id)
# -------------------------------------------------------------------------
def upsert_row(window_id, window_data, labels_dict, driver_id="driver_1"):
    """
    Always write one row per window_id.
    If labels arrive later, they update the same row.
    """
    base_folder = os.environ.get("DATA_DIR", "drowsiness_data")
    driver_folder = os.path.join(base_folder, driver_id)
    os.makedirs(driver_folder, exist_ok=True)
    csv_path = os.path.join(driver_folder, "session_metrics.csv")

    # Base row from metrics (may be None if metrics missing)
    if window_data is None:
        # minimal row if we somehow have labels but no metrics
        row = {
            "window_id": window_id,
            "initial_timestamp": 0.0,
            "video": f"window_{window_id}.mp4",
            "metric_PERCLOS": 0.0,
            "metric_BlinkRate": 0.0,
            "metric_YawnRate": 0.0,
            "metric_Entropy": 0.0,
            "metric_SteeringRate": 0.0,
            "metric_SDLP": 0.0,
            "raw_ear": str([]),
            "raw_mar": str([]),
            "raw_steering": str([]),
            "raw_lane": str([]),
            "raw_hr": str([]),
            "smooth_bpm": str([]),
            "raw_ppg": str([]),
        }
    else:
        row = {
            "window_id": window_id,
            "initial_timestamp": window_data.get("timestamp", 0.0),
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
            "raw_hr": str(window_data["raw_data"]["hr"]),
            "smooth_bpm": str(window_data["raw_data"].get("smooth_bpm", [])),
            "raw_ppg": str(window_data["raw_data"]["ppg"]),
        }

    # Merge labels into row (if any)
    for annotator, lbl in labels_dict.items():
        prefix = annotator.replace(" ", "_")
        row[f"{prefix}_drowsiness_level"] = lbl.get("drowsiness_level", "")
        row[f"{prefix}_notes"] = lbl.get("notes", "")
        row[f"{prefix}_voice_feedback"] = lbl.get("voice_feedback", "")
        row[f"{prefix}_submission_type"] = lbl.get("submission_type", "")
        row[f"{prefix}_instant_bpm"] = lbl.get("instant_bpm", 0.0)
        row[f"{prefix}_smooth_bpm"] = lbl.get("smooth_bpm", 0.0)
        row[f"{prefix}_action_fan"] = int(bool(lbl.get("action_fan", False)))
        row[f"{prefix}_action_voice_command"] = int(bool(lbl.get("action_voice_command", False)))
        row[f"{prefix}_action_steering_vibration"] = int(bool(lbl.get("action_steering_vibration", False)))
        row[f"{prefix}_action_save_video"] = int(bool(lbl.get("action_save_video", False)))

    # New file: write header + row
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
            f.flush()
        return

    # Existing file: read all, upsert by window_id, merge columns
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    # Merge fieldnames
    all_fields = list(existing_fields)
    for k in row.keys():
        if k not in all_fields:
            all_fields.append(k)

    # Find row with same window_id
    w_id_str = str(window_id)
    updated = False
    for r in existing_rows:
        if str(r.get("window_id", "")) == w_id_str:
            r.update(row)
            updated = True
            break

    if not updated:
        new_row = {k: "" for k in all_fields}
        new_row.update(row)
        existing_rows.append(new_row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for r in existing_rows:
            out = {k: r.get(k, "") for k in all_fields}
            writer.writerow(out)
        f.flush()


# -------------------------------------------------------------------------
# Main node
# -------------------------------------------------------------------------
class DriverAssistanceNode(Node):
    def __init__(self):
        super().__init__("driver_assistance_node")

        # declare driver_id
        self.declare_parameter("driver_id", "test_driver")
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

        # Heart Rate & PPG
        self.hr_buffer = deque(maxlen=2000)          # instant_bpm (index 0)
        self.smooth_hr_buffer = deque(maxlen=2000)   # smooth_bpm (index 1)
        self.ppg_buffer = deque(maxlen=20000)        # raw PPG

        self.buffer_lock = threading.Lock()

        # Window timing
        self.window_duration = 60.0
        self.label_collection_time = 10.0
        self.current_window_id = 0
        self.last_window_end_time = self.get_clock().now().nanoseconds / 1e9

        # Guard: avoid multiple triggers for the same window
        self._window_processing = False

        # Video handling
        self.bridge = CvBridge()
        self.video_writer = None
        self.current_video_path = None
        self.finished_video_paths = {}
        data_dir = os.environ.get("DATA_DIR", "drowsiness_data")
        self.video_base_dir = os.path.join(data_dir, self.driver_id, "videos")
        os.makedirs(self.video_base_dir, exist_ok=True)

        # ROS I/O
        self.create_subscription(EarMarValue, "/ear_mar", self.ear_mar_callback, qos_profile_sensor_data)
        self.create_subscription(CarlaEgoVehicleControl, "/carla/ego/vehicle_control_cmd", self.steering_callback, 10)
        self.create_subscription(LanePosition, "/carla/lane_offset", self.lane_offset_callback, qos_profile_sensor_data)
        self.create_subscription(Image, "/flir_camera/image_raw", self.cb_camera, qos_profile_sensor_data)
        self.create_subscription(CombinedAnnotations, "/driver_assistance/combined_annotations", self.combined_annotations_callback, 10)

        self.create_subscription(Float32MultiArray, "/heart_rate_bpm", self.hr_callback, 10)
        self.create_subscription(Float32MultiArray, "/raw_ppg", self.ppg_callback, 10)

        self.store_labels_srv = self.create_service(StoreLabels, "store_labels", self.handle_store_labels)
        self.window_phase_pub = self.create_publisher(Float32MultiArray, "/driver_assistance/window_phase", 10)
        self.create_timer(0.1, self.update_window_phase)

        # For labels & metrics
        self.combined_annotations = {}
        self.pending_metrics = {}

        self.get_logger().info("Driver Assistance Node started.")
        self.start_new_video_writer()

    # ---------------------------------------------------------------------
    # Video handling
    # ---------------------------------------------------------------------
    def start_new_video_writer(self):
        filename = f"window_{self.current_window_id}.mp4"
        self.current_video_path = os.path.join(self.video_base_dir, filename)
        self.video_writer = None
        self.get_logger().info(f"[VIDEO] Prepared recording for {self.current_video_path}")

    def _release_writer_only(self):
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        except Exception as e:
            self.get_logger().error(f"Video release error: {e}")

    # ---------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------
    def cb_camera(self, msg: Image):
        try:
            # Use BGR so OpenCV writes correct colors
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            h, w = cv_img.shape[:2]

            if self.video_writer is None and self.current_video_path:
                self.video_writer = cv2.VideoWriter(
                    self.current_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (w, h),
                )
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

    def hr_callback(self, msg: Float32MultiArray):
        ts = self.get_clock().now().nanoseconds / 1e9
        if len(msg.data) >= 1:
            instant_bpm = float(msg.data[0])
            with self.buffer_lock:
                self.hr_buffer.append((ts, instant_bpm))
        if len(msg.data) >= 2:
            smooth_bpm = float(msg.data[1])
            with self.buffer_lock:
                self.smooth_hr_buffer.append((ts, smooth_bpm))

    def ppg_callback(self, msg: Float32MultiArray):
        ts = self.get_clock().now().nanoseconds / 1e9
        if msg.data:
            with self.buffer_lock:
                for val in msg.data:
                    self.ppg_buffer.append((ts, float(val)))

    def combined_annotations_callback(self, msg: CombinedAnnotations):
        with self.buffer_lock:
            self.combined_annotations[msg.window_id] = msg
        # If metrics already exist, update CSV row immediately
        self._try_upsert_labels_only(msg.window_id)

    def handle_store_labels(self, request, response):
        window_id = request.window_id
        combined = CombinedAnnotations()
        combined.window_id = window_id
        combined.annotator_labels = list(request.annotator_labels)
        combined.is_flagged = False
        with self.buffer_lock:
            self.combined_annotations[window_id] = combined
        self._try_upsert_labels_only(window_id)
        response.success = True
        response.message = f"Stored labels for window {window_id}"
        return response

    # ---------------------------------------------------------------------
    # Window lifecycle
    # ---------------------------------------------------------------------
    def update_window_phase(self):
        now_ros = self.get_clock().now().nanoseconds / 1e9
        time_in_current_window = now_ros - self.last_window_end_time
        remaining_time = max(0.0, self.window_duration - time_in_current_window)
        phase = 0 if remaining_time > self.label_collection_time else 1

        msg = Float32MultiArray()
        msg.data = [float(phase), float(self.current_window_id), remaining_time]
        self.window_phase_pub.publish(msg)

        # Guard: process only once per window
        if remaining_time <= 0.0 and not self._window_processing:
            self._window_processing = True
            self.process_completed_window()

    def process_completed_window(self):
        window_start_time = self.last_window_end_time
        window_id = self.current_window_id
        self.get_logger().info(f"Window {window_id} complete. Computing metrics (async)...")

        finished_path = self.current_video_path
        self._release_writer_only()
        if finished_path:
            self.finished_video_paths[window_id] = finished_path

        self.current_window_id += 1
        self.last_window_end_time += self.window_duration
        self._window_processing = False
        self.start_new_video_writer()

        threading.Thread(
            target=self._async_compute_and_save,
            args=(window_id, window_start_time),
            daemon=True,
        ).start()

    def _async_compute_and_save(self, window_id, window_start_time):
        try:
            window_data = self.compute_metrics(window_start_time)
            with self.buffer_lock:
                # Store metrics so labels arriving later can update same row
                self.pending_metrics[window_id] = window_data

                # Build labels_dict if labels already exist
                combined = self.combined_annotations.get(window_id)
                labels_dict = self._build_labels_dict(combined) if combined else {}

            # ALWAYS write a row (even if labels_dict is empty)
            upsert_row(window_id, window_data, labels_dict, driver_id=self.driver_id)
            self.get_logger().info(f"[ASYNC] Window {window_id} metrics saved to CSV.")

        except Exception as e:
            self.get_logger().error(f"[ASYNC] Metric computation failed for window {window_id}: {e}")

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------
    def compute_metrics(self, window_start_time):
        end_time = window_start_time + self.window_duration

        def robust_fps(ts):
            if len(ts) < 2:
                return 0.0
            dts = [ts[i+1] - ts[i] for i in range(len(ts)-1) if ts[i+1] > ts[i]]
            if not dts:
                return 0.0
            dts.sort()
            med_dt = dts[len(dts)//2]
            return 1.0/med_dt if med_dt > 0 else 0.0

        with self.buffer_lock:
            ear_samples = [(t, v) for t, v in self.ear_buffer if window_start_time <= t < end_time]
            mar_samples = [(t, v) for t, v in self.mar_buffer if window_start_time <= t < end_time]
            steering_samples = [(t, v) for t, v in self.steering_buffer if window_start_time <= t < end_time]
            lane_samples = [(t, v) for t, v in self.lane_offset_buffer if window_start_time <= t < end_time]
            hr_samples = [(t, v) for t, v in self.hr_buffer if window_start_time <= t < end_time]
            smooth_hr_samples = [(t, v) for t, v in self.smooth_hr_buffer if window_start_time <= t < end_time]
            ppg_samples = [(t, v) for t, v in self.ppg_buffer if window_start_time <= t < end_time]

        if not ear_samples:
            # Still return a valid structure; metrics will be zero
            metrics = {
                "PERCLOS": 0.0,
                "BlinkRate": 0.0,
                "YawnRate": 0.0,
                "Entropy": 0.0,
                "SteeringRate": 0.0,
                "SDLP": 0.0,
            }
            raw_data = {
                "ear": [],
                "mar": [],
                "steering": [],
                "lane": [],
                "hr": [],
                "smooth_bpm": [],
                "ppg": [],
            }
            return {"metrics": metrics, "raw_data": raw_data, "timestamp": window_start_time}

        ear_ts, ear_vals = zip(*ear_samples)
        mar_ts, mar_vals = zip(*mar_samples) if mar_samples else ([], [])
        _, steering_vals = zip(*steering_samples) if steering_samples else ([], [])
        _, lane_vals = zip(*lane_samples) if lane_samples else ([], [])

        if hr_samples:
            _, hr_vals = zip(*hr_samples)
            hr_vals = list(map(float, hr_vals))
        else:
            hr_vals = []

        if smooth_hr_samples:
            _, smooth_hr_vals = zip(*smooth_hr_samples)
            smooth_hr_vals = list(map(float, smooth_hr_vals))
        else:
            smooth_hr_vals = []

        if ppg_samples:
            _, ppg_vals = zip(*ppg_samples)
            ppg_vals = list(map(float, ppg_vals))
        else:
            ppg_vals = []

        fps_ear = robust_fps(list(ear_ts))
        if fps_ear <= 0:
            fps_ear = 30.0

        perclos = calculate_perclos(list(ear_vals), self.ear_threshold, self.ear_consec_frames)
        blink_rate = calculate_blink_frequency(list(ear_vals), self.ear_threshold, fps=fps_ear)
        entropy, steering_rate, sdlp = vehicle_feature_extraction(list(steering_vals), list(lane_vals), self.window_duration)

        yawn_rate = 0.0

        metrics = {
            "PERCLOS": float(perclos or 0.0),
            "BlinkRate": float(blink_rate or 0.0),
            "YawnRate": float(yawn_rate),
            "Entropy": float(entropy or 0.0),
            "SteeringRate": float(steering_rate or 0.0),
            "SDLP": float(sdlp or 0.0),
        }

        raw_data = {
            "ear": list(map(float, ear_vals)),
            "mar": list(map(float, mar_vals)),
            "steering": list(map(float, steering_vals)),
            "lane": list(map(float, lane_vals)),
            "hr": hr_vals,
            "smooth_bpm": smooth_hr_vals,
            "ppg": ppg_vals,
        }
        return {"metrics": metrics, "raw_data": raw_data, "timestamp": window_start_time}

    # ---------------------------------------------------------------------
    # Label merge helpers
    # ---------------------------------------------------------------------
    def _build_labels_dict(self, combined: CombinedAnnotations):
        if combined is None:
            return {}
        labels_dict = {}
        for ann in combined.annotator_labels:
            labels_dict[ann.annotator_name] = {
                "drowsiness_level": ann.drowsiness_level,
                "notes": ann.notes,
                "voice_feedback": ann.voice_feedback,
                "submission_type": ann.submission_type,
                "instant_bpm": getattr(ann, 'instant_bpm', 0.0),
                "smooth_bpm": getattr(ann, 'smooth_bpm', 0.0),
                "action_fan": ann.action_fan,
                "action_voice_command": ann.action_voice_command,
                "action_steering_vibration": ann.action_steering_vibration,
                "action_save_video": ann.action_save_video,
            }
        return labels_dict

    def _try_upsert_labels_only(self, window_id):
        with self.buffer_lock:
            combined = self.combined_annotations.get(window_id)
            window_data = self.pending_metrics.get(window_id, None)
        if combined is None:
            return
        labels_dict = self._build_labels_dict(combined)
        upsert_row(window_id, window_data, labels_dict, driver_id=self.driver_id)
        self.get_logger().info(f"[LABEL] Window {window_id} labels upserted into CSV.")

    def destroy_node(self):
        self._release_writer_only()
        super().destroy_node()


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
