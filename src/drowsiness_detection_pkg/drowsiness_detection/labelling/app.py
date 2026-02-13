#!/usr/bin/env python3

"""
This version implements a web application for real-time drowsiness detection labelling.

It uses Flask for the web interface and ROS2 for data communication.

The application allows multiple annotators to label drowsiness levels and actions,
with support for auto-submission of previous labels if no new input is provided.
It also includes tracking for the 'Save Video' action.

CHANGELOG V3.5:
- Integrated Heart Rate data handling (subscription to /heart_rate_bpm).
- Added Raw PPG data handling (subscription to /raw_ppg).
- DataManager updated to store BPM (instant + smooth) and Raw PPG signal.
"""

from waitress import serve
import os
import time
import threading
from collections import deque
from copy import deepcopy
import json
from flask import Flask, render_template, Response, request, jsonify
import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

# Import custom messages
from drowsiness_detection_msg.msg import (
    EarMarValue,
    LanePosition,
    DrowsinessMetricsData,
    Vibration,
    CombinedAnnotations,
    AnnotatorLabels,
)

from drowsiness_detection_msg.srv import StoreLabels
from std_msgs.msg import Float32MultiArray, String, Int32
from carla_msgs.msg import CarlaEgoVehicleControl
from cv_bridge import CvBridge


# --- Data Manager ---
class DataManager:
    """Thread-safe storage for all ROS and Flask shared data for multiple annotators."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_image = None
        
        # ROS Signal Buffers (Existing)
        self.live_ear = deque(maxlen=360)
        self.live_mar = deque(maxlen=360)
        self.live_steering = deque(maxlen=360)
        
        # Heart Rate Data
        self.live_hr_instant = deque(maxlen=360) 
        self.live_hr_smooth = deque(maxlen=360)
        
        # --- NEW: Raw PPG Buffer ---
        # PPG is high frequency (approx 200Hz). 
        # maxlen=2000 stores ~10 seconds of data.
        self.live_ppg = deque(maxlen=2000)

        # Current values (for CSV saving/display text)
        self.instant_bpm = 0.0
        self.smooth_bpm = 0.0
        
        # Metrics & State
        self.latest_metrics = {}
        self.latest_phase_info = {}
        self.selected_labels_buffer = {}
        self.last_submitted_labels = {}

    def set_image(self, img):
        with self.lock:
            self.latest_image = img

    def set_ear_mar(self, ear, mar):
        with self.lock:
            self.live_ear.append(ear)
            self.live_mar.append(mar)

    def set_steering(self, steer):
        with self.lock:
            self.live_steering.append(steer)

    def set_metrics(self, metrics):
        with self.lock:
            self.latest_metrics.update(metrics)

    def set_phase_info(self, phase_info):
        with self.lock:
            self.latest_phase_info.update(phase_info)
            
    def set_heart_rate(self, instant, smooth):
        with self.lock:
            self.instant_bpm = float(instant)
            self.smooth_bpm = float(smooth)
            # Append to history buffers for the graph
            self.live_hr_instant.append(float(instant))
            self.live_hr_smooth.append(float(smooth))

    # --- NEW: Setter for Raw PPG ---
    def set_ppg(self, ppg_values):
        """Receives a list/array of PPG samples and extends the deque."""
        with self.lock:
            self.live_ppg.extend(ppg_values)

    def get_all_live_data(self):
        with self.lock:
            return {
                "live_ear": list(self.live_ear),
                "live_mar": list(self.live_mar),
                "live_steering": list(self.live_steering),
                
                # Heart Rate History
                "live_hr_instant": list(self.live_hr_instant),
                "live_hr_smooth": list(self.live_hr_smooth),
                
                # --- NEW: Send Raw PPG to Frontend ---
                "live_ppg": list(self.live_ppg),
                
                # Single values for text display
                "bpm": {
                    "instant": self.instant_bpm,
                    "smooth": self.smooth_bpm
                },
                
                "latest_metrics": deepcopy(self.latest_metrics),
                "phase_info": deepcopy(self.latest_phase_info),
                "last_submitted_labels": deepcopy(self.last_submitted_labels),
                "selected_labels_buffer": deepcopy(self.selected_labels_buffer),
            }

    def get_image(self):
        with self.lock:
            if self.latest_image is None:
                return None
            return self.latest_image.copy()


# --- ROS Bridge ---
class RosBridge(Node):
    # Voice commands & feedback constants
    VOICE_COMMANDS = "/home/user/voice_files/Trigger_1.mp3"
    VOICE_FEEDBACK = {
        "Yes": "/home/user/voice_files/joke.mp3",
        "No": "/home/user/voice_files/Feedback.mp3",
    }

    def __init__(self, data_manager):
        super().__init__("drowsiness_web_node")
        self.data_manager = data_manager
        self.bridge = CvBridge()
        self.get_logger().info("ROS Bridge Node Initialized.")

        # --- Subscriptions ---
        self.create_subscription(
            Image, "/flir_camera/image_raw", self.cb_camera, qos_profile_sensor_data
        )

        self.create_subscription(
            EarMarValue, "/ear_mar", self.cb_earmar, qos_profile_sensor_data
        )

        self.create_subscription(
            CarlaEgoVehicleControl,
            "/carla/ego/vehicle_control_cmd",
            self.cb_steering,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Float32MultiArray,
            "/driver_assistance/window_phase",
            self.cb_window_phase,
            10,
        )
        
        # Subscription to Heart Rate (BPM)
        self.create_subscription(
            Float32MultiArray,
            "/heart_rate_bpm",
            self.cb_heart_rate,
            10
        )
        
        # --- NEW: Subscription to Raw PPG ---
        # Assumes the HeartRateNode publishes batches of raw data to this topic
        self.create_subscription(
            Float32MultiArray,
            "/raw_ppg",
            self.cb_ppg,
            10
        )

        self.store_labels_client = self.create_client(StoreLabels, "store_labels")

        # --- Publishers ---
        self.vibration_pub = self.create_publisher(Vibration, "/wheel_vibration", 10)
        self.fan_pub = self.create_publisher(Int32, "/fan_speed", 10)
        self.mp4_pub = self.create_publisher(String, "/audio_file", 10)

        self.last_published_window_id = -1
        self.get_logger().info("ROS Bridge Node ready.")

    # --- Callbacks ---
    def cb_camera(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.data_manager.set_image(cv_image)
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def cb_earmar(self, msg):
        self.data_manager.set_ear_mar(msg.ear_value, msg.mar_value)

    def cb_steering(self, msg):
        self.data_manager.set_steering(msg.steer)
        
    def cb_heart_rate(self, msg):
        """Callback for /heart_rate_bpm topic"""
        try:
            if len(msg.data) >= 2:
                self.data_manager.set_heart_rate(msg.data[0], msg.data[1])
        except Exception as e:
            self.get_logger().warn(f"Error processing heart rate msg: {e}")

    def cb_ppg(self, msg):
        """Callback for /raw_ppg topic"""
        try:
            if msg.data:
                # msg.data is a list/array of floats
                self.data_manager.set_ppg(list(msg.data))
        except Exception as e:
            self.get_logger().warn(f"Error processing PPG msg: {e}")

    def cb_window_phase(self, msg):
        # The backend sends [phase, window_id, remaining_time]
        if len(msg.data) >= 3:
            phase_info = {
                "phase_index": int(msg.data[0]),
                "window_id": int(msg.data[1]),
                "remaining_time": float(msg.data[2])
            }
            self.data_manager.set_phase_info(phase_info)

            # Check if window ID changed -> trigger auto-submit for previous window
            if phase_info["window_id"] != self.last_published_window_id:
                self.last_published_window_id = phase_info["window_id"]
                self.auto_submit_previous_labels(phase_info["window_id"])

    def auto_submit_previous_labels(self, current_window_id):
        # Logic to auto-submit labels for the PREVIOUS window if not submitted manually
        prev_window = current_window_id - 1
        if prev_window < 0:
            return

        with self.data_manager.lock:
            annotators = list(self.data_manager.selected_labels_buffer.keys())

        for annotator in annotators:
            with self.data_manager.lock:
                labels = self.data_manager.selected_labels_buffer.get(annotator)
                if not labels:
                    continue
                # Mark as auto-submitted for tracking
                labels["auto_submitted"] = True
                labels["timestamp"] = time.time()
            
            # Call service to store these carried-over labels
            self.call_store_labels_service(prev_window, annotator, labels, "auto_carryover")

    def call_store_labels_service(self, window_id, annotator, labels_to_use, submission_type="manual"):
        if not self.store_labels_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("StoreLabels service not available.")
            return

        req = StoreLabels.Request()
        req.window_id = int(window_id)
        
        try:
            # 1. Prepare the standard labels
            labels_prepared = self.prepare_labels_for_saving(labels_to_use)
            
            # 2. Get current Heart Rate data from DataManager
            current_instant_bpm = 0.0
            current_smooth_bpm = 0.0
            
            with self.data_manager.lock:
                current_instant_bpm = self.data_manager.instant_bpm
                current_smooth_bpm = self.data_manager.smooth_bpm

            # 3. Build the Message
            ann_msg = AnnotatorLabels()
            ann_msg.annotator_name = annotator
            ann_msg.drowsiness_level = str(labels_prepared.get("drowsiness_level", "None"))
            ann_msg.actions = labels_to_use.get("actions", [])
            ann_msg.notes = labels_prepared.get("notes", "")
            ann_msg.voice_feedback = str(labels_prepared.get("voice_feedback", ""))
            ann_msg.submission_type = submission_type
            
            # Heart Rate Data
            ann_msg.instant_bpm = float(current_instant_bpm)
            ann_msg.smooth_bpm = float(current_smooth_bpm)
            
            # Action Booleans
            ann_msg.action_fan = bool(labels_prepared.get("fan", False))
            ann_msg.action_voice_command = bool(labels_prepared.get("voice_command", False))
            ann_msg.action_steering_vibration = bool(labels_prepared.get("steering_vibration", False))
            ann_msg.action_save_video = bool(labels_prepared.get("action_save_video", False))

            req.annotator_labels.append(ann_msg)

            # Async call
            future = self.store_labels_client.call_async(req)
            future.add_done_callback(lambda fut: self._service_result(fut, window_id))

        except Exception as e:
            self.get_logger().error(f"Error preparing StoreLabels service request: {e}")

    def _service_result(self, future, window_id):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f"[SERVICE] Labels stored for window {window_id}")
            else:
                self.get_logger().warn(f"[SERVICE] Label store failed: {res.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def prepare_labels_for_saving(self, labels):
        prepared = deepcopy(labels)
        # One-hot encode actions
        act_list = labels.get("actions", [])
        prepared["voice_command"] = 1 if "Voice Command" in act_list else 0
        prepared["steering_vibration"] = 1 if "Steering Vibration" in act_list else 0
        prepared["fan"] = 1 if "Fan" in act_list else 0
        
        # Voice feedback: Yes=1, No=0, Else=None
        vf = labels.get("voice_feedback")
        if vf == "Yes": prepared["voice_feedback"] = 1
        elif vf == "No": prepared["voice_feedback"] = 0
        else: prepared["voice_feedback"] = None
        
        # Video Flag & Auto Submit
        prepared["action_save_video"] = int(prepared.get("video_save_requested", False))
        prepared["auto_submitted"] = int(prepared.get("auto_submitted", 0))

        # Remove temporary keys
        for key in ["actions", "timestamp", "video_save_requested"]:
            if key in prepared:
                del prepared[key]
        return prepared

    # --- Action Methods ---
    
    def vibrate(self, duration: float, intensity: int):
        out = Vibration()
        out.duration = float(duration)
        out.intensity = int(intensity)
        self.vibration_pub.publish(out)

    def publish_mp4(self, file_path: str):
        if not file_path: return
        msg = String()
        msg.data = file_path
        self.mp4_pub.publish(msg)
        self.get_logger().info(f"Published MP4 file path: {file_path}")
        
    def set_fan_speed(self, speed):
        msg = Int32()
        msg.data = int(speed)
        self.fan_pub.publish(msg)


# --- Flask App Setup ---
data_manager = DataManager()
app = Flask(__name__)
ros_bridge_node = None

def ros_spin():
    global ros_bridge_node
    rclpy.init()
    ros_bridge_node = RosBridge(data_manager)
    executor = MultiThreadedExecutor()
    executor.add_node(ros_bridge_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if ros_bridge_node:
            ros_bridge_node.destroy_node()
        rclpy.shutdown()

# Start ROS thread immediately
threading.Thread(target=ros_spin, daemon=True).start()


# --- FLASK ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = data_manager.get_image()
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_live_data")
def get_live_data():
    return jsonify(data_manager.get_all_live_data())

@app.route("/activate_action", methods=["POST"])
def activate_action():
    req = request.get_json() or {}
    action = req.get("action")
    feedback = req.get("feedback")
    
    if ros_bridge_node is None:
        return jsonify({"status": "error", "message": "ROS node not ready."}), 500

    def handle_action(action, feedback):
        if action == "Steering Vibration":
            ros_bridge_node.get_logger().info("Triggering Vibration via UI")
            ros_bridge_node.vibrate(duration=2.0, intensity=30)
        elif action == "Voice Command":
            ros_bridge_node.get_logger().info("Triggering Voice Command via UI")
            ros_bridge_node.publish_mp4(RosBridge.VOICE_COMMANDS)
        elif action == "Voice Feedback":
            if feedback in RosBridge.VOICE_FEEDBACK:
                ros_bridge_node.publish_mp4(RosBridge.VOICE_FEEDBACK[feedback])
        elif action == "Fan":
            ros_bridge_node.set_fan_speed(3)
        else:
            ros_bridge_node.get_logger().warn(f"Unknown action: {action}")

    threading.Thread(target=handle_action, args=(action, feedback), daemon=True).start()
    return jsonify({"status": "success", "message": f"Action '{action}' triggered."})

@app.route("/store_selected_labels", methods=["POST"])
def store_selected_labels():
    req = request.get_json() or {}
    annotator = req.get("annotator_name")
    
    if not annotator:
        return jsonify({"status": "error", "message": "annotator_name required"}), 400

    # Buffer the labels
    buffered = {
        "drowsiness_level": req.get("drowsiness_level"),
        "actions": req.get("actions", []),
        "notes": req.get("notes", ""),
        "voice_feedback": req.get("voice_feedback", ""),
        "video_save_requested": req.get("video_save_requested", False),
        "timestamp": time.time(),
        "auto_submitted": False,
    }

    with data_manager.lock:
        data_manager.selected_labels_buffer[annotator] = buffered
        
    return jsonify({"status": "success", "annotator_used": annotator})

@app.route("/submit_labels", methods=["POST"])
def submit_labels():
    # Reuse storage logic
    return store_selected_labels()

@app.route("/save_video_segment", methods=["POST"])
def save_video_segment():
    # Just an acknowledgment route, logic handled in store_selected_labels via 'video_save_requested' flag
    return jsonify({"status": "success"})


def main():
    try:
        serve(app, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("Shutting down web interface.")

if __name__ == "__main__":
    main()