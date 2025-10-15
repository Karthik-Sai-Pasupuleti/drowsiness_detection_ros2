#!/usr/bin/env python3

"""
This version contains a interface with voice command feedback button.

This module implements a web application for real-time drowsiness detection labelling.
It uses Flask for the web interface and ROS2 for data communication.
The application allows multiple annotators to label drowsiness levels and actions,
with support for auto-submission of previous labels if no new input is provided.
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
from drowsiness_detection_msg.msg import (
    EarMarValue,
    LanePosition,
    DrowsinessMetricsData,
    Vibration,
)
from std_msgs.msg import Float32MultiArray, String, Int32
from carla_msgs.msg import CarlaEgoVehicleControl
from cv_bridge import CvBridge

from .utils import extract_window_data

# Assume hdf5_manager.py exists and contains a save_to_hdf5 function
from .hdf5_manager import save_to_hdf5


# --- Data Manager ---
class DataManager:
    """Thread-safe storage for all ROS and Flask shared data for multiple annotators."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_image = None
        self.live_ear = deque(maxlen=300)
        self.live_mar = deque(maxlen=300)
        self.live_steering = deque(maxlen=300)
        self.latest_metrics = {}
        self.latest_phase_info = {}
        self.selected_labels_buffer = {}  # annotator -> labels
        self.last_submitted_labels = {}  # annotator -> labels

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

    def get_all_live_data(self):
        with self.lock:
            return {
                "live_ear": list(self.live_ear),
                "live_mar": list(self.live_mar),
                "live_steering": list(self.live_steering),
                "latest_metrics": deepcopy(self.latest_metrics),
                "phase_info": deepcopy(self.latest_phase_info),
                "last_submitted_labels": deepcopy(self.last_submitted_labels),
                "selected_labels_buffer": deepcopy(self.selected_labels_buffer),
            }

    def get_image(self):
        with self.lock:
            return deepcopy(self.latest_image)


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
            Image, "/camera/image_raw", self.cb_camera, qos_profile_sensor_data
        )
        self.create_subscription(
            EarMarValue, "/ear_mar", self.cb_earmar, qos_profile_sensor_data
        )
        self.create_subscription(
            CarlaEgoVehicleControl,
            "/carla/hero/vehicle_control_cmd",
            self.cb_steering,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Float32MultiArray,
            "/driver_assistance/window_phase",
            self.cb_window_phase,
            10,
        )
        self.create_subscription(
            DrowsinessMetricsData,
            "/driver_assistance/window_data",
            self.cb_window_data,
            10,
        )

        # --- New subscriptions/publishers ---
        self.vibration_pub = self.create_publisher(Vibration, "/wheel_vibration", 10)

        self.fan_pub = self.create_publisher(Int32, "/fan_speed", 10)
        self.mp4_pub = self.create_publisher(String, "/audio_file", 10)

        self.get_logger().info("ROS Bridge Node ready.")

    # --- Steering vibration --

    def vibrate(self, duration: float, intensity: int):
        """Publish vibration activation message."""
        out = Vibration()
        out.duration = float(duration)
        out.intensity = int(intensity)
        self.vibration_pub.publish(out)

    # --- Voice commands ---
    def publish_mp4(self, file_path: str):
        if not file_path or not os.path.isfile(file_path):
            self.get_logger().warn(f"Invalid MP4 file path: {file_path}")
            return
        msg = String()
        msg.data = file_path
        self.mp4_pub.publish(msg)
        self.get_logger().info(f"Published MP4 file path: {file_path}")

    # --- Fan control ---
    def set_fan_speed(self, speed_level: int):
        """Set fan speed by publishing to the fan_speed topic."""

        if speed_level not in [0, 1, 2, 3]:
            self.get_logger().warn(f"Invalid fan speed level: {speed_level}")
            return
        self.get_logger().info(f"Setting fan speed to level {speed_level}")
        ros_msg = Int32()
        ros_msg.data = speed_level
        self.fan_pub.publish(ros_msg)

    # --- ROS Data callbacks ---
    def cb_camera(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.data_manager.set_image(cv_img.copy())
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def cb_earmar(self, msg):
        try:
            self.data_manager.set_ear_mar(msg.ear_value, msg.mar_value)
        except Exception as e:
            self.get_logger().error(f"Error in ear_mar_callback: {e}")

    def cb_steering(self, msg):
        self.data_manager.set_steering(msg.steer)

    def cb_window_phase(self, msg):
        if len(msg.data) >= 3:
            phase_info = {
                "phase": int(msg.data[0]),
                "window_id": int(msg.data[1]),
                "remaining_time": msg.data[2],
            }
            self.data_manager.set_phase_info(phase_info)

    def cb_window_data(self, msg):
        try:
            window_data = extract_window_data(msg, self.bridge)
            metrics = window_data.get("metrics")
            self.data_manager.set_metrics(metrics)
            window_id = msg.window_id

            with self.data_manager.lock:
                annotators = set(self.data_manager.selected_labels_buffer.keys()) | set(
                    self.data_manager.last_submitted_labels.keys()
                )
                if not annotators:
                    annotators = {"default_annotator"}

                for annotator in annotators:
                    if annotator in self.data_manager.selected_labels_buffer:
                        labels_to_use = deepcopy(
                            self.data_manager.selected_labels_buffer[annotator]
                        )
                        submission_type = "manual"
                        self.data_manager.last_submitted_labels[annotator] = (
                            labels_to_use
                        )
                        del self.data_manager.selected_labels_buffer[annotator]
                    else:
                        if self.data_manager.last_submitted_labels.get(annotator):
                            labels_to_use = deepcopy(
                                self.data_manager.last_submitted_labels[annotator]
                            )
                            submission_type = "auto"
                            labels_to_use["auto_submitted"] = True
                        else:
                            labels_to_use = {
                                "drowsiness_level": "None",
                                "actions": [],
                                "notes": "Auto-submitted: No previous label found.",
                                "auto_submitted": True,
                            }
                            submission_type = "auto"

                    labels_to_use["submission_type"] = submission_type

                    # --- Prepare labels: split actions into separate columns + add voice feedback ---
                    labels_prepared = prepare_labels_for_saving(labels_to_use)

                    try:
                        save_to_hdf5(window_id, window_data, annotator, labels_prepared)
                        self.get_logger().info(
                            f"Saved window {window_id} for annotator {annotator} ({submission_type})"
                        )
                    except Exception as e:
                        self.get_logger().error(
                            f"Failed to save window {window_id} for {annotator}: {e}"
                        )

        except Exception as e:
            self.get_logger().error(f"Error processing window data: {e}")


def prepare_labels_for_saving(labels):
    prepared = deepcopy(labels)

    # One-hot encode actions (already separate columns)
    prepared["voice_command"] = 1 if "Voice Command" in labels.get("actions", []) else 0
    prepared["steering_vibration"] = (
        1 if "Steering Vibration" in labels.get("actions", []) else 0
    )
    prepared["fan"] = 1 if "Fan" in labels.get("actions", []) else 0

    # Voice feedback: force int or default 0
    vf = labels.get("voice_feedback")
    if vf == "Yes":
        prepared["voice_feedback"] = 1
    elif vf == "No":
        prepared["voice_feedback"] = 0
    else:
        prepared["voice_feedback"] = None  # safe default

    # Auto-submitted: convert bool → int
    prepared["auto_submitted"] = int(prepared.get("auto_submitted", 0))

    # Remove raw 'actions' list to avoid h5py object dtype error
    for key in ["actions", "timestamp"]:
        if key in prepared:
            del prepared[key]
    return prepared


# --- Flask ---
data_manager = DataManager()
app = Flask(__name__)

# Global reference to the RosBridge node for use in Flask routes
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


threading.Thread(target=ros_spin, daemon=True).start()


@app.route("/")
def index():
    return render_template("index_v2.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = data_manager.get_image()
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/get_live_data")
def get_live_data():
    return jsonify(data_manager.get_all_live_data())


# --- NEW ROUTE: For real-time action activation ---
@app.route("/activate_action", methods=["POST"])
def activate_action():
    req = request.get_json() or {}
    action = req.get("action")
    feedback = req.get("feedback")

    if ros_bridge_node is None:
        return jsonify({"status": "error", "message": "ROS node not ready."}), 500

    def handle_action(action, feedback):
        """Run the requested action in a separate thread."""
        if action == "Steering Vibration":
            ros_bridge_node.get_logger().info(
                "Steering vibration action triggered via web interface."
            )
            ros_bridge_node.vibrate(duration=2.0, intensity=30)

        elif action == "Voice Command":
            ros_bridge_node.get_logger().info(
                "Voice command action triggered via web interface (Main command)."
            )
            # Play main voice command only
            voice_text = RosBridge.VOICE_COMMANDS
            ros_bridge_node.publish_mp4(voice_text)

        elif action == "Voice Feedback":
            if feedback in RosBridge.VOICE_FEEDBACK:
                ros_bridge_node.get_logger().info(
                    f"Voice command feedback triggered via web interface: {feedback}"
                )
                feedback_file = RosBridge.VOICE_FEEDBACK[feedback]
                ros_bridge_node.publish_mp4(feedback_file)
            else:
                ros_bridge_node.get_logger().warn(
                    f"Voice Feedback action triggered with invalid feedback: {feedback}"
                )

        elif action == "Fan":
            ros_bridge_node.get_logger().info("Fan action triggered via web interface.")
            ros_bridge_node.set_fan_speed(3)

        else:
            ros_bridge_node.get_logger().warn(f"Unknown action: {action}")

    # Run the action in a separate thread so Flask doesn't block
    threading.Thread(target=handle_action, args=(action, feedback), daemon=True).start()

    return jsonify({"status": "success", "message": f"Action '{action}' triggered."})


@app.route("/store_selected_labels", methods=["POST"])
def store_selected_labels():
    req = request.get_json() or {}
    window_id = req.get("window_id")
    annotator = req.get("annotator_name")
    voice_feedback = req.get("voice_feedback", "")

    if window_id is None or annotator is None:
        return (
            jsonify(
                {"status": "error", "message": "window_id and annotator_name required"}
            ),
            400,
        )

    drowsiness_level = req.get("drowsiness_level")
    notes = req.get("notes", "")

    # The actions sent from the frontend are now just for storage
    actions_to_store = req.get("actions", [])

    buffered = {
        "drowsiness_level": drowsiness_level,
        "actions": actions_to_store,
        "notes": notes,
        "voice_feedback": voice_feedback,
        "timestamp": time.time(),
        "auto_submitted": False,
    }

    with data_manager.lock:
        data_manager.selected_labels_buffer[annotator] = buffered

    return jsonify({"status": "success", "annotator_used": annotator})


@app.route("/submit_labels", methods=["POST"])
def submit_labels():
    # The submit button now only stores the labels, it no longer triggers actions.
    # The actions are triggered by the individual button presses.
    return store_selected_labels()



def main():
    try:
        serve(app, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("Shutting down web interface.")





if __name__ == "__main__":
    main() 
#    serve(app, host="0.0.0.0", port=5000)
#     app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
