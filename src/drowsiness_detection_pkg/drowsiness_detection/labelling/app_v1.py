#!/usr/bin/env python3

"""
This version contains a interface without voice command feedback button.

This module implements a web application for real-time drowsiness detection labelling.
It uses Flask for the web interface and ROS2 for data communication.
The application allows multiple annotators to label drowsiness levels and actions,
with support for auto-submission of previous labels if no new input is provided.
"""


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
from std_msgs.msg import Float32MultiArray
from carla_msgs.msg import CarlaEgoVehicleControl
from cv_bridge import CvBridge

import h5py
import numpy as np
import pandas as pd

from hdf5_manager import save_to_hdf5

# --- HDF5 File Path ---
DRIVER_ID = "driver_1"
HDF5_FILE = f"drowsiness_data/{DRIVER_ID}_session.h5"
CSV_FILE = f"drowsiness_data/{DRIVER_ID}_session.csv"


# --- Data Manager ---
class DataManager:
    """
    Thread-safe storage for all ROS and Flask shared data for multiple annotators.
    """

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
    def __init__(self, data_manager):
        super().__init__("drowsiness_web_node")
        self.data_manager = data_manager
        self.bridge = CvBridge()
        self.get_logger().info("ROS Bridge Node Initialized.")

        self.create_subscription(
            Image, "/camera/image_raw", self.cb_camera, qos_profile_sensor_data
        )
        self.create_subscription(
            Float32MultiArray, "/ear_mar", self.cb_earmar, qos_profile_sensor_data
        )
        self.create_subscription(
            CarlaEgoVehicleControl,
            "/carla/hero/vehicle_control_cmd",
            self.cb_steering,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32MultiArray, "/driver_assistance/metrics", self.cb_metrics, 10
        )
        self.create_subscription(
            Float32MultiArray,
            "/driver_assistance/window_phase",
            self.cb_window_phase,
            10,
        )
        self.create_subscription(
            Float32MultiArray, "/driver_assistance/window_data", self.cb_window_data, 10
        )
        self.get_logger().info("ROS Bridge Node ready.")

    def cb_camera(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.data_manager.set_image(cv_img.copy())
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")

    def cb_earmar(self, msg):
        if len(msg.data) >= 2:
            self.data_manager.set_ear_mar(msg.data[0], msg.data[1])

    def cb_steering(self, msg):
        self.data_manager.set_steering(msg.steer)

    def cb_metrics(self, msg):
        if len(msg.data) == 6:
            metrics = {
                "perclos": msg.data[0],
                "blink_rate": msg.data[1],
                "yawn_rate": msg.data[2],
                "steering_entropy": msg.data[3],
                "steering_rate": msg.data[4],
                "sdlp": msg.data[5],
            }
            self.data_manager.set_metrics(metrics)
        else:
            self.get_logger().warn(f"Unexpected metrics length: {len(msg.data)}")

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
            window_data_str = "".join(chr(int(x)) for x in msg.data)
            window_data = json.loads(window_data_str)
            window_id = window_data.get("window_id")
            if window_id is None:
                self.get_logger().error("No 'window_id' in window data")
                return

            with self.data_manager.lock:
                # Build annotators set: include everyone with a last submission OR buffer
                annotators = set(self.data_manager.selected_labels_buffer.keys()) | set(
                    self.data_manager.last_submitted_labels.keys()
                )
                if not annotators:
                    annotators = {"default_annotator"}  # Fallback default

                for annotator in annotators:
                    if annotator in self.data_manager.selected_labels_buffer:
                        # Manual submission exists
                        labels_to_use = deepcopy(
                            self.data_manager.selected_labels_buffer[annotator]
                        )
                        submission_type = "manual"
                        self.data_manager.last_submitted_labels[annotator] = (
                            labels_to_use
                        )
                        del self.data_manager.selected_labels_buffer[annotator]

                    else:
                        if self.data_manager.last_submitted_labels[annotator]:
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

                    try:
                        save_to_hdf5(window_id, window_data, annotator, labels_to_use)
                        self.get_logger().info(
                            f"Saved window {window_id} for annotator {annotator} ({submission_type})"
                        )
                    except Exception as e:
                        self.get_logger().error(
                            f"Failed to save window {window_id} for {annotator}: {e}"
                        )

        except Exception as e:
            self.get_logger().error(f"Error processing window data: {e}")


# --- Flask ---
data_manager = DataManager()
app = Flask(__name__)


def ros_spin():
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
        ros_bridge_node.destroy_node()
        rclpy.shutdown()


threading.Thread(target=ros_spin, daemon=True).start()


@app.route("/")
def index():
    return render_template("index_v1.html")


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


@app.route("/store_selected_labels", methods=["POST"])
def store_selected_labels():
    req = request.get_json() or {}

    window_id = req.get("window_id")
    if window_id is None:
        return jsonify({"status": "error", "message": "window_id required"}), 400

    annotator = req.get("annotator_name")

    # Only buffer if at least one meaningful field is provided
    if not any([req.get("drowsiness_level"), req.get("actions"), req.get("notes")]):
        return jsonify({"status": "ignored", "message": "No labels provided"}), 200

    buffered = {
        "drowsiness_level": req.get("drowsiness_level"),
        "actions": req.get("actions", []),
        "notes": req.get("notes", ""),
        "timestamp": time.time(),
        "auto_submitted": False,
    }

    with data_manager.lock:
        data_manager.selected_labels_buffer[annotator] = buffered

    return jsonify({"status": "success", "annotator_used": annotator})


@app.route("/submit_labels", methods=["POST"])
def submit_labels():
    return store_selected_labels()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
