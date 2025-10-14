#!/usr/bin/env python3

"""
This module contains a ROS2 node that processes images from a camera
to extract facial landmarks and compute metrics such as Eye Aspect Ratio (EAR)
and Mouth Aspect Ratio (MAR) using mediapipe model.

This node subscribes to the /camera/image_raw topic for input images
and publishes the computed EAR and MAR values on the /ear_mar topic.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from drowsiness_detection_msg.msg import EarMarValue
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
from .utils import calculate_avg_ear, mouth_aspect_ratio

import time

class MediapipeNode(Node):
    def __init__(self):
        super().__init__("mediapipe_node")

        # Mediapipe setup
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Subscription to camera
        self.subscription = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, qos_profile_sensor_data
        )

        # Publisher for EAR/MAR
        self.metrics_pub = self.create_publisher(EarMarValue, "/ear_mar", 10)

        # CV Bridge
        self.bridge = CvBridge()
        self.prev_time = None  # For processing FPS
        self.get_logger().info("Mediapipe node started.")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe processing
        results = self.face_mesh.process(rgb_frame)

        ear, mar = 0.0, 0.0
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            landmark_array = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
            ear = calculate_avg_ear(landmark_array)
            mar = mouth_aspect_ratio(landmark_array)

        ear_mar_msg = EarMarValue()
        ear_mar_msg.header.stamp = msg.header.stamp
        ear_mar_msg.ear_value = float(ear)
        ear_mar_msg.mar_value = float(mar)

        self.metrics_pub.publish(ear_mar_msg)

        # Optional: compute processing FPS
        current_time = time.time()
        if self.prev_time is not None:
            delta = current_time - self.prev_time
            fps = 1.0 / delta if delta > 0 else 0.0
            self.get_logger().info(f"Mediapipe Processing FPS: {fps:.2f}")
        self.prev_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = MediapipeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
