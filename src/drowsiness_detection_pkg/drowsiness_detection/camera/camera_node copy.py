#!/usr/bin/env python3

"""
This module contains a Ros2 node that captures video frames from a camera
and publishes them as sensor_msgs/Image messages on the /camera/image_raw topic.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2
import time


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.publisher_ = self.create_publisher(
            Image, "/camera/image_raw", qos_profile_sensor_data
        )
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            raise RuntimeError("Cannot open camera")

        self.bridge = CvBridge()
        self.get_logger().info("Camera node started.")

        self.prev_time = None  # For FPS calculation

        # Timer to continuously publish the latest frame
        self.timer = self.create_timer(0.0, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        # Convert OpenCV frame to ROS Image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_optical_frame"
        self.publisher_.publish(msg)

        # FPS calculation
        current_time = time.time()
        if self.prev_time is not None:
            delta = current_time - self.prev_time
            fps = 1.0 / delta if delta > 0 else 0.0
            self.get_logger().info(f"Camera FPS: {fps:.2f}")
        self.prev_time = current_time

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()