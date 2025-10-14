#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from drowsiness_detection_msg.msg import EarMarValue
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ..camera.utils import calculate_avg_ear, mouth_aspect_ratio
import time
from concurrent.futures import ThreadPoolExecutor


class MediapipeNode(Node):
    def __init__(self):
        super().__init__("mediapipe_node")

        # Mediapipe Face Landmarker (GPU delegate)
        base_options = python.BaseOptions(
            model_asset_path=r'/home/user/ros2_ws/src/models/face_landmarker.task',
            delegate=python.BaseOptions.Delegate.GPU
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # ROS2 Setup
        self.bridge = CvBridge()
        self.executor_pool = ThreadPoolExecutor(max_workers=2)

        self.subscription = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, qos_profile_sensor_data
        )
        self.metrics_pub = self.create_publisher(EarMarValue, "/ear_mar", 10)

        # Parameters
        self.target_width = 256   # <-- You can change this to 320 or 192
        self.target_height = 256

        self.prev_time = None
        self.get_logger().info(
            f"Mediapipe IR node started (GPU, multithreading, {self.target_width}x{self.target_height})."
        )

        # Keypoint indices for visualization
        self.LEFT_EYE = [362, 380, 374, 263, 386, 385]
        self.RIGHT_EYE = [33, 159, 158, 133, 153, 145]
        self.MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
        
        self.last_cam_time = None
        self.frame_count = 0
        self.last_fps_log_time = time.time()
        self.camera_fps = 0.0

    def image_callback(self, msg):
        """Offload image processing to a separate thread."""
        
        current_time = time.time()

        # --- Calculate Camera Frame Rate ---
        if self.last_cam_time is not None:
            delta = current_time - self.last_cam_time
            if delta > 0:
                self.frame_count += 1
                # Log every 2 seconds to avoid spamming logs
                if current_time - self.last_fps_log_time >= 2.0:
                    self.camera_fps = self.frame_count / (current_time - self.last_fps_log_time)
                    self.get_logger().info(f"Camera Subscriber FPS: {self.camera_fps:.2f}")
                    self.frame_count = 0
                    self.last_fps_log_time = current_time

        self.last_cam_time = current_time
        
        self.executor_pool.submit(self.process_image, msg)

    def process_image(self, msg):
        try:
            # Convert ROS Image → OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Handle IR input (grayscale)
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Reduce resolution for faster inference
            frame = cv2.resize(frame, (self.target_width, self.target_height))

            # Convert to RGB (Mediapipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run Mediapipe detection
            result = self.face_landmarker.detect(mp_image)

            ear, mar = 0.0, 0.0
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                landmark_array = np.array([[lm.x, lm.y] for lm in landmarks])
                ear = calculate_avg_ear(landmark_array)
                mar = mouth_aspect_ratio(landmark_array)


            # Publish EAR and MAR
            ear_mar_msg = EarMarValue()
            ear_mar_msg.header.stamp = msg.header.stamp
            ear_mar_msg.ear_value = float(ear)
            ear_mar_msg.mar_value = float(mar)
            self.metrics_pub.publish(ear_mar_msg)

            # FPS Calculation
            current_time = time.time()
            if self.prev_time is not None:
                delta = current_time - self.prev_time
                fps = 1.0 / delta if delta > 0 else 0.0
                self.get_logger().info(f"Mediapipe Processing FPS: {fps:.2f}")
            self.prev_time = current_time

            # Optional debug visualization (comment out in headless mode)
            cv2.imshow("IR EAR/MAR Visualization", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def draw_keypoints(self, frame, landmarks):
        """Draw left eye, right eye, and mouth keypoints with colors."""
        h, w, _ = frame.shape

        for idx in self.LEFT_EYE:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Blue

        for idx in self.RIGHT_EYE:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Green

        for idx in self.MOUTH:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red


def main(args=None):
    rclpy.init(args=args)
    node = MediapipeNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()