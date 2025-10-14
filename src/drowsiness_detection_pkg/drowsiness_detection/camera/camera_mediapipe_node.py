#!/usr/bin/env python3
"""
ROS2 Node: Combined PySpin camera + Mediapipe EAR/MAR processing
Publishes Mediapipe-annotated frames on /camera/image_raw
Publishes EAR/MAR metrics on /ear_mar
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from drowsiness_detection_msg.msg import EarMarValue
from cv_bridge import CvBridge
import PySpin
import threading
import time
import cv2
import numpy as np
from queue import Queue, Full
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .utils import calculate_avg_ear, mouth_aspect_ratio


class DeviceEventHandler(PySpin.DeviceEventHandler):
    """Handles camera device events"""

    def __init__(self, event_name="EventExposureEnd"):
        super().__init__()
        self.event_name = event_name
        self.count = 0

    def OnDeviceEvent(self, event_name):
        if event_name == self.event_name:
            self.count += 1
            print(f"\tDevice event {event_name} triggered, count={self.count}")
        else:
            print(f"\tDevice event occurred but not {self.event_name}; ignoring...")


class CameraMediapipeNode(Node):
    def __init__(self):
        super().__init__("camera_mediapipe_node")

        # --- ROS publishers
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", qos_profile_sensor_data)
        self.metrics_pub = self.create_publisher(EarMarValue, "/ear_mar", 10)
        self.bridge = CvBridge()

        # --- Frame queue
        self.frame_queue = Queue(maxsize=4)  # buffer max 4 frames

        # --- Mediapipe setup
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

        self.target_width = 480
        self.target_height = 640

        # --- PySpin camera setup
        self.system = PySpin.System.GetInstance()
        self.cameras = self.system.GetCameras()
        if not self.cameras.GetSize():
            self.get_logger().error("No FLIR cameras detected.")
            raise RuntimeError("No FLIR cameras detected.")
        self.cam = self.cameras[0]
        self.get_logger().info(f"Using camera: {self.cam.GetUniqueID()}")
        self.cam.Init()
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()
        self.event_handler = DeviceEventHandler()
        self.cam.RegisterEventHandler(self.event_handler)

        # --- Threading
        self.running = True
        self.prev_time = None
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.camera_thread.start()
        self.worker_thread.start()

        # --- Keypoints for Mediapipe visualization
        self.LEFT_EYE = [362, 380, 374, 263, 386, 385]
        self.RIGHT_EYE = [33, 159, 158, 133, 153, 145]
        self.MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

        self.get_logger().info("Camera + Mediapipe node started.")

    # --- Camera acquisition thread
    def camera_loop(self):
        while self.running and rclpy.ok():
            try:
                image_result = self.cam.GetNextImage()
                if image_result.IsIncomplete():
                    self.get_logger().warn(f"Incomplete image: {image_result.GetImageStatus()}")
                    image_result.Release()
                    continue

                img_cv = image_result.GetNDArray()
                image_result.Release()

                # Enqueue frame for Mediapipe
                try:
                    self.frame_queue.put_nowait(img_cv)
                except Full:
                    self.frame_queue.get_nowait()  # drop oldest
                    self.frame_queue.put_nowait(img_cv)

                # Optional FPS logging
                now = time.time()
                if self.prev_time:
                    fps = 1.0 / (now - self.prev_time)
                    self.get_logger().info(f"Camera FPS: {fps:.2f}")
                self.prev_time = now

                time.sleep(0.001)  # yield thread

            except PySpin.SpinnakerException as ex:
                self.get_logger().error(f"Spinnaker error: {ex}")
            except Exception as e:
                self.get_logger().error(f"Unexpected camera error: {e}")

    # --- Worker thread: Mediapipe processing
    def worker_loop(self):
        while self.running and rclpy.ok():
            frame = self.frame_queue.get()  # blocks until a frame is available
            self.process_frame(frame)

    def process_frame(self, img_frame):
        try:
            start_time = time.time()  # <-- start timing for Mediapipe FPS

            # img_frame is already a NumPy array from PySpin
            frame = img_frame.copy()

            # Handle IR input (grayscale)
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Reduce resolution for faster inference
            frame = cv2.resize(frame, (self.target_width, self.target_height))

            # Convert to RGB (Mediapipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Mediapipe detection
            result = self.face_landmarker.detect(mp_image)

            ear, mar = 0.0, 0.0
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                landmark_array = np.array([[lm.x, lm.y] for lm in landmarks])
                ear = calculate_avg_ear(landmark_array)
                mar = mouth_aspect_ratio(landmark_array)

                # Draw keypoints
                self.draw_keypoints(frame, landmark_array)

            # Publish EAR/MAR
            ear_mar_msg = EarMarValue()
            ear_mar_msg.header.stamp = self.get_clock().now().to_msg()
            ear_mar_msg.ear_value = float(ear)
            ear_mar_msg.mar_value = float(mar)
            self.metrics_pub.publish(ear_mar_msg)

            # Publish annotated frame
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_optical_frame"
            self.image_pub.publish(img_msg)

            # --- Calculate and log Mediapipe FPS
            end_time = time.time()
            mp_fps = 1.0 / (end_time - start_time)
            self.get_logger().info(f"Mediapipe FPS: {mp_fps:.2f}")

        except Exception as e:
            self.get_logger().error(f"Mediapipe processing error: {e}")



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

    def destroy_node(self):
        self.running = False
        self.camera_thread.join()
        self.worker_thread.join()
        try:
            if self.cam is not None:
                self.cam.UnregisterEventHandler(self.event_handler)
                self.cam.EndAcquisition()
                self.cam.DeInit()
            self.cameras.Clear()
            self.system.ReleaseInstance()
        except PySpin.SpinnakerException as ex:
            self.get_logger().warn(f"Error releasing camera: {ex}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraMediapipeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
