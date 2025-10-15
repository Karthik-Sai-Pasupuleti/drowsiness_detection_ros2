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
import cv2
import numpy as np
from queue import Queue, Full
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .utils import calculate_avg_ear, mouth_aspect_ratio


class DeviceEventHandler(PySpin.DeviceEventHandler):
    """Handles camera device events (optional for debugging)."""

    def __init__(self, event_name="EventExposureEnd"):
        super().__init__()
        self.event_name = event_name
        self.count = 0

    def OnDeviceEvent(self, event_name):
        if event_name == self.event_name:
            self.count += 1
            print(f"Device event {event_name} triggered ({self.count})")


class CameraMediapipeNode(Node):
    """Camera node with Mediapipe processing (timer-based acquisition)."""

    def __init__(self):
        super().__init__("camera_mediapipe_node")

        # === ROS Publishers ===
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", qos_profile_sensor_data)
        self.metrics_pub = self.create_publisher(EarMarValue, "/ear_mar", 10)
        self.bridge = CvBridge()

        # === Parameters ===
        self.declare_parameter("camera_fps", 60.0)
        self.target_fps = self.get_parameter("camera_fps").value
        self.frame_period = 1.0 / self.target_fps

        # === Frame Queue ===
        self.frame_queue = Queue(maxsize=4)

        # === Mediapipe Setup ===
        base_options = python.BaseOptions(
            model_asset_path="/home/user/ros2_ws/src/models/face_landmarker.task",
            delegate=python.BaseOptions.Delegate.GPU,
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # Landmark indices for visualization
        self.LEFT_EYE = [362, 380, 374, 263, 386, 385]
        self.RIGHT_EYE = [33, 159, 158, 133, 153, 145]
        self.MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

        # === PySpin Camera Setup ===
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

        # Optional event handler
        self.event_handler = DeviceEventHandler()
        self.cam.RegisterEventHandler(self.event_handler)

        # === Timer-based acquisition ===
        self.create_timer(self.frame_period, self.camera_timer_callback)

        # === Worker Thread for Mediapipe ===
        self.running = True
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()
        self.face_lock = threading.Lock()

        self.get_logger().info("Camera + Mediapipe node started (timer mode).")

    # -------------------------------------------------------------------------
    # Camera Acquisition (Timer Callback)
    # -------------------------------------------------------------------------
    def camera_timer_callback(self):
        """Timer-based acquisition (~60 FPS)."""
        try:
            image_result = self.cam.GetNextImage()
            if image_result.IsIncomplete():
                self.get_logger().warn(f"Incomplete image: {image_result.GetImageStatus()}")
                image_result.Release()
                return

            img_cv = image_result.GetNDArray()
            image_result.Release()

            # Add to queue (drop oldest if full)
            try:
                self.frame_queue.put_nowait(img_cv)
            except Full:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(img_cv)

        except PySpin.SpinnakerException as ex:
            self.get_logger().error(f"Spinnaker error: {ex}")
        except Exception as e:
            self.get_logger().error(f"Unexpected camera error: {e}")

    # -------------------------------------------------------------------------
    # Worker Thread: Mediapipe Processing
    # -------------------------------------------------------------------------
    def worker_loop(self):
        while self.running and rclpy.ok():
            # Drop older frames if queue has more than 1
            while self.frame_queue.qsize() > 1:
                _ = self.frame_queue.get_nowait()
            frame = self.frame_queue.get()
            self.process_frame(frame)


    def process_frame(self, img_frame: np.ndarray):
        """Run Mediapipe on one frame and publish results."""
        try:
            frame = img_frame.copy()

            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # thread safety for Mediapipe
            with self.face_lock:
                result = self.face_landmarker.detect(mp_image)

            ear, mar = 0.0, 0.0
            if result.face_landmarks:
                landmarks = np.array([[lm.x, lm.y] for lm in result.face_landmarks[0]])
                ear = calculate_avg_ear(landmarks)
                mar = mouth_aspect_ratio(landmarks)
                self.draw_keypoints(frame, landmarks)

            # Publish metrics
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

        except Exception as e:
            self.get_logger().error(f"Mediapipe processing error: {e}")

    # -------------------------------------------------------------------------
    # Visualization Helper
    # -------------------------------------------------------------------------
    def draw_keypoints(self, frame: np.ndarray, landmarks: np.ndarray):
        """Draw facial keypoints for eyes and mouth."""
        h, w, _ = frame.shape
        for idx in self.LEFT_EYE:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        for idx in self.RIGHT_EYE:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        for idx in self.MOUTH:
            x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def destroy_node(self):
        """Cleanly release camera and threads."""
        self.running = False
        if hasattr(self, "worker_thread"):
            self.worker_thread.join(timeout=2.0)
        try:
            if self.cam:
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
