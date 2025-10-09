#!/usr/bin/env python3
"""
ROS2 node: Publishes FLIR/Spinnaker camera frames on /camera/image_raw using PySpin
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import PySpin
import time


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


class PySpinCameraNode(Node):
    def __init__(self):
        super().__init__("pyspin_camera_node")

        # --- ROS parameters
        self.declare_parameter("cam_id", None)
        self.cam_id = self.get_parameter("cam_id").get_parameter_value().string_value
        if not self.cam_id:
            self.get_logger().warn(
                "No cam_id specified — using first available camera."
            )

        # --- ROS publisher
        self.publisher_ = self.create_publisher(
            Image, "/camera/image_raw", qos_profile_sensor_data
        )
        self.bridge = CvBridge()

        # --- Initialize PySpin
        self.system = PySpin.System.GetInstance()
        self.cameras = self.system.GetCameras()
        if not self.cameras.GetSize():
            self.get_logger().error("No FLIR cameras detected.")
            raise RuntimeError("No FLIR cameras detected.")

        # Pick camera
        self.cam = None
        if self.cam_id:
            for cam in self.cameras:
                if str(cam.GetUniqueID()) == self.cam_id:
                    self.cam = cam
                    break
        if not self.cam:
            self.cam = self.cameras[0]
            self.get_logger().info(f"Using camera: {self.cam.GetUniqueID()}")

        # Initialize camera
        self.cam.Init()
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()

        # Register device event handler
        self.event_handler = DeviceEventHandler()
        self.cam.RegisterEventHandler(self.event_handler)
        self.get_logger().info(f"Camera initialized: {self.cam.DeviceSerialNumber}")

        # Timer for publishing frames
        self.prev_time = None
        self.timer = self.create_timer(0.0, self.publish_frame)

    def publish_frame(self):
        try:
            image_result = self.cam.GetNextImage(1000)  # timeout in ms
            if image_result.IsIncomplete():
                self.get_logger().warn(
                    f"Incomplete image: {image_result.GetImageStatus()}"
                )
                image_result.Release()
                return

            img_cv = image_result.GetNDArray()
            image_result.Release()

            # Convert to ROS Image
            msg = self.bridge.cv2_to_imgmsg(img_cv, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_optical_frame"
            self.publisher_.publish(msg)

            # Optional FPS logging
            now = time.time()
            if self.prev_time:
                fps = 1.0 / (now - self.prev_time)
                # self.get_logger().info(f"FPS: {fps:.2f}")
            self.prev_time = now

        except PySpin.SpinnakerException as ex:
            self.get_logger().error(f"Spinnaker error: {ex}")

        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")

    def destroy_node(self):
        # Clean up camera and system
        try:
            if self.cam is not None:
                self.cam.UnregisterEventHandler(self.event_handler)
                self.cam.EndAcquisition()
                self.cam.DeInit()
            self.cameras.Clear()
            self.system.ReleaseInstance()
            self.get_logger().info("Camera and system released successfully")
        except PySpin.SpinnakerException as ex:
            self.get_logger().warn(f"Error releasing camera: {ex}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PySpinCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
