#!/usr/bin/env python3
"""
This node now waits for LLM trigger and plots the data to feed into the VLM. Submits 8 frames of
camera feed and 3 frames of graphs (EAR, MAR, SMOOTH_BPM) to the VLM.
"""

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Float32
from cv_bridge import CvBridge
from collections import deque
import threading
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .hybrid_vlm_core import VLMCore

class HybridVLMNode(Node):
    def __init__(self):
        super().__init__('hybrid_vlm_node')
        self.bridge = CvBridge()
        self.vlm_core = VLMCore(logger=self.get_logger())
        
        # 2-second Rolling Buffers (assuming 30 FPS = 60 frames)
        self.buffer_lock = threading.Lock()
        self.frame_buffer = deque(maxlen=60)
        self.ear_buffer = deque(maxlen=60)
        self.mar_buffer = deque(maxlen=60)
        self.bpm_buffer = deque(maxlen=60)
        
        # Latest individual values
        self.current_ear = 0.0
        self.current_mar = 0.0
        self.current_bpm = 0.0
        
        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        # Import your specific EarMarValue message type here if different
        from drowsiness_detection_msg.msg import EarMarValue
        self.create_subscription(EarMarValue, '/ear_mar', self.ear_mar_callback, 10)
        self.create_subscription(Float32, '/smooth_bpm', self.bpm_callback, 10)
        
        # Trigger from LLM
        self.create_subscription(Bool, '/trigger_vlm', self.trigger_callback, 10)
        
        # Publisher back to LLM
        self.vlm_pub = self.create_publisher(String, '/vlm_context', 10)
        
        self.is_processing = False
        self.get_logger().info("Hybrid VLM Node initialized (LLM Triggered).")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.buffer_lock:
            self.frame_buffer.append(frame)
            self.ear_buffer.append(self.current_ear)
            self.mar_buffer.append(self.current_mar)
            self.bpm_buffer.append(self.current_bpm)

    def ear_mar_callback(self, msg):
        self.current_ear = msg.ear_value
        self.current_mar = msg.mar_value

    def bpm_callback(self, msg):
        self.current_bpm = msg.data

    def create_graph_base64(self, data, title, ylabel):
        plt.figure(figsize=(4, 3))
        plt.plot(list(data), color='blue', linewidth=2)
        plt.title(title)
        plt.xlabel("Frames (Last 2 seconds)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')

    def trigger_callback(self, msg):
        if msg.data and not self.is_processing:
            self.get_logger().info("Trigger received from LLM! Processing VLM Request...")
            threading.Thread(target=self.process_vlm_event).start()

    def process_vlm_event(self):
        self.is_processing = True
        with self.buffer_lock:
            # Extract 8 keyframes from the 60-frame buffer
            step = max(1, len(self.frame_buffer) // 8)
            keyframes = [self.frame_buffer[i] for i in range(0, len(self.frame_buffer), step)][:8]
            
            # Generate Graph Base64 Strings
            ear_b64 = self.create_graph_base64(self.ear_buffer, "EAR (2 sec)", "EAR")
            mar_b64 = self.create_graph_base64(self.mar_buffer, "MAR (2 sec)", "MAR")
            bpm_b64 = self.create_graph_base64(self.bpm_buffer, "BPM (2 sec)", "BPM")

        # Encode camera frames
        frame_b64s = []
        for frame in keyframes:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64s.append(base64.b64encode(buffer).decode('utf-8'))

        # Combine all images (8 frames + 3 graphs)
        all_images = frame_b64s + [ear_b64, mar_b64, bpm_b64]
        
        prompt = "Analyze these 8 sequential frames of the driver alongside the 3 graphs for Eye Aspect Ratio, Mouth Aspect Ratio, and Heart Rate. Explain why the metrics look anomalous or if the driver is genuinely drowsy."
        
        result_json = self.vlm_core.query_qwen_vl(all_images, prompt)
        
        if result_json:
            msg = String()
            msg.data = json.dumps(result_json)
            self.vlm_pub.publish(msg)
            self.get_logger().info("VLM analysis published back to LLM.")
        
        self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = HybridVLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
