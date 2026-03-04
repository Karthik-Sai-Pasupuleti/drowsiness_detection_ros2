#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import json
import numpy as np
from collections import deque
from std_msgs.msg import String, Float32
from drowsiness_detection_msg.msg import EarMarValue

class DataAggregatorNode(Node):
    def __init__(self):
        super().__init__('data_aggregator_node')
        
        # Buffers for 20 seconds of data (assuming ~30fps for EAR/MAR and 1fps for BPM)
        self.ear_buffer = deque(maxlen=600)
        self.mar_buffer = deque(maxlen=600)
        self.bpm_buffer = deque(maxlen=20)
        
        # Subscribers
        self.create_subscription(EarMarValue, '/ear_mar', self.ear_mar_callback, 10)
        self.create_subscription(Float32, '/smooth_bpm', self.bpm_callback, 10)
        
        # Publisher
        self.agg_pub = self.create_publisher(String, '/aggregated_metrics_20s', 10)
        
        # Timer (20 seconds)
        self.create_timer(20.0, self.publish_aggregated_data)
        self.get_logger().info("Data Aggregator Node Started. Publishing every 20s.")

    def ear_mar_callback(self, msg):
        self.ear_buffer.append(msg.ear_value)
        self.mar_buffer.append(msg.mar_value)

    def bpm_callback(self, msg):
        self.bpm_buffer.append(msg.data)

    def publish_aggregated_data(self):
        if not self.ear_buffer:
            self.get_logger().warn("No data in buffers to aggregate.")
            return

        metrics = {
            "ear_mean": round(float(np.mean(self.ear_buffer)), 3),
            "ear_min": round(float(np.min(self.ear_buffer)), 3),
            "mar_mean": round(float(np.mean(self.mar_buffer)), 3),
            "mar_max": round(float(np.max(self.mar_buffer)), 3),
            "bpm_mean": round(float(np.mean(self.bpm_buffer)), 1) if self.bpm_buffer else "N/A"
        }

        msg = String()
        msg.data = json.dumps(metrics)
        self.agg_pub.publish(msg)
        self.get_logger().info(f"Published 20s aggregated metrics: {msg.data}")
        
        # Clear buffers for next window
        self.ear_buffer.clear()
        self.mar_buffer.clear()
        self.bpm_buffer.clear()

def main(args=None):
    rclpy.init(args=args)
    node = DataAggregatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
