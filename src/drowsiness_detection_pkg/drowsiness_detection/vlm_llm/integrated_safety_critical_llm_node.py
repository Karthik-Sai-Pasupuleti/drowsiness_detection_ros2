#!/usr/bin/env python3
"""
This node stores 20s aggregated metrics and VLM context for 10 minutes and then summarizes them.
It uses Llama3.1 8b to evaluate the metrics and decide whether to trigger VLM or not.
It publishes vlm trigger to hybrid_vlm_node and receives vlm context from it.
"""
import rclpy
from rclpy.node import Node
import json
import requests
import os
from datetime import datetime
from std_msgs.msg import String, Bool

class PrimaryLLMNode(Node):
    def __init__(self):
        super().__init__('primary_llm_engine')
        
        self.declare_parameter('data_dir', '/root/ws/drowsiness_data/llm_history')
        self.data_dir = self.get_parameter('data_dir').value
        os.makedirs(self.data_dir, exist_ok=True)

        self.latest_metrics = None
        self.latest_vlm_context = None
        self.current_history_context = "No history established yet."
        self.ten_min_buffer = []

        # Subscriptions
        self.create_subscription(String, '/aggregated_metrics_20s', self.metrics_callback, 10)
        self.create_subscription(String, '/vlm_context', self.vlm_callback, 10)

        # Publishers
        self.trigger_pub = self.create_publisher(Bool, '/trigger_vlm', 10)
        self.decision_pub = self.create_publisher(String, '/final_decision', 10)

        # 10 Minute History Timer
        self.create_timer(600.0, self.update_history_context)

        self.get_logger().info("Primary LLM Engine Started (Llama3.1 8b).")

    def metrics_callback(self, msg):
        self.latest_metrics = msg.data
        self.evaluate_short_term()

    def vlm_callback(self, msg):
        self.latest_vlm_context = msg.data
        self.get_logger().info("Received updated VLM Context.")

    def query_llama(self, prompt, expected_schema=None):
        payload = {
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            response = requests.post("http://host.docker.internal:11434/api/generate", json=payload, timeout=30)
            response.raise_for_status()
            return json.loads(response.json()["response"])
        except Exception as e:
            self.get_logger().error(f"Llama3.1 query failed: {e}")
            return None

    def evaluate_short_term(self):
        if not self.latest_metrics:
            return

        prompt = f"""
        You are evaluating live driver metrics every 20 seconds.
        History Context (Last 10m): {self.current_history_context}
        Latest VLM Feedback: {self.latest_vlm_context if self.latest_vlm_context else 'None'}
        Current 20s Metrics: {self.latest_metrics}

        Task 1: Evaluate if this data represents genuine drowsiness or tracking ambiguity (e.g. EAR dropping instantly but BPM normal).
        Task 2: If ambiguity is detected, set "trigger_vlm": true.

        Respond ONLY in valid JSON matching this schema:
        {{
            "current_state": "alert/drowsy/ambiguous",
            "trigger_vlm": boolean,
            "reasoning": "string"
        }}
        """
        
        result = self.query_llama(prompt)

        if result:
            self.ten_min_buffer.append(result)
            self.get_logger().info(f"LLM 20s Decision: {result}")

            # Publish decision
            msg = String()
            msg.data = json.dumps(result)
            self.decision_pub.publish(msg)

            # Trigger VLM if LLM requests it
            if result.get("trigger_vlm", False):
                trig_msg = Bool()
                trig_msg.data = True
                self.trigger_pub.publish(trig_msg)

    def update_history_context(self):
        if not self.ten_min_buffer:
            return
            
        prompt = f"""
        Summarize the driver's state over the last 10 minutes based on these logged 20-second evaluations:
        {json.dumps(self.ten_min_buffer)}
        Output a JSON with a single "summary" string field detailing the baseline fatigue progression.
        """
        
        result = self.query_llama(prompt)

        if result and "summary" in result:
            self.current_history_context = result["summary"]
            self.get_logger().info(f"Updated 10m History: {self.current_history_context}")

            # Save to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.data_dir, f"history_context_{timestamp}.json")
            with open(filepath, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "10m_summary": self.current_history_context,
                    "logs": self.ten_min_buffer
                }, f, indent=4)
                
        self.ten_min_buffer.clear()

def main(args=None):
    rclpy.init(args=args)
    node = PrimaryLLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
