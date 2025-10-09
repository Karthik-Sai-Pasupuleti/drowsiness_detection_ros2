#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray

from drowsiness_detection_pkg.bot.controls import WheelControlVibration, VoiceControl  


class ControlNode(Node):
    def __init__(self):
        super().__init__('wheel_voice_node')

        # Instantiate your existing classes
        self.vibration = WheelControlVibration()
        self.voice = VoiceControl()

        # ROS2 subscriptions
        self.subscription_vibration = self.create_subscription(
            Float32MultiArray,  # [duration, intensity]
            'control/vibrate',
            self.vibration_callback,
            10
        )
        
        self.subscription_voice = self.create_subscription(
            String,
            'control/speak',
            self.voice_callback,
            10
        )

        self.get_logger().info("WheelVoiceNode initialized. Listening to 'wheel/vibrate' and 'wheel/speak'.")

    def vibration_callback(self, msg: Float32MultiArray):
        """Handle vibration commands."""
        if len(msg.data) < 2:
            self.get_logger().warn("Vibration message must have [duration, intensity]")
            return

        duration = float(msg.data[0])
        intensity = int(msg.data[1])

        self.get_logger().info(f"Vibrating wheel: duration={duration}, intensity={intensity}")
        self.vibration.vibrate(duration=duration, intensity=intensity)

    def voice_callback(self, msg: String):
        """Handle text-to-speech commands."""
        text = msg.data.strip()
        self.get_logger().info(f"Speaking: {text}")
        self.voice.text_to_speech(text)

    def destroy(self):
        """Ensure cleanup on shutdown."""
        self.vibration.close()
        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()