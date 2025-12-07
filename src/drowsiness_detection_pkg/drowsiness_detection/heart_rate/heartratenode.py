#!/usr/bin/env python3
"""
HeartRateNode
-------------
ROS2 node that connects to the SiFi / BioPoint sensor via sifi_bridge_py,
computes Instant and Smoothed BPM using the new prominence-based logic,
and publishes them on /heart_rate_bpm as std_msgs/Float32MultiArray:

  msg.data[0] = instant_bpm
  msg.data[1] = smooth_bpm

This is adapted from trail_new_logic.py without its WebSocket / labelling parts.
"""

import time
import threading
from collections import deque

import numpy as np
from scipy import signal

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# --- Try to import SiFi bridge ---
try:
    import sifi_bridge_py as sbp
    SIFI_AVAILABLE = True
except ImportError:
    SIFI_AVAILABLE = False


# --- CONFIGURATION (Signal Processing) ---
ACTUAL_FS = 200          # Hz, sample rate of green PPG
WINDOW_SEC = 10          # seconds of buffer
BUFFER_SIZE = ACTUAL_FS * WINDOW_SEC


def estimate_heart_rate(ppg_data, fs):
    """
    Heart Rate Estimation with Prominence (from trail_new_logic.py).

    Returns:
        bpm (float or None), filtered_signal (np.ndarray)
    """
    if len(ppg_data) < fs * 2:
        return None, []

    sig_arr = signal.detrend(np.array(ppg_data))
    b, a = signal.butter(3, [0.5, 4], 'bandpass', fs=fs)
    filtered = signal.filtfilt(b, a, sig_arr)

    sig_range = np.ptp(filtered)
    peaks, _ = signal.find_peaks(
        filtered,
        distance=fs * 0.4,
        prominence=sig_range * 0.25
    )

    if len(peaks) < 3:
        return None, filtered

    ibi = np.diff(peaks) / fs
    median_ibi = np.median(ibi)

    valid_ibis = ibi[np.abs(ibi - median_ibi) < (0.25 * median_ibi)]
    if len(valid_ibis) == 0:
        return None, filtered

    bpm = 60.0 / np.mean(valid_ibis)
    return bpm, filtered


class HeartRateNode(Node):
    def __init__(self):
        super().__init__("heart_rate_node")

        self.publisher_ = self.create_publisher(
            Float32MultiArray,
            "heart_rate_bpm",
            10
        )

        self.get_logger().info("HeartRateNode initialized.")

        self._instant_bpm = 0.0
        self._smooth_bpm = 0.0
        self._bpm_history = deque(maxlen=3)

        self._running = True
        self._sensor_thread = threading.Thread(
            target=self._sensor_loop,
            daemon=True
        )
        self._sensor_thread.start()

    # --- SiFi sensor loop (adapted from trail_new_logic.py) ---
    def _sensor_loop(self):
        if not SIFI_AVAILABLE:
            self.get_logger().error(
                "sifi_bridge_py not found. HeartRateNode will not stream data."
            )
            return

        self.get_logger().info("Connecting to SiFi / BioPoint sensor...")

        try:
            sb = sbp.SifiBridge()
            sb.connect()
            # Example: Green only, high sensitivity – same as trail_new_logic.py
            sb.configure_ppg(ir=0, green=30, red=0, blue=0,
                             sens=sbp.PpgSensitivity.HIGH)
            sb.start()
            self.get_logger().info(
                f"Sensor started. Streaming Green PPG at {ACTUAL_FS} Hz."
            )
        except Exception as e:
            self.get_logger().error(f"Failed to start sensor: {e}")
            return

        calc_buffer = []
        last_calc_time = 0.0

        try:
            while self._running and rclpy.ok():
                packet = sb.get_ppg()

                if packet and "data" in packet and "g" in packet["data"]:
                    new_green = packet["data"]["g"]
                    if new_green:
                        current_time = time.time()

                        # 1. Maintain rolling buffer
                        calc_buffer.extend(new_green)
                        if len(calc_buffer) > BUFFER_SIZE:
                            calc_buffer = calc_buffer[-BUFFER_SIZE:]

                        # 2. Recalculate HR every 10s
                        if current_time - last_calc_time >= 10.0:
                            if len(calc_buffer) >= ACTUAL_FS * 2:
                                hr, _ = estimate_heart_rate(
                                    calc_buffer, ACTUAL_FS)
                                if hr:
                                    self._instant_bpm = round(hr, 1)
                                    self._bpm_history.append(hr)
                                    self._smooth_bpm = round(
                                        sum(self._bpm_history)
                                        / len(self._bpm_history),
                                        1,
                                    )
                                    self.get_logger().info(
                                        f"HR updated: instant={self._instant_bpm}, "
                                        f"smooth={self._smooth_bpm}"
                                    )

                            last_calc_time = current_time

                        # 3. Publish last known values continuously (10 Hz)
                        msg = Float32MultiArray()
                        msg.data = [float(self._instant_bpm),
                                    float(self._smooth_bpm)]
                        self.publisher_.publish(msg)

                time.sleep(0.1)  # 10 Hz publish/update

        except Exception as e:
            self.get_logger().error(f"Sensor loop error: {e}")
        finally:
            try:
                sb.stop()
                sb.disconnect()
            except Exception:
                pass
            self.get_logger().info("Sensor loop stopped; node shutting down.")

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HeartRateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received; shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
