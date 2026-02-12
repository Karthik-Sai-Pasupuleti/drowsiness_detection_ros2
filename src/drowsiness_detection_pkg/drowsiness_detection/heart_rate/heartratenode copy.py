#!/usr/bin/env python3

"""
HeartRateNode (Robust Threaded Version)
---------------------------------------
Separates sensor reading and ROS publishing into different threads.
If the sensor blocks or fails, the node continues to publish the last known value (or 0.0).
"""

import time
import threading
import sys
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


# --- CONFIGURATION ---
ACTUAL_FS = 200        # Hz
WINDOW_SEC = 10
BUFFER_SIZE = ACTUAL_FS * WINDOW_SEC
CONNECTION_TIMEOUT = 10.0


def estimate_heart_rate(ppg_data, fs):
    """Calculates BPM from Green PPG signal."""
    if len(ppg_data) < fs * 2:
        return None, []

    # Filter
    sig_arr = signal.detrend(np.array(ppg_data))
    b, a = signal.butter(3, [0.5, 4], 'bandpass', fs=fs)
    filtered = signal.filtfilt(b, a, sig_arr)

    # Peak Detection
    sig_range = np.ptp(filtered)
    peaks, _ = signal.find_peaks(filtered, distance=fs*0.4, prominence=sig_range*0.25)

    if len(peaks) < 3:
        return None, filtered

    ibi = np.diff(peaks) / fs
    median_ibi = np.median(ibi)
    valid_ibis = ibi[np.abs(ibi - median_ibi) < (0.25 * median_ibi)]

    if len(valid_ibis) == 0:
        return None, filtered

    return 60.0 / np.mean(valid_ibis), filtered


class HeartRateNode(Node):
    def __init__(self):
        super().__init__("heart_rate_node")
        
        # 1. SHARED STATE (Thread-Safe)
        self.lock = threading.Lock()
        self._instant_bpm = 0.0
        self._smooth_bpm = 0.0
        self._bpm_history = deque(maxlen=3)
        self._running = True

        # 2. PUBLISHER (Run by ROS Timer)
        self.publisher_ = self.create_publisher(Float32MultiArray, "heart_rate_bpm", 10)
        # Create a timer that fires every 0.1s (10 Hz) no matter what the sensor is doing
        self.timer = self.create_timer(0.1, self.publish_callback)

        self.get_logger().info("HeartRateNode initialized (Async Mode).")

        # 3. SENSOR THREAD
        self._sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self._sensor_thread.start()

    def publish_callback(self):
        """Called by ROS2 executor every 0.1s. Publishes latest data."""
        msg = Float32MultiArray()
        with self.lock:
            msg.data = [float(self._instant_bpm), float(self._smooth_bpm)]
        self.publisher_.publish(msg)

    def _check_bluetooth_status(self):
        """Checks if system Bluetooth is powered on."""
        import subprocess
        try:
            result = subprocess.run(['hciconfig'], capture_output=True, text=True)
            if "UP RUNNING" not in result.stdout:
                return False, "Bluetooth Adapter is OFF or not detected."
            return True, "Bluetooth ON"
        except FileNotFoundError:
            try:
                res = subprocess.run(['systemctl', 'is-active', 'bluetooth'], capture_output=True, text=True)
                if res.stdout.strip() != 'active':
                    return False, "Bluetooth Service is not active."
            except Exception:
                pass
            return True, "Bluetooth Status Unknown (Assuming ON)"

    def _sensor_loop(self):
        """Handles the potentially blocking sensor connection and data reading."""
        if not SIFI_AVAILABLE:
            self.get_logger().error("MISSING LIBRARY: 'sifi_bridge_py' is not installed.")
            return

        # Check Bluetooth
        bt_ok, bt_msg = self._check_bluetooth_status()
        if not bt_ok:
            self.get_logger().error(f"CONNECTION FAILED: {bt_msg}")
            self.get_logger().error("-> Please run: sudo hciconfig hci0 up")
            return

        self.get_logger().info("Scanning for BioPoint Sensor...")

        sb = None
        try:
            sb = sbp.SifiBridge()
            sb.connect()
            time.sleep(2.0)
            self.get_logger().info("Sensor Connected! Configuring PPG...")
            sb.configure_ppg(ir=0, green=30, red=0, blue=0, sens=sbp.PpgSensitivity.HIGH)
            sb.start()
            self.get_logger().info(f"Streaming started at {ACTUAL_FS} Hz.")
        except Exception as e:
            self.get_logger().error(f"SENSOR CONNECTION ERROR: {e}")
            if sb:
                try: sb.stop(); sb.disconnect()
                except: pass
            return

        # Data Accumulation Loop
        calc_buffer = []
        last_calc_time = 0.0

        try:
            while self._running and rclpy.ok():
                try:
                    # BLOCKING CALL RISK: If sensor dies, this might hang or throw
                    packet = sb.get_ppg() 
                except Exception as read_err:
                    self.get_logger().warn(f"Read error: {read_err}")
                    time.sleep(1.0)
                    continue

                if packet and "data" in packet and "g" in packet["data"]:
                    new_green = packet["data"]["g"]
                    if new_green:
                        calc_buffer.extend(new_green)
                        if len(calc_buffer) > BUFFER_SIZE:
                            calc_buffer = calc_buffer[-BUFFER_SIZE:]

                # HR Calculation (Every 10s)
                current_time = time.time()
                if current_time - last_calc_time >= 10.0:
                    if len(calc_buffer) >= ACTUAL_FS * 2:
                        hr, _ = estimate_heart_rate(calc_buffer, ACTUAL_FS)
                        if hr:
                            with self.lock:
                                self._instant_bpm = round(hr, 1)
                                self._bpm_history.append(hr)
                                self._smooth_bpm = round(sum(self._bpm_history) / len(self._bpm_history), 1)
                            
                            self.get_logger().info(f"BPM Update: {self._smooth_bpm}")
                    
                    last_calc_time = current_time
                
                # Small sleep to prevent tight loop if get_ppg() returns instantly empty
                time.sleep(0.005)

        except Exception as e:
            self.get_logger().error(f"Sensor Loop Crash: {e}")
        finally:
            if sb:
                try: sb.stop(); sb.disconnect()
                except: pass
            self.get_logger().info("Sensor disconnected.")

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HeartRateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
