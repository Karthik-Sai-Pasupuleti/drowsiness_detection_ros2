#!/usr/bin/env python3

"""
HeartRateNode (Robust + Debug + Kalman Filter + Raw PPG Publishing)
"""

import time
import threading
from collections import deque
import numpy as np
from scipy import signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Try importing SiFi
try:
    import sifi_bridge_py as sbp
    SIFI_AVAILABLE = True
except ImportError:
    SIFI_AVAILABLE = False

# ------------------- CONFIG -------------------
ACTUAL_FS = 200
WINDOW_SEC = 10
BUFFER_SIZE = ACTUAL_FS * WINDOW_SEC
GET_PPG_TIMEOUT = 2.0
TIMEOUT_RETRY_LIMIT = 5
RECONNECT_DELAY = 2.0
HR_INTERVAL = 10.0
DEBUG = True 

# ------------------- KALMAN FILTER -------------------
class KalmanFilter1D:
    def __init__(self, R=5.0, Q=0.1, initial_value=None):
        self.R = R
        self.Q = Q
        self.x = initial_value 
        self.P = 1.0 
        self.initialized = initial_value is not None

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.P = 1.0
            self.initialized = True
            return self.x
        
        x_pred = self.x
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        return self.x

# ------------------- HEART-RATE CALCULATOR -------------------
def estimate_heart_rate(ppg_data, fs):
    if len(ppg_data) < fs * 2:
        return None, []

    # 1. Detrend
    sig_arr = signal.detrend(np.array(ppg_data))
    
    # 2. Bandpass Filter (0.5 - 4 Hz)
    b, a = signal.butter(3, [0.5, 4], "bandpass", fs=fs)
    filtered = signal.filtfilt(b, a, sig_arr)
    
    # 3. Check signal range
    sig_range = np.ptp(filtered)
    if sig_range < 0.05:
        if DEBUG: print(f"DEBUG: Signal too flat (range={sig_range})")
        return None, filtered

    # 4. Find Peaks
    peaks, _ = signal.find_peaks(filtered, distance=fs * 0.3, prominence=sig_range * 0.1)
    
    if len(peaks) < 3:
        if DEBUG: print("DEBUG: Too few peaks for HR")
        return None, filtered
        
    # 5. Calculate IBI
    ibi = np.diff(peaks) / fs
    median_ibi = np.median(ibi)
    valid_ibis = ibi[np.abs(ibi - median_ibi) < 0.5 * median_ibi]
    
    if len(valid_ibis) == 0:
        return None, filtered
        
    # 6. Compute BPM
    bpm = 60.0 / np.mean(valid_ibis)
    return bpm, filtered

# ------------------- HEART RATE NODE -------------------
class HeartRateNode(Node):
    def __init__(self):
        super().__init__("heart_rate_node_debug")
        self.lock = threading.Lock()
        self._instant_bpm = 0.0
        self._smooth_bpm = 0.0
        self._bpm_history = deque(maxlen=3)

        self.kf = KalmanFilter1D(R=5.0, Q=0.1)
        self._running = True

        # Existing Publisher for BPM
        self.publisher_ = self.create_publisher(Float32MultiArray, "heart_rate_bpm", 10)
        
        # --- NEW: Publisher for Raw PPG ---
        self.ppg_publisher_ = self.create_publisher(Float32MultiArray, "raw_ppg", 10)

        self.timer = self.create_timer(0.1, self.publish_callback)
        self._sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self._sensor_thread.start()
        
        self.get_logger().info("HeartRateNode (Robust + Debug + Kalman) started.")

    def get_ppg_with_timeout(self, sb, timeout=GET_PPG_TIMEOUT):
        result = [None]
        def worker():
            try:
                result[0] = sb.get_ppg()
            except Exception as e:
                result[0] = e
        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return TimeoutError("get_ppg() timeout")
        return result[0]

    def publish_callback(self):
        msg = Float32MultiArray()
        with self.lock:
            msg.data = [self._instant_bpm, self._smooth_bpm]
        self.publisher_.publish(msg)

    def _check_bluetooth_status(self):
        import subprocess
        try:
            result = subprocess.run(['hciconfig'], capture_output=True, text=True)
            return "UP RUNNING" in result.stdout
        except Exception:
            return True

    def _connect_sensor(self):
        sb = sbp.SifiBridge()
        sb.connect()
        time.sleep(1.5)
        sb.configure_ppg(ir=0, green=30, red=0, blue=0, sens=sbp.PpgSensitivity.HIGH)
        sb.start()
        return sb

    def _sensor_loop(self):
        if not SIFI_AVAILABLE:
            self.get_logger().error("sifi_bridge_py not installed")
            return

        timeout_streak = 0
        reconnect_needed = False

        # Initial connect loop
        while self._running and rclpy.ok():
            try:
                if not self._check_bluetooth_status():
                    self.get_logger().error("Bluetooth off/unavailable")
                    time.sleep(2)
                    continue
                self.get_logger().info("Connecting to BioPoint sensor...")
                sb = self._connect_sensor()
                self.get_logger().info("Sensor connected and streaming")
                break
            except Exception as e:
                self.get_logger().warn(f"Connect failed: {e}")
                time.sleep(RECONNECT_DELAY)

        calc_buffer = []
        last_hr_time = 0.0
        last_warn_time = 0.0

        while self._running and rclpy.ok():
            if reconnect_needed:
                try: sb.stop(); sb.disconnect()
                except: pass
                self.get_logger().info("Reconnecting to sensor...")
                time.sleep(RECONNECT_DELAY)
                try:
                    sb = self._connect_sensor()
                    self.get_logger().info("Reconnected.")
                    timeout_streak = 0
                    reconnect_needed = False
                except Exception as e:
                    self.get_logger().warn(f"Reconnect failed: {e}")
                    time.sleep(2)
                    continue

            # Safe read
            packet = self.get_ppg_with_timeout(sb)
            now = time.time()

            if isinstance(packet, TimeoutError):
                timeout_streak += 1
                if now - last_warn_time > 5:
                    self.get_logger().warn("Sensor not responding (timeout)")
                    last_warn_time = now
                if timeout_streak >= TIMEOUT_RETRY_LIMIT:
                    reconnect_needed = True
                continue

            if isinstance(packet, Exception):
                timeout_streak += 1
                if timeout_streak >= TIMEOUT_RETRY_LIMIT:
                    reconnect_needed = True
                continue

            timeout_streak = 0

            # Process Data
            if packet and "data" in packet and "g" in packet["data"]:
                gvals = packet["data"]["g"]
                
                if gvals:
                    # --- NEW: Publish Raw PPG Data immediately ---
                    ppg_msg = Float32MultiArray()
                    ppg_msg.data = [float(x) for x in gvals]
                    self.ppg_publisher_.publish(ppg_msg)
                    # ---------------------------------------------

                    calc_buffer.extend(gvals)
                    if len(calc_buffer) > BUFFER_SIZE:
                        calc_buffer = calc_buffer[-BUFFER_SIZE:]

            # HR calculation (unchanged)
            if now - last_hr_time >= HR_INTERVAL:
                if len(calc_buffer) >= ACTUAL_FS * 2:
                    hr, filtered = estimate_heart_rate(calc_buffer, ACTUAL_FS)
                    if hr:
                        with self.lock:
                            self._instant_bpm = round(hr, 1)
                            self._bpm_history.append(hr)
                            smoothed_val = self.kf.update(hr)
                            self._smooth_bpm = round(smoothed_val, 1)
                            
                        self.get_logger().info(f"BPM: {self._smooth_bpm} (Raw: {hr:.1f})")
                    last_hr_time = now
            
            time.sleep(0.005)

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