#!/usr/bin/env python3chekc
"""
HeartRateNode (PPG + Battery Check)
- Connects to BioPoint (MAC: D3:9E:07:2D:DA:A1)
- Checks Battery % on startup
- Publishes Raw PPG & BPM
"""

import time
import threading
from collections import deque
import numpy as np
from scipy import signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32

try:
    import sifi_bridge_py as sbp
    SIFI_AVAILABLE = True
except ImportError:
    SIFI_AVAILABLE = False

TARGET_MAC = "D3:9E:07:2D:DA:A1"
ACTUAL_FS = 200
WINDOW_SEC = 10
BUFFER_SIZE = ACTUAL_FS * WINDOW_SEC
HR_CALC_INTERVAL = 10.0
DEBUG = True

class KalmanFilter1D:
    def __init__(self, R=5.0, Q=0.1, initial_value=None):
        self.R = R
        self.Q = Q
        self.x = initial_value
        self.P = 1.0
        self.initialized = initial_value is not None
    
    def update(self, m):
        if not self.initialized:
            self.x = m
            self.P = 1.0
            self.initialized = True
            return self.x
        x_p = self.x
        P_p = self.P + self.Q
        K = P_p / (P_p + self.R)
        self.x = x_p + K * (m - x_p)
        self.P = (1 - K) * P_p
        return self.x

def estimate_heart_rate(ppg_data, fs):
    if len(ppg_data) < fs * 2:
        return None, []
    
    sig_arr = signal.detrend(np.array(ppg_data))
    b, a = signal.butter(3, [0.5, 4], "bandpass", fs=fs)
    filtered = signal.filtfilt(b, a, sig_arr)
    
    sig_range = np.ptp(filtered)
    if sig_range < 0.05:
        return None, filtered

    peaks, _ = signal.find_peaks(filtered, distance=fs * 0.3, prominence=sig_range * 0.1)
    if len(peaks) < 3:
        return None, filtered

    ibi = np.diff(peaks) / fs
    median_ibi = np.median(ibi)
    valid_ibis = ibi[np.abs(ibi - median_ibi) < 0.5 * median_ibi]
    
    if len(valid_ibis) == 0:
        return None, filtered
    
    bpm = 60.0 / np.mean(valid_ibis)
    return bpm, filtered

def extract_battery_value(bat_data):
    """Extract scalar battery value from list or scalar"""
    if isinstance(bat_data, list):
        return float(bat_data[0]) if len(bat_data) > 0 else 0.0
    return float(bat_data)

class RobustSifiSensor:
    def __init__(self, mac=TARGET_MAC):
        self.mac = mac
        self.sb = sbp.SifiBridge() if SIFI_AVAILABLE else None
        self.battery_cache = 0.0

    def connect_and_configure(self):
        if not self.sb:
            return False
        try:
            try:
                success = self.sb.connect(self.mac)
            except:
                success = self.sb.connect()
            
            if not success:
                return False
            time.sleep(1.5)
            
            self.sb.configure_ppg(ir=0, green=30, red=0, blue=0, sens=sbp.PpgSensitivity.HIGH)
            
            try:
                self.sb.configure_emg(enabled=False)
            except:
                pass
            try:
                self.sb.configure_imu(enabled=False)
            except:
                pass
            try:
                self.sb.configure_eda(enabled=False)
            except:
                pass
            
            self.sb.start()
            return True
        except:
            return False

    def get_battery_level(self):
        """
        Waits for battery data in any stream packet.
        BioPoint sends battery in mixed data/status packets.
        """
        start = time.time()
        while time.time() - start < 5.0:
            try:
                pkt = self.sb.get_data()
                if pkt and 'data' in pkt:
                    if 'battery_%' in pkt['data']:
                        bat_val = extract_battery_value(pkt['data']['battery_%'])
                        self.battery_cache = bat_val
                        return bat_val
            except:
                pass
            time.sleep(0.05)
        return self.battery_cache if self.battery_cache > 0 else None

    def get_ppg_packet(self):
        try:
            return self.sb.get_ppg()
        except:
            return None
    
    def disconnect(self):
        try:
            self.sb.stop()
            time.sleep(0.5)
            self.sb.disconnect()
        except:
            pass

class HeartRateNode(Node):
    def __init__(self):
        super().__init__("heart_rate_node")
        self.lock = threading.Lock()
        
        self._instant_bpm = 0.0
        self._smooth_bpm = 0.0
        self._battery_level = 0.0
        self.kf = KalmanFilter1D(R=5.0, Q=0.1)
        self._running = True
        
        self.publisher_ = self.create_publisher(Float32MultiArray, "heart_rate_bpm", 10)
        self.ppg_publisher_ = self.create_publisher(Float32MultiArray, "raw_ppg", 10)
        self.bat_publisher_ = self.create_publisher(Float32, "heart_rate_battery", 10)
        
        self.timer = self.create_timer(0.1, self.publish_callback)
        self._sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self._sensor_thread.start()
        
        self.get_logger().info("HeartRateNode Started")

    def publish_callback(self):
        msg = Float32MultiArray()
        with self.lock:
            msg.data = [self._instant_bpm, self._smooth_bpm]
        self.publisher_.publish(msg)
        
        bat_msg = Float32()
        with self.lock:
            bat_msg.data = float(self._battery_level)
        self.bat_publisher_.publish(bat_msg)

    def _sensor_loop(self):
        if not SIFI_AVAILABLE:
            self.get_logger().error("sifi_bridge_py not installed")
            return

        sensor = RobustSifiSensor(TARGET_MAC)
        calc_buffer = deque(maxlen=BUFFER_SIZE)
        last_hr_time = 0
        battery_read = False
        buffer_fill_start_time = None
        
        while self._running and rclpy.ok():
            self.get_logger().info(f"Connecting to {TARGET_MAC}...")
            
            if sensor.connect_and_configure():
                if not battery_read:
                    bat = sensor.get_battery_level()
                    if bat is not None:
                        with self.lock:
                            self._battery_level = bat
                        self.get_logger().info(f"\n{'='*30}\n[BATTERY] Sensor Battery: {bat}%\n{'='*30}")
                        battery_read = True
                    else:
                        self.get_logger().warn("[BATTERY] Could not read battery level")
                
                self.get_logger().info("Streaming PPG Data...")
                buffer_fill_start_time = time.time()
            else:
                time.sleep(2.0)
                continue
            
            while self._running and rclpy.ok():
                packet = sensor.get_ppg_packet()
                now = time.time()
                
                if packet and "data" in packet:
                    if "battery_%" in packet["data"]:
                        with self.lock:
                            self._battery_level = extract_battery_value(packet["data"]["battery_%"])
                    
                    if "g" in packet["data"]:
                        gvals = packet["data"]["g"]
                        if gvals:
                            raw_msg = Float32MultiArray()
                            raw_msg.data = [float(x) for x in gvals]
                            self.ppg_publisher_.publish(raw_msg)
                            
                            calc_buffer.extend(gvals)
                            
                            if buffer_fill_start_time is None:
                                buffer_fill_start_time = now
                            
                            elapsed_time = now - buffer_fill_start_time
                            
                            if elapsed_time >= HR_CALC_INTERVAL and len(calc_buffer) >= BUFFER_SIZE:
                                hr, _ = estimate_heart_rate(list(calc_buffer), ACTUAL_FS)
                                if hr:
                                    with self.lock:
                                        self._instant_bpm = round(hr, 1)
                                        self._smooth_bpm = round(self.kf.update(hr), 1)
                                    bat_display = self._battery_level
                                    self.get_logger().info(f"BPM: {self._smooth_bpm} (Raw: {hr:.1f}) | Bat: {bat_display}% | Buffer: {len(calc_buffer)}")
                                
                                buffer_fill_start_time = now
                
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
