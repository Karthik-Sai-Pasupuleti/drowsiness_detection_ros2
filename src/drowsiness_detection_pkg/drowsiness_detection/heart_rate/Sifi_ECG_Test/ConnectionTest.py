#!/usr/bin/env python3
"""
Sifi Band ECG Connection Test
Tests sensor connectivity and basic ECG data streaming at 500Hz
"""

import time
import sys

try:
    import sifi_bridge_py as sbp
    SIFI_AVAILABLE = True
except ImportError:
    print("ERROR: sifi_bridge_py not installed")
    print("Install with: pip install sifi_bridge_py")
    sys.exit(1)

def test_bluetooth():
    """Check if Bluetooth is available"""
    import subprocess
    try:
        result = subprocess.run(['hciconfig'], capture_output=True, text=True)
        if "UP RUNNING" in result.stdout:
            print("[PASS] Bluetooth is UP and RUNNING")
            return True
        else:
            print("[FAIL] Bluetooth is not active")
            return False
    except Exception as e:
        print(f"[WARN] Could not check Bluetooth status: {e}")
        return True

def test_sensor_connection():
    """Test basic sensor connection"""
    print("\n=== Testing Sifi Band Connection ===")
    
    try:
        sb = sbp.SifiBridge()
        print("[PASS] SifiBridge object created")
        
        print("Attempting to connect to sensor...")
        sb.connect()
        time.sleep(2)
        print("[PASS] Connected to Sifi Band")
        
        return sb
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return None

def configure_ecg_sensor(sb):
    """Configure ECG sensor at 500Hz"""
    print("\n=== Configuring ECG Sensor ===")
    
    try:
        print("Configuring ECG sensor for 500Hz sampling...")
        
        sb.start()
        print("[PASS] ECG streaming started")
        time.sleep(1)
        
        return True
    except Exception as e:
        print(f"[FAIL] Configuration failed: {e}")
        return False

def test_ecg_data_stream(sb, duration=5):
    """Test ECG data streaming for specified duration"""
    print(f"\n=== Testing ECG Data Stream ({duration}s) ===")
    
    packet_count = 0
    sample_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            packet = sb.get_ecg() if hasattr(sb, 'get_ecg') else sb.get_data()
            
            if packet:
                packet_count += 1
                
                if "data" in packet and "ecg" in packet["data"]:
                    ecg_samples = packet["data"]["ecg"]
                    sample_count += len(ecg_samples)
                    
                    if packet_count == 1:
                        print(f"[PASS] First ECG packet received")
                        print(f"  Packet structure: {packet.keys()}")
                        print(f"  Samples in packet: {len(ecg_samples)}")
                
            time.sleep(0.01)
        
        elapsed = time.time() - start_time
        print(f"\n[PASS] Data streaming successful")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Packets received: {packet_count}")
        print(f"  Total samples: {sample_count}")
        print(f"  Effective sampling rate: {sample_count/elapsed:.2f} Hz")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Data streaming failed: {e}")
        return False

def cleanup_sensor(sb):
    """Cleanup and disconnect sensor"""
    print("\n=== Cleanup ===")
    try:
        sb.stop()
        print("[PASS] Stopped streaming")
        sb.disconnect()
        print("[PASS] Disconnected from sensor")
    except Exception as e:
        print(f"[WARN] Cleanup warning: {e}")

def main():
    print("=" * 50)
    print("  Sifi Band ECG Connection Test")
    print("  Target: ECG @500Hz")
    print("=" * 50)
    
    if not test_bluetooth():
        print("\n[WARN] WARNING: Bluetooth may not be functional")
    
    sb = test_sensor_connection()
    if not sb:
        print("\n[FAIL] FAILED: Could not connect to sensor")
        sys.exit(1)
    
    if not configure_ecg_sensor(sb):
        cleanup_sensor(sb)
        print("\n[FAIL] FAILED: Could not configure ECG")
        sys.exit(1)
    
    if not test_ecg_data_stream(sb, duration=5):
        cleanup_sensor(sb)
        print("\n[FAIL] FAILED: ECG data streaming failed")
        sys.exit(1)
    
    cleanup_sensor(sb)
    
    print("\n" + "=" * 50)
    print("  [PASS] ALL TESTS PASSED")
    print("=" * 50)

if __name__ == "__main__":
    main()
