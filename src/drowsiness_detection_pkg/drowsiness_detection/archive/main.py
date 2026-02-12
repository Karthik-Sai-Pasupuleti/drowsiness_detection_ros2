import serial
import time
import os

# --- Configuration ---
SERIAL_PORT = "/dev/ttyACM0"  # Change this to your fan controller device
BAUD_RATE = 115200

# Fan levels (percentages)
FAN_LEVELS = {
    "level 0": 0.0,
    "level 1": float(os.getenv("L1_PCT", 30.0)),  # Default 30%
    "level 2": float(os.getenv("L2_PCT", 60.0)),  # Default 60%
    "level 3": 100.0,
}

# --- Initialize Serial ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to fan controller on {SERIAL_PORT}")
except Exception as e:
    print(f"Failed to connect to serial device: {e}")
    exit(1)


# --- Function to set fan speed ---
def set_fan(level: str):
    if level not in FAN_LEVELS:
        print(f"Invalid fan level: {level}")
        return
    duty = FAN_LEVELS[level]
    command = f"set_duty {duty}\n"
    ser.write(command.encode())
    ser.flush()
    print(f"Fan set to {level} ({duty}%)")


# --- Example usage ---
if __name__ == "__main__":
    try:
        while True:
            for level in FAN_LEVELS.keys():
                set_fan(level)
                time.sleep(5)  # Wait 5 seconds before changing level
    except KeyboardInterrupt:
        print("Stopping fan control...")
        set_fan("level 0")
        ser.close()
