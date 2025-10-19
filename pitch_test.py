#!/usr/bin/env python3
"""
Simple pitch motor test - read and optionally write position
Usage:
    python3 pitch_test.py              # Read current position
    python3 pitch_test.py 300          # Write position 300
"""

import sys
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

# Configuration from rt_200.py
PITCH_MOTOR_ID = 5
PITCH_MIN = 0
PITCH_MAX = 600
PORT = "/dev/ttyACM0"

def main():
    # Parse command line
    write_position = None
    if len(sys.argv) > 1:
        try:
            write_position = int(sys.argv[1])
            if write_position < PITCH_MIN or write_position > PITCH_MAX:
                print(f"Error: Position must be between {PITCH_MIN} and {PITCH_MAX}")
                sys.exit(1)
        except ValueError:
            print(f"Usage: {sys.argv[0]} [position]")
            sys.exit(1)

    # Setup motor
    motors = {
        "pitch": Motor(PITCH_MOTOR_ID, "sts3215", MotorNormMode.RANGE_M100_100)
    }

    print(f"Connecting to pitch motor (ID {PITCH_MOTOR_ID}) on {PORT}...")
    motor_bus = FeetechMotorsBus(PORT, motors)
    motor_bus.connect(handshake=False)
    print("✓ Connected")

    # Read current position
    current_pos = motor_bus.read("Present_Position", "pitch", normalize=False)
    print(f"Current pitch position: {int(current_pos)}")

    # Write if requested
    if write_position is not None:
        print(f"Writing position: {write_position}")
        motor_bus.write("Goal_Position", "pitch", write_position, normalize=False)
        print("✓ Position written")

    # Disconnect
    motor_bus.disconnect()
    print("✓ Disconnected")

if __name__ == "__main__":
    main()
