#!/usr/bin/env python3
import glob
from lerobot.motors.feetech import FeetechMotorsBus

# Find USB serial ports
ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')

if not ports:
    print("No USB serial ports found!")
    print("Make sure the motor driver is connected.")
    exit(1)

print(f"Found ports: {ports}")
print()

for port in ports:
    print(f"Scanning {port}...")
    try:
        bus = FeetechMotorsBus(port=port, motors={})
        bus.connect(handshake=False)

        # Try broadcast ping
        motors_found = bus.broadcast_ping()

        if motors_found:
            print(f"✓ Found {len(motors_found)} motor(s) on {port}:")
            for motor_id, model_num in motors_found.items():
                print(f"  - ID {motor_id}: Model {model_num}")
        else:
            print(f"  No motors found on {port}")

        bus.disconnect()
    except Exception as e:
        print(f"  Error: {e}")
    print()
