#!/usr/bin/env python3
"""Read or adjust the Feetech pitch motor homing offset.

Examples:
    python3 tools/hardware/pitch_homing_offset.py
    python3 tools/hardware/pitch_homing_offset.py --delta -100
    python3 tools/hardware/pitch_homing_offset.py --delta -100 --apply

Feetech protocol 0 reports:
    Present_Position = Actual_Position - Homing_Offset

So decreasing Homing_Offset by 100 makes the same physical pitch read about 100
raw units higher, which gives the logical range more room to command upward.
"""

import argparse

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode


PITCH_MOTOR_ID = 5
PORT = "/dev/ttyACM0"


def connect_pitch(port):
    motors = {
        "pitch": Motor(PITCH_MOTOR_ID, "sts3215", MotorNormMode.RANGE_M100_100),
    }
    motor_bus = FeetechMotorsBus(port, motors)
    motor_bus.connect(handshake=False)
    return motor_bus


def main():
    parser = argparse.ArgumentParser(description="Read or adjust pitch motor Homing_Offset")
    parser.add_argument("--port", default=PORT, help=f"Serial port (default: {PORT})")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--delta", type=int, help="Add this signed delta to Homing_Offset")
    group.add_argument("--set", dest="set_offset", type=int, help="Set Homing_Offset exactly")
    parser.add_argument("--apply", action="store_true", help="Actually write the new offset")
    args = parser.parse_args()

    motor_bus = connect_pitch(args.port)
    try:
        present = int(motor_bus.read("Present_Position", "pitch", normalize=False))
        calibration = motor_bus.read_calibration()["pitch"]

        print(f"Present pitch position: {present}")
        print(f"Homing_Offset: {calibration.homing_offset}")
        print(f"Min_Position_Limit: {calibration.range_min}")
        print(f"Max_Position_Limit: {calibration.range_max}")

        if args.delta is None and args.set_offset is None:
            return

        if args.set_offset is not None:
            new_offset = args.set_offset
        else:
            new_offset = calibration.homing_offset + args.delta

        same_physical_new_present = present + calibration.homing_offset - new_offset
        print()
        print(f"New Homing_Offset: {new_offset}")
        print(
            "Same physical pitch would read approximately: "
            f"{same_physical_new_present}"
        )

        if not args.apply:
            print("Dry run only. Re-run with --apply to write persistent motor calibration.")
            return

        new_calibration = {
            "pitch": MotorCalibration(
                id=calibration.id,
                drive_mode=calibration.drive_mode,
                homing_offset=new_offset,
                range_min=calibration.range_min,
                range_max=calibration.range_max,
            )
        }

        print("Writing persistent pitch motor calibration...")
        motor_bus.disable_torque("pitch")
        motor_bus.write_calibration(new_calibration, cache=False)
        motor_bus.enable_torque("pitch")
        updated = motor_bus.read_calibration()["pitch"]
        updated_present = int(motor_bus.read("Present_Position", "pitch", normalize=False))
        print(f"Updated Homing_Offset: {updated.homing_offset}")
        print(f"Updated present pitch position: {updated_present}")
    finally:
        motor_bus.disconnect()


if __name__ == "__main__":
    main()
