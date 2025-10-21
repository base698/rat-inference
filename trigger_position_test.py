#!/usr/bin/env python3
"""
Interactive Trigger Servo Position Test
Type numbers (0-180 degrees) to test different positions
Press 'q' to quit

This is for setting the trigger servo positions in rt_200.py
"""

import time
import os
import sys

# Configuration for trigger servo (PWM0, Pin 15, GPIO12)
PWM_CHIP = 0
PWM_CHANNEL = 0  # PWM0 for GPIO12/Pin15 (trigger servo)
PWM_PERIOD = 20000000  # 20ms period (50Hz) in nanoseconds

class TriggerServoTester:
    def __init__(self, chip=0, channel=0):
        self.chip = chip
        self.channel = channel
        self.pwm_path = f"/sys/class/pwm/pwmchip{chip}/pwm{channel}"
        self.export_path = f"/sys/class/pwm/pwmchip{chip}/export"
        self.unexport_path = f"/sys/class/pwm/pwmchip{chip}/unexport"

    def setup(self):
        """Export and configure PWM channel"""
        # Export PWM channel if not already exported
        if not os.path.exists(self.pwm_path):
            try:
                with open(self.export_path, 'w') as f:
                    f.write(str(self.channel))
                time.sleep(0.2)  # Give time for sysfs to create files
            except IOError as e:
                print(f"Warning: Could not export PWM (might already be exported): {e}")

        # Set period (20ms = 50Hz)
        try:
            with open(f"{self.pwm_path}/period", 'w') as f:
                f.write(str(PWM_PERIOD))
        except IOError as e:
            print(f"Error setting period: {e}")
            return False

        # Enable PWM
        try:
            with open(f"{self.pwm_path}/enable", 'w') as f:
                f.write('1')
        except IOError as e:
            print(f"Error enabling PWM: {e}")
            return False

        return True

    def set_angle(self, angle):
        """Set servo angle (0-180 degrees)"""
        # Clamp angle to valid range
        angle = max(0, min(180, angle))

        # Convert angle to pulse width (nanoseconds)
        # MG996R servo: 0.5ms (500000ns) = 0°, 2.5ms (2500000ns) = 180°
        min_pulse = 500000   # 0.5ms
        max_pulse = 2500000  # 2.5ms
        pulse_width = int(min_pulse + (angle / 180.0) * (max_pulse - min_pulse))

        try:
            with open(f"{self.pwm_path}/duty_cycle", 'w') as f:
                f.write(str(pulse_width))
            return True
        except IOError as e:
            print(f"Error setting duty cycle: {e}")
            return False

    def cleanup(self):
        """Disable and unexport PWM"""
        try:
            # Disable PWM
            with open(f"{self.pwm_path}/enable", 'w') as f:
                f.write('0')
        except:
            pass

        try:
            # Unexport PWM
            with open(self.unexport_path, 'w') as f:
                f.write(str(self.channel))
        except:
            pass

def main():
    print("=" * 60)
    print("Trigger Servo Position Tester")
    print("=" * 60)
    print(f"Testing trigger servo on PWM{PWM_CHANNEL} (Pin 15, GPIO12)")
    print()
    print("Usage:")
    print("  - Type a number (0-180) to set servo angle")
    print("  - Press 'q' or 'quit' to exit")
    print()
    print("Recommended positions to test:")
    print("  - Neutral position (no trigger): try 60-80°")
    print("  - Trigger position (activated): try 10-30°")
    print("=" * 60)
    print()

    servo = TriggerServoTester(chip=PWM_CHIP, channel=PWM_CHANNEL)

    if not servo.setup():
        print("❌ Failed to setup PWM. Make sure you're running with --privileged")
        sys.exit(1)

    print("✓ Servo initialized")
    print()

    try:
        while True:
            try:
                user_input = input("Enter angle (0-180) or 'q' to quit: ").strip()

                # Check for quit
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("Exiting...")
                    break

                # Try to parse as a number
                try:
                    angle = float(user_input)

                    if angle < 0 or angle > 180:
                        print(f"⚠️  Angle {angle}° out of range, clamping to 0-180°")
                        angle = max(0, min(180, angle))

                    servo.set_angle(angle)
                    print(f"✓ Moved to {angle}°")

                except ValueError:
                    print("Invalid input. Please enter a number (0-180) or 'q' to quit")

            except EOFError:
                # Handle Ctrl+D
                print("\nExiting...")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        servo.cleanup()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
