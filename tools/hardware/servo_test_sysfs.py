#!/usr/bin/env python3
"""
Servo Test using sysfs PWM (hardware PWM)
This uses the Linux sysfs interface for hardware PWM instead of Jetson.GPIO

Jetson Orin PWM channels:
- PWM0 (Pin 32, GPIO12): /sys/class/pwm/pwmchip0/pwm0
- PWM1 (Pin 33, GPIO13): /sys/class/pwm/pwmchip0/pwm1
"""

import time
import os
import sys

# Configuration for PWM1 (Pin 33, GPIO13)
PWM_CHIP = 0
PWM_CHANNEL = 1  # PWM1 for GPIO13/Pin33 (use 0 for PWM0/GPIO12/Pin32)
PWM_PERIOD = 20000000  # 20ms period (50Hz) in nanoseconds
PWM_FREQ = 50  # 50Hz for standard servos

class SysfsPWMServo:
    def __init__(self, chip=0, channel=1):
        self.chip = chip
        self.channel = channel
        self.pwm_path = f"/sys/class/pwm/pwmchip{chip}/pwm{channel}"
        self.export_path = f"/sys/class/pwm/pwmchip{chip}/export"
        self.unexport_path = f"/sys/class/pwm/pwmchip{chip}/unexport"

    def setup(self):
        """Export and configure PWM channel"""
        print(f"Setting up PWM{self.channel} via sysfs...")

        # Export PWM channel if not already exported
        if not os.path.exists(self.pwm_path):
            try:
                with open(self.export_path, 'w') as f:
                    f.write(str(self.channel))
                print(f"✓ Exported PWM{self.channel}")
                time.sleep(0.1)  # Give time for sysfs to create files
            except IOError as e:
                print(f"Warning: Could not export PWM (might already be exported): {e}")

        # Set period (20ms = 50Hz)
        try:
            with open(f"{self.pwm_path}/period", 'w') as f:
                f.write(str(PWM_PERIOD))
            print(f"✓ Set period to {PWM_PERIOD}ns (50Hz)")
        except IOError as e:
            print(f"Error setting period: {e}")
            return False

        # Enable PWM
        try:
            with open(f"{self.pwm_path}/enable", 'w') as f:
                f.write('1')
            print(f"✓ PWM enabled")
        except IOError as e:
            print(f"Error enabling PWM: {e}")
            return False

        return True

    def set_angle(self, angle):
        """Set servo angle (0-180 degrees)"""
        if angle < 0 or angle > 180:
            angle = max(0, min(180, angle))

        # Convert angle to pulse width
        # 0° = 1ms (5% duty), 90° = 1.5ms (7.5% duty), 180° = 2ms (10% duty)
        min_pulse = 1000000  # 1ms in nanoseconds
        max_pulse = 2000000  # 2ms in nanoseconds

        pulse_width = int(min_pulse + (angle / 180.0) * (max_pulse - min_pulse))
        duty_percent = (pulse_width / PWM_PERIOD) * 100

        print(f"Setting angle {angle}° → pulse width {pulse_width}ns ({duty_percent:.1f}% duty)")

        try:
            with open(f"{self.pwm_path}/duty_cycle", 'w') as f:
                f.write(str(pulse_width))
            return True
        except IOError as e:
            print(f"Error setting duty cycle: {e}")
            return False

    def cleanup(self):
        """Disable and unexport PWM"""
        print("\nCleaning up PWM...")
        try:
            # Disable PWM
            with open(f"{self.pwm_path}/enable", 'w') as f:
                f.write('0')
            print("✓ PWM disabled")
        except:
            pass

        try:
            # Unexport PWM
            with open(self.unexport_path, 'w') as f:
                f.write(str(self.channel))
            print(f"✓ Unexported PWM{self.channel}")
        except:
            pass

def main():
    print("=" * 60)
    print("Jetson Orin Servo Test (sysfs Hardware PWM)")
    print("=" * 60)
    print(f"Using PWM{PWM_CHANNEL} (Pin 33, GPIO13)")
    print(f"Frequency: {PWM_FREQ}Hz, Period: {PWM_PERIOD}ns")
    print()

    servo = SysfsPWMServo(chip=PWM_CHIP, channel=PWM_CHANNEL)

    if not servo.setup():
        print("\n❌ Failed to setup PWM. Make sure you're running with --privileged")
        sys.exit(1)

    try:
        if len(sys.argv) > 1:
            # Single angle test
            angle = float(sys.argv[1])
            print(f"\nMoving to {angle}°...")
            servo.set_angle(angle)
            print("Holding position for 5 seconds...")
            time.sleep(5)
        else:
            # Sweep test
            print("\nStarting sweep test...")
            angles = [0, 45, 90, 135, 180, 90]
            for angle in angles:
                print(f"\nMoving to {angle}°...")
                servo.set_angle(angle)
                time.sleep(2)

        print("\nTest complete!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        servo.cleanup()

if __name__ == "__main__":
    main()
