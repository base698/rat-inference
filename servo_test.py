#!/usr/bin/env python3
"""
PWM Servo diagnostic and test for Jetson Orin Nano
Helps identify correct PWM chip and tests servo control
"""

import os
import time
import sys

# Servo parameters for MG996R
PWM_PERIOD_NS = 20000000  # 20ms period (50Hz)
MIN_PULSE_WIDTH_NS = 500000   # 0.5ms for 0 degrees
MAX_PULSE_WIDTH_NS = 2500000  # 2.5ms for 180 degrees

class PWMServo:
    def __init__(self, chip, channel=0):
        self.chip = chip
        self.channel = channel
        self.pwm_path = f"/sys/class/pwm/pwmchip{chip}/pwm{channel}"
        self.export_path = f"/sys/class/pwm/pwmchip{chip}/export"
        self.unexport_path = f"/sys/class/pwm/pwmchip{chip}/unexport"

    def setup(self):
        """Initialize PWM channel"""
        try:
            # Export PWM channel if not already exported
            if not os.path.exists(self.pwm_path):
                with open(self.export_path, 'w') as f:
                    f.write(str(self.channel))
                time.sleep(0.2)  # Wait for export to complete

            # Set period
            with open(f"{self.pwm_path}/period", 'w') as f:
                f.write(str(PWM_PERIOD_NS))

            # Set initial duty cycle to middle position
            with open(f"{self.pwm_path}/duty_cycle", 'w') as f:
                f.write(str(1500000))  # 1.5ms - 90 degrees

            # Enable PWM explicitly
            with open(f"{self.pwm_path}/enable", 'w') as f:
                f.write('1')

            print(f"✓ PWM initialized on chip{self.chip}/channel{self.channel}")

            # Verify settings
            with open(f"{self.pwm_path}/period", 'r') as f:
                actual_period = f.read().strip()
            with open(f"{self.pwm_path}/enable", 'r') as f:
                enabled = f.read().strip()

            print(f"  Period: {int(actual_period)/1000000:.1f}ms")
            print(f"  Enabled: {'Yes' if enabled == '1' else 'No'}")

            return True

        except Exception as e:
            print(f"✗ Error setting up PWM chip{self.chip}: {e}")
            return False

    def set_angle(self, angle):
        """Set servo angle (0-180 degrees)"""
        if angle < 0:
            angle = 0
        elif angle > 180:
            angle = 180

        # Calculate pulse width for the angle
        pulse_range = MAX_PULSE_WIDTH_NS - MIN_PULSE_WIDTH_NS
        pulse_width = MIN_PULSE_WIDTH_NS + int((angle / 180.0) * pulse_range)

        try:
            # Set duty cycle
            with open(f"{self.pwm_path}/duty_cycle", 'w') as f:
                f.write(str(pulse_width))

            # Make sure it's enabled
            with open(f"{self.pwm_path}/enable", 'w') as f:
                f.write('1')

            return True

        except Exception as e:
            print(f"Error setting angle: {e}")
            return False

    def sweep_test(self):
        """Perform a sweep test"""
        print("  Sweeping 0° → 180° → 0°...")

        # Go to 0
        self.set_angle(0)
        time.sleep(1)

        # Sweep to 180
        for angle in range(0, 181, 30):
            self.set_angle(angle)
            print(f"    {angle}°", end=" ", flush=True)
            time.sleep(0.5)
        print()

        # Sweep back to 0
        for angle in range(180, -1, -30):
            self.set_angle(angle)
            print(f"    {angle}°", end=" ", flush=True)
            time.sleep(0.5)
        print()

    def cleanup(self):
        """Clean up PWM channel"""
        try:
            # Disable PWM
            with open(f"{self.pwm_path}/enable", 'w') as f:
                f.write('0')
            # Unexport PWM channel
            if os.path.exists(self.pwm_path):
                with open(self.unexport_path, 'w') as f:
                    f.write(str(self.channel))
        except:
            pass

def test_all_pwm_chips():
    """Test all available PWM chips to find the one connected to your servo"""
    print("\n" + "="*60)
    print("PWM CHIP DIAGNOSTIC TEST")
    print("="*60)
    print("\nTesting all PWM chips - watch for servo movement!")
    print("The chip that moves your servo is the correct one.\n")

    pwm_base = "/sys/class/pwm/"
    chips = sorted([d for d in os.listdir(pwm_base) if d.startswith("pwmchip")])

    working_chip = None

    for chip_name in chips:
        chip_num = int(chip_name.replace("pwmchip", ""))
        print(f"\n--- Testing PWM Chip {chip_num} ---")

        servo = PWMServo(chip_num, 0)

        if servo.setup():
            servo.sweep_test()

            response = input("  Did the servo move? (y/n): ").strip().lower()
            if response == 'y':
                working_chip = chip_num
                print(f"  ✓ Great! Chip {chip_num} is connected to your servo!")
                servo.cleanup()
                break
            else:
                print(f"  ✗ Chip {chip_num} is not connected to your servo")

        servo.cleanup()
        time.sleep(0.5)

    return working_chip

def manual_control(chip_num):
    """Manual control mode with the specified PWM chip"""
    print("\n" + "="*60)
    print(f"MANUAL CONTROL MODE - Using PWM Chip {chip_num}")
    print("="*60)

    servo = PWMServo(chip_num, 0)

    if not servo.setup():
        print("Failed to setup PWM!")
        return

    print("\nCommands:")
    print("  - Enter angle (0-180)")
    print("  - 's' for sweep test")
    print("  - 'q' to quit")

    try:
        while True:
            user_input = input("\nEnter command: ").strip().lower()

            if user_input == 'q':
                break
            elif user_input == 's':
                servo.sweep_test()
            else:
                try:
                    angle = float(user_input)
                    if 0 <= angle <= 180:
                        servo.set_angle(angle)
                        print(f"  Set to {angle}°")
                    else:
                        print("  Angle must be between 0 and 180")
                except ValueError:
                    print("  Invalid input!")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        servo.cleanup()
        print("Cleaned up!")

def quick_test(chip_num, angle):
    """Quick test - move servo to specific angle"""
    print(f"Moving servo to {angle}° using PWM Chip {chip_num}...")

    servo = PWMServo(chip_num, 0)

    if not servo.setup():
        print("Failed to setup PWM!")
        return False

    servo.set_angle(angle)
    print(f"✓ Servo set to {angle}°")
    time.sleep(3)

    servo.cleanup()
    return True

def main():
    print("MG996R Servo Test Tool")
    print("Jetson Orin Nano PWM Test")
    print()

    # Check for command line arguments
    if len(sys.argv) > 1:
        # Quick test mode with angle
        try:
            angle = float(sys.argv[1])
            chip = 0  # Use chip 0 (confirmed working)

            if len(sys.argv) > 2:
                chip = int(sys.argv[2])

            print(f"Quick test mode: {angle}° on chip {chip}")
            quick_test(chip, angle)
            return
        except ValueError:
            if sys.argv[1].lower() == 'calibrate':
                # Run full calibration
                pass
            else:
                print("Usage:")
                print("  servo_test.py [angle]        - Move to angle (0-180)")
                print("  servo_test.py [angle] [chip] - Move to angle on specific chip")
                print("  servo_test.py calibrate      - Full calibration mode")
                return

    # First, test all chips to find the right one
    working_chip = test_all_pwm_chips()

    if working_chip is not None:
        print(f"\n✓ SUCCESS: Your servo is connected to PWM Chip {working_chip}")
        print(f"\nFor future use, set: PWM_CHIP = {working_chip}")

        # Ask if user wants to continue with manual control
        response = input("\nWould you like to enter manual control mode? (y/n): ").strip().lower()
        if response == 'y':
            manual_control(working_chip)
    else:
        print("\n✗ No working PWM chip found!")
        print("\nPossible issues:")
        print("1. Servo not connected to a PWM-capable pin")
        print("2. Pin not configured for PWM in device tree")
        print("3. Wiring issue (check connections)")
        print("4. Servo power supply issue")
        print("\nWiring checklist:")
        print("- Servo signal wire → GPIO pin")
        print("- Servo power (red) → 5V external supply")
        print("- Servo ground (brown) → Common ground")
        print("- External supply ground → Jetson ground")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This script needs root privileges to access PWM.")
        print("Please run with sudo: sudo python3 <script_name>")
        sys.exit(1)

    main()
