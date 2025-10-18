#!/usr/bin/env python3
"""
Test all available PWM chips to find which one controls Pin 33
Run this and watch your servo - note which chip makes it move!
"""

import time
import os

PWM_PERIOD = 20000000  # 20ms = 50Hz
TEST_ANGLE = 90  # Center position
PULSE_WIDTH_90 = 1500000  # 1.5ms pulse for 90 degrees

def test_pwm_chip(chip_num):
    """Test a specific PWM chip"""
    pwm_path = f"/sys/class/pwm/pwmchip{chip_num}/pwm0"
    export_path = f"/sys/class/pwm/pwmchip{chip_num}/export"
    unexport_path = f"/sys/class/pwm/pwmchip{chip_num}/unexport"

    print(f"\n{'='*60}")
    print(f"Testing pwmchip{chip_num}")
    print(f"{'='*60}")

    try:
        # Export PWM0
        if not os.path.exists(pwm_path):
            with open(export_path, 'w') as f:
                f.write('0')
            time.sleep(0.1)

        # Set period
        with open(f"{pwm_path}/period", 'w') as f:
            f.write(str(PWM_PERIOD))

        # Set duty cycle to 90 degrees (1.5ms)
        with open(f"{pwm_path}/duty_cycle", 'w') as f:
            f.write(str(PULSE_WIDTH_90))

        # Enable PWM
        with open(f"{pwm_path}/enable", 'w') as f:
            f.write('1')

        print(f"✓ pwmchip{chip_num} activated with 90° pulse (1.5ms)")
        print(f"  ** WATCH YOUR SERVO - Does it move? **")
        print(f"  Holding for 3 seconds...")
        time.sleep(3)

        # Disable PWM
        with open(f"{pwm_path}/enable", 'w') as f:
            f.write('0')

        # Unexport
        with open(unexport_path, 'w') as f:
            f.write('0')

        print(f"✓ pwmchip{chip_num} test complete")

        return True

    except Exception as e:
        print(f"✗ pwmchip{chip_num} failed: {e}")
        # Try to cleanup
        try:
            if os.path.exists(f"{pwm_path}/enable"):
                with open(f"{pwm_path}/enable", 'w') as f:
                    f.write('0')
            if os.path.exists(pwm_path):
                with open(unexport_path, 'w') as f:
                    f.write('0')
        except:
            pass
        return False

def main():
    print("=" * 60)
    print("PWM Chip Servo Test - Testing ALL chips")
    print("=" * 60)
    print("This will test pwmchip0 through pwmchip4")
    print("WATCH YOUR SERVO and note which chip makes it move!")
    print()
    print("Servo should move to 90° (center) for 3 seconds on each chip")
    print()
    input("Press ENTER to start testing...")

    # Test all 5 chips
    for chip in range(5):
        test_pwm_chip(chip)
        time.sleep(1)  # Pause between tests

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    print("Which chip made your servo move?")
    print("Update servo_test_sysfs.py with that chip number:")
    print("  PWM_CHIP = X  # Replace X with the working chip number")

if __name__ == "__main__":
    main()
