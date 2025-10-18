#!/usr/bin/env python3
"""
Find available PWM chips and channels on Jetson
"""

import os
import glob

print("=" * 60)
print("PWM Chip Discovery for Jetson Orin")
print("=" * 60)

# Find all PWM chips
pwm_chips = glob.glob("/sys/class/pwm/pwmchip*")

if not pwm_chips:
    print("❌ No PWM chips found in /sys/class/pwm/")
    print("\nThis could mean:")
    print("  1. PWM is not enabled in the device tree")
    print("  2. Wrong kernel module not loaded")
    print("  3. Need to configure pinmux for PWM")
else:
    print(f"✓ Found {len(pwm_chips)} PWM chip(s):\n")

    for chip_path in sorted(pwm_chips):
        chip_name = os.path.basename(chip_path)
        print(f"📍 {chip_path}")

        # Read npwm (number of PWM channels)
        try:
            with open(f"{chip_path}/npwm", 'r') as f:
                npwm = int(f.read().strip())
            print(f"   Number of PWM channels: {npwm}")

            # Check which channels are already exported
            exported_channels = []
            for i in range(npwm):
                pwm_channel_path = f"{chip_path}/pwm{i}"
                if os.path.exists(pwm_channel_path):
                    exported_channels.append(i)
                    print(f"   ✓ PWM{i} - Already exported")
                else:
                    print(f"   ○ PWM{i} - Available (not exported)")

        except Exception as e:
            print(f"   Error reading chip info: {e}")

        print()

print("\nChecking GPIO pinmux configuration...")
print("-" * 60)

# Check common Jetson Orin GPIO/PWM pins
gpio_pins = {
    "GPIO12 (Pin 32, PWM0)": "/sys/class/gpio/gpio12",
    "GPIO13 (Pin 33, PWM1)": "/sys/class/gpio/gpio13",
}

for name, path in gpio_pins.items():
    if os.path.exists(path):
        print(f"✓ {name} - GPIO available")
    else:
        print(f"○ {name} - Not exported as GPIO")

print("\n" + "=" * 60)
print("Next steps:")
print("=" * 60)
print("1. If you see PWM chips above, note the chip number and channel")
print("2. Update servo_test_sysfs.py with the correct PWM_CHIP and PWM_CHANNEL")
print("3. If no PWM chips found, you may need to:")
print("   - Enable PWM in device tree")
print("   - Load PWM kernel module: modprobe pwm-tegra")
print("   - Configure pinmux for PWM output")
