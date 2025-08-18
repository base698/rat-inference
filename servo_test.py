#!/usr/bin/env python3
"""
Servo Test Script for Raspberry Pi
Tests servo at multiple angles or moves to a specific position
"""

import RPi.GPIO as GPIO
import time
import sys
import signal

# Configuration
SERVO_PIN = 14  
PWM_FREQ = 50   # Standard servo frequency (50Hz = 20ms period)

def angle_to_duty_cycle(angle):
    """
    Convert angle (0-180) to duty cycle (2-12%)
    Most servos use:
    - 1ms pulse (5% duty) = 0 degrees
    - 1.5ms pulse (7.5% duty) = 90 degrees  
    - 2ms pulse (10% duty) = 180 degrees
    
    Adjust these values if your servo needs different pulse widths
    """
    # Map 0-180 degrees to 2-12% duty cycle
    # You may need to adjust these values for your specific servo
    min_duty = 2.5  # Duty cycle for 0 degrees
    max_duty = 12.5  # Duty cycle for 180 degrees
    
    duty = min_duty + (angle / 180.0) * (max_duty - min_duty)
    return duty

def setup_servo():
    """Initialize GPIO and PWM for servo control"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    
    # Create PWM instance
    pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
    pwm.start(0)  # Start with 0% duty cycle (no signal)
    
    return pwm

def move_servo(pwm, angle):
    """Move servo to specified angle"""
    if angle < 0 or angle > 180:
        print(f"Warning: Angle {angle} out of range (0-180). Clamping.")
        angle = max(0, min(180, angle))
    
    duty = angle_to_duty_cycle(angle)
    print(f"Moving to {angle}° (duty cycle: {duty:.1f}%)")
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)  # Give servo time to move
    pwm.ChangeDutyCycle(0)  # Stop sending signal to prevent jitter

def cleanup(signum=None, frame=None):
    """Clean up GPIO on exit"""
    print("\nCleaning up GPIO...")
    GPIO.cleanup()
    sys.exit(0)

def continuous_test(pwm):
    """Continuously test servo at different angles"""
    test_angles = [0, 45, 90, 135, 180]
    
    print("Starting continuous servo test")
    print(f"Testing angles: {test_angles}")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            for angle in test_angles:
                move_servo(pwm, angle)
                time.sleep(3)  # Wait 3 seconds at each position
                
    except KeyboardInterrupt:
        cleanup()

def single_position(pwm, angle):
    """Move servo to a single position and hold"""
    print(f"Moving servo to {angle}°")
    move_servo(pwm, angle)
    print(f"Servo moved to {angle}°")
    
    cleanup()

def main():
    # Set up signal handler for clean exit
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print("Raspberry Pi Servo Test")
    print(f"Using GPIO{SERVO_PIN} (adjust SERVO_PIN in script if needed)\n")
    
    # Setup servo
    pwm = setup_servo()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            angle = float(sys.argv[1])
            single_position(pwm, angle)
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid angle")
            print("Usage: python3 servo_test.py [angle]")
            print("  angle: 0-180 degrees (optional)")
            print("  If no angle provided, runs continuous test")
            cleanup()
    else:
        continuous_test(pwm)
    
    cleanup()

if __name__ == "__main__":
    main()
