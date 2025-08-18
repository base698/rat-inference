#!/usr/bin/env python3
"""
Servo Test Script for Raspberry Pi
Tests servo at multiple angles or moves to a specific position
"""

import RPi.GPIO as GPIO
import time
import sys
import signal
import atexit

# Configuration
SERVO_PIN = 14  
PWM_FREQ = 50   # Standard servo frequency (50Hz = 20ms period)

# Global PWM object for cleanup
pwm_global = None

def angle_to_duty_cycle(angle):
    """
    Convert angle (0-180) to duty cycle
    Standard servos typically use:
    - 0.5ms pulse (2.5% duty @ 50Hz) = 0 degrees
    - 1.5ms pulse (7.5% duty @ 50Hz) = 90 degrees  
    - 2.5ms pulse (12.5% duty @ 50Hz) = 180 degrees
    
    Adjust these values if your servo needs different pulse widths
    """
    # Standard servo calibration - adjust if needed
    # Most servos work with 5-10% duty cycle range
    min_duty = 5.0   # Duty cycle for 0 degrees (1ms pulse)
    max_duty = 10.0  # Duty cycle for 180 degrees (2ms pulse)
    
    duty = min_duty + (angle / 180.0) * (max_duty - min_duty)
    return duty

def setup_servo():
    """Initialize GPIO and PWM for servo control"""
    global pwm_global
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    
    # Create PWM instance
    pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
    pwm.start(0)  # Start with 0% duty cycle (no signal)
    
    pwm_global = pwm  # Store for cleanup
    return pwm

def move_servo(pwm, angle, hold_time=0.5):
    """Move servo to specified angle"""
    if angle < 0 or angle > 180:
        print(f"Warning: Angle {angle} out of range (0-180). Clamping.")
        angle = max(0, min(180, angle))
    
    duty = angle_to_duty_cycle(angle)
    print(f"Moving to {angle}° (duty cycle: {duty:.2f}%)")
    print(f"  Pulse width: {(duty/100) * 20:.2f}ms")  # Debug info
    
    try:
        pwm.ChangeDutyCycle(duty)
        time.sleep(hold_time)  # Give servo time to move
        pwm.ChangeDutyCycle(0)  # Stop sending signal to prevent jitter
    except Exception as e:
        print(f"Error moving servo: {e}")

def cleanup(signum=None, frame=None):
    """Clean up GPIO on exit"""
    global pwm_global
    
    print("\nCleaning up...")
    
    try:
        # Stop PWM before cleanup
        if pwm_global is not None:
            print("Stopping PWM...")
            pwm_global.ChangeDutyCycle(0)
            pwm_global.stop()
            pwm_global = None
    except Exception as e:
        print(f"PWM cleanup error: {e}")
    
    try:
        print("Cleaning up GPIO...")
        GPIO.cleanup()
    except Exception as e:
        print(f"GPIO cleanup error: {e}")
    
    if signum is not None:
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
        pass  # Let main handle cleanup

def single_position(pwm, angle):
    """Move servo to a single position and hold"""
    print(f"\nMoving servo to {angle}°")
    move_servo(pwm, angle, hold_time=1.0)  # Hold longer for single position
    print(f"Servo at {angle}°. Waiting 5 seconds...")
    
    try:
        # Keep the program running instead of immediately cleaning up
        time.sleep(5)  # Hold position for 5 seconds
        print("Returning to center (90°)...")
        move_servo(pwm, 90)  # Return to center before exit
        time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")

def calibration_mode(pwm):
    """Interactive calibration mode to find correct duty cycle values"""
    print("\n=== CALIBRATION MODE ===")
    print("Enter duty cycle values (2.0-12.5) to test, or 'q' to quit")
    print("Typical values: 5.0 = 0°, 7.5 = 90°, 10.0 = 180°")
    print("Or use 'a' prefix for angle (e.g., 'a90' for 90 degrees)\n")
    
    while True:
        try:
            user_input = input("Duty cycle % or angle: ").strip()
            
            if user_input.lower() == 'q':
                break
                
            if user_input.startswith('a'):
                # Angle mode
                angle = float(user_input[1:])
                move_servo(pwm, angle)
            else:
                # Direct duty cycle mode
                duty = float(user_input)
                if duty < 2.0 or duty > 12.5:
                    print("Warning: Duty cycle should be between 2.0 and 12.5")
                
                print(f"Setting duty cycle to {duty}%")
                pwm.ChangeDutyCycle(duty)
                time.sleep(0.5)
                pwm.ChangeDutyCycle(0)
                
        except ValueError:
            print("Invalid input. Enter a number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting calibration mode")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    global pwm_global
    
    # Set up cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print("Raspberry Pi Servo Test")
    print(f"Using GPIO{SERVO_PIN} (adjust SERVO_PIN in script if needed)")
    print("Current calibration: 5.0% = 0°, 7.5% = 90°, 10.0% = 180°\n")
    
    try:
        # Setup servo
        pwm = setup_servo()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1].lower() == 'calibrate':
                calibration_mode(pwm)
            else:
                try:
                    angle = float(sys.argv[1])
                    single_position(pwm, angle)
                except ValueError:
                    print(f"Error: '{sys.argv[1]}' is not a valid angle")
                    print("Usage: python3 servo_test.py [angle|calibrate]")
                    print("  angle: 0-180 degrees (moves to specific angle)")
                    print("  calibrate: Enter calibration mode")
                    print("  If no argument provided, runs continuous test")
        else:
            continuous_test(pwm)
            
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()