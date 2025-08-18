#!/usr/bin/env python3
"""
Basic GPIO diagnostic test - checks if the pin and connections work
Tests without PWM to isolate hardware issues
"""

import RPi.GPIO as GPIO
import time
import sys

SERVO_PIN = 14  # Change this if using different pin

def test_basic_output():
    """Test GPIO pin with simple on/off"""
    print(f"Testing GPIO{SERVO_PIN} with basic on/off signals")
    print("This should NOT move the servo, but tests the pin")
    print("-" * 40)
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        
        print("Pin configured as output successfully")
        
        # Test 1: Simple on/off
        print("\nTest 1: Toggling pin 5 times (1 second intervals)")
        for i in range(5):
            GPIO.output(SERVO_PIN, GPIO.HIGH)
            print(f"  {i+1}. HIGH", end="")
            time.sleep(0.5)
            
            GPIO.output(SERVO_PIN, GPIO.LOW)
            print(" -> LOW")
            time.sleep(0.5)
        
        print("Basic toggle test completed\n")
        
        # Test 2: Hold HIGH
        print("Test 2: Holding HIGH for 2 seconds...")
        GPIO.output(SERVO_PIN, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(SERVO_PIN, GPIO.LOW)
        print("Hold test completed\n")
        
        # Test 3: Rapid toggle
        print("Test 3: Rapid toggle (100 times in 1 second)...")
        for _ in range(100):
            GPIO.output(SERVO_PIN, GPIO.HIGH)
            time.sleep(0.005)
            GPIO.output(SERVO_PIN, GPIO.LOW)
            time.sleep(0.005)
        print("Rapid toggle completed\n")
        
        print("All basic tests passed!")
        
    except Exception as e:
        print(f"\nERROR during test: {e}")
        print("\nPossible issues:")
        print("1. Wrong GPIO pin number")
        print("2. Pin already in use by another process")
        print("3. Hardware connection issue")
        print("4. Insufficient permissions (need sudo?)")
        
    finally:
        print("\nCleaning up GPIO...")
        GPIO.cleanup()
        print("Cleanup complete")

def test_pwm_minimal():
    """Minimal PWM test without servo movement"""
    print(f"\nTesting minimal PWM on GPIO{SERVO_PIN}")
    print("This sends a very brief PWM signal")
    print("-" * 40)
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        
        print("Creating PWM object...")
        pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz
        
        print("Starting PWM with 0% duty...")
        pwm.start(0)
        
        print("Setting 7.5% duty for 0.1 seconds...")
        pwm.ChangeDutyCycle(7.5)
        time.sleep(0.1)
        
        print("Back to 0% duty...")
        pwm.ChangeDutyCycle(0)
        
        print("Stopping PWM...")
        pwm.stop()
        
        print("PWM test completed successfully!")
        
    except Exception as e:
        print(f"\nERROR during PWM test: {e}")
        print("\nThis error suggests:")
        print("- Power supply issue (servo drawing too much current)")
        print("- Short circuit in wiring")
        print("- Damaged GPIO pin")
        print("- Ground connection problem")
        
    finally:
        time.sleep(0.5)
        GPIO.cleanup()
        print("Cleanup complete")

def check_connections():
    """Print connection checklist"""
    print("\n" + "=" * 50)
    print("SERVO CONNECTION CHECKLIST")
    print("=" * 50)
    print(f"""
1. POWER CONNECTIONS:
   □ Servo RED/POWER wire -> 5V pin (NOT 3.3V!)
   □ Servo BROWN/BLACK/GND wire -> Ground pin
   □ Servo ORANGE/YELLOW/SIGNAL wire -> GPIO{SERVO_PIN}

2. POWER SUPPLY:
   □ Using external 5V power supply for servo? (Recommended)
   □ If using Pi's 5V: Is your power adapter 2.5A or higher?
   □ Check voltage with multimeter (should be 4.8-5.2V)

3. WIRING ISSUES:
   □ All connections firm (no loose jumpers)?
   □ No exposed wires touching?
   □ Correct GPIO pin? (GPIO{SERVO_PIN} = physical pin varies by Pi model)

4. SERVO ISSUES:
   □ Servo works with other controllers?
   □ Not mechanically jammed?
   □ Within current limits (small servos: <500mA, large: >1A)

5. COMMON FIXES:
   • Add 1000µF capacitor across servo power
   • Use separate 5V supply for servo
   • Add 1kΩ resistor between GPIO and servo signal
   • Try different GPIO pin
   • Check if pin is damaged (test with LED + resistor)
""")

def main():
    print("GPIO Diagnostic Tool")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "basic":
            test_basic_output()
        elif sys.argv[1] == "pwm":
            test_pwm_minimal()
        elif sys.argv[1] == "check":
            check_connections()
        else:
            print("Usage: python3 gpio_test.py [basic|pwm|check]")
    else:
        # Run all tests
        check_connections()
        input("\nPress Enter to run basic GPIO test...")
        test_basic_output()
        
        input("\nPress Enter to run minimal PWM test...")
        test_pwm_minimal()

if __name__ == "__main__":
    main()