#!/usr/bin/env python3
"""
Real-time rat detection with Raspberry Pi Camera and servo control
Captures images in a loop, runs YOLOv8 inference, and triggers servo on detection
"""

from picamera2 import Picamera2
from ultralytics import YOLO
from datetime import datetime
import time
import os
import argparse
import numpy as np
import RPi.GPIO as GPIO

class RatDetector:
    def __init__(self, model_path, servo_pin=18, servo_enabled=False, confidence_threshold=0.5):
        """
        Initialize the rat detector
        
        Args:
            model_path: Path to YOLO model weights
            servo_pin: GPIO pin for servo control (default: 18)
            servo_enabled: Whether to actually control the servo
            confidence_threshold: Minimum confidence for detection
        """
        self.model = YOLO(model_path)
        self.servo_pin = servo_pin
        self.servo_enabled = servo_enabled
        self.confidence_threshold = confidence_threshold
        self.servo_position = 0  # Track current servo position
        
        # Initialize servo if enabled
        if self.servo_enabled:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.servo_pin, GPIO.OUT)
            self.servo = GPIO.PWM(self.servo_pin, 50)  # 50Hz PWM
            self.servo.start(0)
            self.set_servo_angle(0)  # Start at 0 degrees
            
    def set_servo_angle(self, angle):
        """
        Set servo to specific angle
        
        Args:
            angle: Angle in degrees (0-180)
        """
        if not self.servo_enabled:
            print(f"[SERVO SIMULATION] Would move to {angle} degrees")
            return
            
        # Convert angle to duty cycle (2.5% to 12.5% for 0-180 degrees)
        duty_cycle = 2.5 + (angle / 180.0) * 10.0
        self.servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # Allow servo to move
        self.servo.ChangeDutyCycle(0)  # Stop sending signal
        self.servo_position = angle
        
    def move_servo_forward(self):
        """Move servo 60 degrees forward"""
        new_position = min(self.servo_position + 60, 180)
        print(f"Moving servo forward: {self.servo_position}° -> {new_position}°")
        self.set_servo_angle(new_position)
        
    def move_servo_backward(self):
        """Move servo back to previous position"""
        new_position = max(self.servo_position - 60, 0)
        print(f"Moving servo backward: {self.servo_position}° -> {new_position}°")
        self.set_servo_angle(new_position)
        
    def capture_image(self, picam2, save_path="captures"):
        """
        Capture an image from the camera
        
        Args:
            picam2: Picamera2 instance
            save_path: Directory to save captured images
            
        Returns:
            Path to captured image
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}/capture_{timestamp}.jpg"
        
        # Capture the image
        picam2.capture_file(filename)
        return filename
        
    def run_inference(self, image_path):
        """
        Run YOLO inference on an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (list of detections with rat class, inference time in ms)
        """
        # Start timing
        start_time = time.time()
        
        # Run inference
        results = self.model(
            image_path,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        rat_detections = []
        
        # Process results
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # Get class name
                    class_name = self.model.names[cls] if cls < len(self.model.names) else f"Class_{cls}"
                    
                    # Check if it's a rat (assuming 'rat' is in model classes)
                    # You may need to adjust this based on your model's class names
                    if 'rat' in class_name.lower() or cls == 0:  # Adjust class index as needed
                        rat_detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': xyxy
                        })
                        
        return rat_detections, inference_time
        
    def run(self, capture_interval=1.0, keep_images=False):
        """
        Main loop for continuous detection
        
        Args:
            capture_interval: Time between captures in seconds
            keep_images: Whether to keep captured images
        """
        # Initialize camera
        picam2 = Picamera2()
        
        # Configure camera for still capture
        config = picam2.create_still_configuration(
            main={"size": (1920, 1080)},  # Reduced resolution for faster processing
            lores={"size": (640, 480)},
            display="lores"
        )
        picam2.configure(config)
        picam2.start()
        
        print("Starting real-time rat detection...")
        print(f"Servo control: {'ENABLED' if self.servo_enabled else 'DISABLED (simulation mode)'}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Capture interval: {capture_interval}s")
        print("Press Ctrl+C to stop\n")
        
        try:
            frame_count = 0
            detection_count = 0
            
            while True:
                frame_count += 1
                print(f"\n--- Frame {frame_count} ---")
                
                # Capture image
                image_path = self.capture_image(picam2)
                print(f"Captured: {image_path}")
                
                # Run inference
                detections = self.run_inference(image_path)
                
                if detections:
                    detection_count += len(detections)
                    print(f"🐀 RAT DETECTED! ({len(detections)} detection(s))")
                    
                    for i, det in enumerate(detections, 1):
                        print(f"  Detection {i}:")
                        print(f"    Class: {det['class']}")
                        print(f"    Confidence: {det['confidence']:.3f}")
                        print(f"    Bounding box: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, "
                              f"{det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
                    
                    # Trigger servo action
                    if self.servo_enabled:
                        print("\nTriggering servo action...")
                        self.move_servo_forward()
                        time.sleep(1)  # Hold position
                        self.move_servo_backward()
                    else:
                        print("\n[SERVO DISABLED] Would trigger servo action here")
                else:
                    print("No rats detected")
                
                # Delete image if not keeping
                if not keep_images:
                    os.remove(image_path)
                
                # Wait before next capture
                time.sleep(capture_interval)
                
        except KeyboardInterrupt:
            print(f"\n\nStopping detection...")
            print(f"Total frames processed: {frame_count}")
            print(f"Total detections: {detection_count}")
            if frame_count > 0:
                print(f"Detection rate: {(detection_count/frame_count)*100:.1f}%")
        finally:
            # Cleanup
            picam2.stop()
            picam2.close()
            
            if self.servo_enabled:
                self.servo.stop()
                GPIO.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Real-time rat detection with servo control")
    
    # Model settings
    parser.add_argument("--model", "-m", type=str, default="yolov8n.pt",
                       help="Path to YOLO model (e.g., runs/train/rat_detector/weights/best.pt)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="Confidence threshold for detection")
    
    # Servo settings
    parser.add_argument("--enable-servo", action="store_true",
                       help="Enable servo control (requires GPIO access)")
    parser.add_argument("--servo-pin", type=int, default=18,
                       help="GPIO pin for servo control")
    
    # Capture settings
    parser.add_argument("--interval", "-i", type=float, default=1.0,
                       help="Capture interval in seconds")
    parser.add_argument("--keep-images", action="store_true",
                       help="Keep captured images (default: delete after processing)")
    
    args = parser.parse_args()
    
    # Create detector
    detector = RatDetector(
        model_path=args.model,
        servo_pin=args.servo_pin,
        servo_enabled=args.enable_servo,
        confidence_threshold=args.confidence
    )
    
    # Run detection loop
    detector.run(
        capture_interval=args.interval,
        keep_images=args.keep_images
    )

if __name__ == "__main__":
    main()