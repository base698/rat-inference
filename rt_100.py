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
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
import uvicorn
import threading
import shutil
from pathlib import Path
from PIL import Image

app = FastAPI()
detector_instance = None  # Global reference to detector for API access

@app.get("/show")
async def show_image():
    """Show the most recent captured image with optional list of all images"""
    # Check if show.png exists
    if not os.path.exists("show.png"):
        return HTMLResponse(content="<h1>No image captured yet</h1>", status_code=200)
    
    # Build HTML response
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rat Detector - Latest Image</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            img { max-width: 100%; height: auto; border: 2px solid #333; }
            .links { margin-top: 20px; }
            .links a { display: block; margin: 5px 0; color: #0066cc; }
            h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Latest Captured Image</h1>
        <img src="/show.png" alt="Latest capture">
        <p>Auto-refreshing every 5 seconds...</p>
    """
    
    # Add image list if keep_images is enabled
    if detector_instance and detector_instance.keep_images_enabled:
        image_list = detector_instance.get_image_list()
        if image_list:
            html_content += """
        <div class="links">
            <h2>Captured Images:</h2>
    """
            for img_path in image_list[-50:]:  # Show last 50 images to avoid huge lists
                filename = os.path.basename(img_path)
                html_content += f'            <a href="/{img_path}">{filename}</a>\n'
            html_content += "        </div>\n"
    
    html_content += """
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/show.png")
async def get_latest_image():
    """Serve the latest captured image"""
    if not os.path.exists("show.png"):
        return Response(content="No image available", status_code=404)
    
    with open("show.png", "rb") as f:
        image_data = f.read()
    
    return Response(content=image_data, media_type="image/png")

@app.get("/captures/{file_path:path}")
async def get_captured_image(file_path: str):
    """Serve a specific captured image"""
    full_path = f"captures/{file_path}"
    if not os.path.exists(full_path):
        return Response(content="Image not found", status_code=404)
    
    with open(full_path, "rb") as f:
        image_data = f.read()
    
    return Response(content=image_data, media_type="image/jpeg")

class RatDetector:
    def __init__(self, model_path, servo_pin=14, servo_enabled=False, confidence_threshold=0.5, allow_multiple_triggers=False):
        """
        Initialize the rat detector
        
        Args:
            model_path: Path to YOLO model weights
            servo_pin: GPIO pin for servo control (default: 14)
            servo_enabled: Whether to actually control the servo
            confidence_threshold: Minimum confidence for detection
            allow_multiple_triggers: Whether to allow servo to trigger multiple times
        """
        self.model = YOLO(model_path)
        self.servo_pin = servo_pin
        self.servo_enabled = servo_enabled
        self.confidence_threshold = confidence_threshold
        self.servo_position = 0  # Track current servo position
        self.servo_triggered = False  # Track if servo has been triggered
        self.allow_multiple_triggers = allow_multiple_triggers
        self.image_list_file = "captured_images.txt"  # File to track captured images
        self.keep_images_enabled = False  # Will be set when run() is called
        
        # Initialize servo if enabled
        if self.servo_enabled:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.servo_pin, GPIO.OUT)
            self.servo = GPIO.PWM(self.servo_pin, 50)  # 50Hz PWM
            self.servo.start(0)
            self.set_servo_angle(25)  # Start
            
    def add_image_to_list(self, image_path):
        """Add image path to the list file"""
        with open(self.image_list_file, 'a') as f:
            f.write(f"{image_path}\n")
    
    def get_image_list(self):
        """Read the list of captured images from file"""
        if not os.path.exists(self.image_list_file):
            return []
        with open(self.image_list_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and os.path.exists(line.strip())]
    
    def clear_image_list(self):
        """Clear the image list file"""
        if os.path.exists(self.image_list_file):
            os.remove(self.image_list_file)
    
    def set_servo_angle(self, angle):
        """
        Set servo to specific angle
        
        Args:
            angle: Angle in degrees (0-180)
        """
        if not self.servo_enabled:
            print(f"[SERVO SIMULATION] Would move to {angle} degrees")
            return
            
        # Convert angle to duty cycle (5.0% to 10.0% for 0-180 degrees)
        # Matching servo_test.py calibration
        min_duty = 5.0   # Duty cycle for 0 degrees
        max_duty = 10.0  # Duty cycle for 180 degrees
        duty_cycle = min_duty + (angle / 180.0) * (max_duty - min_duty)
        self.servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # Allow servo to move
        self.servo.ChangeDutyCycle(0)  # Stop sending signal
        self.servo_position = angle
        
    def move_servo_forward(self):
        """Move servo to 35 degrees (trigger position)"""
        print(f"Moving servo to trigger position: {self.servo_position}° -> 80°")
        self.set_servo_angle(90)
        
    def move_servo_backward(self):
        """Move servo back to neutral position (0 degrees)"""
        print(f"Moving servo back to neutral: {self.servo_position}° -> 30°")
        self.set_servo_angle(25)
        
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
        
        # Flip the image 180 degrees (rotate)
        img = Image.open(filename)
        img_rotated = img.rotate(180)
        img_rotated.save(filename)
        
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
        self.keep_images_enabled = keep_images
        
        # Clear the image list file on startup if keep_images is enabled
        if keep_images:
            self.clear_image_list()
        
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
                
                # Copy to show.png for web display
                shutil.copy2(image_path, "show.png")
                
                # Add to image list if keeping images
                if keep_images:
                    self.add_image_to_list(image_path)
                
                # Run inference
                detections, inference_time = self.run_inference(image_path)
                print(f"Inference time: {inference_time:.2f} ms")
                
                if detections:
                    detection_count += len(detections)
                    print(f"🐀 RAT DETECTED! ({len(detections)} detection(s))")
                    
                    for i, det in enumerate(detections, 1):
                        print(f"  Detection {i}:")
                        print(f"    Class: {det['class']}")
                        print(f"    Confidence: {det['confidence']:.3f}")
                        print(f"    Bounding box: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, "
                              f"{det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
                    
                    # Trigger servo action (only if not already triggered or multiple triggers allowed)
                    if not self.servo_triggered or self.allow_multiple_triggers:
                        if self.servo_enabled:
                            print("\nTriggering servo action...")
                            self.move_servo_forward()
                            time.sleep(3)  # Hold position for 3
                            self.move_servo_backward()
                            if not self.allow_multiple_triggers:
                                self.servo_triggered = True
                                print("Servo has been triggered. It will not trigger again.")
                        else:
                            print("\n[SERVO DISABLED] Would trigger servo action here")
                            if not self.allow_multiple_triggers:
                                self.servo_triggered = True  # Mark as triggered even in simulation
                    else:
                        print("\n[SERVO ALREADY TRIGGERED] Skipping servo action")
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

def run_api_server(host="0.0.0.0", port=8000):
    """Run the FastAPI server in a separate thread"""
    uvicorn.run(app, host=host, port=port, log_level="error")

def main():
    parser = argparse.ArgumentParser(description="Real-time rat detection with servo control")
    
    # Model settings
    parser.add_argument("--model", "-m", type=str, default="runs/yolo8n-2025-08-17/weights/best.pt",
                       help="Path to YOLO model (e.g., runs/yolo8n-2025-08-17/weights/best.pt)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="Confidence threshold for detection")
    
    # Servo settings
    parser.add_argument("--enable-servo", action="store_true",
                       help="Enable servo control (requires GPIO access)")
    parser.add_argument("--servo-pin", type=int, default=14,
                       help="GPIO pin for servo control")
    parser.add_argument("--allow-multiple-triggers", action="store_true",
                       help="Allow servo to trigger multiple times (default: only trigger once)")
    
    # Capture settings
    parser.add_argument("--interval", "-i", type=float, default=1.0,
                       help="Capture interval in seconds")
    parser.add_argument("--keep-images", action="store_true",
                       help="Keep captured images (default: delete after processing)")
    
    # API settings
    parser.add_argument("--api-host", type=str, default="0.0.0.0",
                       help="Host for FastAPI server")
    parser.add_argument("--api-port", type=int, default=8000,
                       help="Port for FastAPI server")
    
    args = parser.parse_args()
    
    # Create detector and set global instance
    global detector_instance
    detector = RatDetector(
        model_path=args.model,
        servo_pin=args.servo_pin,
        servo_enabled=args.enable_servo,
        confidence_threshold=args.confidence,
        allow_multiple_triggers=args.allow_multiple_triggers
    )
    detector_instance = detector
    
    # Start API server in a separate thread
    api_thread = threading.Thread(
        target=run_api_server,
        args=(args.api_host, args.api_port),
        daemon=True
    )
    api_thread.start()
    print(f"FastAPI server started at http://{args.api_host}:{args.api_port}/show")
    
    # Run detection loop
    detector.run(
        capture_interval=args.interval,
        keep_images=args.keep_images
    )

if __name__ == "__main__":
    main()
