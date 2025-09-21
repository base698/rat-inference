#!/usr/bin/env python3
"""
Real-time camera tracking with Feetech servo control and rat detection
Controls yaw (ID 1, 1500-3500 raw) and pitch (ID 2, 0-600 raw) servos
Includes Raspberry Pi camera streaming, YOLO inference, and GPIO trigger servo
"""

import os
import time
import argparse
import threading
import base64
import io
import shutil
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn
import numpy as np

# Set library path for macOS
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

# Try importing Raspberry Pi specific libraries
try:
    from picamera2 import Picamera2
    from PIL import Image
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False
    print("Raspberry Pi libraries not available - camera and GPIO features disabled")

# Try importing YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available - detection features disabled")

# Try importing Feetech servo libraries
try:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode
    FEETECH_AVAILABLE = True
except ImportError:
    FEETECH_AVAILABLE = False
    print("Feetech libraries not available - servo tracking features disabled")

app = FastAPI()
tracker_instance = None  # Global reference to tracker for API access

# Servo configuration for tracking servos
YAW_MOTOR_ID = 1
PITCH_MOTOR_ID = 5
YAW_MIN = 1500  # Minimum yaw position (raw)
YAW_MAX = 3500  # Maximum yaw position (raw)
YAW_CENTER = (YAW_MIN + YAW_MAX) // 2  # Center position for yaw
PITCH_MIN = 0    # Minimum pitch position (level, raw)
PITCH_MAX = 600  # Maximum pitch position (~55 degrees down, raw)
PITCH_CENTER = 0 # Start at level position

# GPIO servo configuration (trigger servo)
TRIGGER_SERVO_PIN = 4  # GPIO4 for trigger servo

@app.get("/")
async def root():
    """Root endpoint with control interface and camera view"""
    enable_trigger = tracker_instance and tracker_instance.trigger_servo_enabled

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Tracker Control</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f0f0f0;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{ color: #333; }}
            .main-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }}
            .camera-view {{
                background-color: #000;
                min-height: 480px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 10px;
                position: relative;
            }}
            .camera-view img {{
                max-width: 100%;
                height: auto;
                border-radius: 10px;
            }}
            .status {{
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                background-color: #e7f3ff;
                border: 1px solid #b3d9ff;
            }}
            .detection-status {{
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                background-color: #fff3cd;
                border: 1px solid #ffc107;
            }}
            .detection-active {{
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                animation: blink 1s infinite;
            }}
            @keyframes blink {{
                0%, 50%, 100% {{ opacity: 1; }}
                25%, 75% {{ opacity: 0.5; }}
            }}
            .controls {{
                margin: 20px 0;
                padding: 20px;
                border: 2px solid #ddd;
                border-radius: 10px;
                background-color: #f9f9f9;
            }}
            .control-group {{
                margin: 15px 0;
            }}
            .control-group label {{
                display: inline-block;
                width: 120px;
                font-weight: bold;
            }}
            .control-group input[type="range"] {{
                width: 300px;
                vertical-align: middle;
            }}
            .control-group span {{
                display: inline-block;
                width: 60px;
                text-align: center;
                font-family: monospace;
                background-color: #e0e0e0;
                padding: 2px 5px;
                border-radius: 3px;
            }}
            .button-group {{
                margin: 20px 0;
                text-align: center;
            }}
            button {{
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 4px 8px;
                cursor: pointer;
                border-radius: 5px;
                transition: background-color 0.3s;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            button.center {{
                background-color: #008CBA;
            }}
            button.center:hover {{
                background-color: #007399;
            }}
            button.trigger {{
                background-color: #ff9800;
                padding: 15px 30px;
                font-size: 16px;
            }}
            button.trigger:hover {{
                background-color: #e68900;
            }}
            button:disabled {{
                background-color: #cccccc;
                cursor: not-allowed;
            }}
            .keyboard-info {{
                margin-top: 20px;
                padding: 15px;
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 5px;
            }}
            .keyboard-info h3 {{
                margin-top: 0;
                color: #856404;
            }}
            .key-list {{
                font-family: monospace;
                line-height: 1.8;
            }}
            .key {{
                background-color: #e0e0e0;
                padding: 2px 6px;
                border-radius: 3px;
                border: 1px solid #999;
                font-weight: bold;
            }}
            .servo-status {{
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
            }}
            .servo-status.success {{
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .servo-status.error {{
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}
            .servo-status.info {{
                background-color: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }}
            .detections-log {{
                max-height: 200px;
                overflow-y: auto;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
                margin-top: 10px;
                font-family: monospace;
                font-size: 12px;
            }}
        </style>
        <script>
            let yawValue = {YAW_CENTER};
            let pitchValue = {PITCH_CENTER};
            let streamInterval = null;

            async function updatePosition(yaw = null, pitch = null) {{
                if (yaw !== null) {{
                    yawValue = parseInt(yaw);
                    document.getElementById('yawSlider').value = yawValue;
                    document.getElementById('yawValue').textContent = yawValue;
                }}
                if (pitch !== null) {{
                    pitchValue = parseInt(pitch);
                    document.getElementById('pitchSlider').value = pitchValue;
                    document.getElementById('pitchValue').textContent = pitchValue;
                }}

                try {{
                    const response = await fetch('/set-position', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            yaw: yawValue,
                            pitch: pitchValue
                        }})
                    }});
                    const result = await response.json();
                    if (!result.success) {{
                        console.error('Failed to set position:', result.message);
                    }}
                }} catch (error) {{
                    console.error('Error:', error);
                }}
            }}

            async function centerServos() {{
                await updatePosition({YAW_CENTER}, {PITCH_CENTER});
            }}

            async function triggerServo() {{
                const button = document.getElementById('triggerButton');
                const status = document.getElementById('triggerStatus');

                button.disabled = true;
                status.className = 'servo-status info';
                status.innerHTML = 'Triggering servo...';
                status.style.display = 'block';

                try {{
                    const response = await fetch('/trigger-servo', {{
                        method: 'POST'
                    }});
                    const result = await response.json();

                    if (result.success) {{
                        status.className = 'servo-status success';
                        status.innerHTML = result.message;
                    }} else {{
                        status.className = 'servo-status error';
                        status.innerHTML = result.message;
                    }}
                }} catch (error) {{
                    status.className = 'servo-status error';
                    status.innerHTML = 'Error: ' + error.message;
                }} finally {{
                    setTimeout(() => {{
                        button.disabled = false;
                        status.style.display = 'none';
                    }}, 3000);
                }}
            }}

            async function updateStream() {{
                try {{
                    const response = await fetch('/stream-frame');
                    if (response.ok) {{
                        const data = await response.json();
                        const img = document.getElementById('cameraStream');
                        if (data.image) {{
                            img.src = 'data:image/jpeg;base64,' + data.image;
                            img.style.display = 'block';
                        }}

                        // Update detection status
                        const detectionDiv = document.getElementById('detectionStatus');
                        if (data.detection) {{
                            detectionDiv.className = 'detection-status detection-active';
                            detectionDiv.innerHTML = `<strong>🐀 RAT DETECTED!</strong><br>Confidence: ${{data.confidence?.toFixed(3) || 'N/A'}}`;
                        }} else {{
                            detectionDiv.className = 'detection-status';
                            detectionDiv.innerHTML = 'No detection';
                        }}

                        // Update detections log
                        if (data.recent_detections) {{
                            const log = document.getElementById('detectionsLog');
                            log.innerHTML = data.recent_detections.join('<br>');
                        }}
                    }}
                }} catch (error) {{
                    console.error('Stream update error:', error);
                }}
            }}

            async function getStatus() {{
                try {{
                    const response = await fetch('/status');
                    const status = await response.json();
                    document.getElementById('connectionStatus').textContent =
                        status.connected ? 'Connected' : 'Disconnected';
                    document.getElementById('yawStatus').textContent =
                        'Yaw: ' + status.yaw_position;
                    document.getElementById('pitchStatus').textContent =
                        'Pitch: ' + status.pitch_position;
                    document.getElementById('cameraStatus').textContent =
                        'Camera: ' + (status.camera_active ? 'Active' : 'Inactive');
                    document.getElementById('detectionCount').textContent =
                        'Total Detections: ' + (status.detection_count || 0);
                }} catch (error) {{
                    console.error('Error getting status:', error);
                }}
            }}

            // Keyboard control
            document.addEventListener('keydown', async (event) => {{
                const step = 50;
                switch(event.key) {{
                    case 'ArrowLeft':
                        event.preventDefault();
                        await updatePosition(Math.max({YAW_MIN}, yawValue - step), null);
                        break;
                    case 'ArrowRight':
                        event.preventDefault();
                        await updatePosition(Math.min({YAW_MAX}, yawValue + step), null);
                        break;
                    case 'ArrowUp':
                        event.preventDefault();
                        await updatePosition(null, Math.max({PITCH_MIN}, pitchValue - step));
                        break;
                    case 'ArrowDown':
                        event.preventDefault();
                        await updatePosition(null, Math.min({PITCH_MAX}, pitchValue + step));
                        break;
                    case 'c':
                    case 'C':
                        event.preventDefault();
                        await centerServos();
                        break;
                    case 't':
                    case 'T':
                        event.preventDefault();
                        await triggerServo();
                        break;
                }}
            }});

            // Start stream updates
            window.onload = function() {{
                getStatus();
                setInterval(getStatus, 2000);

                // Update stream at 2 FPS
                updateStream();
                setInterval(updateStream, 500);
            }};
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Camera Tracker Control with Detection</h1>

            <div class="main-grid">
                <div>
                    <h3>Camera View</h3>
                    <div class="camera-view">
                        <img id="cameraStream" style="display:none;" alt="Camera stream">
                        <div id="noStream" style="color: white;">Waiting for camera...</div>
                    </div>

                    <div id="detectionStatus" class="detection-status">
                        No detection
                    </div>

                    <div class="detections-log">
                        <strong>Recent Detections:</strong>
                        <div id="detectionsLog">No detections yet</div>
                    </div>
                </div>

                <div>
                    <div class="status">
                        <h3>System Status</h3>
                        <div id="connectionStatus">Checking...</div>
                        <div id="cameraStatus">Camera: --</div>
                        <div id="yawStatus">Yaw: --</div>
                        <div id="pitchStatus">Pitch: --</div>
                        <div id="detectionCount">Total Detections: 0</div>
                    </div>

                    <div class="controls">
                        <h3>Servo Control</h3>

                        <div class="control-group">
                            <label for="yawSlider">Yaw (L/R):</label>
                            <input type="range" id="yawSlider"
                                   min="{YAW_MIN}"
                                   max="{YAW_MAX}"
                                   value="{YAW_CENTER}"
                                   oninput="updatePosition(this.value, null)">
                            <span id="yawValue">{YAW_CENTER}</span>
                        </div>

                        <div class="control-group">
                            <label for="pitchSlider">Pitch (U/D):</label>
                            <input type="range" id="pitchSlider"
                                   min="{PITCH_MIN}"
                                   max="{PITCH_MAX}"
                                   value="{PITCH_CENTER}"
                                   oninput="updatePosition(null, this.value)">
                            <span id="pitchValue">{PITCH_CENTER}</span>
                        </div>

                        <div class="button-group">
                            <button class="center" onclick="centerServos()">Center Servos</button>
                            <button id="triggerButton" class="trigger" onclick="triggerServo()"
                                    {'disabled' if not enable_trigger else ''}>
                                Trigger Action
                            </button>
                            <div id="triggerStatus" class="servo-status" style="display: none;"></div>
                        </div>
                    </div>

                    <div class="keyboard-info">
                        <h3>Keyboard Controls</h3>
                        <div class="key-list">
                            <div><span class="key">←</span> / <span class="key">→</span> - Yaw left/right</div>
                            <div><span class="key">↑</span> / <span class="key">↓</span> - Pitch up/down</div>
                            <div><span class="key">C</span> - Center both servos</div>
                            <div><span class="key">T</span> - Trigger action servo</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/status")
async def get_status():
    """Get current status of the tracker"""
    if not tracker_instance:
        return JSONResponse({
            "connected": False,
            "yaw_position": "N/A",
            "pitch_position": "N/A",
            "camera_active": False,
            "detection_count": 0
        })

    return JSONResponse({
        "connected": tracker_instance.connected,
        "yaw_position": tracker_instance.current_yaw,
        "pitch_position": tracker_instance.current_pitch,
        "camera_active": tracker_instance.camera_active,
        "detection_count": tracker_instance.detection_count
    })

@app.get("/stream-frame")
async def stream_frame():
    """Get the latest camera frame with detection info"""
    if not tracker_instance or not tracker_instance.camera_active:
        return JSONResponse({
            "image": None,
            "detection": False,
            "confidence": 0,
            "recent_detections": []
        })

    frame_data = tracker_instance.get_latest_frame()
    return JSONResponse(frame_data)

@app.post("/set-position")
async def set_position(request: dict):
    """Set servo positions"""
    if not tracker_instance or not tracker_instance.connected:
        return JSONResponse({
            "success": False,
            "message": "Tracker not connected"
        })

    try:
        yaw = request.get("yaw")
        pitch = request.get("pitch")

        if yaw is not None:
            tracker_instance.set_yaw(yaw)
        if pitch is not None:
            tracker_instance.set_pitch(pitch)

        return JSONResponse({
            "success": True,
            "message": "Position updated"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        })

@app.post("/trigger-servo")
async def trigger_servo():
    """Manually trigger the action servo"""
    if not tracker_instance:
        return JSONResponse(
            content={"success": False, "message": "Tracker not initialized"},
            status_code=500
        )

    if not tracker_instance.trigger_servo_enabled:
        return JSONResponse(
            content={"success": False, "message": "Trigger servo is not enabled. Run with --enable-trigger flag to enable."},
            status_code=400
        )

    try:
        tracker_instance.trigger_action_servo()
        return JSONResponse(
            content={"success": True, "message": "Servo triggered successfully!"},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"Error triggering servo: {str(e)}"},
            status_code=500
        )

class CameraTracker:
    def __init__(self, port="/dev/cu.usbmodem5A680116511", enable_servos=True,
                 no_connect=False, enable_camera=False, enable_trigger=False,
                 model_path=None, confidence_threshold=0.85):
        """
        Initialize the camera tracker

        Args:
            port: Serial port for servo connection
            enable_servos: Whether to actually control tracking servos
            no_connect: Skip connection attempt entirely
            enable_camera: Enable Raspberry Pi camera
            enable_trigger: Enable GPIO trigger servo
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
        """
        self.port = port
        self.enable_servos = enable_servos
        self.no_connect = no_connect
        self.connected = False
        self.motor_bus = None

        # Camera and detection
        self.enable_camera = enable_camera and RPI_AVAILABLE
        self.camera_active = False
        self.picam2 = None
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.detection_count = 0
        self.latest_frame = None
        self.latest_detection = False
        self.recent_detections = []
        self.frame_lock = threading.Lock()

        # Trigger servo
        self.trigger_servo_enabled = enable_trigger and RPI_AVAILABLE
        self.trigger_servo = None

        # Current positions for tracking servos
        self.current_yaw = YAW_CENTER
        self.current_pitch = PITCH_CENTER

        # Create detections directory
        os.makedirs("detections", exist_ok=True)

        # Initialize tracking servos if enabled
        if self.enable_servos and not self.no_connect and FEETECH_AVAILABLE:
            self.connect_servos()

        # Initialize trigger servo if enabled
        if self.trigger_servo_enabled:
            self.init_trigger_servo()

        # Initialize camera if enabled
        if self.enable_camera:
            self.init_camera()

        # Initialize YOLO model if path provided
        if self.model_path and YOLO_AVAILABLE:
            self.init_model()

        # Start camera thread if everything is ready
        if self.camera_active:
            self.start_camera_thread()

    def init_trigger_servo(self):
        """Initialize GPIO trigger servo"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(TRIGGER_SERVO_PIN, GPIO.OUT)
            self.trigger_servo = GPIO.PWM(TRIGGER_SERVO_PIN, 50)  # 50Hz PWM
            self.trigger_servo.start(0)
            self.set_trigger_angle(25)  # Neutral position
            print(f"✓ Trigger servo initialized on GPIO{TRIGGER_SERVO_PIN}")
        except Exception as e:
            print(f"Failed to initialize trigger servo: {e}")
            self.trigger_servo_enabled = False

    def set_trigger_angle(self, angle):
        """Set trigger servo angle (0-180 degrees)"""
        if not self.trigger_servo_enabled or not self.trigger_servo:
            return

        # Convert angle to duty cycle
        min_duty = 5.0   # Duty cycle for 0 degrees
        max_duty = 10.0  # Duty cycle for 180 degrees
        duty_cycle = min_duty + (angle / 180.0) * (max_duty - min_duty)

        self.trigger_servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        self.trigger_servo.ChangeDutyCycle(0)

    def trigger_action_servo(self):
        """Trigger the action servo"""
        if not self.trigger_servo_enabled:
            print("[TRIGGER SIMULATION] Would trigger servo")
            return

        print("Triggering action servo...")
        self.set_trigger_angle(90)  # Move to trigger position
        time.sleep(3)  # Hold for 3 seconds
        self.set_trigger_angle(25)  # Return to neutral
        print("Trigger complete")

    def init_camera(self):
        """Initialize Raspberry Pi camera"""
        try:
            self.picam2 = Picamera2()

            # Configure camera for lower resolution (2 FPS optimization)
            config = self.picam2.create_still_configuration(
                main={"size": (640, 480)},  # Lower resolution for 2 FPS
                lores={"size": (320, 240)},
                display="lores"
            )
            self.picam2.configure(config)

            # Camera settings
            self.picam2.set_controls({
                "AwbMode": 1,
                "ColourGains": (1.0, 1.0),
                "ExposureTime": 25000,
                "AnalogueGain": 1.4,
                "Sharpness": 1.0,
                "Contrast": 1.3,
                "Brightness": 0.3
            })

            self.picam2.start()
            self.camera_active = True
            print("✓ Camera initialized (640x480 @ 2 FPS)")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.camera_active = False

    def init_model(self):
        """Initialize YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✓ YOLO model loaded: {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def connect_servos(self):
        """Connect to Feetech tracking servos"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Connection timed out")

        try:
            # Define motors with their IDs and models
            motors = {
                "yaw": Motor(YAW_MOTOR_ID, "sts3215", MotorNormMode.RANGE_M100_100),
                "pitch": Motor(PITCH_MOTOR_ID, "sts3215", MotorNormMode.RANGE_M100_100)
            }

            # Initialize motor bus
            self.motor_bus = FeetechMotorsBus(self.port, motors)

            # Set a timeout for connection (5 seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)

            try:
                self.motor_bus.connect(handshake=False)
                signal.alarm(0)  # Cancel the alarm
            except TimeoutError:
                print(f"Connection timed out after 5 seconds on {self.port}")
                raise

            # Initialize to center positions
            self.set_yaw(YAW_CENTER)
            self.set_pitch(PITCH_CENTER)

            self.connected = True
            print(f"✓ Connected to tracking servos on {self.port}")
            print(f"  Yaw motor (ID {YAW_MOTOR_ID}): {YAW_MIN}-{YAW_MAX} raw")
            print(f"  Pitch motor (ID {PITCH_MOTOR_ID}): {PITCH_MIN}-{PITCH_MAX} raw")

        except Exception as e:
            print(f"Failed to connect to tracking servos: {e}")
            self.connected = False
            self.motor_bus = None

    def raw_write(self, motor_name, value):
        """Write raw value directly to motor"""
        if not self.motor_bus or not self.connected:
            print(f"[SERVO SIMULATION] Would move {motor_name} to {value}")
            return

        try:
            # Write directly to Goal_Position register
            self.motor_bus.write("Goal_Position", motor_name, int(value), normalize=False)
        except Exception as e:
            print(f"Error writing to {motor_name}: {e}")

    def set_yaw(self, position):
        """Set yaw position"""
        position = max(YAW_MIN, min(YAW_MAX, int(position)))
        self.current_yaw = position

        if self.enable_servos:
            self.raw_write("yaw", position)

    def set_pitch(self, position):
        """Set pitch position"""
        position = max(PITCH_MIN, min(PITCH_MAX, int(position)))
        self.current_pitch = position

        if self.enable_servos:
            self.raw_write("pitch", position)

    def process_frame(self):
        """Capture and process a frame"""
        if not self.camera_active or not self.picam2:
            return

        try:
            # Capture frame
            image = self.picam2.capture_image("main")

            # Rotate 180 degrees
            image = image.rotate(180)

            # Run inference if model is available
            detection = False
            confidence = 0

            if self.model:
                # Save temporary file for inference
                temp_path = "temp_frame.jpg"
                image.save(temp_path)

                # Run inference
                results = self.model(temp_path, conf=self.confidence_threshold, verbose=False)

                # Check for detections
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            cls = int(box.cls)
                            conf = float(box.conf)
                            class_name = self.model.names[cls] if cls < len(self.model.names) else f"Class_{cls}"

                            # Check if it's a rat
                            if 'rat' in class_name.lower() or cls == 0:
                                detection = True
                                confidence = max(confidence, conf)

                                # Save detection
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                detection_path = f"detections/detection_{timestamp}.jpg"
                                shutil.copy2(temp_path, detection_path)

                                self.detection_count += 1
                                detection_msg = f"{datetime.now().strftime('%H:%M:%S')} - Rat detected (conf: {conf:.3f})"
                                self.recent_detections.append(detection_msg)
                                self.recent_detections = self.recent_detections[-10:]  # Keep last 10

                                print(f"🐀 Detection saved: {detection_path}")

                                # Trigger servo if enabled and detection is new
                                if detection and not self.latest_detection and self.trigger_servo_enabled:
                                    threading.Thread(target=self.trigger_action_servo, daemon=True).start()

                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # Convert to base64 for streaming
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=70)  # Lower quality for faster streaming
            img_str = base64.b64encode(buffer.getvalue()).decode()

            # Update latest frame
            with self.frame_lock:
                self.latest_frame = img_str
                self.latest_detection = detection
                self.latest_confidence = confidence

        except Exception as e:
            print(f"Frame processing error: {e}")

    def get_latest_frame(self):
        """Get the latest processed frame data"""
        with self.frame_lock:
            return {
                "image": self.latest_frame,
                "detection": self.latest_detection,
                "confidence": self.latest_confidence if hasattr(self, 'latest_confidence') else 0,
                "recent_detections": self.recent_detections
            }

    def camera_thread(self):
        """Camera processing thread (2 FPS)"""
        while self.camera_active:
            self.process_frame()
            time.sleep(0.5)  # 2 FPS

    def start_camera_thread(self):
        """Start the camera processing thread"""
        thread = threading.Thread(target=self.camera_thread, daemon=True)
        thread.start()
        print("Camera thread started (2 FPS)")

    def disconnect(self):
        """Disconnect and cleanup"""
        # Stop camera
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass

        # Cleanup GPIO
        if self.trigger_servo_enabled and RPI_AVAILABLE:
            try:
                if self.trigger_servo:
                    self.trigger_servo.stop()
                GPIO.cleanup()
            except:
                pass

        # Disconnect tracking servos
        if self.motor_bus:
            try:
                self.motor_bus.disconnect()
                print("Disconnected from servos")
            except:
                pass

        self.connected = False
        self.camera_active = False

def run_api_server(host="0.0.0.0", port=8000):
    """Run the FastAPI server in a separate thread"""
    uvicorn.run(app, host=host, port=port, log_level="error")

def main():
    parser = argparse.ArgumentParser(description="Camera tracker with servo control and rat detection")

    # Servo settings
    parser.add_argument("--port", "-p", type=str, default="/dev/cu.usbmodem5A680116511",
                       help="Serial port for servo connection")
    parser.add_argument("--disable-servos", action="store_true",
                       help="Disable tracking servo control (simulation mode)")
    parser.add_argument("--no-connect", action="store_true",
                       help="Skip servo connection attempt (web interface only)")

    # Camera and detection settings
    parser.add_argument("--enable-camera", action="store_true",
                       help="Enable Raspberry Pi camera")
    parser.add_argument("--model", "-m", type=str, default="runs/yolo11n-2025-08-24/weights/best.pt",
                       help="Path to YOLO model")
    parser.add_argument("--confidence", "-c", type=float, default=0.85,
                       help="Detection confidence threshold")

    # Trigger servo settings
    parser.add_argument("--enable-trigger", action="store_true",
                       help="Enable GPIO trigger servo")

    # API settings
    parser.add_argument("--api-host", type=str, default="0.0.0.0",
                       help="Host for FastAPI server")
    parser.add_argument("--api-port", type=int, default=8000,
                       help="Port for FastAPI server")

    args = parser.parse_args()

    # Create tracker and set global instance
    global tracker_instance
    tracker = CameraTracker(
        port=args.port,
        enable_servos=not args.disable_servos,
        no_connect=args.no_connect,
        enable_camera=args.enable_camera,
        enable_trigger=args.enable_trigger,
        model_path=args.model if args.enable_camera else None,
        confidence_threshold=args.confidence
    )
    tracker_instance = tracker

    print("=" * 60)
    print("Camera Tracker Control System with Detection")
    print("=" * 60)
    print(f"Tracking servos: {'ENABLED' if not args.disable_servos else 'DISABLED'}")
    print(f"Trigger servo: {'ENABLED' if args.enable_trigger else 'DISABLED'}")
    print(f"Camera: {'ENABLED' if args.enable_camera else 'DISABLED'}")
    print(f"Detection: {'ENABLED' if (args.enable_camera and args.model) else 'DISABLED'}")
    if args.enable_camera and args.model:
        print(f"Model: {args.model}")
        print(f"Confidence threshold: {args.confidence}")
    print()

    # Start API server in a separate thread
    api_thread = threading.Thread(
        target=run_api_server,
        args=(args.api_host, args.api_port),
        daemon=True
    )
    api_thread.start()
    print(f"Web interface: http://{args.api_host}:{args.api_port}")
    print()

    try:
        print("Server running. Press Ctrl+C to stop.")
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        tracker.disconnect()

if __name__ == "__main__":
    main()