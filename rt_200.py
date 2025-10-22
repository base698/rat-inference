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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np

# Set library path for macOS
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

# Try importing OpenCV for camera
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - camera features disabled")

# Try importing CSI camera helper
try:
    from csi_camera_capture import CSICameraCapture
    CSI_HELPER_AVAILABLE = True
except ImportError:
    CSI_HELPER_AVAILABLE = False

# Try importing GPIO library
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("GPIO library not available - trigger servo features disabled")

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

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files directory for serving JS, CSS, and other assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Servo configuration for tracking servos
YAW_MOTOR_ID = 1
PITCH_MOTOR_ID = 5
YAW_MIN = 1600  # Minimum yaw position (raw)
YAW_MAX = 3100  # Maximum yaw position (raw)
YAW_CENTER = 2200  # Center position for yaw
PITCH_MIN = 0    # Minimum pitch position (level, raw)
PITCH_MAX = 500  # Maximum pitch position (~55 degrees down, raw)
PITCH_CENTER = 250 # Start at level position

# PWM servo configuration (trigger servo)
# Using sysfs hardware PWM - Pin 15 (GPIO12) = PWM Chip 0
TRIGGER_PWM_CHIP = 0  # PWM chip 0 (confirmed working)
TRIGGER_PWM_CHANNEL = 0
TRIGGER_NEUTRAL_ANGLE = 99  # Neutral position in degrees (rest/safe position)
TRIGGER_ACTION_ANGLE = 38   # Trigger position in degrees (activated trap)

# Tracking configuration
TARGET_CROSSHAIR_X = 291  # Center X position for target
TARGET_CROSSHAIR_Y = 199  # Center Y position for target
CROSSHAIR_SIZE = 20       # Size of crosshair in pixels
VIDEO_FPS = 30            # Video display frame rate
INFERENCE_FPS = 7         # Inference frame rate

@app.get("/")
async def root():
    """Root endpoint with control interface and camera view"""
    enable_trigger = tracker_instance and tracker_instance.trigger_servo_enabled

    # Get actual motor positions if connected, otherwise use center values
    if tracker_instance and tracker_instance.connected:
        initial_yaw = tracker_instance.current_yaw
        initial_pitch = tracker_instance.current_pitch
    else:
        initial_yaw = YAW_CENTER
        initial_pitch = PITCH_CENTER

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
                display: block;
                position: absolute;
                margin-top: -62px;
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
            let yawValue = {initial_yaw};  // Initialized from actual motor position
            let pitchValue = {initial_pitch};  // Initialized from actual motor position
            let desiredYaw = {initial_yaw};  // Desired yaw value from slider
            let desiredPitch = {initial_pitch};  // Desired pitch value from slider
            let positionFetching = false;
            let streamInterval = null;

            function updateSlider(yaw = null, pitch = null) {{
                // Update desired values and UI immediately
                if (yaw !== null) {{
                    desiredYaw = parseInt(yaw, 10);  // Explicit base-10 parsing
                    // Clamp to valid range
                    desiredYaw = Math.max({YAW_MIN}, Math.min({YAW_MAX}, desiredYaw));
                    document.getElementById('yawSlider').value = desiredYaw;
                    document.getElementById('yawValue').textContent = desiredYaw;
                }}
                if (pitch !== null) {{
                    desiredPitch = parseInt(pitch, 10);  // Explicit base-10 parsing
                    // Clamp to valid range
                    desiredPitch = Math.max({PITCH_MIN}, Math.min({PITCH_MAX}, desiredPitch));
                    document.getElementById('pitchSlider').value = desiredPitch;
                    document.getElementById('pitchValue').textContent = desiredPitch;
                }}

                // Trigger position update
                sendPositionUpdate();
            }}

            async function sendPositionUpdate() {{
                // Skip if already sending or if values haven't changed
                if (positionFetching || (desiredYaw === yawValue && desiredPitch === pitchValue)) {{
                    return;
                }}

                try {{
                    positionFetching = true;
                    const response = await fetch('/set-position', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            yaw: desiredYaw,
                            pitch: desiredPitch
                        }})
                    }});
                    const result = await response.json();
                    if (result.success) {{
                        // Update current values on success
                        yawValue = desiredYaw;
                        pitchValue = desiredPitch;
                    }} else {{
                        console.error('Failed to set position:', result.message);
                    }}
                }} catch (error) {{
                    console.error('Error:', error);
                }} finally {{
                    positionFetching = false;
                    // Check if values changed while we were sending
                    if (desiredYaw !== yawValue || desiredPitch !== pitchValue) {{
                        setTimeout(sendPositionUpdate, 50);
                    }}
                }}
            }}

            function centerServos() {{
                updateSlider({YAW_CENTER}, {PITCH_CENTER});
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
                    // Re-enable button after trigger completes (1.5s - slightly longer than the 1s hold time)
                    setTimeout(() => {{
                        button.disabled = false;
                        status.style.display = 'none';
                    }}, 1200);
                }}
            }}

            function updateDetectionInfo(data) {{
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

            let statusFetching = false;
            async function getStatus() {{
                if (statusFetching) return;

                try {{
                    statusFetching = true;
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

                    // Don't update sliders from status - let user control them directly
                    // Status text shows the tracked position, sliders show commanded position
                }} catch (error) {{
                    console.error('Error getting status:', error);
                }} finally {{
                    statusFetching = false;
                }}
            }}

            // Keyboard control
            document.addEventListener('keydown', async (event) => {{
                const step = 5;
                switch(event.key) {{
                    case 'ArrowLeft':
                        event.preventDefault();
                        updateSlider(Math.max({YAW_MIN}, desiredYaw - step), null);
                        break;
                    case 'ArrowRight':
                        event.preventDefault();
                        updateSlider(Math.min({YAW_MAX}, desiredYaw + step), null);
                        break;
                    case 'ArrowUp':
                        event.preventDefault();
                        updateSlider(null, Math.max({PITCH_MIN}, desiredPitch - step));
                        break;
                    case 'ArrowDown':
                        event.preventDefault();
                        updateSlider(null, Math.min({PITCH_MAX}, desiredPitch + step));
                        break;
                    case 'c':
                    case 'C':
                        event.preventDefault();
                        centerServos();
                        break;
                    case 't':
                    case 'T':
                        event.preventDefault();
                        await triggerServo();
                        break;
                }}
            }});

            // Initialize Web Worker
            let worker = null;

            // Start stream updates
            window.onload = function() {{
                getStatus();
                setInterval(getStatus, 2000);

                // Transfer canvas control to worker
                const canvas = document.getElementById('cameraStream');
                const offscreen = canvas.transferControlToOffscreen();

                try {{
                    worker = new Worker('/static/worker.js');

                    // Handle messages from worker
                    worker.onmessage = function(e) {{
                        if (e.data.type === 'frame_data') {{
                            // Worker has drawn the frame, now update detection info
                            updateDetectionInfo(e.data);

                            // Hide "waiting" message on first frame
                            const noStream = document.getElementById('noStream');
                            if (noStream && noStream.style.display !== 'none') {{
                                noStream.style.display = 'none';
                            }}
                        }}
                    }};

                    worker.onerror = function(e) {{
                        console.error('Worker error:', e);
                    }};

                    // Send canvas and start streaming
                    worker.postMessage({{
                        type: 'init',
                        canvas: offscreen
                    }}, [offscreen]);

                }} catch (error) {{
                    console.error('Failed to initialize worker:', error);
                }}
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
                        <canvas id="cameraStream" width="640" height="480" style="max-width: 100%; height: auto; border-radius: 10px;"></canvas>
                        <div id="noStream" style="color: white; position: absolute;">Waiting for camera...</div>
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
                                   value="{initial_yaw}"
                                   oninput="updateSlider(this.value, null)">
                            <span id="yawValue">{initial_yaw}</span>
                        </div>

                        <div class="control-group">
                            <label for="pitchSlider">Pitch (U/D):</label>
                            <input type="range" id="pitchSlider"
                                   min="{PITCH_MIN}"
                                   max="{PITCH_MAX}"
                                   value="{initial_pitch}"
                                   oninput="updateSlider(null, this.value)">
                            <span id="pitchValue">{initial_pitch}</span>
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

    # Don't read positions on every status call - too frequent and causes bus errors
    # Positions are read once on connection and updated when we write to motors
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
                 model_path=None, confidence_threshold=0.85, camera_id=0,
                 use_csi=False, invert_camera=False):
        """
        Initialize the camera tracker

        Args:
            port: Serial port for servo connection
            enable_servos: Whether to actually control tracking servos
            no_connect: Skip connection attempt entirely
            enable_camera: Enable camera
            enable_trigger: Enable GPIO trigger servo
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
            camera_id: Camera device ID (0 for USB, varies for CSI)
            use_csi: Use CSI camera with GStreamer pipeline (Jetson)
            invert_camera: Invert camera 180 degrees for upside-down mounting (default: False)
        """
        self.port = port
        self.enable_servos = enable_servos
        self.no_connect = no_connect
        self.connected = False
        self.motor_bus = None

        # Camera and detection
        self.enable_camera = enable_camera and CV2_AVAILABLE
        self.camera_active = False
        self.camera = None
        self.camera_id = camera_id
        self.use_csi = use_csi
        self.invert_camera = invert_camera
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.detection_count = 0
        self.latest_frame = None
        self.latest_detection = False
        self.latest_bbox = None  # Store latest bounding box (x1, y1, x2, y2)
        self.latest_center_point = None  # Store latest center point (x, y)
        self.recent_detections = []
        self.frame_lock = threading.Lock()
        self.inference_lock = threading.Lock()

        # Trigger servo (uses sysfs PWM, doesn't need GPIO library)
        self.trigger_servo_enabled = enable_trigger
        self.trigger_pwm_path = None

        # Current positions for tracking servos
        self.current_yaw = YAW_CENTER
        self.current_pitch = PITCH_CENTER

        # PID controller state
        self.pid_yaw_integral = 0.0
        self.pid_pitch_integral = 0.0
        self.pid_yaw_prev_error = 0.0
        self.pid_pitch_prev_error = 0.0
        self.pid_last_time = time.time()

        # PID gains (tune these for your system)
        self.pid_yaw_kp = 0.3    # Proportional gain for yaw
        self.pid_yaw_ki = 0.01   # Integral gain for yaw
        self.pid_yaw_kd = 0.1    # Derivative gain for yaw
        self.pid_pitch_kp = 0.3  # Proportional gain for pitch
        self.pid_pitch_ki = 0.01 # Integral gain for pitch
        self.pid_pitch_kd = 0.1  # Derivative gain for pitch

        # Camera and servo calibration
        # Assuming camera FOV (field of view) - adjust these based on your camera specs
        self.camera_fov_horizontal = 60.0  # degrees (typical for many webcams)
        self.camera_fov_vertical = 45.0    # degrees
        self.image_width = 640
        self.image_height = 480

        # Servo range in degrees (estimate - needs calibration)
        # These are the angular ranges that the servos can physically move
        self.yaw_range_degrees = 180.0    # Total yaw range in degrees
        self.pitch_range_degrees = 55.0   # Total pitch range in degrees

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

        # Start camera and inference threads if everything is ready
        if self.camera_active:
            self.start_camera_thread()
            if self.model:
                self.start_inference_thread()

    def init_trigger_servo(self):
        """Initialize PWM trigger servo using sysfs"""
        try:
            pwm_path = f"/sys/class/pwm/pwmchip{TRIGGER_PWM_CHIP}/pwm{TRIGGER_PWM_CHANNEL}"
            export_path = f"/sys/class/pwm/pwmchip{TRIGGER_PWM_CHIP}/export"

            # Export PWM channel if not already exported
            if not os.path.exists(pwm_path):
                with open(export_path, 'w') as f:
                    f.write(str(TRIGGER_PWM_CHANNEL))
                time.sleep(0.2)

            # Store PWM path
            self.trigger_pwm_path = pwm_path

            # Set period (20ms = 50Hz)
            with open(f"{pwm_path}/period", 'w') as f:
                f.write('20000000')  # 20ms in nanoseconds

            # Enable PWM
            with open(f"{pwm_path}/enable", 'w') as f:
                f.write('1')

            # Set to neutral position
            self.set_trigger_angle(TRIGGER_NEUTRAL_ANGLE)

            print(f"✓ Trigger servo initialized on PWM chip {TRIGGER_PWM_CHIP} (Pin 15, GPIO12)")
            print(f"  Neutral: {TRIGGER_NEUTRAL_ANGLE}°, Trigger: {TRIGGER_ACTION_ANGLE}°")
        except Exception as e:
            print(f"Failed to initialize trigger servo: {e}")
            self.trigger_servo_enabled = False
            self.trigger_pwm_path = None

    def set_trigger_angle(self, angle):
        """Set trigger servo angle (0-180 degrees) using sysfs PWM"""
        if not self.trigger_servo_enabled or not hasattr(self, 'trigger_pwm_path') or not self.trigger_pwm_path:
            return

        try:
            # Clamp angle
            angle = max(0, min(180, angle))

            # Convert angle to pulse width (nanoseconds)
            # MG996R: 0.5ms (500000ns) = 0°, 2.5ms (2500000ns) = 180°
            min_pulse = 500000
            max_pulse = 2500000
            pulse_width = int(min_pulse + (angle / 180.0) * (max_pulse - min_pulse))

            # Set duty cycle
            with open(f"{self.trigger_pwm_path}/duty_cycle", 'w') as f:
                f.write(str(pulse_width))

            # Ensure it's enabled
            with open(f"{self.trigger_pwm_path}/enable", 'w') as f:
                f.write('1')

        except Exception as e:
            print(f"Error setting trigger angle: {e}")

    def trigger_action_servo(self):
        """Trigger the action servo"""
        if not self.trigger_servo_enabled:
            print("[TRIGGER SIMULATION] Would trigger servo")
            return

        print("Triggering action servo...")
        self.set_trigger_angle(TRIGGER_ACTION_ANGLE)  # Move to trigger position (30°)
        time.sleep(1)  # Hold for 1 second
        self.set_trigger_angle(TRIGGER_NEUTRAL_ANGLE)  # Return to neutral (70°)
        print("Trigger complete")

    def gstreamer_pipeline(self, sensor_id=0, capture_width=1280, capture_height=720,
                           display_width=640, display_height=480, framerate=30, flip_method=0):
        """
        Generate GStreamer pipeline for Jetson CSI camera
        """
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
            f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink"
        )

    def init_camera(self):
        """Initialize camera (USB or CSI)"""
        try:
            if self.use_csi:
                # Determine flip method based on invert_camera flag
                flip_method = 2 if self.invert_camera else 0  # 2=rotate-180, 0=none

                # Use CSI camera helper (workaround for OpenCV without GStreamer)
                if CSI_HELPER_AVAILABLE:
                    self.camera = CSICameraCapture(
                        sensor_id=self.camera_id,
                        width=640,
                        height=480,
                        fps=VIDEO_FPS,
                        flip_method=flip_method
                    )
                    self.camera.start()
                    flip_status = "inverted" if self.invert_camera else "normal"
                    print(f"✓ CSI Camera initialized with subprocess+GStreamer (640x480 @ {VIDEO_FPS} FPS, {flip_status})")
                else:
                    # Fallback to cv2.VideoCapture with GStreamer
                    pipeline = self.gstreamer_pipeline(
                        sensor_id=self.camera_id,
                        capture_width=1280,
                        capture_height=720,
                        display_width=640,
                        display_height=480,
                        framerate=VIDEO_FPS,
                        flip_method=flip_method
                    )
                    self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    flip_status = "inverted" if self.invert_camera else "normal"
                    print(f"✓ CSI Camera initialized with GStreamer (640x480 @ {VIDEO_FPS} FPS, {flip_status})")
            else:
                # Use regular USB camera
                self.camera = cv2.VideoCapture(self.camera_id)

                # Set format to MJPEG if available (better compatibility)
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
                flip_status = "inverted" if self.invert_camera else "normal"
                print(f"✓ USB Camera {self.camera_id} initialized (640x480 @ {VIDEO_FPS} FPS, {flip_status})")

            if not self.camera.isOpened():
                raise Exception("Failed to open camera")

            self.camera_active = True

        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.camera_active = False
            self.camera = None

    def init_model(self):
        """Initialize YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✓ YOLO model loaded: {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def read_motor_positions(self):
        """Read current positions from motors"""
        if not self.motor_bus or not self.connected:
            return self.current_yaw, self.current_pitch

        try:
            yaw_pos = self.motor_bus.read("Present_Position", "yaw", normalize=False)
            pitch_pos = self.motor_bus.read("Present_Position", "pitch", normalize=False)
            return int(yaw_pos), int(pitch_pos)
        except Exception as e:
            print(f"Error reading motor positions: {e}")
            return self.current_yaw, self.current_pitch

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

            self.connected = True

            # Read actual current positions from motors
            actual_yaw, actual_pitch = self.read_motor_positions()
            self.current_yaw = actual_yaw
            self.current_pitch = actual_pitch

            print(f"✓ Connected to tracking servos on {self.port}")
            print(f"  Yaw motor (ID {YAW_MOTOR_ID}): {YAW_MIN}-{YAW_MAX} raw")
            print(f"  Pitch motor (ID {PITCH_MOTOR_ID}): {PITCH_MIN}-{PITCH_MAX} raw")
            print(f"  Current positions: Yaw={actual_yaw}, Pitch={actual_pitch}")

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
            # Ensure value is a positive integer (unsigned)
            # Feetech servos use 16-bit unsigned position values (0-65535)
            value = int(value) & 0xFFFF  # Mask to 16-bit unsigned
            # Write directly to Goal_Position register
            self.motor_bus.write("Goal_Position", motor_name, value, normalize=False)
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

    def draw_overlays(self, frame):
        """Draw crosshair, bounding box, and center point on image (OpenCV)"""
        # Draw target crosshair (fixed position)
        ch_x, ch_y = TARGET_CROSSHAIR_X, TARGET_CROSSHAIR_Y
        ch_size = CROSSHAIR_SIZE

        # Crosshair lines in green (BGR format)
        cv2.line(frame, (ch_x - ch_size, ch_y), (ch_x + ch_size, ch_y), (0, 255, 0), 2)
        cv2.line(frame, (ch_x, ch_y - ch_size), (ch_x, ch_y + ch_size), (0, 255, 0), 2)
        # Crosshair circle
        cv2.circle(frame, (ch_x, ch_y), 5, (0, 255, 0), 2)

        # Draw bounding box and center point if detection exists
        with self.inference_lock:
            if self.latest_bbox is not None:
                x1, y1, x2, y2 = self.latest_bbox
                # Draw bounding box in red (BGR)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            if self.latest_center_point is not None:
                cx, cy = self.latest_center_point
                # Draw center point in blue (BGR)
                cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)  # Filled circle
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)  # White outline
                # Draw line from center to target in yellow (BGR)
                cv2.line(frame, (cx, cy), (ch_x, ch_y), (0, 255, 255), 2)

        return frame

    def pixels_to_angle(self, pixel_error, image_dimension, fov_degrees):
        """
        Convert pixel error to angular error in degrees

        Args:
            pixel_error: Error in pixels from center
            image_dimension: Width or height of image in pixels
            fov_degrees: Field of view in degrees for this dimension

        Returns:
            Angular error in degrees
        """
        # Calculate degrees per pixel
        degrees_per_pixel = fov_degrees / image_dimension
        # Convert pixel error to angular error
        angle_error = pixel_error * degrees_per_pixel
        return angle_error

    def angle_to_servo_raw(self, angle_delta, axis='yaw'):
        """
        Convert angular change (in degrees) to servo raw units

        Args:
            angle_delta: Desired angular change in degrees
            axis: 'yaw' or 'pitch'

        Returns:
            Servo position change in raw units
        """
        if axis == 'yaw':
            # Yaw servo: 1500-3500 raw = 2000 units over yaw_range_degrees
            raw_range = YAW_MAX - YAW_MIN
            raw_per_degree = raw_range / self.yaw_range_degrees
        else:  # pitch
            # Pitch servo: 0-600 raw = 600 units over pitch_range_degrees
            raw_range = PITCH_MAX - PITCH_MIN
            raw_per_degree = raw_range / self.pitch_range_degrees

        # Convert angle to raw units
        raw_delta = angle_delta * raw_per_degree
        return int(raw_delta)

    def observe(self, center_x, center_y):
        """
        PID controller that takes detected rat center point and returns updated servo coordinates.
        Uses proper angle conversions and smooth control to center the rat in the view.

        The controller:
        1. Converts pixel error to angular error (degrees)
        2. Applies PID control in the angular domain
        3. Converts angular corrections to servo raw units
        4. Smoothly moves servos without overshooting

        Args:
            center_x: X coordinate of detected rat center
            center_y: Y coordinate of detected rat center

        Returns:
            tuple: (desired_yaw, desired_pitch) servo positions in raw units
        """
        # Calculate time delta for derivative and integral calculations
        current_time = time.time()
        dt = current_time - self.pid_last_time
        self.pid_last_time = current_time

        # Prevent division by zero or too small dt
        if dt < 0.001:
            dt = 0.001

        # Calculate pixel error from target crosshair
        # Positive error_x means rat is to the right
        # Positive error_y means rat is below center
        pixel_error_x = center_x - TARGET_CROSSHAIR_X
        pixel_error_y = center_y - TARGET_CROSSHAIR_Y

        # Convert pixel errors to angular errors (degrees)
        angle_error_yaw = self.pixels_to_angle(
            pixel_error_x,
            self.image_width,
            self.camera_fov_horizontal
        )
        angle_error_pitch = self.pixels_to_angle(
            pixel_error_y,
            self.image_height,
            self.camera_fov_vertical
        )

        # ===== YAW PID CONTROL =====
        # Proportional term
        yaw_p = self.pid_yaw_kp * angle_error_yaw

        # Integral term (accumulated error)
        self.pid_yaw_integral += angle_error_yaw * dt
        # Anti-windup: limit integral to prevent excessive buildup
        max_integral = 10.0  # degrees
        self.pid_yaw_integral = max(-max_integral, min(max_integral, self.pid_yaw_integral))
        yaw_i = self.pid_yaw_ki * self.pid_yaw_integral

        # Derivative term (rate of change of error)
        yaw_d = self.pid_yaw_kd * (angle_error_yaw - self.pid_yaw_prev_error) / dt
        self.pid_yaw_prev_error = angle_error_yaw

        # Calculate total yaw correction (in degrees)
        yaw_correction_deg = yaw_p + yaw_i + yaw_d

        # ===== PITCH PID CONTROL =====
        # Proportional term
        pitch_p = self.pid_pitch_kp * angle_error_pitch

        # Integral term (accumulated error)
        self.pid_pitch_integral += angle_error_pitch * dt
        self.pid_pitch_integral = max(-max_integral, min(max_integral, self.pid_pitch_integral))
        pitch_i = self.pid_pitch_ki * self.pid_pitch_integral

        # Derivative term (rate of change of error)
        pitch_d = self.pid_pitch_kd * (angle_error_pitch - self.pid_pitch_prev_error) / dt
        self.pid_pitch_prev_error = angle_error_pitch

        # Calculate total pitch correction (in degrees)
        pitch_correction_deg = pitch_p + pitch_i + pitch_d

        # ===== CONVERT ANGULAR CORRECTIONS TO SERVO RAW UNITS =====
        yaw_correction_raw = self.angle_to_servo_raw(yaw_correction_deg, axis='yaw')
        pitch_correction_raw = self.angle_to_servo_raw(pitch_correction_deg, axis='pitch')

        # Calculate desired servo positions
        desired_yaw = self.current_yaw + yaw_correction_raw
        desired_pitch = self.current_pitch + pitch_correction_raw

        # Clamp to valid servo ranges
        desired_yaw = max(YAW_MIN, min(YAW_MAX, desired_yaw))
        desired_pitch = max(PITCH_MIN, min(PITCH_MAX, desired_pitch))

        # Debug output (optional - can be removed for production)
        print(f"   PID Debug:")
        print(f"     Pixel error: X={pixel_error_x:.1f}px, Y={pixel_error_y:.1f}px")
        print(f"     Angle error: Yaw={angle_error_yaw:.2f}°, Pitch={angle_error_pitch:.2f}°")
        print(f"     Yaw PID: P={yaw_p:.3f}, I={yaw_i:.3f}, D={yaw_d:.3f} -> {yaw_correction_deg:.3f}° ({yaw_correction_raw} raw)")
        print(f"     Pitch PID: P={pitch_p:.3f}, I={pitch_i:.3f}, D={pitch_d:.3f} -> {pitch_correction_deg:.3f}° ({pitch_correction_raw} raw)")

        return desired_yaw, desired_pitch

    def capture_video_frame(self):
        """Capture a video frame at 30 FPS with overlays"""
        if not self.camera_active or not self.camera:
            return

        try:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                return

            # Rotate 180 degrees if invert_camera is enabled and not using CSI with flip_method
            # (CSI camera handles rotation in GStreamer pipeline)
            if self.invert_camera and not self.use_csi:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Resize to 640x480 if needed (camera may not respect resolution settings)
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))

            # Draw overlays
            frame = self.draw_overlays(frame)

            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                return

            # Convert to base64 for streaming
            img_str = base64.b64encode(buffer).decode()

            # Update latest frame
            with self.frame_lock:
                self.latest_frame = img_str

        except Exception as e:
            print(f"Video frame capture error: {e}")

    def run_inference(self):
        """Run inference at 15 FPS"""
        if not self.camera_active or not self.camera or not self.model:
            return

        try:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                return

            # Rotate 180 degrees if invert_camera is enabled and not using CSI with flip_method
            # (CSI camera handles rotation in GStreamer pipeline)
            if self.invert_camera and not self.use_csi:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Resize to 640x480 if needed (camera may not respect resolution settings)
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))

            # Save temporary file for inference
            temp_path = "temp_inference.jpg"
            cv2.imwrite(temp_path, frame)

            # Run inference (YOLO can work directly with numpy arrays or file paths)
            results = self.model(temp_path, conf=self.confidence_threshold, verbose=False)

            # Process detections
            detection = False
            confidence = 0
            bbox = None
            center_point = None

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

                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = (int(x1), int(y1), int(x2), int(y2))

                            # Calculate center point
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            center_point = (center_x, center_y)

                            # Save detection
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            detection_path = f"detections/detection_{timestamp}.jpg"
                            shutil.copy2(temp_path, detection_path)

                            self.detection_count += 1
                            detection_msg = f"{datetime.now().strftime('%H:%M:%S')} - Rat detected (conf: {conf:.3f}) at ({center_x}, {center_y})"
                            self.recent_detections.append(detection_msg)
                            self.recent_detections = self.recent_detections[-10:]

                            print(f"🐀 Detection #{self.detection_count}: {detection_path}")
                            print(f"   Center: ({center_x}, {center_y}), Confidence: {conf:.3f}")

                            # Get updated servo positions from observe function
                            desired_yaw, desired_pitch = self.observe(center_x, center_y)
                            print(f"   Servo update: Yaw {self.current_yaw} -> {desired_yaw}, Pitch {self.current_pitch} -> {desired_pitch}")

                            # Update servo positions
                            self.set_yaw(desired_yaw)
                            self.set_pitch(desired_pitch)

                            # Auto-trigger disabled - only trigger manually via button
                            # if detection and not self.latest_detection and self.trigger_servo_enabled:
                            #     threading.Thread(target=self.trigger_action_servo, daemon=True).start()

            # Update detection state
            with self.inference_lock:
                self.latest_detection = detection
                self.latest_confidence = confidence
                self.latest_bbox = bbox
                self.latest_center_point = center_point

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            print(f"Inference error: {e}")

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
        """Camera processing thread for video at 30 FPS"""
        while self.camera_active:
            self.capture_video_frame()
            time.sleep(1.0 / VIDEO_FPS)  # 30 FPS

    def inference_thread(self):
        """Inference processing thread at 15 FPS"""
        while self.camera_active:
            self.run_inference()
            time.sleep(1.0 / INFERENCE_FPS)  

    def start_camera_thread(self):
        """Start the camera processing thread"""
        thread = threading.Thread(target=self.camera_thread, daemon=True)
        thread.start()
        print(f"Camera thread started ({VIDEO_FPS} FPS)")

    def start_inference_thread(self):
        """Start the inference processing thread"""
        thread = threading.Thread(target=self.inference_thread, daemon=True)
        thread.start()
        print(f"Inference thread started ({INFERENCE_FPS} FPS)")

    def disconnect(self):
        """Disconnect and cleanup"""
        # Stop camera
        if self.camera:
            try:
                self.camera.release()
            except:
                pass

        # Cleanup PWM trigger servo
        if self.trigger_servo_enabled and hasattr(self, 'trigger_pwm_path') and self.trigger_pwm_path:
            try:
                # Disable PWM
                with open(f"{self.trigger_pwm_path}/enable", 'w') as f:
                    f.write('0')
                # Unexport PWM
                unexport_path = f"/sys/class/pwm/pwmchip{TRIGGER_PWM_CHIP}/unexport"
                with open(unexport_path, 'w') as f:
                    f.write(str(TRIGGER_PWM_CHANNEL))
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
    parser.add_argument("--port", "-p", type=str, default="/dev/ttyACM0",
                       help="Serial port for servo connection")
    parser.add_argument("--disable-servos", action="store_true",
                       help="Disable tracking servo control (simulation mode)")
    parser.add_argument("--no-connect", action="store_true",
                       help="Skip servo connection attempt (web interface only)")

    # Camera and detection settings
    parser.add_argument("--enable-camera", action="store_true",
                       help="Enable camera")
    parser.add_argument("--camera-id", type=int, default=0,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--use-csi", action="store_true",
                       help="Use CSI camera with GStreamer pipeline (Jetson)")
    parser.add_argument("--invert-camera", action="store_true",
                       help="Invert camera 180 degrees for upside-down mounting")
    parser.add_argument("--model", "-m", type=str, default="runs/yolo11n-2025-10-20/weights/best.pt",
                       help="Path to YOLO model")
    parser.add_argument("--confidence", "-c", type=float, default=0.75,
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
        confidence_threshold=args.confidence,
        camera_id=args.camera_id,
        use_csi=args.use_csi,
        invert_camera=args.invert_camera
    )
    tracker_instance = tracker

    print("=" * 60)
    print("Camera Tracker Control System with Detection")
    print("=" * 60)
    print(f"Tracking servos: {'ENABLED' if not args.disable_servos else 'DISABLED'}")
    print(f"Trigger servo: {'ENABLED' if args.enable_trigger else 'DISABLED'}")
    print(f"Camera: {'ENABLED' if args.enable_camera else 'DISABLED'}")
    if args.enable_camera:
        camera_type = "CSI (GStreamer)" if args.use_csi else "USB"
        invert_status = "inverted (upside-down)" if args.invert_camera else "normal"
        print(f"Camera type: {camera_type} (ID: {args.camera_id}, {invert_status})")
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
