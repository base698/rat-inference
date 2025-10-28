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
import io
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict
import uvicorn
import numpy as np
from yolo_inference import run_inference as yolo_run_inference, extract_detections

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

# Load configuration from YAML file
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file with fallback to defaults"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using default configuration")
        return None
    except Exception as e:
        print(f"Warning: Error loading {config_path}: {e}, using default configuration")
        return None

# Load config (with fallback to hardcoded defaults)
CONFIG = load_config()

# Servo configuration for tracking servos (with config fallback)
if CONFIG and 'servos' in CONFIG:
    YAW_MOTOR_ID = CONFIG['servos']['yaw']['motor_id']
    PITCH_MOTOR_ID = CONFIG['servos']['pitch']['motor_id']
    YAW_MIN = CONFIG['servos']['yaw']['min']
    YAW_MAX = CONFIG['servos']['yaw']['max']
    YAW_CENTER = CONFIG['servos']['yaw']['center']
    PITCH_MIN = CONFIG['servos']['pitch']['min']
    PITCH_MAX = CONFIG['servos']['pitch']['max']
    PITCH_CENTER = CONFIG['servos']['pitch']['center']
    TRIGGER_PWM_CHIP = CONFIG['servos']['trigger']['pwm_chip']
    TRIGGER_PWM_CHANNEL = CONFIG['servos']['trigger']['pwm_channel']
    TRIGGER_NEUTRAL_ANGLE = CONFIG['servos']['trigger']['neutral_angle']
    TRIGGER_ACTION_ANGLE = CONFIG['servos']['trigger']['action_angle']
else:
    # Default values (fallback)
    YAW_MOTOR_ID = 1
    PITCH_MOTOR_ID = 5
    YAW_MIN = 1600
    YAW_MAX = 3100
    YAW_CENTER = 2200
    PITCH_MIN = 0
    PITCH_MAX = 500
    PITCH_CENTER = 250
    TRIGGER_PWM_CHIP = 0
    TRIGGER_PWM_CHANNEL = 0
    TRIGGER_NEUTRAL_ANGLE = 99
    TRIGGER_ACTION_ANGLE = 38

# Tracking configuration (with config fallback)
if CONFIG and 'tracking' in CONFIG:
    TARGET_CROSSHAIR_X_BASE = CONFIG['tracking']['crosshair']['x_base']
    TARGET_CROSSHAIR_Y_BASE = CONFIG['tracking']['crosshair']['y_base']
    CROSSHAIR_SIZE = CONFIG['tracking']['crosshair_size']
    VIDEO_FPS = CONFIG['tracking']['video_fps']
    INFERENCE_FPS = CONFIG['tracking']['inference_fps']
    # Yaw compensation settings
    YAW_COMPENSATION_ENABLED = CONFIG['tracking']['yaw_compensation']['enabled']
    YAW_COMP_MIN = CONFIG['tracking']['yaw_compensation']['yaw_min']
    YAW_COMP_MAX = CONFIG['tracking']['yaw_compensation']['yaw_max']
    X_OFFSET_AT_MIN = CONFIG['tracking']['yaw_compensation']['x_offset_at_min']
    X_OFFSET_AT_MAX = CONFIG['tracking']['yaw_compensation']['x_offset_at_max']
    # Pitch compensation settings
    PITCH_COMPENSATION_ENABLED = CONFIG['tracking']['pitch_compensation']['enabled']
    PITCH_COMP_MIN = CONFIG['tracking']['pitch_compensation']['pitch_min']
    PITCH_COMP_MAX = CONFIG['tracking']['pitch_compensation']['pitch_max']
    Y_OFFSET_AT_MIN = CONFIG['tracking']['pitch_compensation']['y_offset_at_min']
    Y_OFFSET_AT_MAX = CONFIG['tracking']['pitch_compensation']['y_offset_at_max']
else:
    # Default values (fallback)
    TARGET_CROSSHAIR_X_BASE = 298
    TARGET_CROSSHAIR_Y_BASE = 199
    CROSSHAIR_SIZE = 20
    VIDEO_FPS = 30
    INFERENCE_FPS = 7
    YAW_COMPENSATION_ENABLED = False
    YAW_COMP_MIN = 1600
    YAW_COMP_MAX = 3100
    X_OFFSET_AT_MIN = 0
    X_OFFSET_AT_MAX = 0
    PITCH_COMPENSATION_ENABLED = False
    PITCH_COMP_MIN = 100
    PITCH_COMP_MAX = 550
    Y_OFFSET_AT_MIN = 0
    Y_OFFSET_AT_MAX = -120


def get_target_crosshair_x(current_yaw):
    """
    Calculate the target crosshair X position based on current yaw.

    As the yaw servo pans the camera (yaw changes), the same real-world point
    appears at different horizontal positions in the image. This function compensates
    for that by adjusting the target crosshair X position.

    Args:
        current_yaw: Current yaw servo position (raw value)

    Returns:
        int: Adjusted X position for the target crosshair
    """
    if not YAW_COMPENSATION_ENABLED:
        return int(TARGET_CROSSHAIR_X_BASE)

    # Clamp yaw to valid range
    yaw = max(YAW_COMP_MIN, min(YAW_COMP_MAX, current_yaw))

    # Linear interpolation between min and max yaw
    # t = 0.0 at YAW_COMP_MIN, t = 1.0 at YAW_COMP_MAX
    yaw_range = YAW_COMP_MAX - YAW_COMP_MIN
    if yaw_range == 0:
        t = 0.0
    else:
        t = (yaw - YAW_COMP_MIN) / yaw_range

    # Interpolate X offset
    x_offset = X_OFFSET_AT_MIN + t * (X_OFFSET_AT_MAX - X_OFFSET_AT_MIN)

    # Calculate adjusted X position
    adjusted_x = TARGET_CROSSHAIR_X_BASE + x_offset

    return int(adjusted_x)


def get_target_crosshair_y(current_pitch):
    """
    Calculate the target crosshair Y position based on current pitch.

    As the pitch servo tilts the camera down (pitch increases), the same real-world
    point appears higher in the image. This function compensates for that by adjusting
    the target crosshair Y position.

    Args:
        current_pitch: Current pitch servo position (raw value)

    Returns:
        int: Adjusted Y position for the target crosshair
    """
    if not PITCH_COMPENSATION_ENABLED:
        return int(TARGET_CROSSHAIR_Y_BASE)

    # Clamp pitch to valid range
    pitch = max(PITCH_COMP_MIN, min(PITCH_COMP_MAX, current_pitch))

    # Linear interpolation between min and max pitch
    # t = 0.0 at PITCH_COMP_MIN, t = 1.0 at PITCH_COMP_MAX
    pitch_range = PITCH_COMP_MAX - PITCH_COMP_MIN
    if pitch_range == 0:
        t = 0.0
    else:
        t = (pitch - PITCH_COMP_MIN) / pitch_range

    # Interpolate Y offset
    y_offset = Y_OFFSET_AT_MIN + t * (Y_OFFSET_AT_MAX - Y_OFFSET_AT_MIN)

    # Calculate adjusted Y position
    adjusted_y = TARGET_CROSSHAIR_Y_BASE + y_offset

    return int(adjusted_y)

@app.get("/")
async def root():
    """Root endpoint - serve static HTML file"""
    return FileResponse("static/index.html")

@app.get("/config")
async def get_config():
    """Get configuration values for the UI"""
    if tracker_instance and tracker_instance.connected:
        initial_yaw = tracker_instance.current_yaw
        initial_pitch = tracker_instance.current_pitch
        # Calculate dynamic crosshair positions based on current servo angles
        target_x = get_target_crosshair_x(initial_yaw)
        target_y = get_target_crosshair_y(initial_pitch)
    else:
        initial_yaw = YAW_CENTER
        initial_pitch = PITCH_CENTER
        target_x = get_target_crosshair_x(initial_yaw)
        target_y = get_target_crosshair_y(initial_pitch)

    return JSONResponse({
        "YAW_MIN": YAW_MIN,
        "YAW_MAX": YAW_MAX,
        "YAW_CENTER": YAW_CENTER,
        "PITCH_MIN": PITCH_MIN,
        "PITCH_MAX": PITCH_MAX,
        "PITCH_CENTER": PITCH_CENTER,
        "TARGET_CROSSHAIR_X": target_x,
        "TARGET_CROSSHAIR_Y": target_y,
        "initial_yaw": initial_yaw,
        "initial_pitch": initial_pitch,
        "enable_trigger": tracker_instance.trigger_servo_enabled if tracker_instance else False
    })

@app.get("/status")
async def get_status():
    """Get current status of the tracker"""
    if not tracker_instance:
        return JSONResponse({
            "connected": False,
            "yaw_position": "N/A",
            "pitch_position": "N/A",
            "camera_active": False,
            "detection_count": 0,
            "detection": False,
            "confidence": 0,
            "recent_detections": []
        })

    # Don't read positions on every status call - too frequent and causes bus errors
    # Positions are read once on connection and updated when we write to motors
    status_data = {
        "connected": tracker_instance.connected,
        "yaw_position": tracker_instance.current_yaw,
        "pitch_position": tracker_instance.current_pitch,
        "camera_active": tracker_instance.camera_active,
        "detection_count": tracker_instance.detection_count
    }

    # Add detection data if camera is active
    if tracker_instance.camera_active:
        detection_data = tracker_instance.get_detection_data()
        status_data.update(detection_data)
    else:
        status_data.update({
            "detection": False,
            "confidence": 0,
            "recent_detections": []
        })

    return JSONResponse(status_data)

@app.get("/stream-frame")
async def stream_frame():
    """Get the latest camera frame as bytes"""
    if not tracker_instance or not tracker_instance.camera_active:
        # Return empty JPEG if no camera
        return Response(content=b'', media_type="image/jpeg")

    frame_bytes = tracker_instance.get_latest_frame_bytes()
    if frame_bytes is None:
        return Response(content=b'', media_type="image/jpeg")

    return Response(content=frame_bytes, media_type="image/jpeg")

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

@app.get("/detections/{filename}")
async def get_detection(filename: str):
    """Serve detection image files"""
    detection_path = Path("detections") / filename
    if not detection_path.exists():
        raise HTTPException(status_code=404, detail="Detection image not found")
    return FileResponse(detection_path)

@app.post("/move-to-position")
async def move_to_position(request: Dict):
    """Move tracker to clicked canvas position"""
    if not tracker_instance:
        return JSONResponse({
            "success": False,
            "message": "Tracker not initialized"
        })

    try:
        x = request.get("x")
        y = request.get("y")

        if x is None or y is None:
            return JSONResponse({
                "success": False,
                "message": "Missing x or y coordinates"
            })

        # Use the move_to_pixel function for direct positioning (not PID controller)
        desired_yaw, desired_pitch = tracker_instance.move_to_pixel(x, y)

        # Update servo positions
        tracker_instance.set_yaw(desired_yaw)
        tracker_instance.set_pitch(desired_pitch)

        return JSONResponse({
            "success": True,
            "message": f"Moved to position ({x}, {y})",
            "yaw": desired_yaw,
            "pitch": desired_pitch
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        })

class CameraTracker:
    def __init__(self, port="/dev/cu.usbmodem5A680116511", enable_servos=True,
                 no_connect=False, enable_camera=False, enable_trigger=False,
                 model_path=None, confidence_threshold=0.85, camera_id=0,
                 use_csi=False, invert_camera=False, imgsz=640,
                 calibration_file=None):
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
            imgsz: Inference image size in pixels (default: 640)
            calibration_file: Path to camera calibration file (.npz)
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
        self.imgsz = imgsz
        self.detection_count = 0
        self.latest_frame = None
        self.latest_detection = False
        self.latest_bbox = None  # Store latest bounding box (x1, y1, x2, y2)
        self.latest_center_point = None  # Store latest center point (x, y)
        self.recent_detections = []
        self.frame_lock = threading.Lock()
        self.inference_lock = threading.Lock()

        # Camera calibration
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_enabled = False
        if self.calibration_file:
            self.load_calibration()

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

    def load_calibration(self):
        """Load camera calibration from file"""
        try:
            calib_data = np.load(self.calibration_file)
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            self.calibration_enabled = True
            rms_error = calib_data.get('rms_error', 0)
            print(f"✓ Camera calibration loaded: {self.calibration_file}")
            print(f"  RMS error: {rms_error:.3f} pixels")
            print(f"  Focal length: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
        except Exception as e:
            print(f"⚠ Failed to load calibration: {e}")
            self.calibration_enabled = False

    def undistort_frame(self, frame):
        """Apply camera calibration to undistort frame"""
        if not self.calibration_enabled:
            return frame

        try:
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        except Exception as e:
            print(f"Error undistorting frame: {e}")
            return frame

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
        # Draw target crosshair (dynamic position based on yaw and pitch)
        ch_x = get_target_crosshair_x(self.current_yaw)
        ch_y = get_target_crosshair_y(self.current_pitch)
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

    def move_to_pixel(self, target_x, target_y):
        """
        Directly move servos to point the crosshair at a target pixel position.
        Unlike observe(), this is NOT a PID controller - it's a direct positioning command.

        Args:
            target_x: X coordinate of target position in pixels
            target_y: Y coordinate of target position in pixels

        Returns:
            tuple: (desired_yaw, desired_pitch) servo positions in raw units
        """
        # Calculate pixel offset from current crosshair to target
        # Use dynamic crosshair positions based on current servo angles
        target_crosshair_x = get_target_crosshair_x(self.current_yaw)
        target_crosshair_y = get_target_crosshair_y(self.current_pitch)
        pixel_offset_x = target_x - target_crosshair_x
        pixel_offset_y = target_y - target_crosshair_y

        # Convert pixel offsets to angular offsets (degrees)
        angle_offset_yaw = self.pixels_to_angle(
            pixel_offset_x,
            self.image_width,
            self.camera_fov_horizontal
        )
        angle_offset_pitch = self.pixels_to_angle(
            pixel_offset_y,
            self.image_height,
            self.camera_fov_vertical
        )

        # Convert angular offsets to servo raw units
        # Note: Right arrow increases yaw (e.g., 2734 → 2739), so right click = increase yaw
        yaw_offset_raw = self.angle_to_servo_raw(angle_offset_yaw, axis='yaw')
        # Note: Pitch servo moves same as screen direction (down click = increase pitch)
        pitch_offset_raw = self.angle_to_servo_raw(angle_offset_pitch, axis='pitch')

        # Calculate desired servo positions
        desired_yaw = self.current_yaw + yaw_offset_raw
        desired_pitch = self.current_pitch + pitch_offset_raw

        # Clamp to valid servo ranges
        desired_yaw = max(YAW_MIN, min(YAW_MAX, desired_yaw))
        desired_pitch = max(PITCH_MIN, min(PITCH_MAX, desired_pitch))

        print(f"   Direct positioning:")
        print(f"     Target pixel: ({target_x}, {target_y})")
        print(f"     Pixel offset: X={pixel_offset_x:.1f}px, Y={pixel_offset_y:.1f}px")
        print(f"     Angle offset: Yaw={angle_offset_yaw:.2f}°, Pitch={angle_offset_pitch:.2f}°")
        print(f"     Servo move: Yaw {self.current_yaw} → {desired_yaw} ({yaw_offset_raw:+d}), Pitch {self.current_pitch} → {desired_pitch} ({pitch_offset_raw:+d})")

        return desired_yaw, desired_pitch

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
        # Use dynamic crosshair positions based on current servo angles
        # Positive error_x means rat is to the right
        # Positive error_y means rat is below center
        target_crosshair_x = get_target_crosshair_x(self.current_yaw)
        target_crosshair_y = get_target_crosshair_y(self.current_pitch)
        pixel_error_x = center_x - target_crosshair_x
        pixel_error_y = center_y - target_crosshair_y

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

            # Apply calibration (undistort)
            if self.calibration_enabled:
                frame = self.undistort_frame(frame)

            # Draw overlays
            frame = self.draw_overlays(frame)

            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                return

            # Store as bytes (no base64 encoding)
            img_bytes = buffer.tobytes()

            # Update latest frame
            with self.frame_lock:
                self.latest_frame = img_bytes

        except Exception as e:
            print(f"Video frame capture error: {e}")

    def run_inference(self):
        """Run inference at 7 FPS using shared inference module"""
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

            # Apply calibration (undistort) before inference
            if self.calibration_enabled:
                frame = self.undistort_frame(frame)

            # Run inference using shared inference module (YOLO works directly with numpy arrays)
            results = yolo_run_inference(
                self.model,
                frame,
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                verbose=False
            )

            # Extract detections using shared utility (filter for 'rat' class)
            detections = extract_detections(results, self.model, target_class='rat')

            # Process first detection (if any)
            detection = False
            confidence = 0
            bbox = None
            center_point = None

            if detections:
                # Use the first (highest confidence) detection
                det = detections[0]
                detection = True
                confidence = det['confidence']
                bbox = det['bbox']
                center_point = det['center']
                center_x, center_y = center_point

                # Save detection image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                detection_filename = f"detection_{timestamp}.jpg"
                detection_path = f"detections/{detection_filename}"
                cv2.imwrite(detection_path, frame)

                self.detection_count += 1
                # Format: "time - message | filename" so the UI can parse and link it
                detection_msg = f"{datetime.now().strftime('%H:%M:%S')} - Rat detected (conf: {confidence:.3f}) at ({center_x}, {center_y}) | {detection_filename}"
                self.recent_detections.append(detection_msg)
                self.recent_detections = self.recent_detections[-10:]

                print(f"🐀 Detection #{self.detection_count}: {detection_path}")
                print(f"   Center: ({center_x}, {center_y}), Confidence: {confidence:.3f}")

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

        except Exception as e:
            print(f"Inference error: {e}")

    def get_latest_frame_bytes(self):
        """Get the latest frame as bytes"""
        with self.frame_lock:
            return self.latest_frame

    def get_detection_data(self):
        """Get the latest detection data without image"""
        with self.inference_lock:
            return {
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
    parser.add_argument("--model", "-m", type=str, default="runs/yolo11n-2025-10-24/weights/best.pt",
                       help="Path to YOLO model")
    parser.add_argument("--confidence", "-c", type=float, default=0.75,
                       help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Inference image size in pixels (default: 640)")
    parser.add_argument("--calibration", type=str, default=None,
                       help="Path to camera calibration file (.npz)")

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
        invert_camera=args.invert_camera,
        imgsz=args.imgsz,
        calibration_file=args.calibration
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
        print(f"Inference image size: {args.imgsz}px")
    if args.calibration:
        calib_status = "ENABLED" if tracker.calibration_enabled else "FAILED"
        print(f"Calibration: {calib_status} ({args.calibration})")
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
