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
import uvicorn
import numpy as np
from ratbot.vision.yolo_inference import run_inference as yolo_run_inference, extract_detections
from ratbot.robot import (
    CrosshairAiming,
    DepthCrosshairCompensation,
    PitchCompensation,
    YawCompensation,
)
from ratbot.web import ControlApiConfig, create_control_app

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
    from ratbot.vision.csi_camera import CSICameraCapture
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
    PITCH_COMP_POINTS = sorted(
        [
            (float(point['pitch']), float(point['offset']))
            for point in CONFIG['tracking']['pitch_compensation'].get('points', [])
        ],
        key=lambda point: point[0]
    )
    depth_comp_config = CONFIG['tracking'].get('depth_crosshair_compensation', {})
    DEPTH_CROSSHAIR_COMPENSATION_ENABLED = depth_comp_config.get('enabled', False)
    LASER_VERTICAL_OFFSET_MM = float(depth_comp_config.get('laser_vertical_offset_mm', 55.0))
    LASER_REFERENCE_DISTANCE_MM = float(depth_comp_config.get('reference_distance_mm', 1000.0))
    LASER_MAX_ADJUST_PX = float(depth_comp_config.get('max_adjust_px', 80.0))
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
    PITCH_COMP_POINTS = []
    DEPTH_CROSSHAIR_COMPENSATION_ENABLED = False
    LASER_VERTICAL_OFFSET_MM = 55.0
    LASER_REFERENCE_DISTANCE_MM = 1000.0
    LASER_MAX_ADJUST_PX = 80.0


AIMING = CrosshairAiming(
    x_base=TARGET_CROSSHAIR_X_BASE,
    y_base=TARGET_CROSSHAIR_Y_BASE,
    yaw=YawCompensation(
        enabled=YAW_COMPENSATION_ENABLED,
        yaw_min=YAW_COMP_MIN,
        yaw_max=YAW_COMP_MAX,
        x_offset_at_min=X_OFFSET_AT_MIN,
        x_offset_at_max=X_OFFSET_AT_MAX,
    ),
    pitch=PitchCompensation(
        enabled=PITCH_COMPENSATION_ENABLED,
        pitch_min=PITCH_COMP_MIN,
        pitch_max=PITCH_COMP_MAX,
        y_offset_at_min=Y_OFFSET_AT_MIN,
        y_offset_at_max=Y_OFFSET_AT_MAX,
        points=tuple(PITCH_COMP_POINTS),
    ),
    depth=DepthCrosshairCompensation(
        enabled=DEPTH_CROSSHAIR_COMPENSATION_ENABLED,
        laser_vertical_offset_mm=LASER_VERTICAL_OFFSET_MM,
        reference_distance_mm=LASER_REFERENCE_DISTANCE_MM,
        max_adjust_px=LASER_MAX_ADJUST_PX,
    ),
)


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
    return AIMING.target_x(current_yaw)


def get_target_crosshair_y(current_pitch, camera_bore_offset_mm=82, focal_length_px=None, assumed_distance_mm=5000):
    """
    Calculate the target crosshair Y position based on current pitch using proper geometry.

    Camera is mounted 82mm above the bore. Both are pitched together, so the parallax
    offset depends on the distance to target and focal length.

    Args:
        current_pitch: Current pitch servo position (raw value)
        camera_bore_offset_mm: Vertical distance from camera to bore (default: 82mm)
        focal_length_px: Focal length in pixels (from calibration), or None to use assumed distance method
        assumed_distance_mm: Assumed target distance for compensation (default: 5000mm = 5m)

    Returns:
        int: Adjusted Y position for the target crosshair
    """
    return AIMING.target_y(
        current_pitch,
        camera_bore_offset_mm=camera_bore_offset_mm,
        focal_length_px=focal_length_px,
        assumed_distance_mm=assumed_distance_mm,
    )

def choose_detections_dir():
    """Return a writable detections directory, falling back if old files are root-owned."""
    for candidate in ("detections", os.path.join("run_logs", "detections")):
        try:
            os.makedirs(candidate, exist_ok=True)
            test_path = os.path.join(candidate, ".write_test")
            with open(test_path, "w", encoding="utf-8") as test_file:
                test_file.write("ok")
            os.remove(test_path)
            if candidate != "detections":
                print(f"⚠ detections/ is not writable; saving detections to {candidate}")
            return candidate
        except OSError:
            continue

    raise RuntimeError("No writable detections directory found")

DETECTIONS_DIR = choose_detections_dir()

control_api = create_control_app(
    ControlApiConfig(
        yaw_min=YAW_MIN,
        yaw_max=YAW_MAX,
        yaw_center=YAW_CENTER,
        pitch_min=PITCH_MIN,
        pitch_max=PITCH_MAX,
        pitch_center=PITCH_CENTER,
        detections_dir=DETECTIONS_DIR,
    ),
    get_target_crosshair_x=get_target_crosshair_x,
    get_target_crosshair_y=get_target_crosshair_y,
)
app = control_api.app

class CameraTracker:
    def __init__(self, port="/dev/cu.usbmodem5A680116511", enable_servos=True,
                 no_connect=False, enable_camera=False, enable_trigger=False,
                 model_path=None, confidence_threshold=0.85, camera_id=0,
                 use_csi=False, invert_camera=False, imgsz=640,
                 inference_fps=None, target_classes=None, calibration_file=None,
                 stereo_mode=False, baseline_override=None,
                 tracking_smoothing=0.45, max_yaw_step=45, max_pitch_step=28):
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
            inference_fps: Target inference loop rate
            target_classes: Class names to track from YOLO detections
            calibration_file: Path to camera calibration file (.npz)
            stereo_mode: Use stereo cameras for depth estimation
            baseline_override: Override baseline in mm (if calibration is incorrect)
            tracking_smoothing: Detection center smoothing alpha (0 disables, 1 follows raw detections)
            max_yaw_step: Maximum yaw raw-unit move per inference update
            max_pitch_step: Maximum pitch raw-unit move per inference update
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
        self.camera2 = None  # Second camera for stereo
        self.camera_id = camera_id
        self.use_csi = use_csi
        self.stereo_mode = stereo_mode
        self.invert_camera = invert_camera
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.inference_fps = max(1.0, float(inference_fps or INFERENCE_FPS))
        self.target_classes = self._normalize_target_classes(target_classes)
        self.tracking_smoothing = max(0.0, min(1.0, float(tracking_smoothing)))
        self.max_yaw_step = max(0, int(max_yaw_step))
        self.max_pitch_step = max(0, int(max_pitch_step))
        self.detection_count = 0
        self.latest_frame = None
        self.latest_detection = False
        self.latest_bbox = None  # Store latest bounding box (x1, y1, x2, y2)
        self.latest_center_point = None  # Store latest center point (x, y)
        self.smoothed_tracking_center = None
        self.latest_depth = None  # Store depth in mm at detection point
        self.recent_detections = []
        self.frame_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.motor_lock = threading.Lock()
        self.inference_loop_count = 0
        self.inference_fps_window_start = time.time()

        # Camera calibration
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_enabled = False

        # Stereo calibration
        self.stereo_calibration_enabled = False
        self.K1 = None
        self.D1 = None
        self.K2 = None
        self.D2 = None
        self.R = None
        self.T = None
        self.baseline = None
        self.baseline_override = baseline_override
        self.stereo_matcher = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.stereo_map_left = None
        self.stereo_map_right = None
        self.stereo_focal_length = None
        self.stereo_min_disparity = 0
        self.stereo_num_disparities = 192
        self.stereo_disparity_sign = 1
        self.last_depth_debug = "not computed"

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
        pid_config = CONFIG.get('pid', {}) if CONFIG else {}
        yaw_pid_config = pid_config.get('yaw', {})
        pitch_pid_config = pid_config.get('pitch', {})
        self.pid_yaw_kp = float(yaw_pid_config.get('kp', 0.85))
        self.pid_yaw_ki = float(yaw_pid_config.get('ki', 0.01))
        self.pid_yaw_kd = float(yaw_pid_config.get('kd', 0.03))
        self.pid_pitch_kp = float(pitch_pid_config.get('kp', 0.85))
        self.pid_pitch_ki = float(pitch_pid_config.get('ki', 0.01))
        self.pid_pitch_kd = float(pitch_pid_config.get('kd', 0.03))
        self.pid_max_integral = float(pid_config.get('max_integral', 10.0))

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
        self.detections_dir = DETECTIONS_DIR
        os.makedirs(self.detections_dir, exist_ok=True)

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

    @staticmethod
    def _normalize_target_classes(target_classes):
        if target_classes is None:
            return []

        if isinstance(target_classes, str):
            raw_classes = [target_classes]
        else:
            raw_classes = target_classes

        normalized = []
        for item in raw_classes:
            normalized.extend(part.strip().lower() for part in str(item).split(","))

        normalized = [item for item in normalized if item]
        if any(item in {"*", "all", "any"} for item in normalized):
            return []

        return normalized

    @staticmethod
    def _minimum_tracking_step(raw_delta, pixel_error, deadband_px=8, min_step=6):
        if abs(pixel_error) <= deadband_px or abs(raw_delta) >= min_step:
            return raw_delta

        return min_step if pixel_error > 0 else -min_step

    @staticmethod
    def _limit_tracking_step(current_position, desired_position, max_step):
        if max_step <= 0:
            return desired_position

        delta = desired_position - current_position
        if abs(delta) <= max_step:
            return desired_position

        return current_position + (max_step if delta > 0 else -max_step)

    def smooth_tracking_center(self, center_x, center_y):
        if self.tracking_smoothing <= 0 or self.smoothed_tracking_center is None:
            self.smoothed_tracking_center = (float(center_x), float(center_y))
            return center_x, center_y

        alpha = self.tracking_smoothing
        prev_x, prev_y = self.smoothed_tracking_center
        smooth_x = (alpha * center_x) + ((1.0 - alpha) * prev_x)
        smooth_y = (alpha * center_y) + ((1.0 - alpha) * prev_y)
        self.smoothed_tracking_center = (smooth_x, smooth_y)

        return int(round(smooth_x)), int(round(smooth_y))

    def load_calibration(self):
        """Load camera calibration from file (single or stereo)"""
        try:
            # Check if file exists
            if not os.path.exists(self.calibration_file):
                print(f"⚠ Calibration file not found: {self.calibration_file}")
                self.calibration_enabled = False
                return

            calib_data = np.load(self.calibration_file)

            # Check if it's stereo calibration
            if 'K1' in calib_data and 'K2' in calib_data:
                # Stereo calibration
                self.K1 = calib_data['K1']
                self.D1 = calib_data['D1']
                self.K2 = calib_data['K2']
                self.D2 = calib_data['D2']
                self.R = calib_data['R']
                self.T = calib_data['T']
                self.baseline = calib_data['baseline']
                self.stereo_calibration_enabled = True
                self.calibration_enabled = True

                # Use left camera parameters for single-camera fallback
                self.camera_matrix = self.K1
                self.dist_coeffs = self.D1
                self.init_stereo_rectification()

                # Create stereo matcher for disparity calculation.
                # StereoSGBM is more accurate than StereoBM.
                self.stereo_matcher = cv2.StereoSGBM_create(
                    minDisparity=self.stereo_min_disparity,
                    numDisparities=self.stereo_num_disparities,  # Must be divisible by 16
                    blockSize=5,
                    P1=8 * 3 * 5**2,  # Smoothness penalty
                    P2=32 * 3 * 5**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )

                # Get RMS error from calibration
                rms_error = calib_data.get('rms_error', 0)

                # Apply baseline override if provided
                baseline_from_calib = self.baseline
                if self.baseline_override is not None:
                    self.baseline = self.baseline_override
                    print(f"✓ Stereo calibration loaded: {self.calibration_file}")
                    print(f"  RMS error: {rms_error:.3f} pixels")
                    print(f"  Baseline (calibrated): {baseline_from_calib:.2f} mm")
                    print(f"  Baseline (OVERRIDDEN): {self.baseline:.2f} mm ⚠")
                    print(f"  Left focal length: fx={self.K1[0,0]:.2f}, fy={self.K1[1,1]:.2f}")
                    print(f"  Right focal length: fx={self.K2[0,0]:.2f}, fy={self.K2[1,1]:.2f}")
                else:
                    print(f"✓ Stereo calibration loaded: {self.calibration_file}")
                    print(f"  RMS error: {rms_error:.3f} pixels")
                    print(f"  Baseline: {self.baseline:.2f} mm")
                    print(f"  Left focal length: fx={self.K1[0,0]:.2f}, fy={self.K1[1,1]:.2f}")
                    print(f"  Right focal length: fx={self.K2[0,0]:.2f}, fy={self.K2[1,1]:.2f}")
                if rms_error > 1.0:
                    print(f"  ⚠ Stereo RMS is high ({rms_error:.3f}px); distance may be inaccurate until recalibrated")
            else:
                # Single camera calibration
                self.camera_matrix = calib_data['camera_matrix']
                self.dist_coeffs = calib_data['dist_coeffs']
                self.calibration_enabled = True
                rms_error = calib_data.get('rms_error', 0)
                print(f"✓ Single camera calibration loaded: {self.calibration_file}")
                print(f"  RMS error: {rms_error:.3f} pixels")
                print(f"  Focal length: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
        except Exception as e:
            print(f"⚠ Failed to load calibration: {e}")
            self.calibration_enabled = False
            self.stereo_calibration_enabled = False

    def init_stereo_rectification(self, image_size=(640, 480)):
        """Build rectification maps so stereo disparity is computed on aligned frames."""
        try:
            self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
                self.K1, self.D1,
                self.K2, self.D2,
                image_size,
                self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=-1
            )

            center_point = np.array([[[image_size[0] / 2.0, image_size[1] / 2.0]]], dtype=np.float32)
            rectified_center = cv2.undistortPoints(
                center_point, self.K1, self.D1, R=self.R1, P=self.P1
            )[0, 0]
            rectified_shift_x = (image_size[0] / 2.0) - float(rectified_center[0])
            rectified_shift_y = (image_size[1] / 2.0) - float(rectified_center[1])
            if abs(rectified_shift_x) > 1.0 or abs(rectified_shift_y) > 1.0:
                self.P1[0, 2] += rectified_shift_x
                self.P2[0, 2] += rectified_shift_x
                self.P1[1, 2] += rectified_shift_y
                self.P2[1, 2] += rectified_shift_y

            self.stereo_map_left = cv2.initUndistortRectifyMap(
                self.K1, self.D1, self.R1, self.P1, image_size, cv2.CV_16SC2
            )
            self.stereo_map_right = cv2.initUndistortRectifyMap(
                self.K2, self.D2, self.R2, self.P2, image_size, cv2.CV_16SC2
            )

            self.stereo_focal_length = float(self.P1[0, 0])
            rectified_baseline = abs(float(self.P2[0, 3]) / float(self.P2[0, 0]))
            if self.baseline_override is None:
                self.baseline = rectified_baseline

            # P2[0,3] sign tells us which disparity direction to expect after rectification.
            # Some camera orderings produce negative valid disparities.
            if float(self.P2[0, 3]) > 0:
                self.stereo_disparity_sign = -1
                self.stereo_min_disparity = -192
                self.stereo_num_disparities = 256
            else:
                self.stereo_disparity_sign = 1
                self.stereo_min_disparity = 0
                self.stereo_num_disparities = 192

            print("✓ Stereo rectification initialized")
            print(f"  Rectified focal length: {self.stereo_focal_length:.2f}px")
            print(f"  Rectified baseline: {rectified_baseline:.2f} mm")
            print(f"  Rectified image shift: x={rectified_shift_x:.1f}px, y={rectified_shift_y:.1f}px")
            print(f"  Disparity search: min={self.stereo_min_disparity}, num={self.stereo_num_disparities}, sign={self.stereo_disparity_sign:+d}")
        except Exception as e:
            print(f"⚠ Failed to initialize stereo rectification: {e}")
            self.stereo_calibration_enabled = False
            self.stereo_map_left = None
            self.stereo_map_right = None

    def rectify_stereo_frames(self, frame_left, frame_right):
        """Rectify left/right frames before stereo matching."""
        if self.stereo_map_left is None or self.stereo_map_right is None:
            return frame_left, frame_right

        try:
            left = cv2.remap(
                frame_left,
                self.stereo_map_left[0],
                self.stereo_map_left[1],
                cv2.INTER_LINEAR
            )
            right = cv2.remap(
                frame_right,
                self.stereo_map_right[0],
                self.stereo_map_right[1],
                cv2.INTER_LINEAR
            )
            return left, right
        except Exception as e:
            print(f"Error rectifying stereo frames: {e}")
            return frame_left, frame_right

    def rectify_stereo_point(self, x, y):
        """Map a point from the raw left image into rectified stereo coordinates."""
        if self.R1 is None or self.P1 is None:
            return int(x), int(y)

        try:
            point = np.array([[[float(x), float(y)]]], dtype=np.float32)
            rectified = cv2.undistortPoints(point, self.K1, self.D1, R=self.R1, P=self.P1)
            rx, ry = rectified[0, 0]
            return int(round(rx)), int(round(ry))
        except Exception as e:
            print(f"Error rectifying stereo point: {e}")
            return int(x), int(y)

    def undistort_frame(self, frame, use_left=True):
        """Apply camera calibration to undistort frame"""
        if not self.calibration_enabled:
            return frame

        try:
            if self.stereo_calibration_enabled:
                # Use appropriate camera matrix for stereo
                K = self.K1 if use_left else self.K2
                D = self.D1 if use_left else self.D2
                return cv2.undistort(frame, K, D)
            else:
                return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        except Exception as e:
            print(f"Error undistorting frame: {e}")
            return frame

    def calculate_depth(self, frame_left, frame_right, x, y):
        """
        Calculate depth at a specific pixel location using stereo matching

        Args:
            frame_left: Left camera frame (undistorted)
            frame_right: Right camera frame (undistorted)
            x: X coordinate in image
            y: Y coordinate in image

        Returns:
            Depth in millimeters, or None if calculation fails
        """
        if not self.stereo_calibration_enabled or self.stereo_matcher is None:
            self.last_depth_debug = "stereo disabled"
            return None

        try:
            frame_left, frame_right = self.rectify_stereo_frames(frame_left, frame_right)
            x, y = self.rectify_stereo_point(x, y)

            # Convert to grayscale for stereo matching
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            # Compute disparity map
            disparity = self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0

            if x < 0 or x >= disparity.shape[1] or y < 0 or y >= disparity.shape[0]:
                self.last_depth_debug = f"rectified point out of frame ({x}, {y})"
                return None

            # Use the median of valid disparities around the point to reduce pixel noise.
            d = None
            for radius in (4, 8, 16, 32):
                x1 = max(0, x - radius)
                x2 = min(disparity.shape[1], x + radius + 1)
                y1 = max(0, y - radius)
                y2 = min(disparity.shape[0], y + radius + 1)
                window = disparity[y1:y2, x1:x2]
                if self.stereo_disparity_sign < 0:
                    valid = window[(window < -1) & (window >= self.stereo_min_disparity)]
                else:
                    valid = window[window > 0]
                if valid.size >= 8:
                    d = abs(float(np.median(valid)))
                    break

            if d is None:
                self.last_depth_debug = f"no valid disparity near ({x}, {y})"
                return None

            # Check if disparity is valid (not zero or negative)
            if d <= 0:
                self.last_depth_debug = f"invalid disparity {d:.2f}px"
                return None

            # Calculate depth using formula: depth = (focal_length * baseline) / disparity
            # focal_length is in pixels, baseline is in mm
            focal_length = self.stereo_focal_length or self.K1[0, 0]
            depth_mm = (focal_length * self.baseline) / d
            self.last_depth_debug = f"disp {d:.2f}px"

            return depth_mm

        except Exception as e:
            self.last_depth_debug = f"error: {e}"
            print(f"Error calculating depth: {e}")
            return None

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
        """Initialize camera (USB or CSI), and second camera if stereo mode"""
        try:
            if self.use_csi:
                # Determine flip method based on invert_camera flag
                flip_method = 2 if self.invert_camera else 0  # 2=rotate-180, 0=none

                # Use CSI camera helper (workaround for OpenCV without GStreamer)
                if CSI_HELPER_AVAILABLE:
                    self.camera = CSICameraCapture(
                        sensor_id=0,  # Left camera is sensor 0
                        width=640,
                        height=480,
                        fps=VIDEO_FPS,
                        flip_method=flip_method
                    )
                    self.camera.start()
                    flip_status = "inverted" if self.invert_camera else "normal"
                    print(f"✓ CSI Camera (left) initialized with subprocess+GStreamer (640x480 @ {VIDEO_FPS} FPS, {flip_status})")

                    # Initialize second camera for stereo
                    if self.stereo_mode:
                        self.camera2 = CSICameraCapture(
                            sensor_id=1,  # Right camera is sensor 1
                            width=640,
                            height=480,
                            fps=VIDEO_FPS,
                            flip_method=flip_method
                        )
                        self.camera2.start()
                        print(f"✓ CSI Camera (right) initialized with subprocess+GStreamer (640x480 @ {VIDEO_FPS} FPS, {flip_status})")
                else:
                    # Fallback to cv2.VideoCapture with GStreamer
                    pipeline = self.gstreamer_pipeline(
                        sensor_id=0,
                        capture_width=1280,
                        capture_height=720,
                        display_width=640,
                        display_height=480,
                        framerate=VIDEO_FPS,
                        flip_method=flip_method
                    )
                    self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    flip_status = "inverted" if self.invert_camera else "normal"
                    print(f"✓ CSI Camera (left) initialized with GStreamer (640x480 @ {VIDEO_FPS} FPS, {flip_status})")

                    # Initialize second camera for stereo
                    if self.stereo_mode:
                        pipeline2 = self.gstreamer_pipeline(
                            sensor_id=1,
                            capture_width=1280,
                            capture_height=720,
                            display_width=640,
                            display_height=480,
                            framerate=VIDEO_FPS,
                            flip_method=flip_method
                        )
                        self.camera2 = cv2.VideoCapture(pipeline2, cv2.CAP_GSTREAMER)
                        print(f"✓ CSI Camera (right) initialized with GStreamer (640x480 @ {VIDEO_FPS} FPS, {flip_status})")
            else:
                # Use regular USB camera
                self.camera = cv2.VideoCapture(self.camera_id)

                # Set format to MJPEG if available (better compatibility)
                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
                flip_status = "inverted" if self.invert_camera else "normal"
                print(f"✓ USB Camera (left) {self.camera_id} initialized (640x480 @ {VIDEO_FPS} FPS, {flip_status})")

                # Initialize second USB camera for stereo
                if self.stereo_mode:
                    self.camera2 = cv2.VideoCapture(self.camera_id + 1)
                    self.camera2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera2.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
                    print(f"✓ USB Camera (right) {self.camera_id + 1} initialized (640x480 @ {VIDEO_FPS} FPS, {flip_status})")

            if not self.camera.isOpened():
                raise Exception("Failed to open left camera")

            if self.stereo_mode and not self.camera2.isOpened():
                raise Exception("Failed to open right camera")

            self.camera_active = True

        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.camera_active = False
            self.camera = None
            self.camera2 = None

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
            with self.motor_lock:
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

        previous_alarm_handler = signal.getsignal(signal.SIGALRM)

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
            except TimeoutError:
                print(f"Connection timed out after 5 seconds on {self.port}")
                raise
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_alarm_handler)

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
            with self.motor_lock:
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

    def draw_overlays(self, frame, frame_right=None):
        """Draw crosshair, bounding box, and center point on image (OpenCV)"""
        # Draw target crosshair (dynamic position based on yaw and pitch)
        ch_x = get_target_crosshair_x(self.current_yaw)

        # Get focal length from calibration if available
        # Don't use geometric calculation yet - use empirical linear interpolation
        focal_length_y = None  # Set to None to force fallback to linear interpolation
        # if self.calibration_enabled and self.camera_matrix is not None:
        #     focal_length_y = self.camera_matrix[1, 1]  # fy from camera matrix

        ch_y = get_target_crosshair_y(self.current_pitch,
                                       camera_bore_offset_mm=82,
                                       focal_length_px=focal_length_y,
                                       assumed_distance_mm=5000)
        ch_size = CROSSHAIR_SIZE

        # Debug: print servo positions and crosshair (only every 30 frames to avoid spam)
        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0
        self._frame_counter += 1

        # Calculate depth at crosshair if stereo is available
        crosshair_depth_m = None
        depth_adjust_px = 0.0
        depth_text = None
        depth_color = (0, 255, 0)
        if self.stereo_calibration_enabled and frame_right is not None:
            depth_mm = self.calculate_depth(frame, frame_right, ch_x, ch_y)
            if depth_mm is not None:
                crosshair_depth_m = depth_mm / 1000.0  # Convert to meters
                focal_y = None
                if self.K1 is not None:
                    focal_y = float(self.K1[1, 1])
                elif self.camera_matrix is not None:
                    focal_y = float(self.camera_matrix[1, 1])

                depth_adjust = AIMING.depth_adjust_px(depth_mm, focal_y)
                if depth_adjust is not None:
                    depth_adjust_px = depth_adjust
                    ch_y = int(round(ch_y + depth_adjust_px))
                depth_text = f"{crosshair_depth_m:.2f}m"
            else:
                depth_text = f"-- {self.last_depth_debug}"
                depth_color = (0, 200, 255)
        elif self.stereo_mode:
            depth_text = "-- no right frame/calibration"
            depth_color = (0, 200, 255)

        if self._frame_counter % 30 == 0:  # Print once per second
            depth_debug = depth_text if depth_text is not None else "n/a"
            print(f"[DEBUG] Yaw: {self.current_yaw}, Pitch: {self.current_pitch}, Crosshair: ({ch_x}, {ch_y}), Depth: {depth_debug}, DepthYAdjust: {depth_adjust_px:.1f}px")

        # Crosshair lines in green (BGR format)
        cv2.line(frame, (ch_x - ch_size, ch_y), (ch_x + ch_size, ch_y), (0, 255, 0), 2)
        cv2.line(frame, (ch_x, ch_y - ch_size), (ch_x, ch_y + ch_size), (0, 255, 0), 2)
        # Crosshair circle
        cv2.circle(frame, (ch_x, ch_y), 5, (0, 255, 0), 2)

        # Display distance/status at crosshair when stereo mode is active
        if depth_text is not None:
            # Draw text with black background for visibility
            text_size = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = ch_x + ch_size + 10
            text_y = ch_y
            # Background rectangle
            cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 5), (0, 0, 0), -1)
            # Text in green
            cv2.putText(frame, depth_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, depth_color, 2)

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
        max_integral = self.pid_max_integral  # degrees
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
        yaw_correction_raw = self._minimum_tracking_step(yaw_correction_raw, pixel_error_x)
        pitch_correction_raw = self._minimum_tracking_step(pitch_correction_raw, pixel_error_y)

        # Calculate desired servo positions
        desired_yaw = self.current_yaw + yaw_correction_raw
        desired_pitch = self.current_pitch + pitch_correction_raw

        # Clamp to valid servo ranges
        desired_yaw = max(YAW_MIN, min(YAW_MAX, desired_yaw))
        desired_pitch = max(PITCH_MIN, min(PITCH_MAX, desired_pitch))
        desired_yaw = self._limit_tracking_step(self.current_yaw, desired_yaw, self.max_yaw_step)
        desired_pitch = self._limit_tracking_step(self.current_pitch, desired_pitch, self.max_pitch_step)

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
            # Update servo positions for dynamic crosshair (reads actual position from servos)
            if self.connected and self.motor_bus:
                try:
                    yaw_pos, pitch_pos = self.read_motor_positions()
                    # Only update if we got valid readings
                    if yaw_pos is not None and pitch_pos is not None:
                        self.current_yaw = yaw_pos
                        self.current_pitch = pitch_pos
                except Exception as e:
                    # Silently fail on read errors (don't spam console at 30 FPS)
                    pass

            # Capture frame from left camera
            ret, frame = self.camera.read()
            if not ret:
                return

            # Capture frame from right camera if stereo mode
            frame_right = None
            if self.stereo_mode and self.camera2:
                ret2, frame_right = self.camera2.read()
                if not ret2:
                    frame_right = None

            # Rotate 180 degrees if invert_camera is enabled and not using CSI with flip_method
            # (CSI camera handles rotation in GStreamer pipeline)
            if self.invert_camera and not self.use_csi:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                if frame_right is not None:
                    frame_right = cv2.rotate(frame_right, cv2.ROTATE_180)

            # Resize to 640x480 if needed (camera may not respect resolution settings)
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))
            if frame_right is not None and (frame_right.shape[1] != 640 or frame_right.shape[0] != 480):
                frame_right = cv2.resize(frame_right, (640, 480))

            # Keep the displayed frame unrectified; rectification is only for depth math.
            if self.calibration_enabled and not self.stereo_calibration_enabled:
                frame = self.undistort_frame(frame, use_left=True)

            # Draw overlays (pass right frame for depth calculation)
            frame = self.draw_overlays(frame, frame_right)

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
        """Run one inference pass using the shared inference module"""
        if not self.camera_active or not self.camera or not self.model:
            return

        try:
            # Capture frame from left camera
            ret, frame = self.camera.read()
            if not ret:
                return

            # Capture frame from right camera if stereo mode
            frame_right = None
            if self.stereo_mode and self.camera2:
                ret2, frame_right = self.camera2.read()
                if not ret2:
                    frame_right = None

            # Rotate 180 degrees if invert_camera is enabled and not using CSI with flip_method
            # (CSI camera handles rotation in GStreamer pipeline)
            if self.invert_camera and not self.use_csi:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                if frame_right is not None:
                    frame_right = cv2.rotate(frame_right, cv2.ROTATE_180)

            # Resize to 640x480 if needed (camera may not respect resolution settings)
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))
            if frame_right is not None and (frame_right.shape[1] != 640 or frame_right.shape[0] != 480):
                frame_right = cv2.resize(frame_right, (640, 480))

            # Keep inference on the normal camera image; depth rectifies hidden copies only.
            if self.calibration_enabled and not self.stereo_calibration_enabled:
                frame = self.undistort_frame(frame, use_left=True)

            # Run inference using shared inference module (YOLO works directly with numpy arrays)
            results = yolo_run_inference(
                self.model,
                frame,
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                verbose=False
            )

            # Extract detections using shared utility and keep configured target classes.
            if self.target_classes:
                detections = []
                for target_class in self.target_classes:
                    detections.extend(extract_detections(results, self.model, target_class=target_class))
            else:
                detections = extract_detections(results, self.model)
            detections.sort(key=lambda det: det["confidence"], reverse=True)

            # Process first detection (if any)
            detection = False
            confidence = 0
            bbox = None
            center_point = None

            # Calculate depth if stereo enabled
            depth_mm = None
            if detections:
                # Use the first (highest confidence) detection
                det = detections[0]
                detection = True
                confidence = det['confidence']
                class_name = det.get('class_name', 'object')
                bbox = det['bbox']
                center_point = det['center']
                center_x, center_y = center_point
                tracking_center_x, tracking_center_y = self.smooth_tracking_center(center_x, center_y)
                if (tracking_center_x, tracking_center_y) != (center_x, center_y):
                    print(f"   Tracking center: raw=({center_x}, {center_y}), smoothed=({tracking_center_x}, {tracking_center_y})")

                # Calculate depth if stereo calibration is available
                if self.stereo_calibration_enabled and frame_right is not None:
                    depth_mm = self.calculate_depth(frame, frame_right, center_x, center_y)
                    if depth_mm is not None:
                        depth_m = depth_mm / 1000.0  # Convert to meters
                        print(f"   Depth: {depth_m:.2f}m ({depth_mm:.1f}mm)")

                # Save detection image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                detection_filename = f"detection_{timestamp}.jpg"
                detection_path = os.path.join(self.detections_dir, detection_filename)
                cv2.imwrite(detection_path, frame)

                self.detection_count += 1
                # Format: "time - message | filename" so the UI can parse and link it
                depth_str = f" @ {depth_mm/1000.0:.2f}m" if depth_mm else ""
                detection_msg = f"{datetime.now().strftime('%H:%M:%S')} - {class_name} detected (conf: {confidence:.3f}) at ({center_x}, {center_y}){depth_str} | {detection_filename}"
                self.recent_detections.append(detection_msg)
                self.recent_detections = self.recent_detections[-10:]

                print(f"🎯 Detection #{self.detection_count}: {detection_path}")
                print(f"   Class: {class_name}, Center: ({center_x}, {center_y}), Confidence: {confidence:.3f}")

                # Get updated servo positions from observe function
                desired_yaw, desired_pitch = self.observe(tracking_center_x, tracking_center_y)
                print(f"   Servo update: Yaw {self.current_yaw} -> {desired_yaw}, Pitch {self.current_pitch} -> {desired_pitch}")

                # Update servo positions
                self.set_yaw(desired_yaw)
                self.set_pitch(desired_pitch)

                # Auto-trigger disabled - only trigger manually via button
                # if detection and not self.latest_detection and self.trigger_servo_enabled:
                #     threading.Thread(target=self.trigger_action_servo, daemon=True).start()
            else:
                self.smoothed_tracking_center = None

            # Update detection state
            with self.inference_lock:
                self.latest_detection = detection
                self.latest_confidence = confidence
                self.latest_bbox = bbox
                self.latest_center_point = center_point
                self.latest_depth = depth_mm

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
        """Inference processing thread"""
        while self.camera_active:
            loop_start = time.time()
            self.run_inference()
            self.inference_loop_count += 1
            window_elapsed = time.time() - self.inference_fps_window_start
            if window_elapsed >= 5.0:
                actual_fps = self.inference_loop_count / window_elapsed
                print(f"Inference actual FPS: {actual_fps:.1f} (target {self.inference_fps:g})", flush=True)
                self.inference_loop_count = 0
                self.inference_fps_window_start = time.time()
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, (1.0 / self.inference_fps) - elapsed))

    def start_camera_thread(self):
        """Start the camera processing thread"""
        thread = threading.Thread(target=self.camera_thread, daemon=True)
        thread.start()
        print(f"Camera thread started ({VIDEO_FPS} FPS)")

    def start_inference_thread(self):
        """Start the inference processing thread"""
        thread = threading.Thread(target=self.inference_thread, daemon=True)
        thread.start()
        print(f"Inference thread started ({self.inference_fps:g} FPS target)")

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
    detection_config = CONFIG.get('detection', {}) if CONFIG else {}
    tracking_config = CONFIG.get('tracking', {}) if CONFIG else {}
    auto_tracking_config = tracking_config.get('auto_tracking', {})

    parser = argparse.ArgumentParser(description="Camera tracker with servo control and rat detection")

    # Servo settings
    parser.add_argument("--port", "-p", type=str, default="/dev/ttyACM0",
                       help="Serial port for servo connection")
    parser.add_argument("--disable-servos", action="store_true",
                       help="Disable tracking servo control (simulation mode)")
    parser.add_argument("--no-connect", action="store_true",
                       help="Skip servo connection attempt (web interface only)")

    # Camera and detection settings
    parser.add_argument("--video-only", action="store_true",
                       help="Convenience mode: CSI camera web stream, no servos, no detection")
    parser.add_argument("--enable-camera", action="store_true",
                       help="Enable camera")
    parser.add_argument("--camera-id", type=int, default=0,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--use-csi", action="store_true",
                       help="Use CSI camera with GStreamer pipeline (Jetson)")
    parser.add_argument("--invert-camera", action="store_true",
                       help="Invert camera 180 degrees for upside-down mounting")
    parser.add_argument("--model", "-m", type=str, default=detection_config.get("model_path", "runs/yolo11n-2025-10-24/weights/best.pt"),
                       help="Path to YOLO model")
    parser.add_argument("--disable-detection", action="store_true",
                       help="Disable YOLO detection while keeping the camera stream enabled")
    parser.add_argument("--confidence", "-c", type=float, default=float(detection_config.get("confidence_threshold", 0.75)),
                       help="Detection confidence threshold")
    parser.add_argument("--target-class", action="append", default=None,
                       help="YOLO class name to track. Repeat or comma-separate values. Default: all model classes")
    parser.add_argument("--imgsz", type=int, default=int(detection_config.get("imgsz", 640)),
                       help="Inference image size in pixels (default: 640)")
    parser.add_argument("--inference-fps", type=float, default=None,
                       help=f"Target inference loop FPS (default: {INFERENCE_FPS})")
    parser.add_argument("--tracking-smoothing", type=float, default=float(auto_tracking_config.get("smoothing_alpha", 0.45)),
                       help="Detection center smoothing alpha for auto tracking: 0 disables, 1 follows raw detections (default: 0.45)")
    parser.add_argument("--max-yaw-step", type=int, default=int(auto_tracking_config.get("max_yaw_step", 45)),
                       help="Maximum yaw raw-unit move per inference update for smoother tracking (default: 45)")
    parser.add_argument("--max-pitch-step", type=int, default=int(auto_tracking_config.get("max_pitch_step", 28)),
                       help="Maximum pitch raw-unit move per inference update for smoother tracking (default: 28)")
    parser.add_argument("--calibration", type=str, default="camera_calibration.npz",
                       help="Path to camera calibration file (.npz, default: camera_calibration.npz)")
    parser.add_argument("--stereo", action="store_true",
                       help="Enable stereo mode for depth estimation (requires stereo calibration)")
    parser.add_argument("--baseline-override", type=float, default=None,
                       help="Override stereo baseline in mm (use if calibration baseline is incorrect)")

    # Trigger servo settings
    parser.add_argument("--enable-trigger", action="store_true",
                       help="Enable GPIO trigger servo")

    # API settings
    parser.add_argument("--api-host", type=str, default="0.0.0.0",
                       help="Host for FastAPI server")
    parser.add_argument("--api-port", type=int, default=8000,
                       help="Port for FastAPI server")

    args = parser.parse_args()

    if args.target_class is None:
        configured_targets = detection_config.get("target_classes")
        if isinstance(configured_targets, str):
            args.target_class = [configured_targets]
        elif configured_targets:
            args.target_class = list(configured_targets)

    if args.video_only:
        args.enable_camera = True
        args.use_csi = True
        args.disable_servos = True
        args.no_connect = True
        args.disable_detection = True

    if args.stereo and args.calibration == "camera_calibration.npz":
        default_stereo_calibration = "tools/vision/calibration/output_recal/stereo_calibration.npz"
        if os.path.exists(default_stereo_calibration):
            args.calibration = default_stereo_calibration

    model_path = None
    if args.enable_camera and not args.disable_detection:
        model_path = args.model

    # Create tracker and hand it to the web controller
    tracker = CameraTracker(
        port=args.port,
        enable_servos=not args.disable_servos,
        no_connect=args.no_connect,
        enable_camera=args.enable_camera,
        enable_trigger=args.enable_trigger,
        model_path=model_path,
        confidence_threshold=args.confidence,
        camera_id=args.camera_id,
        use_csi=args.use_csi,
        invert_camera=args.invert_camera,
        imgsz=args.imgsz,
        inference_fps=args.inference_fps,
        target_classes=args.target_class,
        calibration_file=args.calibration,
        stereo_mode=args.stereo,
        baseline_override=args.baseline_override,
        tracking_smoothing=args.tracking_smoothing,
        max_yaw_step=args.max_yaw_step,
        max_pitch_step=args.max_pitch_step
    )
    control_api.set_tracker(tracker)

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
    else:
        print("Tip: use --video-only for Jetson CSI camera web streaming without servos or detection")
    print(f"Detection: {'ENABLED' if model_path else 'DISABLED'}")
    if model_path:
        print(f"Model: {args.model}")
        target_classes = ", ".join(tracker.target_classes) if tracker.target_classes else "all"
        print(f"Target classes: {target_classes}")
        print(f"Confidence threshold: {args.confidence}")
        print(f"Inference image size: {args.imgsz}px")
        print(f"Inference FPS target: {tracker.inference_fps:g}")
        print(f"Tracking smoothing: alpha={tracker.tracking_smoothing:g}, max yaw/pitch step={tracker.max_yaw_step}/{tracker.max_pitch_step}")
    calib_status = "ENABLED" if tracker.calibration_enabled else "DISABLED"
    if tracker.calibration_enabled:
        calib_type = "STEREO" if tracker.stereo_calibration_enabled else "SINGLE"
        print(f"Calibration: {calib_status} ({calib_type}, {args.calibration})")
    elif args.calibration != "camera_calibration.npz":
        # Only show warning if user explicitly specified a file
        print(f"Calibration: {calib_status} (file not found)")

    if args.stereo:
        stereo_status = "ENABLED" if tracker.stereo_calibration_enabled else "DISABLED (no stereo calibration)"
        print(f"Stereo depth: {stereo_status}")
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
