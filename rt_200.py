#!/usr/bin/env python3
"""
Real-time camera tracking with Feetech servo control and rat detection
Controls yaw (ID 1, 1600-3100 raw) and pitch (ID 5, 1-500 raw) servos
Includes Raspberry Pi camera streaming, YOLO inference, and GPIO trigger servo
"""

import os
import time
import threading
import io
import shutil
import yaml
from datetime import datetime
import uvicorn
import numpy as np
from ratbot.vision.yolo_inference import run_inference as yolo_run_inference, extract_detections
from ratbot.vision.camera_source import CameraSource
from ratbot.vision.overlay import OverlayRenderer
from ratbot.vision.stereo_depth import StereoDepthService
from ratbot.runtime_config import parse_runtime_config
from ratbot.robot import (
    AngularBeliefController,
    AngularTargetBelief,
    CrosshairAiming,
    DepthCrosshairCompensation,
    ObservationConfig,
    PitchCompensation,
    ServoBounds,
    TrackingObservationConverter,
    TrackingServoController,
    TriggerServoConfig,
    TriggerServoController,
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
CSICameraCapture = None
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
FeetechMotorsBus = None
Motor = None
MotorNormMode = None
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
    PITCH_MIN = 1
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
                 tracking_smoothing=0.45, max_yaw_step=45, max_pitch_step=45,
                 tracking_control_fps=20, belief_update_alpha=0.45,
                 belief_miss_decay=0.94, belief_min_confidence=0.15,
                 belief_max_age=1.5, belief_deadband_raw=4,
                 belief_min_step_raw=3, pitch_tracking_scale=1.0,
                 belief_reseed_distance_raw=160, belief_velocity_alpha=0.45,
                 belief_velocity_decay=0.96, belief_max_velocity_raw_per_s=600,
                 belief_max_prediction_age=0.45, belief_reseed_confirmations=2,
                 belief_reseed_match_distance_raw=120,
                 belief_reseed_max_interval=0.8, belief_pitch_update_alpha=None,
                 belief_pitch_velocity_alpha=None,
                 belief_max_pitch_velocity_raw_per_s=None):
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
            tracking_smoothing: Deprecated compatibility parameter; belief smoothing uses belief_update_alpha
            max_yaw_step: Maximum yaw raw-unit move per inference update
            max_pitch_step: Maximum pitch raw-unit move per control update
            tracking_control_fps: Servo control loop rate for angular target belief tracking
            belief_update_alpha: Weight of new observations when updating target belief
            belief_miss_decay: Confidence decay applied on inference ticks without detections
            belief_min_confidence: Minimum belief confidence required for control loop movement
            belief_max_age: Maximum age in seconds for target belief movement
            belief_deadband_raw: Raw servo-unit error treated as centered
            belief_min_step_raw: Minimum raw servo-unit correction while outside deadband
            pitch_tracking_scale: Multiplier for vertical image error to pitch raw observation
            belief_reseed_distance_raw: Observation jump that starts a fresh belief
            belief_velocity_alpha: Smoothing alpha for angular belief velocity
            belief_velocity_decay: Velocity decay applied on inference ticks without detections
            belief_max_velocity_raw_per_s: Maximum predicted target speed in raw units/sec
            belief_max_prediction_age: Maximum seconds to extrapolate after the latest detection
            belief_reseed_confirmations: Matching far-jump detections required before reseed
            belief_reseed_match_distance_raw: Raw-unit distance for matching pending reseeds
            belief_reseed_max_interval: Maximum seconds between matching pending reseeds
            belief_pitch_update_alpha: Optional pitch-only belief update alpha
            belief_pitch_velocity_alpha: Optional pitch-only velocity update alpha
            belief_max_pitch_velocity_raw_per_s: Optional pitch-only velocity cap
        """
        self.port = port
        self.enable_servos = enable_servos
        self.no_connect = no_connect

        # Camera and detection
        self.camera_source = CameraSource(
            enabled=enable_camera and CV2_AVAILABLE,
            camera_id=camera_id,
            use_csi=use_csi,
            stereo_mode=stereo_mode,
            invert_camera=invert_camera,
            video_fps=VIDEO_FPS,
            csi_capture_factory=(
                CSICameraCapture if CSI_HELPER_AVAILABLE else None
            ),
        )
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.inference_fps = max(1.0, float(inference_fps or INFERENCE_FPS))
        self.target_classes = self._normalize_target_classes(target_classes)
        self.max_yaw_step = max(0, int(max_yaw_step))
        self.max_pitch_step = max(0, int(max_pitch_step))
        self.tracking_control_fps = max(1.0, float(tracking_control_fps))
        self.belief_update_alpha = max(0.0, min(1.0, float(belief_update_alpha)))
        self.belief_miss_decay = max(0.0, min(1.0, float(belief_miss_decay)))
        self.belief_min_confidence = max(0.0, min(1.0, float(belief_min_confidence)))
        self.belief_max_age = max(0.0, float(belief_max_age))
        self.belief_deadband_raw = max(0, int(belief_deadband_raw))
        self.belief_min_step_raw = max(0, int(belief_min_step_raw))
        self.pitch_tracking_scale = max(0.1, float(pitch_tracking_scale))
        self.belief_reseed_distance_raw = max(0.0, float(belief_reseed_distance_raw))
        self.belief_velocity_alpha = max(0.0, min(1.0, float(belief_velocity_alpha)))
        self.belief_velocity_decay = max(0.0, min(1.0, float(belief_velocity_decay)))
        self.belief_max_velocity_raw_per_s = max(0.0, float(belief_max_velocity_raw_per_s))
        self.belief_max_prediction_age = max(0.0, float(belief_max_prediction_age))
        self.belief_reseed_confirmations = max(1, int(belief_reseed_confirmations))
        self.belief_reseed_match_distance_raw = max(0.0, float(belief_reseed_match_distance_raw))
        self.belief_reseed_max_interval = max(0.0, float(belief_reseed_max_interval))
        self.belief_pitch_update_alpha = (
            self.belief_update_alpha if belief_pitch_update_alpha is None
            else max(0.0, min(1.0, float(belief_pitch_update_alpha)))
        )
        self.belief_pitch_velocity_alpha = (
            self.belief_velocity_alpha if belief_pitch_velocity_alpha is None
            else max(0.0, min(1.0, float(belief_pitch_velocity_alpha)))
        )
        self.belief_max_pitch_velocity_raw_per_s = (
            self.belief_max_velocity_raw_per_s if belief_max_pitch_velocity_raw_per_s is None
            else max(0.0, float(belief_max_pitch_velocity_raw_per_s))
        )
        self.detection_count = 0
        self.latest_frame = None
        self.latest_detection = False
        self.latest_bbox = None  # Store latest bounding box (x1, y1, x2, y2)
        self.latest_center_point = None  # Store latest center point (x, y)
        self.latest_depth = None  # Store depth in mm at detection point
        self.recent_detections = []
        self.frame_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.inference_loop_count = 0
        self.inference_fps_window_start = time.time()

        tracking_config = CONFIG.get('tracking', {}) if CONFIG else {}
        self.stereo_depth = StereoDepthService(
            calibration_file=calibration_file,
            baseline_override=baseline_override,
            min_texture_std=tracking_config.get('depth_min_texture_std', 4.0),
            max_valid_mm=tracking_config.get('depth_max_valid_mm', 6000.0),
        )
        self.overlay_renderer = OverlayRenderer(
            stereo_depth=self.stereo_depth,
            aiming=AIMING,
            crosshair_x=get_target_crosshair_x,
            crosshair_y=get_target_crosshair_y,
            crosshair_size=CROSSHAIR_SIZE,
            depth_adjust_smoothing_alpha=tracking_config.get(
                'depth_adjust_smoothing_alpha', 0.20
            ),
            depth_adjust_missing_decay=tracking_config.get(
                'depth_adjust_missing_decay', 0.85
            ),
        )
        self.motor_readback_fps = max(0.0, float(tracking_config.get('motor_readback_fps', 10)))
        self.motor_readback_interval = 1.0 / self.motor_readback_fps if self.motor_readback_fps > 0 else 0.0
        self.last_motor_readback_time = 0.0
        self.stereo_depth.last_depth_debug = "not computed"

        servo_bounds = ServoBounds(YAW_MIN, YAW_MAX, PITCH_MIN, PITCH_MAX)
        self.tracking_servos = TrackingServoController(
            port=port,
            enabled=enable_servos,
            bounds=servo_bounds,
            yaw_center=YAW_CENTER,
            pitch_center=PITCH_CENTER,
            yaw_motor_id=YAW_MOTOR_ID,
            pitch_motor_id=PITCH_MOTOR_ID,
            bus_factory=FeetechMotorsBus if FEETECH_AVAILABLE else None,
            motor_factory=Motor if FEETECH_AVAILABLE else None,
            norm_mode=getattr(MotorNormMode, "RANGE_M100_100", None),
        )
        self.trigger_servo = TriggerServoController(
            enabled=enable_trigger,
            config=TriggerServoConfig(
                pwm_chip=TRIGGER_PWM_CHIP,
                pwm_channel=TRIGGER_PWM_CHANNEL,
                neutral_angle=TRIGGER_NEUTRAL_ANGLE,
                action_angle=TRIGGER_ACTION_ANGLE,
            ),
        )

        self.observation_converter = TrackingObservationConverter(
            config=ObservationConfig(
                image_width=640,
                image_height=480,
                horizontal_fov_degrees=60.0,
                vertical_fov_degrees=45.0,
                yaw_min=YAW_MIN,
                yaw_max=YAW_MAX,
                pitch_min=PITCH_MIN,
                pitch_max=PITCH_MAX,
                yaw_range_degrees=180.0,
                pitch_range_degrees=55.0,
                pitch_tracking_scale=self.pitch_tracking_scale,
            ),
            aiming=AIMING,
            stereo_depth=self.stereo_depth,
            crosshair_x=get_target_crosshair_x,
            crosshair_y=get_target_crosshair_y,
        )

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

        self.target_belief = AngularTargetBelief(
            update_alpha=belief_update_alpha,
            miss_decay=belief_miss_decay,
            min_confidence=belief_min_confidence,
            max_age=belief_max_age,
            reseed_distance_raw=belief_reseed_distance_raw,
            velocity_alpha=belief_velocity_alpha,
            velocity_decay=belief_velocity_decay,
            max_velocity_raw_per_s=belief_max_velocity_raw_per_s,
            max_prediction_age=belief_max_prediction_age,
            reseed_confirmations=belief_reseed_confirmations,
            reseed_match_distance_raw=belief_reseed_match_distance_raw,
            reseed_max_interval=belief_reseed_max_interval,
            pitch_update_alpha=self.belief_pitch_update_alpha,
            pitch_velocity_alpha=self.belief_pitch_velocity_alpha,
            max_pitch_velocity_raw_per_s=self.belief_max_pitch_velocity_raw_per_s,
        )
        self.tracking_controller = AngularBeliefController(
            robot=self,
            belief=self.target_belief,
            bounds=servo_bounds,
            control_fps=tracking_control_fps,
            max_yaw_step=self.max_yaw_step,
            max_pitch_step=self.max_pitch_step,
            deadband_raw=belief_deadband_raw,
            min_step_raw=belief_min_step_raw
        )

        # Create detections directory
        self.detections_dir = DETECTIONS_DIR
        os.makedirs(self.detections_dir, exist_ok=True)

        # Initialize tracking servos if enabled
        if self.enable_servos and not self.no_connect and FEETECH_AVAILABLE:
            self.tracking_servos.connect()

        # Initialize trigger servo if enabled
        if self.trigger_servo_enabled:
            self.trigger_servo.initialize()

        # Initialize camera if enabled
        if self.enable_camera:
            self.camera_source.initialize()

        # Initialize YOLO model if path provided
        if self.model_path and YOLO_AVAILABLE:
            self.init_model()

        # Start camera and inference threads if everything is ready
        if self.camera_active:
            self.start_camera_thread()
            if self.model:
                self.start_inference_thread()
                self.start_tracking_control_thread()

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

    @property
    def connected(self):
        return self.tracking_servos.connected

    @connected.setter
    def connected(self, connected):
        self.tracking_servos.connected = bool(connected)

    @property
    def current_yaw(self):
        return self.tracking_servos.current_yaw

    @current_yaw.setter
    def current_yaw(self, position):
        self.tracking_servos.current_yaw = int(position)

    @property
    def current_pitch(self):
        return self.tracking_servos.current_pitch

    @current_pitch.setter
    def current_pitch(self, position):
        self.tracking_servos.current_pitch = int(position)

    @property
    def calibration_enabled(self):
        """Compatibility view of the extracted calibration service state."""
        return self.stereo_depth.calibration_enabled

    @property
    def stereo_calibration_enabled(self):
        """Compatibility view of the extracted stereo calibration state."""
        return self.stereo_depth.stereo_calibration_enabled

    @property
    def trigger_servo_enabled(self):
        return self.trigger_servo.enabled

    @trigger_servo_enabled.setter
    def trigger_servo_enabled(self, enabled):
        self.trigger_servo.enabled = bool(enabled)

    def read_motor_positions(self):
        return self.tracking_servos.read_positions()

    def set_yaw(self, position):
        self.tracking_servos.set_yaw(position)

    def set_pitch(self, position):
        self.tracking_servos.set_pitch(position)

    def trigger_action_servo(self):
        self.trigger_servo.fire()

    @property
    def enable_camera(self):
        return self.camera_source.enabled

    @property
    def camera_active(self):
        return self.camera_source.active

    @camera_active.setter
    def camera_active(self, active):
        self.camera_source.active = bool(active)

    @property
    def camera(self):
        return self.camera_source.left

    @property
    def camera2(self):
        return self.camera_source.right

    @property
    def camera_id(self):
        return self.camera_source.camera_id

    @property
    def use_csi(self):
        return self.camera_source.use_csi

    @property
    def stereo_mode(self):
        return self.camera_source.stereo_mode

    @property
    def invert_camera(self):
        return self.camera_source.invert_camera

    def draw_overlays(self, frame, frame_right=None):
        """Render a consistent snapshot of detection and aiming state."""
        with self.inference_lock:
            bbox = self.latest_bbox
            center_point = self.latest_center_point

        return self.overlay_renderer.render(
            frame=frame,
            frame_right=frame_right,
            current_yaw=self.current_yaw,
            current_pitch=self.current_pitch,
            stereo_mode=self.stereo_mode,
            bbox=bbox,
            center_point=center_point,
        )

    def angle_to_servo_raw(self, angle_delta, axis='yaw'):
        return self.observation_converter.angle_to_servo_raw(angle_delta, axis)

    def servo_raw_to_angle(self, raw_delta, axis='yaw'):
        return self.observation_converter.servo_raw_to_angle(raw_delta, axis)

    def pixel_to_target_position(self, target_x, target_y, depth_mm=None):
        return self.observation_converter.to_servo_target(
            target_x=target_x,
            target_y=target_y,
            current_yaw=self.current_yaw,
            current_pitch=self.current_pitch,
            depth_mm=depth_mm,
        )












    def init_model(self):
        """Initialize YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✓ YOLO model loaded: {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None













    def update_target_belief(self, center_x, center_y, confidence, depth_mm=None):
        """Update angular target belief from one detection observation."""
        observation = self.pixel_to_target_position(center_x, center_y, depth_mm=depth_mm)
        belief = self.target_belief.update(observation["yaw"], observation["pitch"], confidence)

        print(
            "   Target belief: "
            f"obs=({observation['yaw']}, {observation['pitch']}), "
            f"belief=({belief['yaw']:.1f}, {belief['pitch']:.1f}), "
            f"conf={belief['confidence']:.2f}, "
            f"vel=({belief['yaw_velocity']:.0f}, {belief['pitch_velocity']:.0f}) raw/s, "
            f"pixel_error=({observation['pixel_error_x']:.1f}, {observation['pixel_error_y']:.1f})"
            f"{' reseed=' + belief['reseed_reason'] if belief.get('reseeded') else ''}"
            f"{' ignored=' + belief['ignored_reason'] if belief.get('ignored') else ''}"
        )

    def decay_target_belief(self):
        """Decay angular target belief confidence after an inference tick without detections."""
        self.target_belief.decay()

    def clear_target_belief(self):
        """Clear target belief and reset tracking controller state."""
        self.target_belief.clear()
        self.tracking_controller.reset()
        print("   Target belief cleared")
        return True

    def get_active_target_belief(self):
        """Return active angular target belief if it is fresh and confident enough."""
        return self.target_belief.get_active()


    def move_to_pixel(self, target_x, target_y):
        """
        Directly move servos to point the crosshair at a target pixel position.
        This is a direct positioning command, not the belief control loop.

        Args:
            target_x: X coordinate of target position in pixels
            target_y: Y coordinate of target position in pixels

        Returns:
            tuple: (desired_yaw, desired_pitch) servo positions in raw units
        """
        observation = self.pixel_to_target_position(target_x, target_y)
        desired_yaw = observation["yaw"]
        desired_pitch = observation["pitch"]

        print(f"   Direct positioning:")
        print(f"     Target pixel: ({target_x}, {target_y})")
        print(f"     Pixel offset: X={observation['pixel_error_x']:.1f}px, Y={observation['pixel_error_y']:.1f}px")
        print(f"     Angle offset: Yaw={observation['angle_error_yaw']:.2f}°, Pitch={observation['angle_error_pitch']:.2f}°")
        print(f"     Servo move: Yaw {self.current_yaw} → {desired_yaw} ({observation['yaw_offset_raw']:+d}), Pitch {self.current_pitch} → {desired_pitch} ({observation['pitch_offset_raw']:+d})")

        return desired_yaw, desired_pitch


    def track_target_belief(self):
        """Move servos toward the current angular target belief."""
        self.tracking_controller.track_once()

    def capture_video_frame(self):
        """Capture a video frame at 30 FPS with overlays"""
        if not self.camera_active or not self.camera:
            return

        try:
            # Update servo positions for dynamic crosshair (reads actual position from servos)
            readback_due = (
                self.motor_readback_interval <= 0
                or time.time() - self.last_motor_readback_time >= self.motor_readback_interval
            )
            if self.connected and self.tracking_servos.motor_bus and readback_due:
                try:
                    self.last_motor_readback_time = time.time()
                    yaw_pos, pitch_pos = self.read_motor_positions()
                    # Only update if we got valid readings
                    if yaw_pos is not None and pitch_pos is not None:
                        self.current_yaw = yaw_pos
                        self.current_pitch = pitch_pos
                except Exception as e:
                    # Silently fail on read errors (don't spam console at 30 FPS)
                    pass

            frame, frame_right = self.camera_source.read_frames()
            if frame is None:
                return

            # Keep the displayed frame unrectified; rectification is only for depth math.
            if self.stereo_depth.calibration_enabled and not self.stereo_depth.stereo_calibration_enabled:
                frame = self.stereo_depth.undistort_frame(frame, use_left=True)

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
            frame, frame_right = self.camera_source.read_frames()
            if frame is None:
                return

            # Keep inference on the normal camera image; depth rectifies hidden copies only.
            if self.stereo_depth.calibration_enabled and not self.stereo_depth.stereo_calibration_enabled:
                frame = self.stereo_depth.undistort_frame(frame, use_left=True)

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

                # Calculate depth if stereo calibration is available
                if self.stereo_depth.stereo_calibration_enabled and frame_right is not None:
                    depth_mm = self.stereo_depth.calculate_depth(frame, frame_right, center_x, center_y)
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

                # Detection updates angular belief; the control loop moves servos toward it.
                self.update_target_belief(center_x, center_y, confidence, depth_mm=depth_mm)

                # Auto-trigger disabled - only trigger manually via button
                # if detection and not self.latest_detection and self.trigger_servo_enabled:
                #     threading.Thread(target=self.trigger_action_servo, daemon=True).start()
            else:
                self.decay_target_belief()

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

    def tracking_control_thread(self):
        """Servo control thread that continuously moves toward angular target belief."""
        self.tracking_controller.run()

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

    def start_tracking_control_thread(self):
        """Start the angular target belief servo control thread"""
        self.tracking_controller.start()

    def disconnect(self):
        """Disconnect and cleanup"""
        self.camera_source.close()

        self.trigger_servo.close()
        self.tracking_servos.close()
        self.camera_active = False

def run_api_server(host="0.0.0.0", port=8000):
    """Run the FastAPI server in a separate thread"""
    uvicorn.run(app, host=host, port=port, log_level="error")

def main():
    runtime = parse_runtime_config(
        argv=None,
        raw_config=CONFIG,
        inference_fps_default=INFERENCE_FPS,
        path_exists=os.path.exists,
    )
    settings = runtime.tracker

    tracker = CameraTracker(**settings.to_tracker_kwargs())
    control_api.set_tracker(tracker)

    print("=" * 60)
    print("Camera Tracker Control System with Detection")
    print("=" * 60)
    print(f"Tracking servos: {'ENABLED' if settings.enable_servos else 'DISABLED'}")
    print(f"Trigger servo: {'ENABLED' if settings.enable_trigger else 'DISABLED'}")
    print(f"Camera: {'ENABLED' if settings.enable_camera else 'DISABLED'}")
    if settings.enable_camera:
        camera_type = "CSI (GStreamer)" if settings.use_csi else "USB"
        invert_status = "inverted (upside-down)" if settings.invert_camera else "normal"
        print(f"Camera type: {camera_type} (ID: {settings.camera_id}, {invert_status})")
    else:
        print("Tip: use --video-only for Jetson CSI camera web streaming without servos or detection")
    print(f"Detection: {'ENABLED' if settings.model_path else 'DISABLED'}")
    if settings.model_path:
        print(f"Model: {settings.requested_model}")
        target_classes = ", ".join(tracker.target_classes) if tracker.target_classes else "all"
        print(f"Target classes: {target_classes}")
        print(f"Confidence threshold: {settings.confidence_threshold}")
        print(f"Inference image size: {settings.imgsz}px")
        print(f"Inference FPS target: {tracker.inference_fps:g}")
        print(f"Tracking control FPS target: {tracker.tracking_control_fps:g}")
        print(f"Angular belief: alpha={tracker.belief_update_alpha:g}, pitch_alpha={tracker.belief_pitch_update_alpha:g}, miss_decay={tracker.belief_miss_decay:g}, min_conf={tracker.belief_min_confidence:g}, max_age={tracker.belief_max_age:g}s, reseed={tracker.belief_reseed_distance_raw:g} raw x{tracker.belief_reseed_confirmations}")
        print(f"Belief velocity: alpha={tracker.belief_velocity_alpha:g}, pitch_alpha={tracker.belief_pitch_velocity_alpha:g}, decay={tracker.belief_velocity_decay:g}, max={tracker.belief_max_velocity_raw_per_s:g}/{tracker.belief_max_pitch_velocity_raw_per_s:g} raw/s, predict={tracker.belief_max_prediction_age:g}s")
        print(f"Tracking limits: max yaw/pitch step={tracker.max_yaw_step}/{tracker.max_pitch_step}, deadband={tracker.belief_deadband_raw} raw, pitch_scale={tracker.pitch_tracking_scale:g}, motor_readback={tracker.motor_readback_fps:g} FPS")
    calib_status = "ENABLED" if tracker.stereo_depth.calibration_enabled else "DISABLED"
    if tracker.stereo_depth.calibration_enabled:
        calib_type = "STEREO" if tracker.stereo_depth.stereo_calibration_enabled else "SINGLE"
        print(f"Calibration: {calib_status} ({calib_type}, {settings.calibration_file})")
    elif settings.calibration_file != "camera_calibration.npz":
        print(f"Calibration: {calib_status} (file not found)")

    if settings.stereo_mode:
        stereo_status = "ENABLED" if tracker.stereo_depth.stereo_calibration_enabled else "DISABLED (no stereo calibration)"
        print(f"Stereo depth: {stereo_status}")
    print()

    api_thread = threading.Thread(
        target=run_api_server,
        args=(runtime.api_host, runtime.api_port),
        daemon=True,
    )
    api_thread.start()
    print(f"Web interface: http://{runtime.api_host}:{runtime.api_port}")
    print()

    try:
        print("Server running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        tracker.disconnect()


if __name__ == "__main__":
    main()
