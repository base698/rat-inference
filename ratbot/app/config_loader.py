"""Configuration loading and derived constants for the ratbot app.

Executed once at import, matching the original rt_200 module behavior
(config.yaml is read from the process working directory).
"""

from ratbot.app.deps import *  # noqa: F401,F403

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

# Camera geometry (with config fallback)
camera_config = CONFIG.get('camera', {}) if CONFIG else {}
CAMERA_WIDTH = int(camera_config.get('width', 640))
CAMERA_HEIGHT = int(camera_config.get('height', 480))
CAMERA_FOV_HORIZONTAL = float(camera_config.get('fov_horizontal', 60.0))
CAMERA_FOV_VERTICAL = float(camera_config.get('fov_vertical', 45.0))


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

