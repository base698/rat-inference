#!/usr/bin/env python3
"""
Real-time camera tracking with Feetech servo control and rat detection
Controls yaw (ID 1, 1600-3100 raw) and pitch (ID 5, 1-500 raw) servos
Includes Raspberry Pi camera streaming, YOLO inference, and GPIO trigger servo
"""

from ratbot.app.deps import *  # noqa: F401,F403
from ratbot.app.config_loader import *  # noqa: F401,F403
from ratbot.app.server import app, control_api  # noqa: F401
from ratbot.app.video import VideoPipelineMixin

class CameraTracker(VideoPipelineMixin):
    def __init__(self, port="/dev/cu.usbmodem5A680116511", enable_servos=True,
                 no_connect=False, enable_camera=False, enable_trigger=False,
                 model_path=None, confidence_threshold=0.85, camera_id=0,
                 use_csi=False, invert_camera=False, camera_width=640,
                 camera_height=480, camera_fov_horizontal=60.0,
                 camera_fov_vertical=45.0, imgsz=640,
                 inference_device=None,
                 inference_fps=None, target_classes=None, calibration_file=None,
                 stereo_mode=False, baseline_override=None,
                 tracking_smoothing=0.45, max_yaw_step=45, max_pitch_step=45,
                 max_yaw_speed_raw_per_s=None,
                 max_pitch_speed_raw_per_s=None,
                 tracking_control_fps=20, belief_update_alpha=0.45,
                 belief_miss_decay=0.94, belief_min_confidence=0.15,
                 belief_max_age=1.5, belief_deadband_raw=4,
                 belief_min_step_raw=3, pitch_tracking_scale=1.0,
                 belief_reseed_distance_raw=160, belief_velocity_alpha=0.45,
                 belief_velocity_decay=0.96, belief_max_velocity_raw_per_s=600,
                 belief_max_prediction_age=0.45, belief_reseed_confirmations=2,
                 belief_reseed_match_distance_raw=120,
                 belief_reseed_max_interval=0.8,
                 belief_reseed_min_confidence=0.55,
                 belief_pitch_update_alpha=None, belief_pitch_velocity_alpha=None,
                 belief_max_pitch_velocity_raw_per_s=None,
                 world_tracking=False, world_gate_distance_mm=750.0,
                 world_confirm_hits=3, world_max_misses=5,
                 world_delete_after_seconds=4.0,
                 world_reidentify_after_seconds=8.0,
                 world_process_acceleration_std_mm_s2=300.0,
                 world_min_depth_confidence=0.2,
                 world_aim_latency_seconds=0.12,
                 world_yaw_center_raw=None, world_pitch_center_raw=None,
                 world_yaw_raw_per_degree=None,
                 world_pitch_raw_per_degree=None,
                 world_yaw_sign=1.0, world_pitch_sign=-1.0,
                 world_camera_translation_mm=(0.0, 0.0, 0.0),
                 world_camera_mount_rpy_degrees=(0.0, 0.0, 0.0),
                 world_log_path=None, world_recordings_dir="run_logs/tracks",
                 world_api_recording_enabled=False,
                 world_api_selection_enabled=False,
                 world_actuation_enabled=False,
                 world_calibration_validated=False):
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
            inference_device: YOLO device passed to Ultralytics ('0', 'cuda:0', 'cpu', or None for auto)
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
            max_yaw_speed_raw_per_s: Optional yaw raw-unit speed cap for control interpolation
            max_pitch_speed_raw_per_s: Optional pitch raw-unit speed cap for control interpolation
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
            belief_reseed_min_confidence: Minimum confidence for far-jump reseeds
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
            output_width=camera_width,
            output_height=camera_height,
            csi_capture_factory=(
                CSICameraCapture if CSI_HELPER_AVAILABLE else None
            ),
        )
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_fov_horizontal = float(camera_fov_horizontal)
        self.camera_fov_vertical = float(camera_fov_vertical)
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.inference_device = inference_device
        self.inference_runtime_device = None
        self._runtime_device_logged = False
        self.inference_fps = max(1.0, float(inference_fps or INFERENCE_FPS))
        self.target_classes = self._normalize_target_classes(target_classes)
        self.max_yaw_step = max(0, int(max_yaw_step))
        self.max_pitch_step = max(0, int(max_pitch_step))
        self.max_yaw_speed_raw_per_s = (
            None if max_yaw_speed_raw_per_s is None
            else max(0.0, float(max_yaw_speed_raw_per_s))
        )
        self.max_pitch_speed_raw_per_s = (
            None if max_pitch_speed_raw_per_s is None
            else max(0.0, float(max_pitch_speed_raw_per_s))
        )
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
        self.belief_reseed_min_confidence = max(
            self.belief_min_confidence,
            min(1.0, float(belief_reseed_min_confidence)),
        )
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
        self.world_tracking = bool(world_tracking)
        self.world_calibration_validated = bool(world_calibration_validated)
        if world_actuation_enabled and not self.world_tracking:
            raise ValueError("world actuation requires world tracking")
        if world_actuation_enabled and not self.world_calibration_validated:
            raise ValueError(
                "world actuation requires explicit calibration validation"
            )
        self.world_actuation_enabled = bool(world_actuation_enabled)
        self.world_min_depth_confidence = max(
            0.0, min(1.0, float(world_min_depth_confidence))
        )
        self.world_aim_latency_seconds = max(0.0, float(world_aim_latency_seconds))
        self.world_log_path = str(world_log_path) if world_log_path else None
        self.track_recordings = TrackRecordingStore(world_recordings_dir)
        self.world_api_recording_enabled = bool(world_api_recording_enabled)
        self.world_api_selection_enabled = bool(world_api_selection_enabled)
        self.latest_tracks = []
        self.latest_track_assignments = []
        self.detection_count = 0
        self.latest_frame = None
        self.latest_raw_frame = None
        self.pose_history = deque(maxlen=90)  # (monotonic_t, yaw, pitch) at ~30Hz
        self.camera_latency_s = float(os.environ.get("RATBOT_CAMERA_LATENCY_S", "0.10"))
        self.latest_detection = False
        self.latest_bbox = None  # Store latest bounding box (x1, y1, x2, y2)
        self.latest_center_point = None  # Store latest center point (x, y)
        self.latest_depth = None  # Store depth in mm at detection point
        self.recent_detections = []
        self.frame_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.inference_loop_count = 0
        self.inference_fps_window_start = time.time()
        self.last_inference_fps = 0.0

        tracking_config = CONFIG.get('tracking', {}) if CONFIG else {}
        snapshot_config = tracking_config.get('detection_snapshots', {})
        self.detection_snapshot_enabled = bool(snapshot_config.get('enabled', True))
        self.detection_snapshot_min_interval_seconds = max(
            0.0, float(snapshot_config.get('min_interval_seconds', 2.0))
        )
        self.detection_snapshot_max_age_seconds = max(
            0.0, float(snapshot_config.get('max_age_days', 1.0)) * 86400.0
        )
        self.detection_snapshot_max_files = max(
            0, int(snapshot_config.get('max_files', 1000))
        )
        self.detection_snapshot_prune_interval_seconds = max(
            1.0, float(snapshot_config.get('prune_interval_seconds', 300.0))
        )
        self._last_detection_snapshot_time = 0.0
        self._last_detection_snapshot_prune_time = 0.0
        self.stereo_depth = StereoDepthService(
            calibration_file=calibration_file,
            baseline_override=baseline_override,
            min_texture_std=tracking_config.get('depth_min_texture_std', 4.0),
            min_valid_mm=tracking_config.get('depth_min_valid_mm', 0.0),
            max_valid_mm=tracking_config.get('depth_max_valid_mm', 6000.0),
            image_size=(self.camera_width, self.camera_height),
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
        self.servo_acceleration = int(tracking_config.get('servo_acceleration', 0) or 0)
        self.servo_max_speed = int(tracking_config.get('servo_max_speed', 0) or 0)
        vc = tracking_config.get('velocity_control') or {}
        self.velocity_control_enabled = bool(vc.get('enabled', False))
        self.velocity_control_cfg = vc
        self.measured_yaw_velocity = 0.0
        self.measured_pitch_velocity = 0.0
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
                image_width=self.camera_width,
                image_height=self.camera_height,
                horizontal_fov_degrees=self.camera_fov_horizontal,
                vertical_fov_degrees=self.camera_fov_vertical,
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
            reseed_min_confidence=self.belief_reseed_min_confidence,
            pitch_update_alpha=self.belief_pitch_update_alpha,
            pitch_velocity_alpha=self.belief_pitch_velocity_alpha,
            max_pitch_velocity_raw_per_s=self.belief_max_pitch_velocity_raw_per_s,
        )
        self.world_transformer = TurretFrameTransformer(
            ServoKinematicsConfig(
                yaw_center_raw=(YAW_CENTER if world_yaw_center_raw is None else world_yaw_center_raw),
                pitch_center_raw=(PITCH_CENTER if world_pitch_center_raw is None else world_pitch_center_raw),
                yaw_raw_per_degree=(
                    (YAW_MAX - YAW_MIN) / 180.0
                    if world_yaw_raw_per_degree is None
                    else world_yaw_raw_per_degree
                ),
                pitch_raw_per_degree=(
                    (PITCH_MAX - PITCH_MIN) / 55.0
                    if world_pitch_raw_per_degree is None
                    else world_pitch_raw_per_degree
                ),
                yaw_sign=float(world_yaw_sign),
                pitch_sign=float(world_pitch_sign),
                camera_translation_mm=tuple(world_camera_translation_mm),
                camera_mount_rpy_degrees=tuple(world_camera_mount_rpy_degrees),
                yaw_min_raw=YAW_MIN,
                yaw_max_raw=YAW_MAX,
                pitch_min_raw=PITCH_MIN,
                pitch_max_raw=PITCH_MAX,
            )
        )
        self.world_tracker = MultiTargetTracker(
            TrackManagerConfig(
                gate_distance_mm=float(world_gate_distance_mm),
                confirm_hits=int(world_confirm_hits),
                max_misses=int(world_max_misses),
                delete_after_seconds=float(world_delete_after_seconds),
                reidentify_after_seconds=float(world_reidentify_after_seconds),
                process_acceleration_std_mm_s2=float(
                    world_process_acceleration_std_mm_s2
                ),
                # With physical actuation, auto-select only the lone confirmed
                # visible track. Multi-target scenes still require explicit UI
                # selection so the robot does not redirect to another object.
                auto_select=True,
                auto_select_single_only=self.world_actuation_enabled,
            )
        )
        self.world_belief = WorldTrackBeliefAdapter(
            self.world_tracker,
            self.world_transformer,
            aim_latency_seconds=self.world_aim_latency_seconds,
            min_confidence=belief_min_confidence,
            max_age_seconds=belief_max_age,
            robot=self,
        )
        controller_belief = (
            self.world_belief
            if self.world_tracking and self.world_actuation_enabled
            else self.target_belief
        )
        if self.velocity_control_enabled:
            vc = self.velocity_control_cfg
            self.tracking_controller = VelocityFormController(
                robot=self,
                belief=controller_belief,
                bounds=servo_bounds,
                control_fps=tracking_control_fps,
                kp_yaw=float(vc.get('kp_yaw', 6.0)),
                kp_pitch=float(vc.get('kp_pitch', 5.5)),
                max_yaw_velocity=float(vc.get(
                    'max_yaw_velocity_raw_per_s',
                    self.max_yaw_step * tracking_control_fps)),
                max_pitch_velocity=float(vc.get(
                    'max_pitch_velocity_raw_per_s',
                    self.max_pitch_step * tracking_control_fps)),
                max_accel=float(vc.get('max_accel_raw_per_s2', 3500.0)),
                deadband_raw=belief_deadband_raw,
                damping_yaw=float(vc.get('damping_yaw', 0.0)),
                damping_pitch=float(vc.get('damping_pitch', 0.0)),
                reconcile_rate=float(vc.get('reconcile_rate', 2.0)),
            )
            print("Tracking controller: velocity-form (FPS-independent gains)")
        else:
            self.tracking_controller = AngularBeliefController(
                robot=self,
                belief=controller_belief,
                bounds=servo_bounds,
                control_fps=tracking_control_fps,
                max_yaw_step=self.max_yaw_step,
                max_pitch_step=self.max_pitch_step,
                max_yaw_speed_raw_per_s=self.max_yaw_speed_raw_per_s,
                max_pitch_speed_raw_per_s=self.max_pitch_speed_raw_per_s,
                deadband_raw=belief_deadband_raw,
                min_step_raw=belief_min_step_raw
            )

        # Create detections directory
        self.detections_dir = DETECTIONS_DIR
        os.makedirs(self.detections_dir, exist_ok=True)
        self._prune_detection_snapshots()

        # Initialize tracking servos if enabled
        if self.enable_servos and not self.no_connect and FEETECH_AVAILABLE:
            self.tracking_servos.connect()
            if self.tracking_servos.connected:
                self.tracking_servos.configure_motion(
                    acceleration=self.servo_acceleration,
                    max_speed=self.servo_max_speed,
                )

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
            print(f"  YOLO inference device request: {self.inference_device or 'auto'}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def _effective_inference_device(self, runtime_device):
        """Return the device to expose after first inference."""
        if runtime_device is not None:
            return str(runtime_device)
        if not self.inference_device:
            return None
        device = str(self.inference_device)
        if device.isdigit():
            return f"cuda:{device}"
        return device

    def _detection_snapshot_files(self):
        try:
            names = os.listdir(self.detections_dir)
        except OSError:
            return []

        snapshots = []
        for name in names:
            if not name.startswith("detection_") or not name.lower().endswith(".jpg"):
                continue
            path = os.path.join(self.detections_dir, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            if not os.path.isfile(path):
                continue
            snapshots.append((stat.st_mtime, path))
        return snapshots

    def _prune_detection_snapshots(self, now=None):
        """Delete stale detection snapshots and keep the snapshot directory bounded."""
        now = time.time() if now is None else float(now)
        self._last_detection_snapshot_prune_time = now
        snapshots = self._detection_snapshot_files()
        if not snapshots:
            return 0

        deleted = 0
        kept = []
        for modified_time, path in snapshots:
            too_old = (
                self.detection_snapshot_max_age_seconds > 0
                and now - modified_time > self.detection_snapshot_max_age_seconds
            )
            if too_old:
                try:
                    os.remove(path)
                    deleted += 1
                except OSError:
                    pass
            else:
                kept.append((modified_time, path))

        if self.detection_snapshot_max_files > 0:
            overflow = max(0, len(kept) - self.detection_snapshot_max_files)
            for _, path in sorted(kept)[:overflow]:
                try:
                    os.remove(path)
                    deleted += 1
                except OSError:
                    pass

        if deleted:
            print(f"Pruned {deleted} old detection snapshot(s)")
        return deleted

    def _save_detection_snapshot(self, frame, now=None):
        """Save a throttled detection snapshot, returning (filename, path) or None."""
        if not self.detection_snapshot_enabled:
            return None
        now = time.time() if now is None else float(now)
        if now - self._last_detection_snapshot_time < self.detection_snapshot_min_interval_seconds:
            return None
        if now - self._last_detection_snapshot_prune_time >= self.detection_snapshot_prune_interval_seconds:
            self._prune_detection_snapshots(now=now)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        detection_filename = f"detection_{timestamp}.jpg"
        detection_path = os.path.join(self.detections_dir, detection_filename)
        if not cv2.imwrite(detection_path, frame):
            return None
        self._last_detection_snapshot_time = now
        return detection_filename, detection_path













    def update_world_tracks(
        self,
        detections,
        stereo_measurements,
        *,
        timestamp,
        yaw_raw,
        pitch_raw,
    ):
        """Transform valid stereo detections and update stable base-frame tracks."""
        measurements_3d = []
        measurement_records = []
        for detection, stereo in zip(detections, stereo_measurements):
            if stereo is None or stereo.confidence < self.world_min_depth_confidence:
                continue
            position_base = self.world_transformer.camera_to_base(
                stereo.point_camera_mm,
                yaw_raw,
                pitch_raw,
            )
            covariance_base = self.world_transformer.camera_covariance_to_base(
                stereo.covariance_camera,
                yaw_raw,
                pitch_raw,
            )
            measurements_3d.append(
                Detection3D(
                    position_base_mm=position_base,
                    covariance_base=covariance_base,
                    confidence=max(
                        0.0,
                        min(1.0, float(detection["confidence"]) * stereo.confidence),
                    ),
                    classification=detection.get("class_name"),
                    measurement_time=float(timestamp),
                    bbox=detection.get("bbox"),
                    center=detection.get("center"),
                    depth_mm=stereo.depth_mm,
                    depth_confidence=stereo.confidence,
                )
            )
            measurement_records.append(
                {
                    "center": list(detection.get("center")) if detection.get("center") is not None else None,
                    "bbox": list(detection.get("bbox")) if detection.get("bbox") is not None else None,
                    "class": detection.get("class_name"),
                    "detection_confidence": float(detection["confidence"]),
                    "depth_confidence": stereo.confidence,
                    "camera_point_mm": stereo.point_camera_mm.tolist(),
                    "base_point_mm": position_base.tolist(),
                    "base_covariance": covariance_base.tolist(),
                    "depth_mm": stereo.depth_mm,
                    "disparity_px": stereo.disparity_px,
                    "disparity_iqr_px": stereo.disparity_iqr_px,
                    "valid_ratio": stereo.valid_ratio,
                }
            )

        tracks = self.world_tracker.update(measurements_3d, float(timestamp))
        selected = self.world_tracker.get_selected_track(timestamp=float(timestamp))
        with self.inference_lock:
            self.latest_tracks = tracks
            self.latest_track_assignments = list(self.world_tracker.last_assignments)
            if selected is not None and selected.misses == 0:
                self.latest_bbox = selected.bbox
                self.latest_center_point = selected.center
        if self.world_log_path or self.track_recordings.status()["recording"]:
            self._log_world_update(
                timestamp=float(timestamp),
                measurement_count=len(measurements_3d),
                detection_count=len(detections),
                yaw_raw=yaw_raw,
                pitch_raw=pitch_raw,
                tracks=tracks,
                measurement_records=measurement_records,
            )
        print(
            "   World tracks: "
            f"valid_3d={len(measurements_3d)}/{len(detections)}, "
            f"tracks={len(tracks)}, selected={self.world_tracker.selected_track_id}"
        )
        return tracks

    def _log_world_update(
        self,
        *,
        timestamp,
        measurement_count,
        detection_count,
        yaw_raw,
        pitch_raw,
        tracks,
        measurement_records,
    ):
        """Append one replayable JSONL update when configured."""
        predicted = self.world_tracker.get_selected_track(
            timestamp=timestamp,
            prediction_horizon=self.world_aim_latency_seconds,
        )
        predicted_aim = None
        if predicted is not None:
            predicted_aim = self.world_transformer.base_position_to_servo_raw(
                predicted.position
            )
        record = {
            "schema": "ratbot.world_tracks.v1",
            "recorded_at": datetime.now().astimezone().isoformat(),
            "monotonic_time": timestamp,
            "control_monotonic_time": time.monotonic(),
            "image_size": {
                "width": self.camera_width,
                "height": self.camera_height,
            },
            "pose_raw": {"yaw": yaw_raw, "pitch": pitch_raw},
            "commanded_pose_raw": {
                "yaw": self.current_yaw,
                "pitch": self.current_pitch,
            },
            "detection_count": detection_count,
            "valid_3d_measurement_count": measurement_count,
            "measurements": measurement_records,
            "selected_track_id": self.world_tracker.selected_track_id,
            "predicted_aim": predicted_aim,
            "assignments": list(self.world_tracker.last_assignments),
            "tracks": [track.to_dict() for track in tracks],
        }
        try:
            self.track_recordings.append(record)
        except (OSError, TypeError, ValueError) as exc:
            self.track_recordings.fail(str(exc))
            print(f"Track-recording error; recording stopped: {exc}")
        log_path = self.world_log_path
        if not log_path:
            return
        try:
            parent = os.path.dirname(os.path.abspath(log_path))
            os.makedirs(parent, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError as exc:
            print(f"World-track log error: {exc}")

    def update_target_belief(self, center_x, center_y, confidence, depth_mm=None,
                             pose_yaw=None, pose_pitch=None):
        """Update angular target belief from one detection observation.

        pose_yaw/pose_pitch anchor the pixel error to the servo pose captured
        with the frame; using the live position instead injects the turret's
        own motion into the observation and makes the loop hunt.
        """
        observation = self.observation_converter.to_servo_target(
            target_x=center_x,
            target_y=center_y,
            current_yaw=self.current_yaw if pose_yaw is None else pose_yaw,
            current_pitch=self.current_pitch if pose_pitch is None else pose_pitch,
            depth_mm=depth_mm,
        )
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
        """Clear autonomous target state and reset tracking controller state."""
        self.target_belief.clear()
        self.world_tracker.clear()
        self.latest_tracks = []
        self.latest_track_assignments = []
        self.tracking_controller.reset()
        print("   Target tracking state cleared")
        return True

    def get_world_tracks(self):
        return [track.to_dict() for track in self.world_tracker.get_tracks()]

    def _world_tracking_parameters(self):
        config = self.world_tracker.config
        return {
            "gate_distance_mm": config.gate_distance_mm,
            "confirm_hits": config.confirm_hits,
            "max_misses": config.max_misses,
            "delete_after_seconds": config.delete_after_seconds,
            "reidentify_after_seconds": config.reidentify_after_seconds,
            "process_acceleration_std_mm_s2": config.process_acceleration_std_mm_s2,
            "confidence_decay": config.confidence_decay,
            "min_depth_confidence": self.world_min_depth_confidence,
            "image_width": self.camera_width,
            "image_height": self.camera_height,
        }

    def get_track_recording_status(self):
        return self.track_recordings.status()

    def start_track_recording(self):
        if not self.world_tracking:
            raise RuntimeError("world tracking must be enabled before recording")
        return self.track_recordings.start(self._world_tracking_parameters())

    def stop_track_recording(self):
        return self.track_recordings.stop()

    def list_track_recordings(self):
        return self.track_recordings.list_recordings()

    def load_track_recording(self, recording_id):
        return self.track_recordings.load(recording_id)

    def delete_track_recording(self, recording_id):
        return self.track_recordings.delete(recording_id)

    def reprocess_track_recording(self, recording_id, parameters):
        return self.track_recordings.reprocess(recording_id, parameters)

    def select_world_target(self, target_id):
        return self.world_tracker.select_target(int(target_id))

    def clear_world_selection(self):
        self.world_tracker.clear_selection()
        return True

    def get_active_target_belief(self):
        """Return the active belief for the configured tracking mode."""
        if self.world_tracking:
            return self.world_belief.get_active()
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



    def capture_world_measurement_context(self):
        """Snapshot acquisition time with synchronous measured servo readback.

        Cameras currently expose no hardware exposure timestamp. This samples the
        monotonic clock immediately after frame acquisition, then obtains actual
        Present_Position values instead of reusing commanded servo goals.
        """
        measurement_time = time.monotonic()
        pose_yaw, pose_pitch = self.tracking_servos.read_measured_positions()
        return measurement_time, int(pose_yaw), int(pose_pitch)

    def run_inference(self):
        """Run one inference pass using the shared inference module"""
        if not self.camera_active or not self.camera or not self.model:
            return

        try:
            frame, frame_right = self.camera_source.read_frames()
            if frame is None:
                return
            # CameraSource does not yet expose hardware exposure timestamps.
            # Snapshot monotonic acquisition time with synchronous Present_Position
            # readback; commanded servo goals are not valid transform poses.
            measurement_time, pose_yaw, pose_pitch = (
                self.capture_world_measurement_context()
                if self.world_tracking
                else (time.monotonic(), self.current_yaw, self.current_pitch)
            )

            # Keep inference on the normal camera image; depth rectifies hidden copies only.
            if self.stereo_depth.calibration_enabled and not self.stereo_depth.stereo_calibration_enabled:
                frame = self.stereo_depth.undistort_frame(frame, use_left=True)

            # Run inference using shared inference module (YOLO works directly with numpy arrays)
            results = yolo_run_inference(
                self.model,
                frame,
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                device=self.inference_device,
                verbose=False
            )
            if not self._runtime_device_logged:
                runtime_device = getattr(self.model, "device", None)
                self.inference_runtime_device = self._effective_inference_device(
                    runtime_device
                )
                print(
                    f"YOLO runtime/effective device: {self.inference_runtime_device or 'unknown'}",
                    flush=True,
                )
                self._runtime_device_logged = True

            # Extract detections using shared utility and keep configured target classes.
            if self.target_classes:
                detections = []
                for target_class in self.target_classes:
                    detections.extend(extract_detections(results, self.model, target_class=target_class))
            else:
                detections = extract_detections(results, self.model)
            detections.sort(key=lambda det: det["confidence"], reverse=True)
            stereo_measurements = [None] * len(detections)
            if (
                detections
                and self.stereo_depth.stereo_calibration_enabled
                and frame_right is not None
            ):
                stereo_measurements = self.stereo_depth.calculate_depths(
                    frame,
                    frame_right,
                    [det["center"] for det in detections],
                )

            if self.world_tracking:
                self.update_world_tracks(
                    detections,
                    stereo_measurements,
                    timestamp=measurement_time,
                    yaw_raw=pose_yaw,
                    pitch_raw=pose_pitch,
                )

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

                # Reuse the batched stereo pass; do not recompute disparity per target.
                stereo_measurement = stereo_measurements[0]
                if stereo_measurement is not None:
                    depth_mm = stereo_measurement.depth_mm
                    depth_m = depth_mm / 1000.0
                    print(
                        f"   Depth: {depth_m:.2f}m ({depth_mm:.1f}mm), "
                        f"quality={stereo_measurement.confidence:.2f}, "
                        f"valid={stereo_measurement.valid_ratio:.2f}, "
                        f"iqr={stereo_measurement.disparity_iqr_px:.2f}px"
                    )

                snapshot = self._save_detection_snapshot(frame)

                self.detection_count += 1
                # Format: "time - message | filename" so the UI can parse and link it
                depth_str = f" @ {depth_mm/1000.0:.2f}m" if depth_mm else ""
                detection_msg = f"{datetime.now().strftime('%H:%M:%S')} - {class_name} detected (conf: {confidence:.3f}) at ({center_x}, {center_y}){depth_str}"
                if snapshot:
                    detection_msg = f"{detection_msg} | {snapshot[0]}"
                self.recent_detections.append(detection_msg)
                self.recent_detections = self.recent_detections[-10:]

                if snapshot:
                    print(f"🎯 Detection #{self.detection_count}: {snapshot[1]}")
                else:
                    print(f"🎯 Detection #{self.detection_count}: snapshot skipped")
                print(f"   Class: {class_name}, Center: ({center_x}, {center_y}), Confidence: {confidence:.3f}")

                # World mode already updated every valid 3D detection above.  The
                # legacy angular belief remains the default/fallback mode.
                if not self.world_tracking:
                    frame_pose_yaw, frame_pose_pitch = self.pose_at(
                        measurement_time - self.camera_latency_s
                    )
                    self.update_target_belief(
                        center_x,
                        center_y,
                        confidence,
                        depth_mm=depth_mm,
                        pose_yaw=frame_pose_yaw,
                        pose_pitch=frame_pose_pitch,
                    )

                # Auto-trigger disabled - only trigger manually via button
                # if detection and not self.latest_detection and self.trigger_servo_enabled:
                #     threading.Thread(target=self.trigger_action_servo, daemon=True).start()
            else:
                if not self.world_tracking:
                    self.decay_target_belief()

            if self.world_tracking:
                selected_track = self.world_tracker.get_selected_track(
                    timestamp=measurement_time
                )
                if selected_track is not None and selected_track.misses == 0:
                    bbox = selected_track.bbox
                    center_point = selected_track.center
                    confidence = selected_track.confidence
                    depth_mm = float(np.linalg.norm(selected_track.position))

            # Update detection state
            with self.inference_lock:
                self.latest_detection = detection
                self.latest_confidence = confidence
                self.latest_bbox = bbox
                self.latest_center_point = center_point
                self.latest_depth = depth_mm

        except Exception as e:
            print(f"Inference error: {e}")



    def get_detection_data(self):
        """Get the latest detection data without image"""
        with self.inference_lock:
            return {
                "detection": self.latest_detection,
                "confidence": self.latest_confidence if hasattr(self, 'latest_confidence') else 0,
                "depth_mm": self.latest_depth,
                "inference_device": self.inference_device or "auto",
                "inference_runtime_device": self.inference_runtime_device,
                "recent_detections": self.recent_detections,
                "world_tracking": self.world_tracking,
                "world_actuation_enabled": self.world_actuation_enabled,
                "world_calibration_validated": self.world_calibration_validated,
                "world_api_selection_enabled": self.world_api_selection_enabled,
                "selected_track_id": self.world_tracker.selected_track_id,
                "tracks": [track.to_dict() for track in self.latest_tracks],
                "track_assignments": list(self.latest_track_assignments),
            }


    def inference_thread(self):
        """Inference processing thread"""
        while self.camera_active:
            loop_start = time.time()
            self.run_inference()
            self.inference_loop_count += 1
            window_elapsed = time.time() - self.inference_fps_window_start
            if window_elapsed >= 5.0:
                actual_fps = self.inference_loop_count / window_elapsed
                self.last_inference_fps = actual_fps
                print(f"Inference actual FPS: {actual_fps:.1f} (target {self.inference_fps:g})", flush=True)
                self.inference_loop_count = 0
                self.inference_fps_window_start = time.time()
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, (1.0 / self.inference_fps) - elapsed))

    def tracking_control_thread(self):
        """Servo control thread that continuously moves toward angular target belief."""
        self.tracking_controller.run()


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
        print(f"Inference device: {settings.inference_device or 'auto'}")
        print(f"Inference FPS target: {tracker.inference_fps:g}")
        print(f"Tracking control FPS target: {tracker.tracking_control_fps:g}")
        print(f"Angular belief: alpha={tracker.belief_update_alpha:g}, pitch_alpha={tracker.belief_pitch_update_alpha:g}, miss_decay={tracker.belief_miss_decay:g}, min_conf={tracker.belief_min_confidence:g}, max_age={tracker.belief_max_age:g}s, reseed={tracker.belief_reseed_distance_raw:g} raw x{tracker.belief_reseed_confirmations} @ conf>={tracker.belief_reseed_min_confidence:g}")
        print(f"Belief velocity: alpha={tracker.belief_velocity_alpha:g}, pitch_alpha={tracker.belief_pitch_velocity_alpha:g}, decay={tracker.belief_velocity_decay:g}, max={tracker.belief_max_velocity_raw_per_s:g}/{tracker.belief_max_pitch_velocity_raw_per_s:g} raw/s, predict={tracker.belief_max_prediction_age:g}s")
        yaw_speed = tracker.max_yaw_speed_raw_per_s
        pitch_speed = tracker.max_pitch_speed_raw_per_s
        speed_limits = (
            f"{yaw_speed:g}/{pitch_speed:g} raw/s"
            if yaw_speed and pitch_speed
            else "off"
        )
        print(f"Tracking limits: max yaw/pitch step={tracker.max_yaw_step}/{tracker.max_pitch_step}, speed={speed_limits}, deadband={tracker.belief_deadband_raw} raw, pitch_scale={tracker.pitch_tracking_scale:g}, motor_readback={tracker.motor_readback_fps:g} FPS")
        world_status = "ENABLED" if settings.world_tracking else "DISABLED (legacy angular mode)"
        print(f"World-frame tracking: {world_status}")
        if settings.world_tracking:
            world_config = tracker.world_tracker.config
            print(
                "World-frame ID retention: "
                f"gate={world_config.gate_distance_mm:g}mm, "
                f"delete={world_config.delete_after_seconds:g}s, "
                f"re-id={world_config.reidentify_after_seconds:g}s"
            )
            print(
                "WARNING: world-frame motion assumes a stationary base and "
                "verified servo signs/scales plus camera extrinsics"
            )
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
