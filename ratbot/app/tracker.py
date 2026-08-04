"""CameraTracker: wiring, properties, and lifecycle for the tracker runtime.

Pipeline behavior lives in the mixins:
  VideoPipelineMixin      - frames, overlays, pose history (ratbot/app/video.py)
  DetectionPipelineMixin  - YOLO inference + snapshots (ratbot/app/detection.py)
  BeliefGlueMixin         - angular belief observations + control thread
  WorldGlueMixin          - world-frame tracking, recordings, selection
"""

from ratbot.app.deps import *  # noqa: F401,F403
from ratbot.robot.controllers import make_tracking_controller
from ratbot.app.config_loader import *  # noqa: F401,F403
from ratbot.app.video import VideoPipelineMixin
from ratbot.app.detection import DetectionPipelineMixin
from ratbot.app.belief_glue import BeliefGlueMixin
from ratbot.app.world_glue import WorldGlueMixin


class CameraTracker(WorldGlueMixin, BeliefGlueMixin, DetectionPipelineMixin, VideoPipelineMixin):
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
        # Explicit name wins; legacy velocity_control.enabled keeps working.
        self.tracking_controller_name = str(
            tracking_config.get('controller')
            or ('velocity' if self.velocity_control_enabled else 'angular')
        )
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
        self.tracking_controller = make_tracking_controller(
            self.tracking_controller_name,
            robot=self,
            belief=controller_belief,
            bounds=servo_bounds,
            control_fps=tracking_control_fps,
            options=self.velocity_control_cfg,
        )
        print(f"Tracking controller: {self.tracking_controller_name} "
              f"({type(self.tracking_controller).__name__})")

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



































































    def disconnect(self):
        """Disconnect and cleanup"""
        self.camera_source.close()

        self.trigger_servo.close()
        self.tracking_servos.close()
        self.camera_active = False

