"""Typed command-line and runtime composition for the RT-200 application."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence


DEFAULT_STEREO_CALIBRATION = (
    "tools/vision/calibration/output_recal/stereo_calibration.npz"
)


@dataclass(frozen=True)
class TrackerRuntimeConfig:
    port: str
    enable_servos: bool
    no_connect: bool
    enable_camera: bool
    enable_trigger: bool
    model_path: str | None
    requested_model: str
    disable_detection: bool
    confidence_threshold: float
    camera_id: int
    use_csi: bool
    invert_camera: bool
    camera_width: int
    camera_height: int
    camera_fov_horizontal: float
    camera_fov_vertical: float
    imgsz: int
    inference_fps: float | None
    target_classes: tuple[str, ...]
    calibration_file: str
    stereo_mode: bool
    baseline_override: float | None
    tracking_smoothing: float
    max_yaw_step: int
    max_pitch_step: int
    max_yaw_speed_raw_per_s: float | None
    max_pitch_speed_raw_per_s: float | None
    tracking_control_fps: float
    belief_update_alpha: float
    belief_miss_decay: float
    belief_min_confidence: float
    belief_max_age: float
    belief_deadband_raw: int
    belief_min_step_raw: int
    pitch_tracking_scale: float
    belief_reseed_distance_raw: float
    belief_velocity_alpha: float
    belief_velocity_decay: float
    belief_max_velocity_raw_per_s: float
    belief_max_prediction_age: float
    belief_reseed_confirmations: int
    belief_reseed_match_distance_raw: float
    belief_reseed_max_interval: float
    belief_reseed_min_confidence: float
    belief_pitch_update_alpha: float | None
    belief_pitch_velocity_alpha: float | None
    belief_max_pitch_velocity_raw_per_s: float | None
    world_tracking: bool
    world_gate_distance_mm: float
    world_confirm_hits: int
    world_max_misses: int
    world_delete_after_seconds: float
    world_process_acceleration_std_mm_s2: float
    world_min_depth_confidence: float
    world_aim_latency_seconds: float
    world_yaw_center_raw: float | None
    world_pitch_center_raw: float | None
    world_yaw_raw_per_degree: float | None
    world_pitch_raw_per_degree: float | None
    world_yaw_sign: float
    world_pitch_sign: float
    world_camera_translation_mm: tuple[float, float, float]
    world_camera_mount_rpy_degrees: tuple[float, float, float]
    world_log_path: str | None
    world_recordings_dir: str
    world_api_recording_enabled: bool
    world_api_selection_enabled: bool
    world_actuation_enabled: bool
    world_calibration_validated: bool

    def to_tracker_kwargs(self) -> dict[str, Any]:
        values = asdict(self)
        values["target_classes"] = list(self.target_classes)
        values.pop("requested_model")
        values.pop("disable_detection")
        return values


@dataclass(frozen=True)
class ApplicationRuntimeConfig:
    tracker: TrackerRuntimeConfig
    api_host: str
    api_port: int


def _sections(raw_config: Mapping[str, Any] | None):
    raw_config = raw_config or {}
    detection = raw_config.get("detection", {})
    tracking = raw_config.get("tracking", {})
    auto_tracking = tracking.get("auto_tracking", {})
    return detection, auto_tracking


def _world_section(raw_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    raw_config = raw_config or {}
    return raw_config.get("tracking", {}).get("world_frame", {})


def _camera_section(raw_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    raw_config = raw_config or {}
    return raw_config.get("camera", {})


def _strict_bool(
    values: Mapping[str, Any], key: str, *, default: bool = False
) -> bool:
    value = values.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"tracking.world_frame.{key} must be a YAML boolean")
    return value


def _triple(value, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    value = default if value is None else value
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("world-frame translation/RPY values must contain exactly 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def build_argument_parser(
    raw_config: Mapping[str, Any] | None,
    *,
    inference_fps_default: float,
) -> argparse.ArgumentParser:
    detection, auto = _sections(raw_config)
    world = _world_section(raw_config)
    parser = argparse.ArgumentParser(
        description="Camera tracker with servo control and rat detection"
    )
    help_by_dest = {
        "port": "Serial port for servo connection",
        "disable_servos": "Disable tracking servo control (simulation mode)",
        "no_connect": "Skip servo connection attempt (web interface only)",
        "video_only": "Convenience mode: CSI camera web stream, no servos, no detection",
        "enable_camera": "Enable camera",
        "camera_id": "Camera device ID (default: 0)",
        "use_csi": "Use CSI camera with GStreamer pipeline (Jetson)",
        "invert_camera": "Invert camera 180 degrees for upside-down mounting",
        "model": "Path to YOLO model",
        "disable_detection": "Disable YOLO detection while keeping the camera stream enabled",
        "confidence": "Detection confidence threshold",
        "target_class": "YOLO class name to track. Repeat or comma-separate values. Default: all model classes",
        "imgsz": "Inference image size in pixels (default: 640)",
        "inference_fps": f"Target inference loop FPS (default: {inference_fps_default:g})",
        "tracking_smoothing": "Detection center smoothing alpha for auto tracking: 0 disables, 1 follows raw detections (default: 0.45)",
        "max_yaw_step": "Maximum yaw raw-unit move per inference update for smoother tracking (default: 45)",
        "max_pitch_step": "Maximum pitch raw-unit move per control update for smoother tracking (default: 45)",
        "max_yaw_speed_raw_per_s": "Optional yaw raw-unit speed cap for smoother control interpolation",
        "max_pitch_speed_raw_per_s": "Optional pitch raw-unit speed cap for smoother control interpolation",
        "tracking_control_fps": "Servo control loop FPS for angular target belief tracking (default: 20)",
        "belief_update_alpha": "Angular belief update alpha from new detections: 0 ignores, 1 jumps to observation (default: 0.45)",
        "belief_pitch_update_alpha": "Optional pitch-only angular belief update alpha (default: same as --belief-update-alpha)",
        "belief_miss_decay": "Belief confidence decay per inference tick without detection (default: 0.94)",
        "belief_min_confidence": "Minimum angular belief confidence required to keep moving (default: 0.15)",
        "belief_max_age": "Maximum age in seconds for angular belief movement (default: 1.5)",
        "belief_deadband_raw": "Raw servo-unit error treated as centered by angular belief control (default: 4)",
        "belief_min_step_raw": "Minimum raw servo-unit correction while outside angular belief deadband (default: 3)",
        "pitch_tracking_scale": "Multiplier for vertical image error to pitch raw observation; higher uses more down/up pitch travel (default: 1.0)",
        "belief_reseed_distance_raw": "Raw-unit observation jump that starts a fresh angular belief instead of smoothing (default: 160)",
        "belief_velocity_alpha": "Angular belief velocity smoothing alpha from detections (default: 0.45)",
        "belief_pitch_velocity_alpha": "Optional pitch-only angular belief velocity alpha (default: same as --belief-velocity-alpha)",
        "belief_velocity_decay": "Angular belief velocity decay per inference tick without detection (default: 0.96)",
        "belief_max_velocity_raw_per_s": "Maximum angular belief target speed in raw servo units/sec (default: 600)",
        "belief_max_pitch_velocity_raw_per_s": "Optional pitch-only angular belief target speed cap (default: same as --belief-max-velocity-raw-per-s)",
        "belief_max_prediction_age": "Maximum seconds to extrapolate target belief after the latest detection (default: 0.45)",
        "belief_reseed_confirmations": "Matching far-jump detections required before reseeding angular belief (default: 2)",
        "belief_reseed_match_distance_raw": "Raw-unit distance for matching pending reseed detections (default: 120)",
        "belief_reseed_max_interval": "Maximum seconds between matching pending reseed detections (default: 0.8)",
        "belief_reseed_min_confidence": "Minimum confidence allowed to start/confirm a far-jump angular belief reseed (default: 0.55)",
        "calibration": "Path to camera calibration file (.npz, default: camera_calibration.npz)",
        "stereo": "Enable stereo mode for depth estimation (requires stereo calibration)",
        "baseline_override": "Override stereo baseline in mm (use if calibration baseline is incorrect)",
        "world_tracking": "Enable opt-in multi-target tracking in the fixed turret-base 3D frame",
        "enable_trigger": "Enable GPIO trigger servo",
        "api_host": "Host for FastAPI server",
        "api_port": "Port for FastAPI server",
    }
    parser.add_argument("--port", "-p", type=str, default="/dev/ttyACM0")
    parser.add_argument("--disable-servos", action="store_true")
    parser.add_argument("--no-connect", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    parser.add_argument("--enable-camera", action="store_true")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--use-csi", action="store_true")
    parser.add_argument("--invert-camera", action="store_true")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=detection.get(
            "model_path", "runs/yolo11n-2025-10-24/weights/best.pt"
        ),
    )
    parser.add_argument("--disable-detection", action="store_true")
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=float(detection.get("confidence_threshold", 0.75)),
    )
    parser.add_argument("--target-class", action="append", default=None)
    parser.add_argument(
        "--imgsz", type=int, default=int(detection.get("imgsz", 640))
    )
    parser.add_argument("--inference-fps", type=float, default=None)
    parser.add_argument(
        "--tracking-smoothing",
        type=float,
        default=float(auto.get("smoothing_alpha", 0.45)),
    )
    parser.add_argument(
        "--max-yaw-step", type=int, default=int(auto.get("max_yaw_step", 45))
    )
    parser.add_argument(
        "--max-pitch-step",
        type=int,
        default=int(auto.get("max_pitch_step", 45)),
    )
    parser.add_argument(
        "--max-yaw-speed-raw-per-s",
        type=float,
        default=auto.get("max_yaw_speed_raw_per_s"),
    )
    parser.add_argument(
        "--max-pitch-speed-raw-per-s",
        type=float,
        default=auto.get("max_pitch_speed_raw_per_s"),
    )
    parser.add_argument(
        "--tracking-control-fps",
        type=float,
        default=float(auto.get("control_fps", 20)),
    )
    parser.add_argument(
        "--belief-update-alpha",
        type=float,
        default=float(auto.get("belief_update_alpha", 0.45)),
    )
    parser.add_argument(
        "--belief-pitch-update-alpha",
        type=float,
        default=auto.get("belief_pitch_update_alpha"),
    )
    parser.add_argument(
        "--belief-miss-decay",
        type=float,
        default=float(auto.get("belief_miss_decay", 0.94)),
    )
    parser.add_argument(
        "--belief-min-confidence",
        type=float,
        default=float(auto.get("belief_min_confidence", 0.15)),
    )
    parser.add_argument(
        "--belief-max-age",
        type=float,
        default=float(auto.get("belief_max_age", 1.5)),
    )
    parser.add_argument(
        "--belief-deadband-raw",
        type=int,
        default=int(auto.get("belief_deadband_raw", 4)),
    )
    parser.add_argument(
        "--belief-min-step-raw",
        type=int,
        default=int(auto.get("belief_min_step_raw", 3)),
    )
    parser.add_argument(
        "--pitch-tracking-scale",
        type=float,
        default=float(auto.get("pitch_tracking_scale", 1.0)),
    )
    parser.add_argument(
        "--belief-reseed-distance-raw",
        type=float,
        default=float(auto.get("belief_reseed_distance_raw", 160)),
    )
    parser.add_argument(
        "--belief-velocity-alpha",
        type=float,
        default=float(auto.get("belief_velocity_alpha", 0.45)),
    )
    parser.add_argument(
        "--belief-pitch-velocity-alpha",
        type=float,
        default=auto.get("belief_pitch_velocity_alpha"),
    )
    parser.add_argument(
        "--belief-velocity-decay",
        type=float,
        default=float(auto.get("belief_velocity_decay", 0.96)),
    )
    parser.add_argument(
        "--belief-max-velocity-raw-per-s",
        type=float,
        default=float(auto.get("belief_max_velocity_raw_per_s", 600)),
    )
    parser.add_argument(
        "--belief-max-pitch-velocity-raw-per-s",
        type=float,
        default=auto.get("belief_max_pitch_velocity_raw_per_s"),
    )
    parser.add_argument(
        "--belief-max-prediction-age",
        type=float,
        default=float(auto.get("belief_max_prediction_age", 0.45)),
    )
    parser.add_argument(
        "--belief-reseed-confirmations",
        type=int,
        default=int(auto.get("belief_reseed_confirmations", 2)),
    )
    parser.add_argument(
        "--belief-reseed-match-distance-raw",
        type=float,
        default=float(auto.get("belief_reseed_match_distance_raw", 120)),
    )
    parser.add_argument(
        "--belief-reseed-max-interval",
        type=float,
        default=float(auto.get("belief_reseed_max_interval", 0.8)),
    )
    parser.add_argument(
        "--belief-reseed-min-confidence",
        type=float,
        default=float(auto.get("belief_reseed_min_confidence", 0.55)),
    )
    parser.add_argument(
        "--calibration", type=str, default="camera_calibration.npz"
    )
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--baseline-override", type=float, default=None)
    parser.add_argument(
        "--world-tracking",
        action="store_true",
        default=_strict_bool(world, "enabled"),
    )
    parser.add_argument("--enable-trigger", action="store_true")
    parser.add_argument("--api-host", type=str, default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8000)
    for action in parser._actions:
        if action.dest in help_by_dest:
            action.help = help_by_dest[action.dest]
    return parser


def parse_runtime_config(
    *,
    argv: Sequence[str] | None,
    raw_config: Mapping[str, Any] | None,
    inference_fps_default: float,
    path_exists: Callable[[str], bool],
) -> ApplicationRuntimeConfig:
    parser = build_argument_parser(
        raw_config,
        inference_fps_default=inference_fps_default,
    )
    args = parser.parse_args(argv)
    detection, _ = _sections(raw_config)
    camera = _camera_section(raw_config)
    world = _world_section(raw_config)

    target_classes = args.target_class
    if target_classes is None:
        configured_targets = detection.get("target_classes")
        if isinstance(configured_targets, str):
            target_classes = [configured_targets]
        elif configured_targets:
            target_classes = list(configured_targets)
        else:
            target_classes = []

    if args.video_only:
        args.enable_camera = True
        args.use_csi = True
        args.disable_servos = True
        args.no_connect = True
        args.disable_detection = True

    calibration = args.calibration
    if (
        args.stereo
        and calibration == "camera_calibration.npz"
        and path_exists(DEFAULT_STEREO_CALIBRATION)
    ):
        calibration = DEFAULT_STEREO_CALIBRATION

    model_path = None
    if args.enable_camera and not args.disable_detection:
        model_path = args.model

    tracker = TrackerRuntimeConfig(
        port=args.port,
        enable_servos=not args.disable_servos,
        no_connect=args.no_connect,
        enable_camera=args.enable_camera,
        enable_trigger=args.enable_trigger,
        model_path=model_path,
        requested_model=args.model,
        disable_detection=args.disable_detection,
        confidence_threshold=args.confidence,
        camera_id=args.camera_id,
        use_csi=args.use_csi,
        invert_camera=args.invert_camera,
        camera_width=int(camera.get("width", 640)),
        camera_height=int(camera.get("height", 480)),
        camera_fov_horizontal=float(camera.get("fov_horizontal", 60.0)),
        camera_fov_vertical=float(camera.get("fov_vertical", 45.0)),
        imgsz=args.imgsz,
        inference_fps=args.inference_fps,
        target_classes=tuple(target_classes),
        calibration_file=calibration,
        stereo_mode=args.stereo or args.world_tracking,
        baseline_override=args.baseline_override,
        tracking_smoothing=args.tracking_smoothing,
        max_yaw_step=args.max_yaw_step,
        max_pitch_step=args.max_pitch_step,
        max_yaw_speed_raw_per_s=args.max_yaw_speed_raw_per_s,
        max_pitch_speed_raw_per_s=args.max_pitch_speed_raw_per_s,
        tracking_control_fps=args.tracking_control_fps,
        belief_update_alpha=args.belief_update_alpha,
        belief_miss_decay=args.belief_miss_decay,
        belief_min_confidence=args.belief_min_confidence,
        belief_max_age=args.belief_max_age,
        belief_deadband_raw=args.belief_deadband_raw,
        belief_min_step_raw=args.belief_min_step_raw,
        pitch_tracking_scale=args.pitch_tracking_scale,
        belief_reseed_distance_raw=args.belief_reseed_distance_raw,
        belief_velocity_alpha=args.belief_velocity_alpha,
        belief_velocity_decay=args.belief_velocity_decay,
        belief_max_velocity_raw_per_s=args.belief_max_velocity_raw_per_s,
        belief_max_prediction_age=args.belief_max_prediction_age,
        belief_reseed_confirmations=args.belief_reseed_confirmations,
        belief_reseed_match_distance_raw=args.belief_reseed_match_distance_raw,
        belief_reseed_max_interval=args.belief_reseed_max_interval,
        belief_reseed_min_confidence=args.belief_reseed_min_confidence,
        belief_pitch_update_alpha=args.belief_pitch_update_alpha,
        belief_pitch_velocity_alpha=args.belief_pitch_velocity_alpha,
        belief_max_pitch_velocity_raw_per_s=(
            args.belief_max_pitch_velocity_raw_per_s
        ),
        world_tracking=args.world_tracking,
        world_gate_distance_mm=float(world.get("gate_distance_mm", 750.0)),
        world_confirm_hits=int(world.get("confirm_hits", 3)),
        world_max_misses=int(world.get("max_misses", 5)),
        world_delete_after_seconds=float(world.get("delete_after_seconds", 1.5)),
        world_process_acceleration_std_mm_s2=float(
            world.get("process_acceleration_std_mm_s2", 300.0)
        ),
        world_min_depth_confidence=float(
            world.get("min_depth_confidence", 0.2)
        ),
        world_aim_latency_seconds=float(world.get("aim_latency_seconds", 0.12)),
        world_yaw_center_raw=world.get("yaw_center_raw"),
        world_pitch_center_raw=world.get("pitch_center_raw"),
        world_yaw_raw_per_degree=world.get("yaw_raw_per_degree"),
        world_pitch_raw_per_degree=world.get("pitch_raw_per_degree"),
        world_yaw_sign=float(world.get("yaw_sign", 1.0)),
        world_pitch_sign=float(world.get("pitch_sign", -1.0)),
        world_camera_translation_mm=_triple(world.get("camera_translation_mm")),
        world_camera_mount_rpy_degrees=_triple(
            world.get("camera_mount_rpy_degrees")
        ),
        world_log_path=world.get("log_path"),
        world_recordings_dir=str(world.get("recordings_dir", "run_logs/tracks")),
        world_api_recording_enabled=_strict_bool(
            world, "allow_remote_recording"
        ),
        world_api_selection_enabled=_strict_bool(
            world, "allow_remote_selection"
        ),
        world_actuation_enabled=_strict_bool(world, "actuation_enabled"),
        world_calibration_validated=_strict_bool(
            world, "calibration_validated"
        ),
    )
    return ApplicationRuntimeConfig(
        tracker=tracker,
        api_host=args.api_host,
        api_port=args.api_port,
    )
