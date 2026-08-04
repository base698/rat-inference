#!/usr/bin/env python3
"""
Real-time camera tracking with Feetech servo control and rat detection
Controls yaw (ID 1, 1600-3100 raw) and pitch (ID 5, 1-500 raw) servos
Includes Raspberry Pi camera streaming, YOLO inference, and GPIO trigger servo
"""

from ratbot.app.deps import *  # noqa: F401,F403
from ratbot.app.config_loader import *  # noqa: F401,F403
from ratbot.app.server import app, control_api  # noqa: F401
from ratbot.app.tracker import CameraTracker  # noqa: F401

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
