"""Angular belief observation glue, conversions, and the tracking-control thread."""

from ratbot.app.deps import *  # noqa: F401,F403
from ratbot.app.config_loader import *  # noqa: F401,F403


class BeliefGlueMixin:
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
    def tracking_control_thread(self):
        """Servo control thread that continuously moves toward angular target belief."""
        self.tracking_controller.run()
    def start_tracking_control_thread(self):
        """Start the angular target belief servo control thread"""
        self.tracking_controller.start()
