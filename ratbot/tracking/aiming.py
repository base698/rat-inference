"""Convert a selected world-frame target into the legacy servo-belief surface."""

from __future__ import annotations

import math
import time


class WorldTrackBeliefAdapter:
    """Read selected 3D tracks through ``AngularBeliefController``'s API."""

    def __init__(
        self,
        manager,
        transformer,
        *,
        aim_latency_seconds=0.12,
        min_confidence=0.2,
        max_age_seconds=0.8,
        fresh_center_max_age_seconds=None,
        prefer_fresh_center=False,
        robot=None,
        clock=time.monotonic,
    ):
        self.manager = manager
        self.transformer = transformer
        self.aim_latency_seconds = max(0.0, float(aim_latency_seconds))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_age_seconds = max(0.0, float(max_age_seconds))
        self.fresh_center_max_age_seconds = (
            min(self.max_age_seconds, 0.20)
            if fresh_center_max_age_seconds is None
            else max(0.0, float(fresh_center_max_age_seconds))
        )
        self.prefer_fresh_center = bool(prefer_fresh_center)
        self.robot = robot
        self.clock = clock

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(upper, value))

    def _pitch_tracking_scale(self):
        observation = getattr(self.robot, "observation_converter", None)
        config = getattr(observation, "config", None)
        return float(getattr(config, "pitch_tracking_scale", 1.0))

    def _laser_vertical_offset_mm(self):
        observation = getattr(self.robot, "observation_converter", None)
        aiming = getattr(observation, "aiming", None)
        depth = getattr(aiming, "depth", None)
        if depth is None or not getattr(depth, "enabled", False):
            return 0.0
        return float(getattr(depth, "laser_vertical_offset_mm", 0.0))

    def _optical_axis_aim(self, x, y, z):
        current_yaw = float(self.robot.current_yaw)
        current_pitch = float(self.robot.current_pitch)
        laser_y_offset_mm = self._laser_vertical_offset_mm()
        yaw_error_degrees = math.degrees(math.atan2(x, z))
        pitch_error_degrees = math.degrees(math.atan2(y - laser_y_offset_mm, z))
        yaw_delta_raw = self.robot.angle_to_servo_raw(yaw_error_degrees, axis="yaw")
        pitch_delta_raw = self.robot.angle_to_servo_raw(
            pitch_error_degrees,
            axis="pitch",
        )

        config = self.transformer.config
        yaw = self._clamp(
            current_yaw + yaw_delta_raw,
            config.yaw_min_raw,
            config.yaw_max_raw,
        )
        pitch = self._clamp(
            current_pitch + pitch_delta_raw,
            config.pitch_min_raw,
            config.pitch_max_raw,
        )
        return {
            "yaw": int(round(yaw)),
            "pitch": int(round(pitch)),
            "yaw_degrees": yaw_error_degrees,
            "pitch_degrees": pitch_error_degrees,
            "yaw_delta_raw": yaw_delta_raw,
            "pitch_delta_raw": pitch_delta_raw,
            "aim_source": "world_ray",
            "laser_y_offset_mm": laser_y_offset_mm,
        }

    def _camera_intrinsics(self):
        stereo_depth = getattr(self.robot, "stereo_depth", None)
        intrinsics = getattr(stereo_depth, "K1", None)
        if intrinsics is None:
            intrinsics = getattr(stereo_depth, "camera_matrix", None)
        if intrinsics is not None:
            return (
                float(intrinsics[0, 0]),
                float(intrinsics[1, 1]),
                float(intrinsics[0, 2]),
                float(intrinsics[1, 2]),
            )

        observation = getattr(self.robot, "observation_converter", None)
        config = getattr(observation, "config", None)
        if config is None:
            raise ValueError("camera intrinsics are not available for world aiming")
        fx = (float(config.image_width) / 2.0) / math.tan(
            math.radians(float(config.horizontal_fov_degrees)) / 2.0
        )
        fy = (float(config.image_height) / 2.0) / math.tan(
            math.radians(float(config.vertical_fov_degrees)) / 2.0
        )
        return fx, fy, float(config.image_width) / 2.0, float(config.image_height) / 2.0

    def _project_camera_point(self, x, y, z):
        fx, fy, cx, cy = self._camera_intrinsics()
        return (fx * x / z) + cx, (fy * y / z) + cy

    def _crosshair_relative_aim(self, x, y, z):
        if not hasattr(self.robot, "pixel_to_target_position"):
            return self._optical_axis_aim(x, y, z)
        target_x, target_y = self._project_camera_point(x, y, z)
        return self._image_point_aim(target_x, target_y, depth_mm=z)

    def _image_point_aim(self, target_x, target_y, depth_mm=None):
        observation = self.robot.pixel_to_target_position(
            target_x,
            target_y,
            depth_mm=depth_mm,
        )
        return {
            "yaw": int(round(observation["yaw"])),
            "pitch": int(round(observation["pitch"])),
            "yaw_degrees": float(observation.get("angle_error_yaw", 0.0)),
            "pitch_degrees": float(observation.get("angle_error_pitch", 0.0)),
        }

    def _turret_relative_aim(self, position_base_mm):
        current_yaw = float(self.robot.current_yaw)
        current_pitch = float(self.robot.current_pitch)
        point_camera = self.transformer.base_to_camera(
            position_base_mm,
            current_yaw,
            current_pitch,
        )
        x, y, z = map(float, point_camera)
        if not all(math.isfinite(value) for value in (x, y, z)) or z <= 1e-6:
            raise ValueError("selected world target is outside the current camera frame")

        aim = self._optical_axis_aim(x, y, z)
        aim["point_camera_mm"] = (x, y, z)
        return aim

    def _aim_for_position(self, position_base_mm):
        if self.robot is None:
            return self.transformer.base_position_to_servo_raw(position_base_mm)
        return self._turret_relative_aim(position_base_mm)

    def _aim_for_track(self, track, age):
        if (
            self.prefer_fresh_center
            and self.robot is not None
            and hasattr(self.robot, "pixel_to_target_position")
            and track.center is not None
            and track.misses == 0
            and age <= self.fresh_center_max_age_seconds
        ):
            depth_mm = None
            try:
                point_camera = self.transformer.base_to_camera(
                    track.position,
                    float(self.robot.current_yaw),
                    float(self.robot.current_pitch),
                )
                if float(point_camera[2]) > 1e-6:
                    depth_mm = float(point_camera[2])
            except ValueError:
                depth_mm = None
            return self._image_point_aim(
                track.center[0],
                track.center[1],
                depth_mm=depth_mm,
            )
        return self._aim_for_position(track.position)

    def get_active(self):
        now = float(self.clock())
        if self.robot is None:
            prediction_horizon = self.aim_latency_seconds
            track = self.manager.get_selected_track(
                timestamp=now,
                prediction_horizon=prediction_horizon,
            )
        else:
            # Extrapolate the track to "now + aim latency" so the PID chases
            # where the target IS, not where it was one pipeline-latency ago.
            prediction_horizon = self.aim_latency_seconds
            track = self.manager.get_selected_track(
                timestamp=now,
                prediction_horizon=prediction_horizon,
            )
        if track is None:
            return None
        age = max(0.0, now - track.last_seen_time)
        if (
            track.status not in {"confirmed", "lost"}
            or track.confidence < self.min_confidence
            or (self.max_age_seconds > 0 and age > self.max_age_seconds)
        ):
            return None

        try:
            aim = self._aim_for_track(track, age)
            if self.robot is None:
                base_position = track.position - track.velocity * self.aim_latency_seconds
                base_aim = self._aim_for_position(base_position)
                velocity_dt = 0.01
                future_position = track.position + track.velocity * velocity_dt
                future_aim = self._aim_for_position(future_position)
                yaw_velocity = (future_aim["yaw"] - aim["yaw"]) / velocity_dt
                pitch_velocity = (future_aim["pitch"] - aim["pitch"]) / velocity_dt
            else:
                base_aim = aim
                yaw_velocity = 0.0
                pitch_velocity = 0.0
        except ValueError:
            return None
        return {
            "yaw": aim["yaw"],
            "pitch": aim["pitch"],
            "base_yaw": base_aim["yaw"],
            "base_pitch": base_aim["pitch"],
            "yaw_velocity": yaw_velocity,
            "pitch_velocity": pitch_velocity,
            "prediction_dt": prediction_horizon,
            "confidence": track.confidence,
            "age": age,
            "track_id": track.id,
            "position_mm": track.position.copy(),
            "velocity_mm_s": track.velocity.copy(),
            "aim_source": aim.get("aim_source"),
            "yaw_error_degrees": aim.get("yaw_degrees"),
            "pitch_error_degrees": aim.get("pitch_degrees"),
            "point_camera_mm": aim.get("point_camera_mm"),
            "laser_y_offset_mm": aim.get("laser_y_offset_mm"),
        }
