"""Convert image-space detections into servo-space target observations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationConfig:
    image_width: int
    image_height: int
    horizontal_fov_degrees: float
    vertical_fov_degrees: float
    yaw_min: int
    yaw_max: int
    pitch_min: int
    pitch_max: int
    yaw_range_degrees: float
    pitch_range_degrees: float
    pitch_tracking_scale: float = 1.0


class TrackingObservationConverter:
    """Map a detected image point to the raw servo pose that would center it."""

    def __init__(
        self,
        config: ObservationConfig,
        aiming,
        stereo_depth,
        crosshair_x,
        crosshair_y,
    ):
        self.config = config
        self.aiming = aiming
        self.stereo_depth = stereo_depth
        self.crosshair_x = crosshair_x
        self.crosshair_y = crosshair_y

    @staticmethod
    def pixels_to_angle(pixel_error, image_dimension, fov_degrees):
        """Convert pixel error to angular error in degrees."""
        return pixel_error * (fov_degrees / image_dimension)

    def angle_to_servo_raw(self, angle_delta, axis="yaw"):
        """Convert an angular change to raw servo units."""
        if axis == "yaw":
            raw_range = self.config.yaw_max - self.config.yaw_min
            raw_per_degree = raw_range / self.config.yaw_range_degrees
        else:
            raw_range = self.config.pitch_max - self.config.pitch_min
            raw_per_degree = raw_range / self.config.pitch_range_degrees
        return int(angle_delta * raw_per_degree)

    def servo_raw_to_angle(self, raw_delta, axis="yaw"):
        """Convert a raw servo-unit error to degrees."""
        if axis == "yaw":
            raw_range = self.config.yaw_max - self.config.yaw_min
            raw_per_degree = raw_range / self.config.yaw_range_degrees
        else:
            raw_range = self.config.pitch_max - self.config.pitch_min
            raw_per_degree = raw_range / self.config.pitch_range_degrees
        if raw_per_degree == 0:
            return 0.0
        return raw_delta / raw_per_degree

    def target_crosshair_y(self, current_pitch, depth_mm=None):
        """Return the vertical aiming reference at an optional target depth."""
        ch_y = self.crosshair_y(current_pitch)
        if depth_mm is None:
            return ch_y

        focal_y = None
        if self.stereo_depth.K1 is not None:
            focal_y = float(self.stereo_depth.K1[1, 1])
        elif self.stereo_depth.camera_matrix is not None:
            focal_y = float(self.stereo_depth.camera_matrix[1, 1])

        depth_adjust = self.aiming.depth_adjust_px(depth_mm, focal_y)
        if depth_adjust is None:
            return ch_y
        return int(round(ch_y + depth_adjust))

    def to_servo_target(
        self,
        target_x,
        target_y,
        current_yaw,
        current_pitch,
        depth_mm=None,
    ):
        """Convert a detected point into a bounded raw-servo observation."""
        target_crosshair_x = self.crosshair_x(current_yaw)
        target_crosshair_y = self.target_crosshair_y(current_pitch, depth_mm)
        pixel_offset_x = target_x - target_crosshair_x
        pixel_offset_y = target_y - target_crosshair_y

        angle_offset_yaw = self.pixels_to_angle(
            pixel_offset_x,
            self.config.image_width,
            self.config.horizontal_fov_degrees,
        )
        angle_offset_pitch = self.pixels_to_angle(
            pixel_offset_y,
            self.config.image_height,
            self.config.vertical_fov_degrees,
        )

        yaw_offset_raw = self.angle_to_servo_raw(angle_offset_yaw, axis="yaw")
        pitch_offset_raw = int(
            round(
                self.angle_to_servo_raw(angle_offset_pitch, axis="pitch")
                * self.config.pitch_tracking_scale
            )
        )
        observed_yaw = max(
            self.config.yaw_min,
            min(self.config.yaw_max, current_yaw + yaw_offset_raw),
        )
        observed_pitch = max(
            self.config.pitch_min,
            min(self.config.pitch_max, current_pitch + pitch_offset_raw),
        )

        return {
            "yaw": observed_yaw,
            "pitch": observed_pitch,
            "pixel_error_x": pixel_offset_x,
            "pixel_error_y": pixel_offset_y,
            "angle_error_yaw": angle_offset_yaw,
            "angle_error_pitch": angle_offset_pitch,
            "yaw_offset_raw": yaw_offset_raw,
            "pitch_offset_raw": pitch_offset_raw,
        }
