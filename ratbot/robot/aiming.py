"""Crosshair and laser-depth aiming compensation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


PitchPoint = Tuple[float, float]


@dataclass(frozen=True)
class YawCompensation:
    enabled: bool = False
    yaw_min: float = 1600.0
    yaw_max: float = 3100.0
    x_offset_at_min: float = 0.0
    x_offset_at_max: float = 0.0


@dataclass(frozen=True)
class PitchCompensation:
    enabled: bool = False
    pitch_min: float = 100.0
    pitch_max: float = 550.0
    y_offset_at_min: float = 0.0
    y_offset_at_max: float = -120.0
    points: Sequence[PitchPoint] = field(default_factory=tuple)


@dataclass(frozen=True)
class DepthCrosshairCompensation:
    enabled: bool = False
    laser_vertical_offset_mm: float = 55.0
    reference_distance_mm: float = 1000.0
    max_adjust_px: float = 80.0


@dataclass(frozen=True)
class CrosshairAiming:
    x_base: float = 298.0
    y_base: float = 199.0
    yaw: YawCompensation = field(default_factory=YawCompensation)
    pitch: PitchCompensation = field(default_factory=PitchCompensation)
    depth: DepthCrosshairCompensation = field(default_factory=DepthCrosshairCompensation)

    def target_x(self, current_yaw: float) -> int:
        """Calculate the target crosshair X position for the current yaw."""
        if not self.yaw.enabled:
            return int(self.x_base)

        yaw = max(self.yaw.yaw_min, min(self.yaw.yaw_max, current_yaw))
        yaw_range = self.yaw.yaw_max - self.yaw.yaw_min
        t = 0.0 if yaw_range == 0 else (yaw - self.yaw.yaw_min) / yaw_range
        x_offset = self.yaw.x_offset_at_min + t * (
            self.yaw.x_offset_at_max - self.yaw.x_offset_at_min
        )
        return int(self.x_base + x_offset)

    def target_y(
        self,
        current_pitch: float,
        camera_bore_offset_mm: float = 82.0,
        focal_length_px: Optional[float] = None,
        assumed_distance_mm: float = 5000.0,
    ) -> int:
        """Calculate the target crosshair Y position for the current pitch."""
        if not self.pitch.enabled:
            return int(self.y_base)

        pitch_raw = max(self.pitch.pitch_min, min(self.pitch.pitch_max, current_pitch))
        pitch_range = self.pitch.pitch_max - self.pitch.pitch_min
        t = 0.0 if pitch_range == 0 else (pitch_raw - self.pitch.pitch_min) / pitch_range

        if focal_length_px is not None:
            y_offset = (focal_length_px * camera_bore_offset_mm) / assumed_distance_mm
        elif self.pitch.points:
            y_offset = self._interpolate_pitch_points(pitch_raw)
        else:
            y_offset = self.pitch.y_offset_at_min + t * (
                self.pitch.y_offset_at_max - self.pitch.y_offset_at_min
            )

        return int(self.y_base + y_offset)

    def depth_adjust_px(self, depth_mm: float, focal_y_px: Optional[float]) -> Optional[float]:
        """Calculate crosshair Y adjustment from measured depth and laser offset."""
        if (
            not self.depth.enabled
            or focal_y_px is None
            or depth_mm <= 0
            or self.depth.reference_distance_mm <= 0
        ):
            return None

        adjustment = focal_y_px * self.depth.laser_vertical_offset_mm * (
            (1.0 / depth_mm) - (1.0 / self.depth.reference_distance_mm)
        )
        return max(-self.depth.max_adjust_px, min(self.depth.max_adjust_px, adjustment))

    def _interpolate_pitch_points(self, pitch_raw: float) -> float:
        points = tuple(sorted(self.pitch.points, key=lambda point: point[0]))
        if pitch_raw <= points[0][0]:
            return points[0][1]
        if pitch_raw >= points[-1][0]:
            return points[-1][1]

        for (pitch_a, offset_a), (pitch_b, offset_b) in zip(points, points[1:]):
            if pitch_a <= pitch_raw <= pitch_b:
                segment_t = (pitch_raw - pitch_a) / (pitch_b - pitch_a)
                return offset_a + segment_t * (offset_b - offset_a)

        return self.pitch.y_offset_at_min
