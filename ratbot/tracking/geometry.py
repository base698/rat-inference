"""Projection and rigid transforms for the RT-200 turret.

Coordinate conventions:
- OpenCV camera: +X right, +Y down, +Z forward.
- Turret base: +x forward, +y left, +z up.
- Positive physical yaw turns left about base +z.
- Positive physical pitch points upward.  The standard right-handed +y
  rotation points the x axis downward, so upward pitch uses ``Ry(-pitch)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ServoKinematicsConfig:
    yaw_center_raw: float
    pitch_center_raw: float
    yaw_raw_per_degree: float
    pitch_raw_per_degree: float
    yaw_sign: float = 1.0
    pitch_sign: float = -1.0
    camera_translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_mount_rpy_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_min_raw: float = -math.inf
    yaw_max_raw: float = math.inf
    pitch_min_raw: float = -math.inf
    pitch_max_raw: float = math.inf

    def __post_init__(self):
        values = (
            self.yaw_center_raw,
            self.pitch_center_raw,
            self.yaw_raw_per_degree,
            self.pitch_raw_per_degree,
            self.yaw_sign,
            self.pitch_sign,
            *self.camera_translation_mm,
            *self.camera_mount_rpy_degrees,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("kinematics values must be finite")
        if self.yaw_raw_per_degree <= 0 or self.pitch_raw_per_degree <= 0:
            raise ValueError("raw servo units per degree must be positive")
        if self.yaw_sign not in (-1, 1) or self.pitch_sign not in (-1, 1):
            raise ValueError("servo signs must be +1 or -1")
        if self.yaw_min_raw > self.yaw_max_raw or self.pitch_min_raw > self.pitch_max_raw:
            raise ValueError("servo minimum cannot exceed maximum")


class TurretFrameTransformer:
    """Convert stereo points between camera and fixed turret-base frames."""

    _CAMERA_TO_BASE_NEUTRAL = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=float,
    )

    def __init__(self, config: ServoKinematicsConfig):
        self.config = config
        self._camera_translation = np.asarray(config.camera_translation_mm, dtype=float)
        roll, pitch, yaw = map(math.radians, config.camera_mount_rpy_degrees)
        self._mount_rotation = self._rotation_z(yaw) @ self._rotation_y(pitch) @ self._rotation_x(roll)

    @staticmethod
    def _rotation_x(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    def servo_raw_to_physical_degrees(self, yaw_raw: float, pitch_raw: float) -> tuple[float, float]:
        yaw = self.config.yaw_sign * (
            float(yaw_raw) - self.config.yaw_center_raw
        ) / self.config.yaw_raw_per_degree
        pitch = self.config.pitch_sign * (
            float(pitch_raw) - self.config.pitch_center_raw
        ) / self.config.pitch_raw_per_degree
        return yaw, pitch

    def _pan_tilt_rotation(self, yaw_raw: float, pitch_raw: float) -> np.ndarray:
        yaw_degrees, pitch_degrees = self.servo_raw_to_physical_degrees(yaw_raw, pitch_raw)
        return self._rotation_z(math.radians(yaw_degrees)) @ self._rotation_y(
            -math.radians(pitch_degrees)
        )

    def camera_rotation_to_base(self, yaw_raw: float, pitch_raw: float) -> np.ndarray:
        return (
            self._pan_tilt_rotation(yaw_raw, pitch_raw)
            @ self._mount_rotation
            @ self._CAMERA_TO_BASE_NEUTRAL
        )

    def camera_to_base_transform(
        self,
        yaw_raw: float,
        pitch_raw: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return camera-to-base rotation and translated optical-center origin."""
        pan_tilt = self._pan_tilt_rotation(yaw_raw, pitch_raw)
        return (
            self.camera_rotation_to_base(yaw_raw, pitch_raw),
            pan_tilt @ self._camera_translation,
        )

    @staticmethod
    def pixel_depth_to_camera(
        u: float,
        v: float,
        depth_mm: float,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        intrinsics = np.asarray(intrinsics, dtype=float)
        if intrinsics.shape != (3, 3):
            raise ValueError("camera intrinsics must be a 3x3 matrix")
        depth_mm = float(depth_mm)
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        if fx <= 0 or fy <= 0 or not math.isfinite(depth_mm) or depth_mm <= 0:
            raise ValueError("focal lengths and depth must be positive")
        return np.array(
            [
                (float(u) - cx) * depth_mm / fx,
                (float(v) - cy) * depth_mm / fy,
                depth_mm,
            ],
            dtype=float,
        )

    def camera_to_base(
        self,
        point_camera_mm: Iterable[float],
        yaw_raw: float,
        pitch_raw: float,
    ) -> np.ndarray:
        point = np.asarray(tuple(point_camera_mm), dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point_camera_mm must contain three finite values")
        rotation, translation = self.camera_to_base_transform(yaw_raw, pitch_raw)
        return rotation @ point + translation

    def base_to_camera(
        self,
        point_base_mm: Iterable[float],
        yaw_raw: float,
        pitch_raw: float,
    ) -> np.ndarray:
        """Inverse rigid transform, useful for calibration and simulation tests."""
        point = np.asarray(tuple(point_base_mm), dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point_base_mm must contain three finite values")
        rotation, translation = self.camera_to_base_transform(yaw_raw, pitch_raw)
        return rotation.T @ (point - translation)

    def camera_covariance_to_base(
        self,
        covariance_camera: np.ndarray,
        yaw_raw: float,
        pitch_raw: float,
    ) -> np.ndarray:
        covariance = np.asarray(covariance_camera, dtype=float)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("camera covariance must be a finite 3x3 matrix")
        rotation = self.camera_rotation_to_base(yaw_raw, pitch_raw)
        return rotation @ covariance @ rotation.T

    def base_position_to_angles(self, position_base_mm: Iterable[float]) -> tuple[float, float]:
        x, y, z = map(float, position_base_mm)
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise ValueError("base position must be finite")
        horizontal = math.hypot(x, y)
        if horizontal == 0 and z == 0:
            raise ValueError("cannot aim at the turret origin")
        return math.degrees(math.atan2(y, x)), math.degrees(math.atan2(z, horizontal))

    def _raw_from_physical_degrees(
        self,
        yaw_degrees: float,
        pitch_degrees: float,
    ) -> tuple[float, float]:
        yaw_raw = self.config.yaw_center_raw + (
            yaw_degrees * self.config.yaw_raw_per_degree / self.config.yaw_sign
        )
        pitch_raw = self.config.pitch_center_raw + (
            pitch_degrees * self.config.pitch_raw_per_degree / self.config.pitch_sign
        )
        return (
            max(self.config.yaw_min_raw, min(self.config.yaw_max_raw, yaw_raw)),
            max(self.config.pitch_min_raw, min(self.config.pitch_max_raw, pitch_raw)),
        )

    def _optical_axis_residual(
        self,
        position_base_mm: np.ndarray,
        yaw_raw: float,
        pitch_raw: float,
    ) -> np.ndarray:
        camera = self.base_to_camera(position_base_mm, yaw_raw, pitch_raw)
        if float(np.linalg.norm(camera)) <= 1e-9:
            raise ValueError("cannot aim the camera at its optical center")
        # Angular residuals remain well scaled across target distance.
        return np.array(
            [
                math.atan2(float(camera[0]), float(camera[2])),
                math.atan2(float(camera[1]), float(camera[2])),
            ],
            dtype=float,
        )

    def base_position_to_servo_raw(self, position_base_mm: Iterable[float]) -> dict[str, float | int]:
        position = np.asarray(tuple(position_base_mm), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("base position must contain three finite values")
        yaw_degrees, pitch_degrees = self.base_position_to_angles(position)
        yaw_raw, pitch_raw = self._raw_from_physical_degrees(
            yaw_degrees,
            pitch_degrees,
        )

        has_extrinsics = bool(
            np.linalg.norm(self._camera_translation) > 1e-12
            or not np.allclose(self._mount_rotation, np.eye(3), atol=1e-12)
        )
        if has_extrinsics:
            epsilon_raw = 0.5
            converged = False
            for _ in range(20):
                residual = self._optical_axis_residual(position, yaw_raw, pitch_raw)
                if float(np.linalg.norm(residual)) < 1e-7:
                    converged = True
                    break
                jacobian = np.column_stack(
                    [
                        (
                            self._optical_axis_residual(
                                position,
                                yaw_raw + epsilon_raw,
                                pitch_raw,
                            )
                            - self._optical_axis_residual(
                                position,
                                yaw_raw - epsilon_raw,
                                pitch_raw,
                            )
                        )
                        / (2.0 * epsilon_raw),
                        (
                            self._optical_axis_residual(
                                position,
                                yaw_raw,
                                pitch_raw + epsilon_raw,
                            )
                            - self._optical_axis_residual(
                                position,
                                yaw_raw,
                                pitch_raw - epsilon_raw,
                            )
                        )
                        / (2.0 * epsilon_raw),
                    ]
                )
                try:
                    delta = np.linalg.solve(jacobian, -residual)
                except np.linalg.LinAlgError as exc:
                    raise ValueError("camera aiming geometry is singular") from exc
                delta = np.clip(delta, -250.0, 250.0)
                next_yaw = max(
                    self.config.yaw_min_raw,
                    min(self.config.yaw_max_raw, yaw_raw + float(delta[0])),
                )
                next_pitch = max(
                    self.config.pitch_min_raw,
                    min(self.config.pitch_max_raw, pitch_raw + float(delta[1])),
                )
                if next_yaw == yaw_raw and next_pitch == pitch_raw:
                    break
                yaw_raw, pitch_raw = next_yaw, next_pitch
            if not converged:
                residual = self._optical_axis_residual(position, yaw_raw, pitch_raw)
                if float(np.linalg.norm(residual)) >= 1e-4:
                    raise ValueError("target cannot be aligned within servo limits")
            yaw_degrees, pitch_degrees = self.servo_raw_to_physical_degrees(
                yaw_raw,
                pitch_raw,
            )

        return {
            "yaw": int(round(yaw_raw)),
            "pitch": int(round(pitch_raw)),
            "yaw_degrees": yaw_degrees,
            "pitch_degrees": pitch_degrees,
        }
