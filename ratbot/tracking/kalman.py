"""Constant-velocity 3D Kalman filtering for target tracks."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


class ConstantVelocityKalman3D:
    """Six-state ``[x,y,z,vx,vy,vz]`` Kalman filter in millimetres."""

    def __init__(
        self,
        position_mm: Iterable[float],
        measurement_covariance: np.ndarray,
        timestamp: float,
        *,
        process_acceleration_std_mm_s2: float = 250.0,
        initial_velocity_std_mm_s: float = 1000.0,
    ):
        position = np.asarray(tuple(position_mm), dtype=float)
        covariance = self._validated_covariance(measurement_covariance)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must contain three finite values")
        if not math.isfinite(float(timestamp)):
            raise ValueError("timestamp must be finite")
        if process_acceleration_std_mm_s2 < 0 or initial_velocity_std_mm_s < 0:
            raise ValueError("noise standard deviations cannot be negative")

        self.state = np.zeros(6, dtype=float)
        self.state[:3] = position
        self.covariance = np.zeros((6, 6), dtype=float)
        self.covariance[:3, :3] = covariance
        self.covariance[3:, 3:] = np.eye(3) * float(initial_velocity_std_mm_s) ** 2
        self.timestamp = float(timestamp)
        self.process_acceleration_std_mm_s2 = float(process_acceleration_std_mm_s2)

    @staticmethod
    def _validated_covariance(covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("measurement covariance must be a finite 3x3 matrix")
        covariance = (covariance + covariance.T) / 2.0
        if np.min(np.linalg.eigvalsh(covariance)) <= 0:
            raise ValueError("measurement covariance must be positive definite")
        return covariance

    @property
    def position(self) -> np.ndarray:
        return self.state[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:]

    def _transition(self, dt: float) -> np.ndarray:
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        return transition

    def _process_covariance(self, dt: float) -> np.ndarray:
        variance = self.process_acceleration_std_mm_s2 ** 2
        q = np.zeros((6, 6), dtype=float)
        q[:3, :3] = np.eye(3) * (dt ** 4 / 4.0) * variance
        q[:3, 3:] = np.eye(3) * (dt ** 3 / 2.0) * variance
        q[3:, :3] = q[:3, 3:]
        q[3:, 3:] = np.eye(3) * (dt ** 2) * variance
        return q

    def predict_to(self, timestamp: float) -> np.ndarray:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        dt = timestamp - self.timestamp
        if dt <= 0:
            return self.state
        transition = self._transition(dt)
        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T
            + self._process_covariance(dt)
        )
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self.timestamp = timestamp
        return self.state

    def predicted_state(self, timestamp: float) -> np.ndarray:
        timestamp = float(timestamp)
        dt = max(0.0, timestamp - self.timestamp)
        return self._transition(dt) @ self.state

    def predicted_covariance(self, timestamp: float) -> np.ndarray:
        timestamp = float(timestamp)
        dt = max(0.0, timestamp - self.timestamp)
        transition = self._transition(dt)
        return transition @ self.covariance @ transition.T + self._process_covariance(dt)

    def update(self, position_mm: Iterable[float], measurement_covariance: np.ndarray) -> np.ndarray:
        measurement = np.asarray(tuple(position_mm), dtype=float)
        covariance = self._validated_covariance(measurement_covariance)
        if measurement.shape != (3,) or not np.all(np.isfinite(measurement)):
            raise ValueError("measurement must contain three finite values")

        observation = np.zeros((3, 6), dtype=float)
        observation[:, :3] = np.eye(3)
        residual = measurement - observation @ self.state
        innovation = observation @ self.covariance @ observation.T + covariance
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation)
        self.state = self.state + gain @ residual
        identity = np.eye(6)
        correction = identity - gain @ observation
        self.covariance = (
            correction @ self.covariance @ correction.T
            + gain @ covariance @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        return self.state
