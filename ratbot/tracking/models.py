"""Public data models for fixed-frame multi-target tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Detection3D:
    position_base_mm: np.ndarray
    covariance_base: np.ndarray
    confidence: float
    classification: Optional[str]
    measurement_time: float
    bbox: Optional[tuple[int, int, int, int]] = None
    center: Optional[tuple[int, int]] = None
    depth_mm: Optional[float] = None
    depth_confidence: Optional[float] = None

    def __post_init__(self):
        position = np.asarray(self.position_base_mm, dtype=float)
        covariance = np.asarray(self.covariance_base, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_base_mm must contain three finite values")
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("covariance_base must be a finite 3x3 matrix")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between zero and one")
        object.__setattr__(self, "position_base_mm", position.copy())
        object.__setattr__(self, "covariance_base", covariance.copy())


@dataclass(frozen=True)
class TargetTrackSnapshot:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    confidence: float
    status: str
    classification: Optional[str]
    first_seen_time: float
    last_seen_time: float
    hits: int
    consecutive_hits: int
    misses: int
    selected: bool
    bbox: Optional[tuple[int, int, int, int]] = None
    center: Optional[tuple[int, int]] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "position_mm": self.position.tolist(),
            "velocity_mm_s": self.velocity.tolist(),
            "covariance": self.covariance.tolist(),
            "confidence": self.confidence,
            "status": self.status,
            "classification": self.classification,
            "first_seen_time": self.first_seen_time,
            "last_seen_time": self.last_seen_time,
            "hits": self.hits,
            "consecutive_hits": self.consecutive_hits,
            "misses": self.misses,
            "selected": self.selected,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "center": list(self.center) if self.center is not None else None,
        }
