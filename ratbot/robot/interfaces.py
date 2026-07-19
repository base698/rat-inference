"""Shared robot-control interfaces used by web and runtime modules."""

from __future__ import annotations

from typing import Mapping, Optional, Protocol, Tuple, Union, runtime_checkable


NumericPosition = Union[int, float]
DetectionData = Mapping[str, object]


@runtime_checkable
class TrackerRobot(Protocol):
    """Minimum robot surface needed by the tracker web controller."""

    connected: bool
    current_yaw: int
    current_pitch: int
    camera_active: bool
    detection_count: int
    trigger_servo_enabled: bool

    def get_detection_data(self) -> DetectionData:
        """Return current detection status for the UI."""

    def get_latest_frame_bytes(self) -> Optional[bytes]:
        """Return the latest JPEG frame bytes, or None if no frame is available."""

    def set_yaw(self, yaw: NumericPosition) -> None:
        """Move the yaw axis to a raw servo position."""

    def set_pitch(self, pitch: NumericPosition) -> None:
        """Move the pitch axis to a raw servo position."""

    def trigger_action_servo(self) -> None:
        """Fire the optional trigger servo."""

    def clear_target_belief(self) -> bool:
        """Clear any active target belief used by autonomous tracking."""

    def get_world_tracks(self) -> list[dict[str, object]]:
        """Return snapshots of all fixed-frame tracks."""

    def select_world_target(self, target_id: int) -> bool:
        """Select a stable fixed-frame target ID, returning false if absent."""

    def clear_world_selection(self) -> bool:
        """Clear fixed-frame target selection without deleting tracks."""

    def move_to_pixel(self, x: NumericPosition, y: NumericPosition) -> Tuple[int, int]:
        """Convert an image-space target into desired yaw and pitch positions."""
