"""pose_at(): latency-compensated pose lookup (the anti-hunting anchor)."""
from collections import deque

from ratbot.app.video import VideoPipelineMixin


class PoseHost(VideoPipelineMixin):
    def __init__(self):
        self.pose_history = deque(maxlen=90)
        self.current_yaw = 2200
        self.current_pitch = 250


def test_empty_history_falls_back_to_current():
    h = PoseHost()
    assert h.pose_at(123.0) == (2200, 250)


def test_returns_newest_entry_at_or_before_target_time():
    h = PoseHost()
    h.pose_history.extend([(1.0, 2100, 240), (2.0, 2150, 245), (3.0, 2190, 249)])
    assert h.pose_at(2.5) == (2150, 245)
    assert h.pose_at(3.0) == (2190, 249)
    assert h.pose_at(99.0) == (2190, 249)


def test_target_before_all_history_uses_oldest():
    h = PoseHost()
    h.pose_history.extend([(5.0, 2100, 240), (6.0, 2150, 245)])
    assert h.pose_at(1.0) == (2100, 240)
