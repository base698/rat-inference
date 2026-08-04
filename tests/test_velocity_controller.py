"""Unit tests for VelocityFormController (no hardware)."""
import time

from ratbot.robot.belief import ServoBounds, VelocityFormController


class FakeRobot:
    def __init__(self):
        self.current_yaw = 2200
        self.current_pitch = 250
        self.measured_yaw_velocity = 0.0
        self.measured_pitch_velocity = 0.0
        self.camera_active = True
        self.writes = []

    def set_yaw(self, position):
        self.writes.append(("yaw", position))

    def set_pitch(self, position):
        self.writes.append(("pitch", position))


class FakeBelief:
    def __init__(self, yaw=None, pitch=None):
        self.yaw = yaw
        self.pitch = pitch

    def get_active(self):
        if self.yaw is None:
            return None
        return {"yaw": self.yaw, "pitch": self.pitch, "confidence": 1.0}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make(robot, belief, **kw):
    bounds = ServoBounds(1600, 3100, 1, 500)
    defaults = dict(control_fps=20, kp_yaw=6.0, kp_pitch=5.5,
                    max_yaw_velocity=1200, max_pitch_velocity=760,
                    max_accel=3500, deadband_raw=8, reconcile_rate=0.0,
                    clock=FakeClock())
    defaults.update(kw)
    return VelocityFormController(robot, belief, bounds, **defaults)


def run_ticks(ctrl, n, dt=0.05):
    for _ in range(n):
        ctrl.clock.now += dt
        ctrl.track_once()


def test_converges_toward_target_without_overshoot():
    robot = FakeRobot()
    belief = FakeBelief(yaw=2500, pitch=350)
    ctrl = make(robot, belief)
    run_ticks(ctrl, 100)
    assert abs(ctrl.cmd_yaw - 2500) <= 10
    assert abs(ctrl.cmd_pitch - 350) <= 10
    # commanded trajectory never passes the target
    assert max(p for a, p in robot.writes if a == "yaw") <= 2501


def test_acceleration_limit_shapes_first_ticks():
    robot = FakeRobot()
    belief = FakeBelief(yaw=3000, pitch=250)
    ctrl = make(robot, belief, max_accel=1000)
    run_ticks(ctrl, 1)
    # after one 50ms tick velocity can be at most 50 raw/s -> step <= 2.5 raw
    assert abs(ctrl.cmd_yaw - 2200) <= 3
    run_ticks(ctrl, 2)
    assert abs(ctrl.vel_yaw) <= 1000 * 0.05 * 3 + 1e-6


def test_bounds_clamped():
    robot = FakeRobot()
    belief = FakeBelief(yaw=99999, pitch=-50)
    ctrl = make(robot, belief)
    run_ticks(ctrl, 300)
    assert ctrl.cmd_yaw <= 3100
    assert ctrl.cmd_pitch >= 1


def test_idle_resets_anchor():
    robot = FakeRobot()
    belief = FakeBelief(yaw=2500, pitch=350)
    ctrl = make(robot, belief)
    run_ticks(ctrl, 40)
    belief.yaw = None
    run_ticks(ctrl, 1)
    assert ctrl.cmd_yaw is None and ctrl.vel_yaw == 0.0


def test_deadband_holds_still():
    robot = FakeRobot()
    belief = FakeBelief(yaw=2204, pitch=252)  # within deadband of current
    ctrl = make(robot, belief)
    run_ticks(ctrl, 20)
    assert all(abs(p - 2200) <= 1 for a, p in robot.writes if a == "yaw")


def test_reconcile_leak_pulls_commanded_toward_measured():
    robot = FakeRobot()
    belief = FakeBelief(yaw=2500, pitch=350)
    ctrl = make(robot, belief, reconcile_rate=2.0, kp_yaw=0.0, kp_pitch=0.0)
    run_ticks(ctrl, 1)          # anchors cmd at measured (2200)
    ctrl.cmd_yaw = 2300         # simulate accumulated commanded-vs-physical offset
    run_ticks(ctrl, 20)         # 1s of leak at 2/s, no proportional drive
    assert abs(ctrl.cmd_yaw - 2200) < 30   # decayed most of the 100-raw offset


def test_velocity_damping_opposes_measured_motion():
    robot = FakeRobot()
    robot.measured_yaw_velocity = 200.0    # turret already moving +
    belief = FakeBelief(yaw=2200, pitch=250)  # zero error
    ctrl = make(robot, belief, deadband_raw=0, damping_yaw=0.5)
    run_ticks(ctrl, 1)
    assert ctrl.vel_yaw < 0    # damping term pushes against measured velocity
