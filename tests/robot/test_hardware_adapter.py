"""TrackingServoController: sign decode, motion shaping, state reads, clamping."""
import unittest

from ratbot.robot.belief import ServoBounds
from ratbot.robot.hardware import TrackingServoController


class FakeBus:
    def __init__(self, reads=None, fail_registers=()):
        self.writes = []
        self.reads = dict(reads or {})
        self.fail_registers = set(fail_registers)

    def write(self, register, motor, value, normalize=False):
        if register in self.fail_registers:
            raise RuntimeError(f"register {register} unsupported")
        self.writes.append((register, motor, value))

    def read(self, register, motor, normalize=False):
        if register in self.fail_registers:
            raise RuntimeError(f"register {register} unsupported")
        return self.reads[(register, motor)]


def make_controller(bus=None, connected=True):
    c = TrackingServoController(
        port="/dev/null", enabled=True,
        bounds=ServoBounds(1600, 3100, 1, 500),
        yaw_center=2200, pitch_center=250,
        yaw_motor_id=1, pitch_motor_id=5,
    )
    c.motor_bus = bus
    c.connected = connected and bus is not None
    return c


class SignDecodeTests(unittest.TestCase):
    def test_positive_passthrough(self):
        self.assertEqual(TrackingServoController._decode_feetech_signed(500), 500)

    def test_sign_bit_means_negative(self):
        self.assertEqual(TrackingServoController._decode_feetech_signed(0x8000 | 500), -500)

    def test_zero(self):
        self.assertEqual(TrackingServoController._decode_feetech_signed(0), 0)


class ConfigureMotionTests(unittest.TestCase):
    def test_writes_both_motors_and_skips_zeroes(self):
        bus = FakeBus()
        c = make_controller(bus)
        c.configure_motion(acceleration=30, max_speed=0)
        self.assertEqual(
            bus.writes,
            [("Acceleration", "yaw", 30), ("Acceleration", "pitch", 30)],
        )

    def test_unsupported_register_does_not_raise(self):
        bus = FakeBus(fail_registers={"Goal_Velocity"})
        c = make_controller(bus)
        c.configure_motion(acceleration=30, max_speed=2000)  # must not raise
        self.assertIn(("Acceleration", "yaw", 30), bus.writes)

    def test_disconnected_is_noop(self):
        c = make_controller(bus=None, connected=False)
        c.configure_motion(acceleration=30, max_speed=100)  # must not raise


class ReadStateTests(unittest.TestCase):
    def test_reads_positions_and_signed_velocities(self):
        bus = FakeBus(reads={
            ("Present_Position", "yaw"): 2300,
            ("Present_Position", "pitch"): 260,
            ("Present_Velocity", "yaw"): 0x8000 | 120,
            ("Present_Velocity", "pitch"): 40,
        })
        c = make_controller(bus)
        self.assertEqual(c.read_state(), (2300, 260, -120.0, 40.0))

    def test_velocity_read_failure_degrades_to_zero(self):
        bus = FakeBus(
            reads={("Present_Position", "yaw"): 2300, ("Present_Position", "pitch"): 260},
            fail_registers={"Present_Velocity"},
        )
        c = make_controller(bus)
        self.assertEqual(c.read_state(), (2300, 260, 0.0, 0.0))

    def test_disconnected_returns_current_state(self):
        c = make_controller(bus=None, connected=False)
        self.assertEqual(c.read_state(), (2200, 250, 0.0, 0.0))


class SetPositionTests(unittest.TestCase):
    def test_clamps_to_bounds_and_tracks_commanded(self):
        bus = FakeBus()
        c = make_controller(bus)
        c.set_yaw(99999)
        c.set_pitch(-50)
        self.assertEqual(c.commanded_yaw, 3100)
        self.assertEqual(c.commanded_pitch, 1)
        self.assertIn(("Goal_Position", "yaw", 3100), bus.writes)
        self.assertIn(("Goal_Position", "pitch", 1), bus.writes)


if __name__ == "__main__":
    unittest.main()
