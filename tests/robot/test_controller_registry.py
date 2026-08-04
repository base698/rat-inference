"""Controller registry: config-selected, drop-in tracking controllers."""
import unittest

from ratbot.robot.belief import (
    AngularBeliefController,
    ServoBounds,
    VelocityFormController,
)
from ratbot.robot.controllers import (
    RLControllerStub,
    available_controllers,
    make_tracking_controller,
)


class FakeRobot:
    max_yaw_step = 62
    max_pitch_step = 38
    max_yaw_speed_raw_per_s = None
    max_pitch_speed_raw_per_s = None
    belief_deadband_raw = 8
    belief_min_step_raw = 3
    current_yaw = 2200
    current_pitch = 250
    measured_yaw_velocity = 0.0
    measured_pitch_velocity = 0.0

    def set_yaw(self, position):
        self.last_yaw = position

    def set_pitch(self, position):
        self.last_pitch = position


class FakeBelief:
    def __init__(self, active=None):
        self.active = active

    def get_active(self):
        return self.active


BOUNDS = ServoBounds(1600, 3100, 1, 500)


def build(name, belief=None, options=None):
    return make_tracking_controller(
        name, robot=FakeRobot(), belief=belief or FakeBelief(),
        bounds=BOUNDS, control_fps=20, options=options,
    )


class RegistryTests(unittest.TestCase):
    def test_builtin_names_registered(self):
        self.assertEqual(available_controllers(), ["angular", "rl", "velocity"])

    def test_angular_builds_with_robot_defaults(self):
        c = build("angular")
        self.assertIsInstance(c, AngularBeliefController)
        self.assertEqual(c.max_yaw_step, 62)
        self.assertEqual(c.deadband_raw, 8)

    def test_velocity_builds_with_options(self):
        c = build("velocity", options={"kp_yaw": 9.0})
        self.assertIsInstance(c, VelocityFormController)
        self.assertEqual(c.kp_yaw, 9.0)
        self.assertEqual(c.max_yaw_velocity, 62 * 20)

    def test_unknown_name_raises_with_available_list(self):
        with self.assertRaises(ValueError) as ctx:
            build("quantum")
        self.assertIn("angular", str(ctx.exception))
        self.assertIn("velocity", str(ctx.exception))


class RLStubTests(unittest.TestCase):
    def test_stub_holds_position_on_hardware(self):
        belief = FakeBelief({
            "yaw": 2500.0, "pitch": 350.0, "confidence": 1.0, "age": 0.1,
            "yaw_velocity": 0.0, "pitch_velocity": 0.0,
        })
        c = build("rl", belief=belief)
        self.assertIsInstance(c, RLControllerStub)
        start = c.clock()
        for _ in range(10):
            c.last_time = c.clock() - 0.05
            c.track_once()
        # zero-action policy: commanded position stays anchored at measured
        self.assertLessEqual(abs(c.cmd_yaw - 2200), 1)

    def test_observation_vector_shape_and_content(self):
        belief = {
            "yaw": 2300.0, "pitch": 260.0, "confidence": 0.9, "age": 0.05,
            "yaw_velocity": 40.0, "pitch_velocity": -5.0,
        }
        c = build("rl")
        c.cmd_yaw, c.cmd_pitch = 2200.0, 250.0
        obs = c.build_observation(belief, 12.0, -3.0)
        self.assertEqual(len(obs), 10)
        self.assertEqual(obs[0], 100.0)   # yaw error
        self.assertEqual(obs[1], 10.0)    # pitch error
        self.assertEqual(obs[2], 12.0)    # measured yaw velocity
        self.assertEqual(obs[6], 0.9)     # confidence

    def test_custom_policy_drives_velocity(self):
        class GoRight(RLControllerStub):
            def policy(self, observation):
                return 1.0, 0.0

        belief = FakeBelief({
            "yaw": 2500.0, "pitch": 350.0, "confidence": 1.0, "age": 0.1,
            "yaw_velocity": 0.0, "pitch_velocity": 0.0,
        })
        c = GoRight(robot=FakeRobot(), belief=belief, bounds=BOUNDS,
                    control_fps=20, max_yaw_velocity=1000, max_pitch_velocity=500,
                    max_accel=100000, deadband_raw=0)
        c.last_time = c.clock() - 0.05
        c.track_once()
        self.assertGreater(c.cmd_yaw, 2200)   # policy action moved the goal
        self.assertEqual(c.vel_pitch, 0.0)


if __name__ == "__main__":
    unittest.main()
