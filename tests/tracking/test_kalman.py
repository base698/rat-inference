"""Tests for the constant-velocity 3D Kalman filter."""

from __future__ import annotations

import unittest

import numpy as np

from ratbot.tracking.kalman import ConstantVelocityKalman3D


class ConstantVelocityKalman3DTests(unittest.TestCase):
    def test_initial_state_contains_position_zero_velocity_and_finite_covariance(self):
        kalman = ConstantVelocityKalman3D(
            position_mm=[1000, -200, 50],
            measurement_covariance=np.diag([25, 36, 49]),
            timestamp=10.0,
        )

        np.testing.assert_allclose(kalman.position, [1000, -200, 50])
        np.testing.assert_allclose(kalman.velocity, [0, 0, 0])
        self.assertTrue(np.all(np.isfinite(kalman.covariance)))
        self.assertTrue(np.all(np.diag(kalman.covariance) > 0))

    def test_prediction_advances_position_and_grows_uncertainty(self):
        kalman = ConstantVelocityKalman3D([0, 0, 0], np.eye(3), timestamp=1.0)
        kalman.state[3:] = [100, -50, 25]
        before = np.diag(kalman.covariance).copy()

        kalman.predict_to(3.0)

        np.testing.assert_allclose(kalman.position, [200, -100, 50])
        self.assertTrue(np.all(np.diag(kalman.covariance)[:3] > before[:3]))
        self.assertEqual(kalman.timestamp, 3.0)

    def test_measurement_update_moves_position_and_reduces_uncertainty(self):
        kalman = ConstantVelocityKalman3D([0, 0, 0], np.eye(3) * 100, timestamp=1.0)
        before = np.diag(kalman.covariance)[:3].copy()

        kalman.update([100, 0, 0], np.eye(3) * 4)

        self.assertGreater(kalman.position[0], 90)
        self.assertLess(kalman.position[0], 100)
        self.assertTrue(np.all(np.diag(kalman.covariance)[:3] < before))

    def test_prediction_snapshot_does_not_mutate_authoritative_state(self):
        kalman = ConstantVelocityKalman3D([10, 20, 30], np.eye(3), timestamp=1.0)
        kalman.state[3:] = [1, 2, 3]

        predicted = kalman.predicted_state(4.0)

        np.testing.assert_allclose(predicted[:3], [13, 26, 39])
        np.testing.assert_allclose(kalman.position, [10, 20, 30])
        self.assertEqual(kalman.timestamp, 1.0)

    def test_nonincreasing_timestamp_does_not_run_filter_backward(self):
        kalman = ConstantVelocityKalman3D([0, 0, 0], np.eye(3), timestamp=5.0)
        kalman.state[3:] = [100, 0, 0]

        kalman.predict_to(4.0)

        np.testing.assert_allclose(kalman.position, [0, 0, 0])
        self.assertEqual(kalman.timestamp, 5.0)

    def test_invalid_covariance_is_rejected(self):
        with self.assertRaises(ValueError):
            ConstantVelocityKalman3D([0, 0, 0], np.eye(2), timestamp=0)
        with self.assertRaises(ValueError):
            ConstantVelocityKalman3D([0, 0, 0], np.diag([1, -1, 1]), timestamp=0)


if __name__ == "__main__":
    unittest.main()
