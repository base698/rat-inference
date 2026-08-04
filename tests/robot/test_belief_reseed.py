"""AngularTargetBelief: alpha blending, far-jump reseed gate, miss decay.

These gates caused real field behavior (staccato lateral tracking) — the
tests pin the mechanics so tuning stays intentional.
"""
import unittest
from unittest.mock import patch

from ratbot.robot.belief import AngularTargetBelief


def make(**kw):
    defaults = dict(
        update_alpha=0.5, miss_decay=0.9, min_confidence=0.15, max_age=1.5,
        reseed_distance_raw=160, reseed_confirmations=2,
        reseed_match_distance_raw=120, reseed_max_interval=0.8,
        reseed_min_confidence=0.55, velocity_alpha=0.45,
        max_velocity_raw_per_s=600, max_prediction_age=0.45,
    )
    defaults.update(kw)
    return AngularTargetBelief(**defaults)


class BlendingTests(unittest.TestCase):
    def test_small_updates_blend_with_alpha(self):
        b = make()
        with patch("ratbot.robot.belief.time.time", return_value=100.0):
            b.update(2200, 250, confidence=1.0)   # seeds
        with patch("ratbot.robot.belief.time.time", return_value=100.1):
            b.update(2300, 250, confidence=1.0)   # +100 raw, within reseed gate
        # alpha 0.5 -> belief lands halfway-ish (velocity prediction adds a bit)
        self.assertGreater(b.yaw, 2200)
        self.assertLess(b.yaw, 2300)


class ReseedGateTests(unittest.TestCase):
    def test_far_jump_is_not_applied_immediately(self):
        b = make()
        with patch("ratbot.robot.belief.time.time", return_value=100.0):
            b.update(2200, 250, confidence=1.0)
        with patch("ratbot.robot.belief.time.time", return_value=100.1):
            b.update(2600, 250, confidence=1.0)   # 400 raw jump > 160 gate
        self.assertLess(abs(b.yaw - 2200), 40)    # belief held near original

    def test_two_matching_far_detections_reseed(self):
        b = make()
        with patch("ratbot.robot.belief.time.time", return_value=100.0):
            b.update(2200, 250, confidence=1.0)
        with patch("ratbot.robot.belief.time.time", return_value=100.1):
            b.update(2600, 250, confidence=1.0)   # pending reseed 1
        with patch("ratbot.robot.belief.time.time", return_value=100.2):
            b.update(2610, 250, confidence=1.0)   # matches within 120 -> reseed
        self.assertGreater(b.yaw, 2500)

    def test_low_confidence_far_jump_ignored(self):
        b = make()
        with patch("ratbot.robot.belief.time.time", return_value=100.0):
            b.update(2200, 250, confidence=1.0)
        for t in (100.1, 100.2, 100.3):
            with patch("ratbot.robot.belief.time.time", return_value=t):
                b.update(2600, 250, confidence=0.3)   # below reseed_min_confidence
        self.assertLess(abs(b.yaw - 2200), 40)


class DecayTests(unittest.TestCase):
    def test_get_active_none_after_max_age(self):
        b = make(max_age=1.0)
        with patch("ratbot.robot.belief.time.time", return_value=100.0):
            b.update(2200, 250, confidence=1.0)
        with patch("ratbot.robot.belief.time.time", return_value=100.2):
            self.assertIsNotNone(b.get_active())
        with patch("ratbot.robot.belief.time.time", return_value=102.0):
            self.assertIsNone(b.get_active())


if __name__ == "__main__":
    unittest.main()
