"""Tests for the isolated video overlay renderer."""

from __future__ import annotations

import unittest

import numpy as np

from ratbot.vision.overlay import OverlayRenderer


class FakeStereoDepth:
    def __init__(self, depth_mm=None, enabled=False):
        self.depth_mm = depth_mm
        self.stereo_calibration_enabled = enabled
        self.K1 = np.array([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0], [0.0, 0.0, 1.0]])
        self.camera_matrix = None
        self.last_depth_debug = "fake failure"
        self.calls = []

    def calculate_depth(self, left, right, x, y):
        self.calls.append((x, y))
        return self.depth_mm


class FakeAiming:
    def __init__(self, depth_adjust=None):
        self.depth_adjust = depth_adjust
        self.calls = []

    def depth_adjust_px(self, depth_mm, focal_y):
        self.calls.append((depth_mm, focal_y))
        return self.depth_adjust


class OverlayRendererTests(unittest.TestCase):
    def make_renderer(self, stereo=None, aiming=None, alpha=0.5, decay=0.5):
        return OverlayRenderer(
            stereo_depth=stereo or FakeStereoDepth(),
            aiming=aiming or FakeAiming(),
            crosshair_x=lambda yaw: 100,
            crosshair_y=lambda pitch, **kwargs: 120,
            crosshair_size=10,
            depth_adjust_smoothing_alpha=alpha,
            depth_adjust_missing_decay=decay,
        )

    def test_depth_adjustment_smoothing_initializes_interpolates_and_decays(self):
        renderer = self.make_renderer(alpha=0.5, decay=0.5)

        self.assertEqual(renderer.smooth_depth_adjust(20), 20)
        self.assertEqual(renderer.smooth_depth_adjust(10), 15)
        self.assertEqual(renderer.smooth_depth_adjust(None), 7.5)

    def test_missing_adjustment_eventually_resets_renderer_state(self):
        renderer = self.make_renderer(decay=0.01)
        renderer.smooth_depth_adjust(10)

        value = renderer.smooth_depth_adjust(None)

        self.assertEqual(value, 0.0)
        self.assertFalse(renderer.depth_adjust_initialized)

    def test_render_draws_crosshair_bbox_center_and_target_line(self):
        renderer = self.make_renderer()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        actual = renderer.render(
            frame,
            frame_right=None,
            current_yaw=2000,
            current_pitch=250,
            stereo_mode=False,
            bbox=(10, 20, 50, 70),
            center_point=(30, 45),
        )

        self.assertIs(actual, frame)
        self.assertTrue(np.any(frame[20:71, 10:51]))
        self.assertTrue(np.any(frame[110:131, 90:111]))

    def test_render_uses_depth_and_focal_length_to_shift_crosshair(self):
        stereo = FakeStereoDepth(depth_mm=1000.0, enabled=True)
        aiming = FakeAiming(depth_adjust=12.0)
        renderer = self.make_renderer(stereo=stereo, aiming=aiming)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        right = frame.copy()

        renderer.render(
            frame,
            frame_right=right,
            current_yaw=2000,
            current_pitch=250,
            stereo_mode=True,
            bbox=None,
            center_point=None,
        )

        self.assertEqual(stereo.calls, [(100, 120)])
        self.assertEqual(aiming.calls, [(1000.0, 200.0)])
        self.assertTrue(np.any(frame[131:134, 90:111]))

    def test_render_reports_failed_depth_without_adjusting_crosshair(self):
        stereo = FakeStereoDepth(depth_mm=None, enabled=True)
        renderer = self.make_renderer(stereo=stereo)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        renderer.render(
            frame,
            frame_right=frame.copy(),
            current_yaw=2000,
            current_pitch=250,
            stereo_mode=True,
            bbox=None,
            center_point=None,
        )

        self.assertEqual(renderer.display_depth_adjust_px, 0.0)
        self.assertTrue(np.any(frame))


if __name__ == "__main__":
    unittest.main()
