"""Tests for the isolated camera source adapter."""

from __future__ import annotations

import threading
import time
import unittest

import cv2
import numpy as np

from ratbot.vision.camera_source import CameraSource


class FakeCapture:
    def __init__(self, frame=None, opened=True):
        self.frame = frame
        self.opened = opened
        self.settings = []
        self.released = False

    def set(self, key, value):
        self.settings.append((key, value))
        return True

    def isOpened(self):
        return self.opened

    def read(self):
        if self.frame is None:
            return False, None
        return True, self.frame.copy()

    def release(self):
        self.released = True


class ReadMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0


class MonitoredCapture(FakeCapture):
    def __init__(self, frame, monitor):
        super().__init__(frame)
        self.monitor = monitor

    def read(self):
        with self.monitor.lock:
            self.monitor.active += 1
            self.monitor.max_active = max(
                self.monitor.max_active,
                self.monitor.active,
            )
        time.sleep(0.02)
        with self.monitor.lock:
            self.monitor.active -= 1
        return super().read()


class CameraSourceTests(unittest.TestCase):
    def test_gstreamer_pipeline_includes_sensor_geometry_and_flip(self):
        pipeline = CameraSource.gstreamer_pipeline(
            sensor_id=1,
            capture_width=1280,
            capture_height=720,
            display_width=640,
            display_height=480,
            framerate=30,
            flip_method=2,
        )

        self.assertIn("sensor-id=1", pipeline)
        self.assertIn("width=(int)1280", pipeline)
        self.assertIn("height=(int)720", pipeline)
        self.assertIn("framerate=(fraction)30/1", pipeline)
        self.assertIn("flip-method=2", pipeline)

    def test_usb_initialization_configures_both_cameras_for_stereo(self):
        captures = {2: FakeCapture(), 3: FakeCapture()}
        source = CameraSource(
            enabled=True,
            camera_id=2,
            use_csi=False,
            stereo_mode=True,
            invert_camera=False,
            video_fps=30,
            capture_factory=lambda camera_id, *args: captures[camera_id],
        )

        source.initialize()

        self.assertTrue(source.active)
        self.assertIs(source.left, captures[2])
        self.assertIs(source.right, captures[3])
        self.assertEqual(len(captures[2].settings), 4)
        self.assertEqual(len(captures[3].settings), 4)

    def test_failed_camera_open_leaves_source_inactive(self):
        failed_capture = FakeCapture(opened=False)
        source = CameraSource(
            enabled=True,
            camera_id=0,
            use_csi=False,
            stereo_mode=False,
            invert_camera=False,
            video_fps=30,
            capture_factory=lambda camera_id, *args: failed_capture,
        )

        source.initialize()

        self.assertFalse(source.active)
        self.assertTrue(failed_capture.released)
        self.assertIsNone(source.left)
        self.assertIsNone(source.right)

    def test_complete_frame_pair_reads_are_serialized(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        monitor = ReadMonitor()
        source = CameraSource(
            enabled=True,
            camera_id=0,
            use_csi=False,
            stereo_mode=False,
            invert_camera=False,
            video_fps=30,
        )
        source.left = MonitoredCapture(frame, monitor)
        source.active = True
        start = threading.Barrier(3)

        def read_once():
            start.wait()
            source.read_frames()

        readers = [threading.Thread(target=read_once) for _ in range(2)]
        for reader in readers:
            reader.start()
        start.wait()
        for reader in readers:
            reader.join(timeout=1)

        self.assertEqual(monitor.max_active, 1)

    def test_read_frames_rotates_and_resizes_usb_frames(self):
        left = np.zeros((2, 3, 3), dtype=np.uint8)
        left[0, 0] = (1, 2, 3)
        right = np.zeros((2, 3, 3), dtype=np.uint8)
        right[0, 0] = (4, 5, 6)
        source = CameraSource(
            enabled=True,
            camera_id=0,
            use_csi=False,
            stereo_mode=True,
            invert_camera=True,
            video_fps=30,
        )
        source.left = FakeCapture(left)
        source.right = FakeCapture(right)
        source.active = True

        actual_left, actual_right = source.read_frames()

        self.assertEqual(actual_left.shape, (480, 640, 3))
        self.assertEqual(actual_right.shape, (480, 640, 3))
        expected_left = cv2.resize(cv2.rotate(left, cv2.ROTATE_180), (640, 480))
        expected_right = cv2.resize(cv2.rotate(right, cv2.ROTATE_180), (640, 480))
        np.testing.assert_array_equal(actual_left, expected_left)
        np.testing.assert_array_equal(actual_right, expected_right)

    def test_right_read_failure_keeps_left_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        source = CameraSource(
            enabled=True,
            camera_id=0,
            use_csi=False,
            stereo_mode=True,
            invert_camera=False,
            video_fps=30,
        )
        source.left = FakeCapture(frame)
        source.right = FakeCapture(None)
        source.active = True

        actual_left, actual_right = source.read_frames()

        self.assertIsNotNone(actual_left)
        self.assertIsNone(actual_right)

    def test_close_releases_both_cameras_and_deactivates_source(self):
        left = FakeCapture()
        right = FakeCapture()
        source = CameraSource(
            enabled=True,
            camera_id=0,
            use_csi=False,
            stereo_mode=True,
            invert_camera=False,
            video_fps=30,
        )
        source.left = left
        source.right = right
        source.active = True

        source.close()

        self.assertTrue(left.released)
        self.assertTrue(right.released)
        self.assertFalse(source.active)
        self.assertIsNone(source.left)
        self.assertIsNone(source.right)


if __name__ == "__main__":
    unittest.main()
