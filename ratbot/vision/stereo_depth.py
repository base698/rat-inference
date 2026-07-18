"""Camera calibration, rectification, and stereo depth processing."""

from __future__ import annotations

import os

import cv2
import numpy as np


class StereoDepthService:
    """Own camera calibration state and calculate depth from stereo frames."""

    def __init__(self, calibration_file=None, baseline_override=None, min_texture_std=4.0, max_valid_mm=6000.0):
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_enabled = False
        self.stereo_calibration_enabled = False
        self.K1 = None
        self.D1 = None
        self.K2 = None
        self.D2 = None
        self.R = None
        self.T = None
        self.baseline = None
        self.baseline_override = baseline_override
        self.stereo_matcher = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.stereo_map_left = None
        self.stereo_map_right = None
        self.stereo_focal_length = None
        self.stereo_min_disparity = 0
        self.stereo_num_disparities = 192
        self.stereo_disparity_sign = 1
        self.depth_min_texture_std = float(min_texture_std)
        self.depth_max_valid_mm = float(max_valid_mm)
        self.last_depth_debug = "not computed"

        if self.calibration_file:
            self.load_calibration()

    def load_calibration(self):
        """Load camera calibration from file (single or stereo)"""
        try:
            # Check if file exists
            if not os.path.exists(self.calibration_file):
                print(f"⚠ Calibration file not found: {self.calibration_file}")
                self.calibration_enabled = False
                return

            calib_data = np.load(self.calibration_file)

            # Check if it's stereo calibration
            if 'K1' in calib_data and 'K2' in calib_data:
                # Stereo calibration
                self.K1 = calib_data['K1']
                self.D1 = calib_data['D1']
                self.K2 = calib_data['K2']
                self.D2 = calib_data['D2']
                self.R = calib_data['R']
                self.T = calib_data['T']
                self.baseline = calib_data['baseline']
                self.stereo_calibration_enabled = True
                self.calibration_enabled = True

                # Use left camera parameters for single-camera fallback
                self.camera_matrix = self.K1
                self.dist_coeffs = self.D1
                self.init_stereo_rectification()

                # Create stereo matcher for disparity calculation.
                # StereoSGBM is more accurate than StereoBM.
                self.stereo_matcher = cv2.StereoSGBM_create(
                    minDisparity=self.stereo_min_disparity,
                    numDisparities=self.stereo_num_disparities,  # Must be divisible by 16
                    blockSize=5,
                    P1=8 * 3 * 5**2,  # Smoothness penalty
                    P2=32 * 3 * 5**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )

                # Get RMS error from calibration
                rms_error = calib_data.get('rms_error', 0)

                # Apply baseline override if provided
                baseline_from_calib = self.baseline
                if self.baseline_override is not None:
                    self.baseline = self.baseline_override
                    print(f"✓ Stereo calibration loaded: {self.calibration_file}")
                    print(f"  RMS error: {rms_error:.3f} pixels")
                    print(f"  Baseline (calibrated): {baseline_from_calib:.2f} mm")
                    print(f"  Baseline (OVERRIDDEN): {self.baseline:.2f} mm ⚠")
                    print(f"  Left focal length: fx={self.K1[0,0]:.2f}, fy={self.K1[1,1]:.2f}")
                    print(f"  Right focal length: fx={self.K2[0,0]:.2f}, fy={self.K2[1,1]:.2f}")
                else:
                    print(f"✓ Stereo calibration loaded: {self.calibration_file}")
                    print(f"  RMS error: {rms_error:.3f} pixels")
                    print(f"  Baseline: {self.baseline:.2f} mm")
                    print(f"  Left focal length: fx={self.K1[0,0]:.2f}, fy={self.K1[1,1]:.2f}")
                    print(f"  Right focal length: fx={self.K2[0,0]:.2f}, fy={self.K2[1,1]:.2f}")
                if rms_error > 1.0:
                    print(f"  ⚠ Stereo RMS is high ({rms_error:.3f}px); distance may be inaccurate until recalibrated")
            else:
                # Single camera calibration
                self.camera_matrix = calib_data['camera_matrix']
                self.dist_coeffs = calib_data['dist_coeffs']
                self.calibration_enabled = True
                rms_error = calib_data.get('rms_error', 0)
                print(f"✓ Single camera calibration loaded: {self.calibration_file}")
                print(f"  RMS error: {rms_error:.3f} pixels")
                print(f"  Focal length: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
        except Exception as e:
            print(f"⚠ Failed to load calibration: {e}")
            self.calibration_enabled = False
            self.stereo_calibration_enabled = False

    def init_stereo_rectification(self, image_size=(640, 480)):
        """Build rectification maps so stereo disparity is computed on aligned frames."""
        try:
            self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
                self.K1, self.D1,
                self.K2, self.D2,
                image_size,
                self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=-1
            )

            center_point = np.array([[[image_size[0] / 2.0, image_size[1] / 2.0]]], dtype=np.float32)
            rectified_center = cv2.undistortPoints(
                center_point, self.K1, self.D1, R=self.R1, P=self.P1
            )[0, 0]
            rectified_shift_x = (image_size[0] / 2.0) - float(rectified_center[0])
            rectified_shift_y = (image_size[1] / 2.0) - float(rectified_center[1])
            if abs(rectified_shift_x) > 1.0 or abs(rectified_shift_y) > 1.0:
                self.P1[0, 2] += rectified_shift_x
                self.P2[0, 2] += rectified_shift_x
                self.P1[1, 2] += rectified_shift_y
                self.P2[1, 2] += rectified_shift_y

            self.stereo_map_left = cv2.initUndistortRectifyMap(
                self.K1, self.D1, self.R1, self.P1, image_size, cv2.CV_16SC2
            )
            self.stereo_map_right = cv2.initUndistortRectifyMap(
                self.K2, self.D2, self.R2, self.P2, image_size, cv2.CV_16SC2
            )

            self.stereo_focal_length = float(self.P1[0, 0])
            rectified_baseline = abs(float(self.P2[0, 3]) / float(self.P2[0, 0]))
            if self.baseline_override is None:
                self.baseline = rectified_baseline

            # P2[0,3] sign tells us which disparity direction to expect after rectification.
            # Some camera orderings produce negative valid disparities.
            if float(self.P2[0, 3]) > 0:
                self.stereo_disparity_sign = -1
                self.stereo_min_disparity = -192
                self.stereo_num_disparities = 256
            else:
                self.stereo_disparity_sign = 1
                self.stereo_min_disparity = 0
                self.stereo_num_disparities = 192

            print("✓ Stereo rectification initialized")
            print(f"  Rectified focal length: {self.stereo_focal_length:.2f}px")
            print(f"  Rectified baseline: {rectified_baseline:.2f} mm")
            print(f"  Rectified image shift: x={rectified_shift_x:.1f}px, y={rectified_shift_y:.1f}px")
            print(f"  Disparity search: min={self.stereo_min_disparity}, num={self.stereo_num_disparities}, sign={self.stereo_disparity_sign:+d}")
        except Exception as e:
            print(f"⚠ Failed to initialize stereo rectification: {e}")
            self.stereo_calibration_enabled = False
            self.stereo_map_left = None
            self.stereo_map_right = None

    def rectify_stereo_frames(self, frame_left, frame_right):
        """Rectify left/right frames before stereo matching."""
        if self.stereo_map_left is None or self.stereo_map_right is None:
            return frame_left, frame_right

        try:
            left = cv2.remap(
                frame_left,
                self.stereo_map_left[0],
                self.stereo_map_left[1],
                cv2.INTER_LINEAR
            )
            right = cv2.remap(
                frame_right,
                self.stereo_map_right[0],
                self.stereo_map_right[1],
                cv2.INTER_LINEAR
            )
            return left, right
        except Exception as e:
            print(f"Error rectifying stereo frames: {e}")
            return frame_left, frame_right

    def rectify_stereo_point(self, x, y):
        """Map a point from the raw left image into rectified stereo coordinates."""
        if self.R1 is None or self.P1 is None:
            return int(x), int(y)

        try:
            point = np.array([[[float(x), float(y)]]], dtype=np.float32)
            rectified = cv2.undistortPoints(point, self.K1, self.D1, R=self.R1, P=self.P1)
            rx, ry = rectified[0, 0]
            return int(round(rx)), int(round(ry))
        except Exception as e:
            print(f"Error rectifying stereo point: {e}")
            return int(x), int(y)

    def undistort_frame(self, frame, use_left=True):
        """Apply camera calibration to undistort frame"""
        if not self.calibration_enabled:
            return frame

        try:
            if self.stereo_calibration_enabled:
                # Use appropriate camera matrix for stereo
                K = self.K1 if use_left else self.K2
                D = self.D1 if use_left else self.D2
                return cv2.undistort(frame, K, D)
            else:
                return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        except Exception as e:
            print(f"Error undistorting frame: {e}")
            return frame

    def calculate_depth(self, frame_left, frame_right, x, y):
        """
        Calculate depth at a specific pixel location using stereo matching

        Args:
            frame_left: Left camera frame (undistorted)
            frame_right: Right camera frame (undistorted)
            x: X coordinate in image
            y: Y coordinate in image

        Returns:
            Depth in millimeters, or None if calculation fails
        """
        if not self.stereo_calibration_enabled or self.stereo_matcher is None:
            self.last_depth_debug = "stereo disabled"
            return None

        try:
            frame_left, frame_right = self.rectify_stereo_frames(frame_left, frame_right)
            x, y = self.rectify_stereo_point(x, y)

            # Convert to grayscale for stereo matching
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            # Compute disparity map
            disparity = self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0

            if x < 0 or x >= disparity.shape[1] or y < 0 or y >= disparity.shape[0]:
                self.last_depth_debug = f"rectified point out of frame ({x}, {y})"
                return None

            texture_radius = 16
            tx1 = max(0, x - texture_radius)
            tx2 = min(gray_left.shape[1], x + texture_radius + 1)
            ty1 = max(0, y - texture_radius)
            ty2 = min(gray_left.shape[0], y + texture_radius + 1)
            texture_std = float(np.std(gray_left[ty1:ty2, tx1:tx2]))
            if texture_std < self.depth_min_texture_std:
                self.last_depth_debug = f"low texture {texture_std:.1f}"
                return None

            # Use the median of valid disparities around the point to reduce pixel noise.
            d = None
            for radius in (4, 8, 16, 32):
                x1 = max(0, x - radius)
                x2 = min(disparity.shape[1], x + radius + 1)
                y1 = max(0, y - radius)
                y2 = min(disparity.shape[0], y + radius + 1)
                window = disparity[y1:y2, x1:x2]
                if self.stereo_disparity_sign < 0:
                    valid = window[(window < -1) & (window >= self.stereo_min_disparity)]
                else:
                    valid = window[window > 0]
                if valid.size >= 8:
                    d = abs(float(np.median(valid)))
                    break

            if d is None:
                self.last_depth_debug = f"no valid disparity near ({x}, {y})"
                return None

            # Check if disparity is valid (not zero or negative)
            if d <= 0:
                self.last_depth_debug = f"invalid disparity {d:.2f}px"
                return None

            # Calculate depth using formula: depth = (focal_length * baseline) / disparity
            # focal_length is in pixels, baseline is in mm
            focal_length = self.stereo_focal_length or self.K1[0, 0]
            depth_mm = (focal_length * self.baseline) / d
            if self.depth_max_valid_mm > 0 and depth_mm > self.depth_max_valid_mm:
                self.last_depth_debug = f"depth too far {depth_mm / 1000.0:.2f}m"
                return None

            self.last_depth_debug = f"disp {d:.2f}px"

            return depth_mm

        except Exception as e:
            self.last_depth_debug = f"error: {e}"
            print(f"Error calculating depth: {e}")
            return None
