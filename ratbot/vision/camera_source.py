"""USB and CSI camera initialization, capture, and frame normalization."""

from __future__ import annotations

import threading

import cv2


class CameraSource:
    """Own left/right camera handles and return normalized frame pairs."""

    def __init__(
        self,
        enabled,
        camera_id,
        use_csi,
        stereo_mode,
        invert_camera,
        video_fps,
        capture_factory=cv2.VideoCapture,
        csi_capture_factory=None,
    ):
        self.enabled = bool(enabled)
        self.camera_id = int(camera_id)
        self.use_csi = bool(use_csi)
        self.stereo_mode = bool(stereo_mode)
        self.invert_camera = bool(invert_camera)
        self.video_fps = video_fps
        self.capture_factory = capture_factory
        self.csi_capture_factory = csi_capture_factory
        self.left = None
        self.right = None
        self.active = False
        self.capture_lock = threading.Lock()

    @staticmethod
    def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=0,
    ):
        """Generate a GStreamer pipeline for a Jetson CSI camera."""
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
            f"height=(int){capture_height}, format=(string)NV12, "
            f"framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, "
            f"height=(int){display_height}, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        )

    def initialize(self):
        """Initialize the selected USB or CSI camera pair."""
        if not self.enabled:
            return

        try:
            if self.use_csi:
                self._initialize_csi()
            else:
                self._initialize_usb()

            if not self.left.isOpened():
                raise RuntimeError("Failed to open left camera")
            if self.stereo_mode and not self.right.isOpened():
                raise RuntimeError("Failed to open right camera")
            self.active = True
        except Exception as exc:
            print(f"Failed to initialize camera: {exc}")
            self.close()

    def _initialize_csi(self):
        flip_method = 2 if self.invert_camera else 0
        flip_status = "inverted" if self.invert_camera else "normal"

        if self.csi_capture_factory is not None:
            self.left = self.csi_capture_factory(
                sensor_id=0,
                width=640,
                height=480,
                fps=self.video_fps,
                flip_method=flip_method,
            )
            self.left.start()
            print(
                "✓ CSI Camera (left) initialized with subprocess+GStreamer "
                f"(640x480 @ {self.video_fps} FPS, {flip_status})"
            )
            if self.stereo_mode:
                self.right = self.csi_capture_factory(
                    sensor_id=1,
                    width=640,
                    height=480,
                    fps=self.video_fps,
                    flip_method=flip_method,
                )
                self.right.start()
                print(
                    "✓ CSI Camera (right) initialized with subprocess+GStreamer "
                    f"(640x480 @ {self.video_fps} FPS, {flip_status})"
                )
            return

        left_pipeline = self.gstreamer_pipeline(
            sensor_id=0,
            framerate=self.video_fps,
            flip_method=flip_method,
        )
        self.left = self.capture_factory(left_pipeline, cv2.CAP_GSTREAMER)
        print(
            "✓ CSI Camera (left) initialized with GStreamer "
            f"(640x480 @ {self.video_fps} FPS, {flip_status})"
        )
        if self.stereo_mode:
            right_pipeline = self.gstreamer_pipeline(
                sensor_id=1,
                framerate=self.video_fps,
                flip_method=flip_method,
            )
            self.right = self.capture_factory(right_pipeline, cv2.CAP_GSTREAMER)
            print(
                "✓ CSI Camera (right) initialized with GStreamer "
                f"(640x480 @ {self.video_fps} FPS, {flip_status})"
            )

    def _initialize_usb(self):
        flip_status = "inverted" if self.invert_camera else "normal"
        self.left = self.capture_factory(self.camera_id)
        self._configure_usb(self.left)
        print(
            f"✓ USB Camera (left) {self.camera_id} initialized "
            f"(640x480 @ {self.video_fps} FPS, {flip_status})"
        )
        if self.stereo_mode:
            self.right = self.capture_factory(self.camera_id + 1)
            self._configure_usb(self.right)
            print(
                f"✓ USB Camera (right) {self.camera_id + 1} initialized "
                f"(640x480 @ {self.video_fps} FPS, {flip_status})"
            )

    def _configure_usb(self, camera):
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, self.video_fps)

    def read_frames(self):
        """Read and normalize a left frame and optional right frame."""
        with self.capture_lock:
            return self._read_frames_locked()

    def _read_frames_locked(self):
        if not self.active or self.left is None:
            return None, None

        ret, frame = self.left.read()
        if not ret:
            return None, None

        frame_right = None
        if self.stereo_mode and self.right is not None:
            ret_right, frame_right = self.right.read()
            if not ret_right:
                frame_right = None

        if self.invert_camera and not self.use_csi:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            if frame_right is not None:
                frame_right = cv2.rotate(frame_right, cv2.ROTATE_180)

        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480))
        if frame_right is not None and (
            frame_right.shape[1] != 640 or frame_right.shape[0] != 480
        ):
            frame_right = cv2.resize(frame_right, (640, 480))

        return frame, frame_right

    def close(self):
        """Release camera handles and deactivate the source."""
        with self.capture_lock:
            self.active = False
            for camera in (self.left, self.right):
                if camera is None:
                    continue
                try:
                    camera.release()
                except Exception:
                    pass
            self.left = None
            self.right = None
