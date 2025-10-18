"""
CSI Camera capture using subprocess + GStreamer
Workaround for OpenCV without GStreamer support
"""
import subprocess
import numpy as np
import cv2
import threading
import queue

class CSICameraCapture:
    def __init__(self, sensor_id=0, width=640, height=480, fps=30, flip_method=2):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.process = None
        self.thread = None

    def start(self):
        """Start capturing frames"""
        # GStreamer pipeline that outputs raw BGR frames to stdout
        gst_cmd = [
            'gst-launch-1.0',
            'nvarguscamerasrc', f'sensor-id={self.sensor_id}',
            '!', f'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate={self.fps}/1',
            '!', 'nvvidconv', f'flip-method={self.flip_method}',
            '!', f'video/x-raw, width={self.width}, height={self.height}, format=BGR',
            '!', 'videoconvert',
            '!', 'video/x-raw, format=BGR',
            '!', 'fdsink', 'fd=1'
        ]

        self.process = subprocess.Popen(
            gst_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.width * self.height * 3
        )

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        return True

    def _capture_loop(self):
        """Background thread that reads frames from gstreamer"""
        frame_size = self.width * self.height * 3  # BGR = 3 bytes per pixel

        while self.running:
            try:
                # Read one frame worth of bytes
                raw_frame = self.process.stdout.read(frame_size)

                if len(raw_frame) != frame_size:
                    break

                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))

                # Put in queue (drop oldest if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.frame_queue.put(frame)

            except Exception as e:
                print(f"Capture error: {e}")
                break

    def read(self):
        """Read a frame (OpenCV-compatible interface)"""
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None

    def isOpened(self):
        """Check if camera is opened"""
        return self.running and self.process and self.process.poll() is None

    def release(self):
        """Stop capturing"""
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread:
            self.thread.join(timeout=2)
