"""Camera frame capture, overlays, pose history, and the camera thread."""

from ratbot.app.deps import *  # noqa: F401,F403
from ratbot.app.config_loader import *  # noqa: F401,F403


class VideoPipelineMixin:
    def draw_overlays(self, frame, frame_right=None):
        """Render a consistent snapshot of detection and aiming state."""
        with self.inference_lock:
            bbox = self.latest_bbox
            center_point = self.latest_center_point
            tracks = list(self.latest_tracks) if self.world_tracking else []

        return self.overlay_renderer.render(
            frame=frame,
            frame_right=frame_right,
            current_yaw=self.current_yaw,
            current_pitch=self.current_pitch,
            stereo_mode=self.stereo_mode,
            bbox=bbox,
            center_point=center_point,
            tracks=tracks,
        )
    def pose_at(self, target_time):
        """Servo pose closest to target_time from the pose history.

        Compensates for camera pipeline latency: the frame content is older
        than the moment we read it, so the pose paired with a detection must
        come from when the photons actually arrived.
        """
        best = None
        for entry in reversed(self.pose_history):
            if entry[0] <= target_time:
                best = entry
                break
            best = entry
        if best is None:
            return self.current_yaw, self.current_pitch
        return best[1], best[2]
    def capture_video_frame(self):
        """Capture a video frame at 30 FPS with overlays"""
        if not self.camera_active or not self.camera:
            return

        try:
            # Update servo positions for dynamic crosshair (reads actual position from servos)
            readback_due = (
                self.motor_readback_interval <= 0
                or time.time() - self.last_motor_readback_time >= self.motor_readback_interval
            )
            if self.connected and self.tracking_servos.motor_bus and readback_due:
                try:
                    self.last_motor_readback_time = time.time()
                    yaw_pos, pitch_pos, yaw_vel, pitch_vel = (
                        self.tracking_servos.read_state()
                    )
                    # Only update if we got valid readings
                    if yaw_pos is not None and pitch_pos is not None:
                        self.current_yaw = yaw_pos
                        self.current_pitch = pitch_pos
                        self.measured_yaw_velocity = yaw_vel
                        self.measured_pitch_velocity = pitch_vel
                        self.pose_history.append((time.monotonic(), yaw_pos, pitch_pos))
                except Exception as e:
                    # Silently fail on read errors (don't spam console at 30 FPS)
                    pass

            frame, frame_right = self.camera_source.read_frames()
            if frame is None:
                return

            # Keep the displayed frame unrectified; rectification is only for depth math.
            if self.stereo_depth.calibration_enabled and not self.stereo_depth.stereo_calibration_enabled:
                frame = self.stereo_depth.undistort_frame(frame, use_left=True)

            # Keep a clean copy for dataset capture (no overlays)
            ret_raw, raw_buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            raw_bytes = raw_buffer.tobytes() if ret_raw else None

            # Draw overlays (pass right frame for depth calculation)
            frame = self.draw_overlays(frame, frame_right)

            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                return

            # Store as bytes (no base64 encoding)
            img_bytes = buffer.tobytes()

            # Update latest frame
            with self.frame_lock:
                self.latest_frame = img_bytes
                if raw_bytes is not None:
                    self.latest_raw_frame = raw_bytes

        except Exception as e:
            print(f"Video frame capture error: {e}")
    def get_latest_frame_bytes(self):
        """Get the latest frame as bytes"""
        with self.frame_lock:
            return self.latest_frame
    def get_latest_raw_frame_bytes(self):
        """Get the latest camera frame without overlays as bytes"""
        with self.frame_lock:
            return self.latest_raw_frame
    def camera_thread(self):
        """Camera processing thread for video at 30 FPS"""
        while self.camera_active:
            self.capture_video_frame()
            time.sleep(1.0 / VIDEO_FPS)  # 30 FPS
    def start_camera_thread(self):
        """Start the camera processing thread"""
        thread = threading.Thread(target=self.camera_thread, daemon=True)
        thread.start()
        print(f"Camera thread started ({VIDEO_FPS} FPS)")
