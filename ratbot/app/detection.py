"""YOLO model init, the inference hot path, detection snapshots, and the inference thread."""

from ratbot.app.deps import *  # noqa: F401,F403
from ratbot.app.config_loader import *  # noqa: F401,F403


class DetectionPipelineMixin:
    def init_model(self):
        """Initialize YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✓ YOLO model loaded: {self.model_path}")
            print(f"  YOLO inference device request: {self.inference_device or 'auto'}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
    def _effective_inference_device(self, runtime_device):
        """Return the device to expose after first inference."""
        if runtime_device is not None:
            return str(runtime_device)
        if not self.inference_device:
            return None
        device = str(self.inference_device)
        if device.isdigit():
            return f"cuda:{device}"
        return device
    def _detection_snapshot_files(self):
        try:
            names = os.listdir(self.detections_dir)
        except OSError:
            return []

        snapshots = []
        for name in names:
            if not name.startswith("detection_") or not name.lower().endswith(".jpg"):
                continue
            path = os.path.join(self.detections_dir, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            if not os.path.isfile(path):
                continue
            snapshots.append((stat.st_mtime, path))
        return snapshots
    def _prune_detection_snapshots(self, now=None):
        """Delete stale detection snapshots and keep the snapshot directory bounded."""
        now = time.time() if now is None else float(now)
        self._last_detection_snapshot_prune_time = now
        snapshots = self._detection_snapshot_files()
        if not snapshots:
            return 0

        deleted = 0
        kept = []
        for modified_time, path in snapshots:
            too_old = (
                self.detection_snapshot_max_age_seconds > 0
                and now - modified_time > self.detection_snapshot_max_age_seconds
            )
            if too_old:
                try:
                    os.remove(path)
                    deleted += 1
                except OSError:
                    pass
            else:
                kept.append((modified_time, path))

        if self.detection_snapshot_max_files > 0:
            overflow = max(0, len(kept) - self.detection_snapshot_max_files)
            for _, path in sorted(kept)[:overflow]:
                try:
                    os.remove(path)
                    deleted += 1
                except OSError:
                    pass

        if deleted:
            print(f"Pruned {deleted} old detection snapshot(s)")
        return deleted
    def _save_detection_snapshot(self, frame, now=None):
        """Save a throttled detection snapshot, returning (filename, path) or None."""
        if not self.detection_snapshot_enabled:
            return None
        now = time.time() if now is None else float(now)
        if now - self._last_detection_snapshot_time < self.detection_snapshot_min_interval_seconds:
            return None
        if now - self._last_detection_snapshot_prune_time >= self.detection_snapshot_prune_interval_seconds:
            self._prune_detection_snapshots(now=now)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        detection_filename = f"detection_{timestamp}.jpg"
        detection_path = os.path.join(self.detections_dir, detection_filename)
        if not cv2.imwrite(detection_path, frame):
            return None
        self._last_detection_snapshot_time = now
        return detection_filename, detection_path
    def capture_world_measurement_context(self):
        """Snapshot acquisition time with synchronous measured servo readback.

        Cameras currently expose no hardware exposure timestamp. This samples the
        monotonic clock immediately after frame acquisition, then obtains actual
        Present_Position values instead of reusing commanded servo goals.
        """
        measurement_time = time.monotonic()
        pose_yaw, pose_pitch = self.tracking_servos.read_measured_positions()
        return measurement_time, int(pose_yaw), int(pose_pitch)
    def run_inference(self):
        """Run one inference pass using the shared inference module"""
        if not self.camera_active or not self.camera or not self.model:
            return

        try:
            frame, frame_right = self.camera_source.read_frames()
            if frame is None:
                return
            # CameraSource does not yet expose hardware exposure timestamps.
            # Snapshot monotonic acquisition time with synchronous Present_Position
            # readback; commanded servo goals are not valid transform poses.
            measurement_time, pose_yaw, pose_pitch = (
                self.capture_world_measurement_context()
                if self.world_tracking
                else (time.monotonic(), self.current_yaw, self.current_pitch)
            )

            # Keep inference on the normal camera image; depth rectifies hidden copies only.
            if self.stereo_depth.calibration_enabled and not self.stereo_depth.stereo_calibration_enabled:
                frame = self.stereo_depth.undistort_frame(frame, use_left=True)

            # Run inference using shared inference module (YOLO works directly with numpy arrays)
            results = yolo_run_inference(
                self.model,
                frame,
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                device=self.inference_device,
                verbose=False
            )
            if not self._runtime_device_logged:
                runtime_device = getattr(self.model, "device", None)
                self.inference_runtime_device = self._effective_inference_device(
                    runtime_device
                )
                print(
                    f"YOLO runtime/effective device: {self.inference_runtime_device or 'unknown'}",
                    flush=True,
                )
                self._runtime_device_logged = True

            # Extract detections using shared utility and keep configured target classes.
            if self.target_classes:
                detections = []
                for target_class in self.target_classes:
                    detections.extend(extract_detections(results, self.model, target_class=target_class))
            else:
                detections = extract_detections(results, self.model)
            detections.sort(key=lambda det: det["confidence"], reverse=True)
            stereo_measurements = [None] * len(detections)
            if (
                detections
                and self.stereo_depth.stereo_calibration_enabled
                and frame_right is not None
            ):
                stereo_measurements = self.stereo_depth.calculate_depths(
                    frame,
                    frame_right,
                    [det["center"] for det in detections],
                )

            if self.world_tracking:
                self.update_world_tracks(
                    detections,
                    stereo_measurements,
                    timestamp=measurement_time,
                    yaw_raw=pose_yaw,
                    pitch_raw=pose_pitch,
                )

            # Process first detection (if any)
            detection = False
            confidence = 0
            bbox = None
            center_point = None

            # Calculate depth if stereo enabled
            depth_mm = None
            if detections:
                # Use the first (highest confidence) detection
                det = detections[0]
                detection = True
                confidence = det['confidence']
                class_name = det.get('class_name', 'object')
                bbox = det['bbox']
                center_point = det['center']
                center_x, center_y = center_point

                # Reuse the batched stereo pass; do not recompute disparity per target.
                stereo_measurement = stereo_measurements[0]
                if stereo_measurement is not None:
                    depth_mm = stereo_measurement.depth_mm
                    depth_m = depth_mm / 1000.0
                    print(
                        f"   Depth: {depth_m:.2f}m ({depth_mm:.1f}mm), "
                        f"quality={stereo_measurement.confidence:.2f}, "
                        f"valid={stereo_measurement.valid_ratio:.2f}, "
                        f"iqr={stereo_measurement.disparity_iqr_px:.2f}px"
                    )

                snapshot = self._save_detection_snapshot(frame)

                self.detection_count += 1
                # Format: "time - message | filename" so the UI can parse and link it
                depth_str = f" @ {depth_mm/1000.0:.2f}m" if depth_mm else ""
                detection_msg = f"{datetime.now().strftime('%H:%M:%S')} - {class_name} detected (conf: {confidence:.3f}) at ({center_x}, {center_y}){depth_str}"
                if snapshot:
                    detection_msg = f"{detection_msg} | {snapshot[0]}"
                self.recent_detections.append(detection_msg)
                self.recent_detections = self.recent_detections[-10:]

                if snapshot:
                    print(f"🎯 Detection #{self.detection_count}: {snapshot[1]}")
                else:
                    print(f"🎯 Detection #{self.detection_count}: snapshot skipped")
                print(f"   Class: {class_name}, Center: ({center_x}, {center_y}), Confidence: {confidence:.3f}")

                # World mode already updated every valid 3D detection above.  The
                # legacy angular belief remains the default/fallback mode — and
                # also drives actuation in world SHADOW mode (world tracking on
                # for the 3D view, actuation off), so feed it there too.
                if not self.world_tracking or not self.world_actuation_enabled:
                    frame_pose_yaw, frame_pose_pitch = self.pose_at(
                        measurement_time - self.camera_latency_s
                    )
                    self.update_target_belief(
                        center_x,
                        center_y,
                        confidence,
                        depth_mm=depth_mm,
                        pose_yaw=frame_pose_yaw,
                        pose_pitch=frame_pose_pitch,
                    )

                # Auto-trigger disabled - only trigger manually via button
                # if detection and not self.latest_detection and self.trigger_servo_enabled:
                #     threading.Thread(target=self.trigger_action_servo, daemon=True).start()
            else:
                if not self.world_tracking:
                    self.decay_target_belief()

            if self.world_tracking:
                selected_track = self.world_tracker.get_selected_track(
                    timestamp=measurement_time
                )
                if selected_track is not None and selected_track.misses == 0:
                    bbox = selected_track.bbox
                    center_point = selected_track.center
                    confidence = selected_track.confidence
                    depth_mm = float(np.linalg.norm(selected_track.position))

            # Update detection state
            with self.inference_lock:
                self.latest_detection = detection
                self.latest_confidence = confidence
                self.latest_bbox = bbox
                self.latest_center_point = center_point
                self.latest_depth = depth_mm

        except Exception as e:
            print(f"Inference error: {e}")
    def get_detection_data(self):
        """Get the latest detection data without image"""
        with self.inference_lock:
            return {
                "detection": self.latest_detection,
                "confidence": self.latest_confidence if hasattr(self, 'latest_confidence') else 0,
                "depth_mm": self.latest_depth,
                "inference_device": self.inference_device or "auto",
                "inference_runtime_device": self.inference_runtime_device,
                "recent_detections": self.recent_detections,
                "world_tracking": self.world_tracking,
                "world_actuation_enabled": self.world_actuation_enabled,
                "world_calibration_validated": self.world_calibration_validated,
                "world_api_selection_enabled": self.world_api_selection_enabled,
                "selected_track_id": self.world_tracker.selected_track_id,
                "tracks": [track.to_dict() for track in self.latest_tracks],
                "track_assignments": list(self.latest_track_assignments),
            }
    def inference_thread(self):
        """Inference processing thread"""
        while self.camera_active:
            loop_start = time.time()
            self.run_inference()
            self.inference_loop_count += 1
            window_elapsed = time.time() - self.inference_fps_window_start
            if window_elapsed >= 5.0:
                actual_fps = self.inference_loop_count / window_elapsed
                self.last_inference_fps = actual_fps
                print(f"Inference actual FPS: {actual_fps:.1f} (target {self.inference_fps:g})", flush=True)
                self.inference_loop_count = 0
                self.inference_fps_window_start = time.time()
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, (1.0 / self.inference_fps) - elapsed))
    def start_inference_thread(self):
        """Start the inference processing thread"""
        thread = threading.Thread(target=self.inference_thread, daemon=True)
        thread.start()
        print(f"Inference thread started ({self.inference_fps:g} FPS target)")
