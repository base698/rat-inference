"""World-frame tracking glue: track updates, logging, recordings, selection."""

from ratbot.app.deps import *  # noqa: F401,F403
from ratbot.app.config_loader import *  # noqa: F401,F403


class WorldGlueMixin:
    def update_world_tracks(
        self,
        detections,
        stereo_measurements,
        *,
        timestamp,
        yaw_raw,
        pitch_raw,
    ):
        """Transform valid stereo detections and update stable base-frame tracks."""
        measurements_3d = []
        measurement_records = []
        for detection, stereo in zip(detections, stereo_measurements):
            if stereo is None or stereo.confidence < self.world_min_depth_confidence:
                continue
            position_base = self.world_transformer.camera_to_base(
                stereo.point_camera_mm,
                yaw_raw,
                pitch_raw,
            )
            covariance_base = self.world_transformer.camera_covariance_to_base(
                stereo.covariance_camera,
                yaw_raw,
                pitch_raw,
            )
            measurements_3d.append(
                Detection3D(
                    position_base_mm=position_base,
                    covariance_base=covariance_base,
                    confidence=max(
                        0.0,
                        min(1.0, float(detection["confidence"]) * stereo.confidence),
                    ),
                    classification=detection.get("class_name"),
                    measurement_time=float(timestamp),
                    bbox=detection.get("bbox"),
                    center=detection.get("center"),
                    depth_mm=stereo.depth_mm,
                    depth_confidence=stereo.confidence,
                )
            )
            measurement_records.append(
                {
                    "center": list(detection.get("center")) if detection.get("center") is not None else None,
                    "bbox": list(detection.get("bbox")) if detection.get("bbox") is not None else None,
                    "class": detection.get("class_name"),
                    "detection_confidence": float(detection["confidence"]),
                    "depth_confidence": stereo.confidence,
                    "camera_point_mm": stereo.point_camera_mm.tolist(),
                    "base_point_mm": position_base.tolist(),
                    "base_covariance": covariance_base.tolist(),
                    "depth_mm": stereo.depth_mm,
                    "disparity_px": stereo.disparity_px,
                    "disparity_iqr_px": stereo.disparity_iqr_px,
                    "valid_ratio": stereo.valid_ratio,
                }
            )

        tracks = self.world_tracker.update(measurements_3d, float(timestamp))
        selected = self.world_tracker.get_selected_track(timestamp=float(timestamp))
        with self.inference_lock:
            self.latest_tracks = tracks
            self.latest_track_assignments = list(self.world_tracker.last_assignments)
            if selected is not None and selected.misses == 0:
                self.latest_bbox = selected.bbox
                self.latest_center_point = selected.center
        if self.world_log_path or self.track_recordings.status()["recording"]:
            self._log_world_update(
                timestamp=float(timestamp),
                measurement_count=len(measurements_3d),
                detection_count=len(detections),
                yaw_raw=yaw_raw,
                pitch_raw=pitch_raw,
                tracks=tracks,
                measurement_records=measurement_records,
            )
        print(
            "   World tracks: "
            f"valid_3d={len(measurements_3d)}/{len(detections)}, "
            f"tracks={len(tracks)}, selected={self.world_tracker.selected_track_id}"
        )
        return tracks
    def _log_world_update(
        self,
        *,
        timestamp,
        measurement_count,
        detection_count,
        yaw_raw,
        pitch_raw,
        tracks,
        measurement_records,
    ):
        """Append one replayable JSONL update when configured."""
        predicted = self.world_tracker.get_selected_track(
            timestamp=timestamp,
            prediction_horizon=self.world_aim_latency_seconds,
        )
        predicted_aim = None
        if predicted is not None:
            predicted_aim = self.world_transformer.base_position_to_servo_raw(
                predicted.position
            )
        record = {
            "schema": "ratbot.world_tracks.v1",
            "recorded_at": datetime.now().astimezone().isoformat(),
            "monotonic_time": timestamp,
            "control_monotonic_time": time.monotonic(),
            "image_size": {
                "width": self.camera_width,
                "height": self.camera_height,
            },
            "pose_raw": {"yaw": yaw_raw, "pitch": pitch_raw},
            "commanded_pose_raw": {
                "yaw": self.current_yaw,
                "pitch": self.current_pitch,
            },
            "detection_count": detection_count,
            "valid_3d_measurement_count": measurement_count,
            "measurements": measurement_records,
            "selected_track_id": self.world_tracker.selected_track_id,
            "predicted_aim": predicted_aim,
            "assignments": list(self.world_tracker.last_assignments),
            "tracks": [track.to_dict() for track in tracks],
        }
        try:
            self.track_recordings.append(record)
        except (OSError, TypeError, ValueError) as exc:
            self.track_recordings.fail(str(exc))
            print(f"Track-recording error; recording stopped: {exc}")
        log_path = self.world_log_path
        if not log_path:
            return
        try:
            parent = os.path.dirname(os.path.abspath(log_path))
            os.makedirs(parent, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError as exc:
            print(f"World-track log error: {exc}")
    def get_world_tracks(self):
        return [track.to_dict() for track in self.world_tracker.get_tracks()]
    def _world_tracking_parameters(self):
        config = self.world_tracker.config
        return {
            "gate_distance_mm": config.gate_distance_mm,
            "confirm_hits": config.confirm_hits,
            "max_misses": config.max_misses,
            "delete_after_seconds": config.delete_after_seconds,
            "reidentify_after_seconds": config.reidentify_after_seconds,
            "process_acceleration_std_mm_s2": config.process_acceleration_std_mm_s2,
            "confidence_decay": config.confidence_decay,
            "min_depth_confidence": self.world_min_depth_confidence,
            "image_width": self.camera_width,
            "image_height": self.camera_height,
        }
    def get_track_recording_status(self):
        return self.track_recordings.status()
    def start_track_recording(self):
        if not self.world_tracking:
            raise RuntimeError("world tracking must be enabled before recording")
        return self.track_recordings.start(self._world_tracking_parameters())
    def stop_track_recording(self):
        return self.track_recordings.stop()
    def list_track_recordings(self):
        return self.track_recordings.list_recordings()
    def load_track_recording(self, recording_id):
        return self.track_recordings.load(recording_id)
    def delete_track_recording(self, recording_id):
        return self.track_recordings.delete(recording_id)
    def reprocess_track_recording(self, recording_id, parameters):
        return self.track_recordings.reprocess(recording_id, parameters)
    def select_world_target(self, target_id):
        return self.world_tracker.select_target(int(target_id))
    def clear_world_selection(self):
        self.world_tracker.clear_selection()
        return True
