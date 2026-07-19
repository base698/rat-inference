"""OpenCV rendering for the RT-200 video stream."""

from __future__ import annotations

import cv2


class OverlayRenderer:
    """Render aiming, depth, and detection overlays onto camera frames."""

    def __init__(
        self,
        stereo_depth,
        aiming,
        crosshair_x,
        crosshair_y,
        crosshair_size,
        depth_adjust_smoothing_alpha,
        depth_adjust_missing_decay,
    ):
        self.stereo_depth = stereo_depth
        self.aiming = aiming
        self.crosshair_x = crosshair_x
        self.crosshair_y = crosshair_y
        self.crosshair_size = int(crosshair_size)
        self.depth_adjust_smoothing_alpha = max(
            0.0,
            min(1.0, float(depth_adjust_smoothing_alpha)),
        )
        self.depth_adjust_missing_decay = max(
            0.0,
            min(1.0, float(depth_adjust_missing_decay)),
        )
        self.display_depth_adjust_px = 0.0
        self.depth_adjust_initialized = False
        self.frame_counter = 0

    def smooth_depth_adjust(self, target_adjust_px):
        """Smooth visual depth compensation without affecting tracking math."""
        if target_adjust_px is None:
            self.display_depth_adjust_px *= self.depth_adjust_missing_decay
            if abs(self.display_depth_adjust_px) < 0.2:
                self.display_depth_adjust_px = 0.0
                self.depth_adjust_initialized = False
            return self.display_depth_adjust_px

        target_adjust_px = float(target_adjust_px)
        if not self.depth_adjust_initialized:
            self.display_depth_adjust_px = target_adjust_px
            self.depth_adjust_initialized = True
            return self.display_depth_adjust_px

        alpha = self.depth_adjust_smoothing_alpha
        self.display_depth_adjust_px += alpha * (
            target_adjust_px - self.display_depth_adjust_px
        )
        return self.display_depth_adjust_px

    def render(
        self,
        frame,
        frame_right,
        current_yaw,
        current_pitch,
        stereo_mode,
        bbox,
        center_point,
        tracks=(),
    ):
        """Draw crosshair, depth status, boxes, stable IDs, and selected target."""
        ch_x = self.crosshair_x(current_yaw)
        focal_length_y = None
        ch_y = self.crosshair_y(
            current_pitch,
            camera_bore_offset_mm=82,
            focal_length_px=focal_length_y,
            assumed_distance_mm=5000,
        )
        ch_size = self.crosshair_size
        self.frame_counter += 1

        depth_adjust_target_px = None
        depth_text = None
        depth_color = (0, 255, 0)
        if self.stereo_depth.stereo_calibration_enabled and frame_right is not None:
            depth_mm = self.stereo_depth.calculate_depth(
                frame,
                frame_right,
                ch_x,
                ch_y,
            )
            if depth_mm is not None:
                focal_y = None
                if self.stereo_depth.K1 is not None:
                    focal_y = float(self.stereo_depth.K1[1, 1])
                elif self.stereo_depth.camera_matrix is not None:
                    focal_y = float(self.stereo_depth.camera_matrix[1, 1])

                depth_adjust = self.aiming.depth_adjust_px(depth_mm, focal_y)
                if depth_adjust is not None:
                    depth_adjust_target_px = depth_adjust
                depth_text = f"{depth_mm / 1000.0:.2f}m"
            else:
                depth_text = f"-- {self.stereo_depth.last_depth_debug}"
                depth_color = (0, 200, 255)
        elif stereo_mode:
            depth_text = "-- no right frame/calibration"
            depth_color = (0, 200, 255)

        depth_adjust_px = self.smooth_depth_adjust(depth_adjust_target_px)
        if abs(depth_adjust_px) >= 0.2:
            ch_y = int(round(ch_y + depth_adjust_px))

        if self.frame_counter % 30 == 0:
            depth_debug = depth_text if depth_text is not None else "n/a"
            print(
                f"[DEBUG] Yaw: {current_yaw}, Pitch: {current_pitch}, "
                f"Crosshair: ({ch_x}, {ch_y}), Depth: {depth_debug}, "
                f"DepthYAdjust: {depth_adjust_px:.1f}px"
            )

        cv2.line(
            frame,
            (ch_x - ch_size, ch_y),
            (ch_x + ch_size, ch_y),
            (0, 255, 0),
            2,
        )
        cv2.line(
            frame,
            (ch_x, ch_y - ch_size),
            (ch_x, ch_y + ch_size),
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, (ch_x, ch_y), 5, (0, 255, 0), 2)

        if depth_text is not None:
            text_size = cv2.getTextSize(
                depth_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )[0]
            text_x = ch_x + ch_size + 10
            text_y = ch_y
            cv2.rectangle(
                frame,
                (text_x - 2, text_y - text_size[1] - 2),
                (text_x + text_size[0] + 2, text_y + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                depth_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                depth_color,
                2,
            )

        for track in tracks:
            track_bbox = (
                track.get("bbox") if isinstance(track, dict) else track.bbox
            )
            if track_bbox is None:
                continue
            track_id = track.get("id") if isinstance(track, dict) else track.id
            selected = (
                bool(track.get("selected"))
                if isinstance(track, dict)
                else bool(track.selected)
            )
            status = (
                track.get("status", "")
                if isinstance(track, dict)
                else track.status
            )
            x1, y1, x2, y2 = map(int, track_bbox)
            color = (0, 0, 255) if selected else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if selected else 2)
            marker = "*" if selected else ""
            cv2.putText(
                frame,
                f"ID {track_id}{marker} {status}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        if bbox is not None and not tracks:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if center_point is not None:
            cx, cy = center_point
            cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)
            cv2.line(frame, (cx, cy), (ch_x, ch_y), (0, 255, 255), 2)

        return frame
