"""FastAPI controller for a camera/servo tracker."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ratbot.robot import TrackerRobot


@dataclass(frozen=True)
class ControlApiConfig:
    """Values the web UI needs from the robot configuration."""

    yaw_min: int
    yaw_max: int
    yaw_center: int
    pitch_min: int
    pitch_max: int
    pitch_center: int
    static_dir: str = "static"
    detections_dir: str = "detections"


class TrackerControlApi:
    """Web controller facade around a tracker-like robot object."""

    def __init__(
        self,
        config: ControlApiConfig,
        get_target_crosshair_x: Callable[[int], int],
        get_target_crosshair_y: Callable[[int], int],
    ):
        self.config = config
        self._get_target_crosshair_x = get_target_crosshair_x
        self._get_target_crosshair_y = get_target_crosshair_y
        self._tracker: Optional[TrackerRobot] = None
        self.app = FastAPI()

        Path(self.config.static_dir).mkdir(exist_ok=True)
        self.app.mount(
            "/static",
            StaticFiles(directory=self.config.static_dir),
            name="static",
        )
        self._register_routes()

    def set_tracker(self, tracker: TrackerRobot) -> None:
        self._tracker = tracker

    @property
    def tracker(self) -> Optional[TrackerRobot]:
        return self._tracker

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/")
        async def root():
            """Root endpoint - serve static HTML file"""
            return FileResponse(Path(self.config.static_dir) / "index.html")

        @app.get("/config")
        async def get_config():
            """Get configuration values for the UI"""
            tracker = self.tracker
            if tracker and tracker.connected:
                initial_yaw = tracker.current_yaw
                initial_pitch = tracker.current_pitch
            else:
                initial_yaw = self.config.yaw_center
                initial_pitch = self.config.pitch_center

            return JSONResponse({
                "YAW_MIN": self.config.yaw_min,
                "YAW_MAX": self.config.yaw_max,
                "YAW_CENTER": self.config.yaw_center,
                "PITCH_MIN": self.config.pitch_min,
                "PITCH_MAX": self.config.pitch_max,
                "PITCH_CENTER": self.config.pitch_center,
                "TARGET_CROSSHAIR_X": self._get_target_crosshair_x(initial_yaw),
                "TARGET_CROSSHAIR_Y": self._get_target_crosshair_y(initial_pitch),
                "initial_yaw": initial_yaw,
                "initial_pitch": initial_pitch,
                "enable_trigger": tracker.trigger_servo_enabled if tracker else False,
                "world_tracking": bool(getattr(tracker, "world_tracking", False)) if tracker else False,
            })

        @app.get("/status")
        async def get_status():
            """Get current status of the tracker"""
            tracker = self.tracker
            if not tracker:
                return JSONResponse({
                    "connected": False,
                    "yaw_position": "N/A",
                    "pitch_position": "N/A",
                    "camera_active": False,
                    "detection_count": 0,
                    "detection": False,
                    "confidence": 0,
                    "recent_detections": [],
                })

            status_data = {
                "connected": tracker.connected,
                "yaw_position": tracker.current_yaw,
                "pitch_position": tracker.current_pitch,
                "camera_active": tracker.camera_active,
                "detection_count": tracker.detection_count,
            }

            if tracker.camera_active:
                status_data.update(tracker.get_detection_data())
            else:
                status_data.update({
                    "detection": False,
                    "confidence": 0,
                    "recent_detections": [],
                })

            return JSONResponse(status_data)

        @app.get("/stream-frame")
        async def stream_frame():
            """Get the latest camera frame as bytes"""
            tracker = self.tracker
            if not tracker or not tracker.camera_active:
                return Response(content=b"", media_type="image/jpeg")

            frame_bytes = tracker.get_latest_frame_bytes()
            if frame_bytes is None:
                return Response(content=b"", media_type="image/jpeg")

            return Response(content=frame_bytes, media_type="image/jpeg")

        @app.post("/set-position")
        async def set_position(request: dict):
            """Set servo positions"""
            tracker = self.tracker
            if not tracker or not tracker.connected:
                return JSONResponse({
                    "success": False,
                    "message": "Tracker not connected",
                })

            try:
                yaw = request.get("yaw")
                pitch = request.get("pitch")

                tracker.clear_target_belief()

                if yaw is not None:
                    tracker.set_yaw(yaw)
                if pitch is not None:
                    tracker.set_pitch(pitch)

                return JSONResponse({
                    "success": True,
                    "message": "Position updated",
                })
            except Exception as exc:
                return JSONResponse({
                    "success": False,
                    "message": str(exc),
                })

        @app.post("/trigger-servo")
        async def trigger_servo():
            """Manually trigger the action servo"""
            tracker = self.tracker
            if not tracker:
                return JSONResponse(
                    content={"success": False, "message": "Tracker not initialized"},
                    status_code=500,
                )

            if not tracker.trigger_servo_enabled:
                return JSONResponse(
                    content={
                        "success": False,
                        "message": (
                            "Trigger servo is not enabled. "
                            "Run with --enable-trigger flag to enable."
                        ),
                    },
                    status_code=400,
                )

            try:
                tracker.trigger_action_servo()
                return JSONResponse(
                    content={"success": True, "message": "Servo triggered successfully!"},
                    status_code=200,
                )
            except Exception as exc:
                return JSONResponse(
                    content={
                        "success": False,
                        "message": f"Error triggering servo: {str(exc)}",
                    },
                    status_code=500,
                )

        @app.post("/clear-belief")
        async def clear_belief():
            """Clear autonomous target belief without moving the servos."""
            tracker = self.tracker
            if not tracker:
                return JSONResponse(
                    content={"success": False, "message": "Tracker not initialized"},
                    status_code=500,
                )

            try:
                tracker.clear_target_belief()
                return JSONResponse(
                    content={"success": True, "message": "Target belief cleared"},
                    status_code=200,
                )
            except Exception as exc:
                return JSONResponse(
                    content={
                        "success": False,
                        "message": f"Error clearing target belief: {str(exc)}",
                    },
                    status_code=500,
                )

        @app.get("/tracks")
        async def get_tracks():
            """Return all fixed-frame tracks and explicit selection state."""
            tracker = self.tracker
            if not tracker:
                return JSONResponse(
                    content={"world_tracking": False, "selected_track_id": None, "tracks": []},
                    status_code=200,
                )
            tracks = tracker.get_world_tracks()
            selected = next(
                (item["id"] for item in tracks if item.get("selected")),
                None,
            )
            return JSONResponse(
                content={
                    "world_tracking": bool(getattr(tracker, "world_tracking", False)),
                    "world_actuation_enabled": bool(
                        getattr(tracker, "world_actuation_enabled", False)
                    ),
                    "remote_selection_enabled": bool(
                        getattr(tracker, "world_api_selection_enabled", False)
                    ),
                    "selected_track_id": selected,
                    "tracks": tracks,
                },
                status_code=200,
            )

        @app.post("/tracks/select")
        async def select_track(request: dict):
            """Select a stable target ID without silently switching targets."""
            tracker = self.tracker
            if not tracker:
                return JSONResponse(
                    content={"success": False, "message": "Tracker not initialized"},
                    status_code=500,
                )
            if not bool(getattr(tracker, "world_api_selection_enabled", False)):
                return JSONResponse(
                    content={
                        "success": False,
                        "message": "Remote world-track selection is disabled",
                    },
                    status_code=403,
                )
            target_id = request.get("track_id")
            if isinstance(target_id, bool) or not isinstance(target_id, int):
                return JSONResponse(
                    content={"success": False, "message": "track_id must be an integer"},
                    status_code=400,
                )
            if not tracker.select_world_target(target_id):
                return JSONResponse(
                    content={"success": False, "message": "Track not found"},
                    status_code=404,
                )
            return JSONResponse(
                content={"success": True, "selected_track_id": int(target_id)},
                status_code=200,
            )

        @app.post("/tracks/clear-selection")
        async def clear_track_selection():
            """Clear autonomous target selection without deleting tracks."""
            tracker = self.tracker
            if not tracker:
                return JSONResponse(
                    content={"success": False, "message": "Tracker not initialized"},
                    status_code=500,
                )
            if not bool(getattr(tracker, "world_api_selection_enabled", False)):
                return JSONResponse(
                    content={
                        "success": False,
                        "message": "Remote world-track selection is disabled",
                    },
                    status_code=403,
                )
            tracker.clear_world_selection()
            return JSONResponse(
                content={"success": True, "selected_track_id": None},
                status_code=200,
            )

        @app.get("/detections/{filename}")
        async def get_detection(filename: str):
            """Serve detection image files"""
            detection_path = Path(self.config.detections_dir) / filename
            if not detection_path.exists():
                raise HTTPException(status_code=404, detail="Detection image not found")
            return FileResponse(detection_path)

        @app.post("/move-to-position")
        async def move_to_position(request: dict):
            """Move tracker to clicked canvas position"""
            tracker = self.tracker
            if not tracker:
                return JSONResponse({
                    "success": False,
                    "message": "Tracker not initialized",
                })

            try:
                x = request.get("x")
                y = request.get("y")

                if x is None or y is None:
                    return JSONResponse({
                        "success": False,
                        "message": "Missing x or y coordinates",
                    })

                desired_yaw, desired_pitch = tracker.move_to_pixel(x, y)
                tracker.set_yaw(desired_yaw)
                tracker.set_pitch(desired_pitch)

                return JSONResponse({
                    "success": True,
                    "message": f"Moved to position ({x}, {y})",
                    "yaw": desired_yaw,
                    "pitch": desired_pitch,
                })
            except Exception as exc:
                return JSONResponse({
                    "success": False,
                    "message": str(exc),
                })


def create_control_app(
    config: ControlApiConfig,
    get_target_crosshair_x: Callable[[int], int],
    get_target_crosshair_y: Callable[[int], int],
) -> TrackerControlApi:
    return TrackerControlApi(
        config=config,
        get_target_crosshair_x=get_target_crosshair_x,
        get_target_crosshair_y=get_target_crosshair_y,
    )
