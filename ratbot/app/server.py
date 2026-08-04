"""FastAPI control app construction for the ratbot app."""

from ratbot.app.deps import *  # noqa: F401,F403
from ratbot.app.config_loader import *  # noqa: F401,F403

control_api = create_control_app(
    ControlApiConfig(
        yaw_min=YAW_MIN,
        yaw_max=YAW_MAX,
        yaw_center=YAW_CENTER,
        pitch_min=PITCH_MIN,
        pitch_max=PITCH_MAX,
        pitch_center=PITCH_CENTER,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        detections_dir=DETECTIONS_DIR,
    ),
    get_target_crosshair_x=get_target_crosshair_x,
    get_target_crosshair_y=get_target_crosshair_y,
)
app = control_api.app




