"""Application wiring for the rt_200 tracker runtime."""

from ratbot.app.tracker import CameraTracker  # noqa: F401,E402
from ratbot.app.server import app, control_api  # noqa: F401,E402
