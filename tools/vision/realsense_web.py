#!/usr/bin/env python3
"""Browser-based RealSense D435 explorer.

The headless sibling of ``realsense_explorer.py``: same tour of the D435's
capabilities (device introspection, synchronized color + depth + infrared,
depth-to-color alignment, post-processing filters, IR emitter control, visual
presets, point cloud export) but driven from a web page instead of a desktop
OpenCV window. That is what makes it usable on the Jetson, which we drive over
SSH with no display attached.

Usage (on the Jetson):
    python3 tools/vision/realsense_web.py                 # serves on :8090
    python3 tools/vision/realsense_web.py --port 8091
    python3 tools/vision/realsense_web.py --info          # capabilities, no server

Then open http://<jetson>:8090/ from any machine on the network.

Live view layout (2x2 mosaic, same as the desktop explorer):
    [ color         ][ depth colormap ]
    [ infrared left ][ color+depth overlay ]

Controls are buttons on the page; click anywhere on the mosaic to probe the
depth at that pixel.

Hardware background, briefly: the D435 computes depth on-camera by stereo
matching its two IR imagers (85 deg FOV, global shutter). The IR emitter
projects a dot pattern to add texture so matching works on blank surfaces.
The RGB camera is a separate sensor (rolling shutter, different FOV) at a
physical offset, which is why depth must be *aligned* to color before you
can look up the distance of an RGB pixel.

Note on Python versions: Intel ships aarch64 wheels for pyrealsense2 on
cp39/310/312 but not cp311, so on the Jetson this runs under the system
Python 3.10 (which already carries cv2/fastapi/numpy), not the repo's 3.11
venv. Hence plain ``python3``, no ``uv run``.
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not installed. On the Jetson: "
             "python3 -m pip install --user pyrealsense2")

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                                   StreamingResponse)
except ImportError:
    sys.exit("fastapi/uvicorn not installed. On the Jetson they ship with the "
             "system python3; elsewhere: uv sync")

WIDTH, HEIGHT, FPS = 640, 480, 30
BOUNDARY = "frame"


def describe_device(dev: "rs.device", verbose: bool = False) -> dict:
    """Everything the device will tell us about itself, as JSON-able data."""
    info = {}
    for field in (rs.camera_info.name, rs.camera_info.serial_number,
                  rs.camera_info.firmware_version, rs.camera_info.usb_type_descriptor,
                  rs.camera_info.product_line):
        if dev.supports(field):
            info[str(field).split(".")[-1]] = dev.get_info(field)

    sensors = []
    for sensor in dev.query_sensors():
        entry = {"name": sensor.get_info(rs.camera_info.name), "options": {}}
        if sensor.is_depth_sensor():
            entry["depth_scale_m_per_unit"] = sensor.as_depth_sensor().get_depth_scale()
        for opt in sensor.get_supported_options():
            if sensor.is_option_read_only(opt):
                continue
            try:
                rng = sensor.get_option_range(opt)
                entry["options"][str(opt).split(".")[-1]] = {
                    "value": sensor.get_option(opt),
                    "min": rng.min, "max": rng.max, "step": rng.step,
                }
            except RuntimeError:
                pass  # some controls refuse to be read while streaming
        profiles = sensor.get_stream_profiles()
        entry["stream_profile_count"] = len(profiles)
        if verbose:
            entry["stream_profiles"] = [
                f"{p.stream_name()} {str(p.format()).split('.')[-1]} "
                f"{p.as_video_stream_profile().width()}x"
                f"{p.as_video_stream_profile().height()} @ {p.fps()}fps"
                for p in profiles
            ]
        sensors.append(entry)
    info["sensors"] = sensors
    return info


def describe_calibration(profile: "rs.pipeline_profile") -> dict:
    """Intrinsics/extrinsics — what you need to turn pixels into 3D rays.

    Directly relevant to the rat tracker: fx/fy give you degrees-per-pixel for
    servo aiming, and depth turns a detection box into a range. This is the
    factory calibration, which is the D435's headline advantage over the CSI
    stereo pair — no checkerboard session required.
    """
    out = {}
    for stream in (rs.stream.depth, rs.stream.color):
        try:
            vsp = profile.get_stream(stream).as_video_stream_profile()
        except RuntimeError:
            continue  # stream not enabled in this session
        i = vsp.get_intrinsics()
        out[vsp.stream_name().lower()] = {
            "fx": round(i.fx, 2), "fy": round(i.fy, 2),
            "ppx": round(i.ppx, 2), "ppy": round(i.ppy, 2),
            "width": i.width, "height": i.height,
            "fov_x_deg": round(2 * np.degrees(np.arctan2(i.width / 2, i.fx)), 1),
            "fov_y_deg": round(2 * np.degrees(np.arctan2(i.height / 2, i.fy)), 1),
            "deg_per_px_x": round(np.degrees(np.arctan2(1, i.fx)), 4),
            "distortion": str(i.model).split(".")[-1],
        }
    try:
        ext = (profile.get_stream(rs.stream.depth)
               .get_extrinsics_to(profile.get_stream(rs.stream.color)))
        out["depth_to_color_translation_mm"] = [round(v * 1000, 2) for v in ext.translation]
    except RuntimeError:
        pass
    return out


class Worker(threading.Thread):
    """Owns the RealSense pipeline on one thread.

    librealsense controls are not safe to poke from arbitrary request threads
    while another thread is pumping frames, so HTTP handlers never touch the
    device directly — they drop a command on a queue and this thread applies it
    between frames. The thread publishes the latest encoded JPEG under a lock;
    every viewer's MJPEG stream reads from that single slot.
    """

    # Probe stream combinations from richest to plainest and keep the first that
    # actually produces frames. On a USB 3 Jetson port the first one wins; the
    # fallbacks matter when the camera lands on a USB 2 hub.
    STREAM_SETS = [
        ("depth + color + IR", ("color", "ir")),
        ("depth + color", ("color",)),
        ("depth + IR", ("ir",)),
        ("depth only", ()),
    ]

    def __init__(self, output_dir: Path, jpeg_quality: int) -> None:
        super().__init__(daemon=True)
        self.output_dir = output_dir
        self.jpeg_quality = jpeg_quality
        self.commands: "queue.Queue[tuple]" = queue.Queue()
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self._jpeg: bytes | None = None
        self._seq = 0
        self.stop_event = threading.Event()

        last_err = None
        for label, extras in self.STREAM_SETS:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
            if "color" in extras:
                cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
            if "ir" in extras:
                cfg.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
            try:
                self.profile = self.pipeline.start(cfg)
                self.pipeline.wait_for_frames(10000)  # probe (also covers AE warmup)
                self.stream_label = label
                print(f"streaming: {label}")
                break
            except RuntimeError as e:
                last_err = e
                print(f"{label}: no frames arrived, trying a smaller stream set...")
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    pass
        else:
            raise RuntimeError(f"no stream combination produced frames ({last_err})")

        self.has_color = "color" in extras
        self.has_ir = "ir" in extras
        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        # Reproject depth into the color frame so pixel coords line up. Without
        # color there is nothing to align to (depth already shares the left IR
        # imager's viewpoint, so depth and IR line up natively).
        self.align = rs.align(rs.stream.color) if self.has_color else None
        self.colorizer = rs.colorizer()  # histogram-equalized depth colormap

        # Post-processing filters: smooth depth spatially, average over time,
        # and fill holes. Trades latency/detail for stability.
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.filters_on = False

        self.emitter_on = True
        self.preset_idx = 0
        self.preset_names = self._preset_names()
        self.probe_xy = (WIDTH // 2, HEIGHT // 2)

        self.calibration = describe_calibration(self.profile)
        self.status_msg = ""
        self.status_until = 0.0
        self.stats = {"fps": 0.0, "latency_ms": 0.0, "depth_coverage_pct": 0.0,
                      "probe_distance_m": 0.0, "encode_ms": 0.0}
        self.last_artifact: str | None = None

    def _preset_names(self) -> list[tuple[int, str]]:
        names = []
        if self.depth_sensor.supports(rs.option.visual_preset):
            rng = self.depth_sensor.get_option_range(rs.option.visual_preset)
            for v in range(int(rng.min), int(rng.max) + 1):
                desc = self.depth_sensor.get_option_value_description(
                    rs.option.visual_preset, v)
                if desc and desc != "UNKNOWN":
                    names.append((v, desc))
        return names

    def flash(self, msg: str) -> None:
        print(msg)
        self.status_msg, self.status_until = msg, time.time() + 4

    # --- command plumbing -------------------------------------------------
    def submit(self, action: str, value=None) -> None:
        self.commands.put((action, value))

    def _drain_commands(self, depth_frame, texture_frame) -> None:
        while True:
            try:
                action, value = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._apply(action, value, depth_frame, texture_frame)
            except RuntimeError as e:
                self.flash(f"{action} failed: {e}")

    def _apply(self, action, value, depth_frame, texture_frame) -> None:
        if action == "filters":
            self.filters_on = not self.filters_on if value is None else bool(value)
            self.flash(f"filters {'ON' if self.filters_on else 'OFF'}")
        elif action == "emitter":
            self.emitter_on = not self.emitter_on if value is None else bool(value)
            self.depth_sensor.set_option(rs.option.emitter_enabled, float(self.emitter_on))
            self.flash(f"IR emitter {'ON' if self.emitter_on else 'OFF'}"
                       " (watch the dot pattern in the IR view)")
        elif action == "preset":
            if not self.preset_names:
                self.flash("no visual presets supported")
                return
            self.preset_idx = (self.preset_idx + 1) % len(self.preset_names)
            preset_value, name = self.preset_names[self.preset_idx]
            self.depth_sensor.set_option(rs.option.visual_preset, float(preset_value))
            self.flash(f"visual preset: {name}")
        elif action == "probe":
            x, y = value
            self.probe_xy = (max(0, min(WIDTH - 1, int(x))),
                             max(0, min(HEIGHT - 1, int(y))))
        elif action == "pointcloud":
            if texture_frame is None:
                self.flash("point cloud export needs a color or IR stream")
                return
            pc = rs.pointcloud()
            pc.map_to(texture_frame)  # texture-map points with the color image
            points = pc.calculate(depth_frame)
            out = self.output_dir / f"realsense_cloud_{int(time.time())}.ply"
            points.export_to_ply(str(out), texture_frame)
            self.last_artifact = out.name
            self.flash(f"point cloud saved: {out.name} "
                       "(open with MeshLab / Blender / macOS Preview)")
        elif action == "snapshot":
            out = self.output_dir / f"realsense_shot_{int(time.time())}.png"
            with self.lock:
                jpeg = self._jpeg
            if jpeg is None:
                return
            cv2.imwrite(str(out), cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                               cv2.IMREAD_COLOR))
            self.last_artifact = out.name
            self.flash(f"saved {out.name}")

    # --- frame loop -------------------------------------------------------
    def run(self) -> None:
        fps, frames_seen, t0 = 0.0, 0, time.time()
        try:
            while not self.stop_event.is_set():
                try:
                    frames = self.pipeline.wait_for_frames(5000)
                except RuntimeError as e:
                    self.flash(f"frame wait failed: {e}")
                    continue
                if self.align:
                    frames = self.align.process(frames)
                depth = frames.get_depth_frame()
                if not depth:
                    continue

                # Latency from capture to "about to encode". Global time is on
                # by default, which puts frame timestamps in the host clock
                # domain and makes this subtraction meaningful.
                latency_ms = 0.0
                if depth.get_frame_timestamp_domain() == rs.timestamp_domain.global_time:
                    latency_ms = max(0.0, time.time() * 1000 - depth.get_timestamp())

                raw_depth = depth  # filters change the frame; probing wants either
                if self.filters_on:
                    depth = self.spatial.process(depth)
                    depth = self.temporal.process(depth)
                    depth = self.hole_filling.process(depth).as_depth_frame()

                texture = None  # frame used to texture point cloud exports
                depth_img = np.asanyarray(self.colorizer.colorize(depth).get_data())
                views, base_img = [], None
                if self.has_color:
                    texture = frames.get_color_frame()
                    base_img = np.asanyarray(texture.get_data()).copy()
                    views.append((base_img, "color"))
                views.append((depth_img, "aligned depth" if self.align else "depth"))
                if self.has_ir:
                    ir = frames.get_infrared_frame(1)
                    texture = texture or ir
                    ir_img = cv2.cvtColor(np.asanyarray(ir.get_data()), cv2.COLOR_GRAY2BGR)
                    if base_img is None:
                        base_img = ir_img
                    views.append((ir_img, "infrared L"))
                if base_img is not None:
                    views.append((cv2.addWeighted(base_img, 0.5, depth_img, 0.5, 0),
                                  "overlay"))

                self._drain_commands(raw_depth, texture)

                frames_seen += 1
                if time.time() - t0 >= 1.0:
                    fps, frames_seen, t0 = frames_seen / (time.time() - t0), 0, time.time()

                # Fraction of pixels with a valid range — the single most useful
                # number when comparing depth backends on the same scene.
                depth_np = np.asanyarray(depth.get_data())
                coverage = float((depth_np > 0).mean() * 100)

                mx, my = self.probe_xy
                dist = depth.get_distance(mx, my)
                for img, _ in views:
                    cv2.drawMarker(img, (mx, my), (255, 255, 255),
                                   cv2.MARKER_CROSS, 14, 1)
                label = f"{dist:.3f} m" if dist > 0 else "no depth"
                cv2.putText(views[0][0], f"({mx},{my}) {label}", (10, HEIGHT - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                header = (f"{fps:.0f} fps | {latency_ms:.0f} ms | "
                          f"filters {'ON' if self.filters_on else 'off'} | "
                          f"emitter {'ON' if self.emitter_on else 'off'}")
                cv2.putText(views[0][0], header, (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if time.time() < self.status_until:
                    cv2.putText(depth_img, self.status_msg, (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                for img, name in views:
                    cv2.putText(img, name, (10, HEIGHT - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                imgs = [v[0] for v in views]
                if len(imgs) <= 2:
                    mosaic = np.hstack(imgs)
                else:
                    while len(imgs) < 4:
                        imgs.append(np.zeros_like(imgs[0]))
                    mosaic = np.vstack([np.hstack(imgs[:2]), np.hstack(imgs[2:])])

                t_enc = time.time()
                ok, buf = cv2.imencode(".jpg", mosaic,
                                       [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ok:
                    continue
                encode_ms = (time.time() - t_enc) * 1000

                with self.frame_ready:
                    self._jpeg = buf.tobytes()
                    self._seq += 1
                    self.stats = {
                        "fps": round(fps, 1),
                        "latency_ms": round(latency_ms, 1),
                        "encode_ms": round(encode_ms, 1),
                        "depth_coverage_pct": round(coverage, 1),
                        "probe_distance_m": round(dist, 3),
                    }
                    self.frame_ready.notify_all()
        finally:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass

    def frames(self):
        """Yield encoded JPEGs as they are produced, one per viewer."""
        last = -1
        while not self.stop_event.is_set():
            with self.frame_ready:
                if not self.frame_ready.wait_for(lambda: self._seq != last, timeout=5.0):
                    continue
                last, jpeg = self._seq, self._jpeg
            if jpeg:
                yield (b"--" + BOUNDARY.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")

    def snapshot_state(self) -> dict:
        with self.lock:
            stats = dict(self.stats)
        preset = (self.preset_names[self.preset_idx][1]
                  if self.preset_names else "n/a")
        return {
            "stats": stats,
            "filters_on": self.filters_on,
            "emitter_on": self.emitter_on,
            "preset": preset,
            "probe": {"x": self.probe_xy[0], "y": self.probe_xy[1]},
            "streams": self.stream_label,
            "status": self.status_msg if time.time() < self.status_until else "",
            "last_artifact": self.last_artifact,
        }


PAGE = """<!doctype html>
<title>RealSense D435 explorer</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:16px}
 h1{font-size:16px;margin:0 0 12px;font-weight:600}
 h1 span{color:#888;font-weight:400}
 #wrap{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
 /* Cap the mosaic column so the control panel sits beside it on a laptop
    screen. The cap goes on the column, not just the <img>: the mosaic is
    1280x960 natively, and a flex item sizes to that max-content width and
    shoves the panel onto the next row. */
 #col{flex:0 1 860px;min-width:320px;max-width:860px}
 #view{width:100%;cursor:crosshair;border:1px solid #333;border-radius:4px;display:block}
 aside{min-width:260px;flex:1}
 button{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:7px 12px;margin:0 6px 6px 0;cursor:pointer;font:inherit}
 button:hover{background:#2d2d2d;border-color:#666}
 table{border-collapse:collapse;width:100%;margin:8px 0}
 td{padding:2px 8px 2px 0;vertical-align:top}
 td:first-child{color:#888;white-space:nowrap}
 .on{color:#4ade80}.off{color:#888}
 #status{min-height:20px;color:#fbbf24;margin:8px 0}
 details{margin-top:12px}summary{cursor:pointer;color:#888}
 pre{background:#000;padding:10px;border-radius:4px;overflow:auto;max-height:340px;font-size:12px}
 .hint{color:#666;font-size:12px}
</style>
<h1>RealSense D435 <span id="streams"></span></h1>
<div id="wrap">
  <div id="col">
    <img id="view" src="/stream.mjpg">
    <p class="hint">Click the image to probe depth at that pixel (any panel).</p>
  </div>
  <aside>
    <div>
      <button onclick="ctl('filters')">Filters</button>
      <button onclick="ctl('emitter')">IR emitter</button>
      <button onclick="ctl('preset')">Next preset</button>
      <button onclick="ctl('snapshot')">Snapshot</button>
      <button onclick="ctl('pointcloud')">Point cloud</button>
    </div>
    <div id="status"></div>
    <table>
      <tr><td>fps</td><td id="fps"></td></tr>
      <tr><td>latency</td><td id="lat"></td></tr>
      <tr><td>jpeg encode</td><td id="enc"></td></tr>
      <tr><td>depth coverage</td><td id="cov"></td></tr>
      <tr><td>probe</td><td id="probe"></td></tr>
      <tr><td>filters</td><td id="filters"></td></tr>
      <tr><td>emitter</td><td id="emitter"></td></tr>
      <tr><td>preset</td><td id="preset"></td></tr>
      <tr><td>saved</td><td id="artifact"></td></tr>
    </table>
    <details><summary>Calibration (intrinsics / extrinsics)</summary><pre id="calib"></pre></details>
    <details><summary>Device &amp; sensors</summary><pre id="info"></pre></details>
  </aside>
</div>
<script>
const view = document.getElementById('view');
function ctl(action, value=null){
  fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action, value})});
}
view.addEventListener('click', e => {
  const r = view.getBoundingClientRect();
  // Map the click into one panel's 640x480 coordinate space: scale to natural
  // pixels, then modulo the panel size so any of the four panels works.
  const sx = view.naturalWidth / r.width, sy = view.naturalHeight / r.height;
  const x = Math.round((e.clientX - r.left) * sx) % 640;
  const y = Math.round((e.clientY - r.top) * sy) % 480;
  ctl('probe', [x, y]);
});
const t = (id, v) => document.getElementById(id).textContent = v;
async function poll(){
  try {
    const s = await (await fetch('/api/status')).json();
    t('fps', s.stats.fps);
    t('lat', s.stats.latency_ms + ' ms');
    t('enc', s.stats.encode_ms + ' ms');
    t('cov', s.stats.depth_coverage_pct + ' %');
    t('probe', `(${s.probe.x}, ${s.probe.y}) ` +
      (s.stats.probe_distance_m > 0 ? s.stats.probe_distance_m + ' m' : 'no depth'));
    t('streams', s.streams);
    t('preset', s.preset);
    t('status', s.status);
    for (const [id, on] of [['filters', s.filters_on], ['emitter', s.emitter_on]]) {
      const el = document.getElementById(id);
      el.textContent = on ? 'ON' : 'off';
      el.className = on ? 'on' : 'off';
    }
    const a = document.getElementById('artifact');
    a.innerHTML = s.last_artifact
      ? `<a href="/files/${s.last_artifact}" style="color:#60a5fa">${s.last_artifact}</a>` : '-';
  } catch (e) {}
}
setInterval(poll, 500); poll();
fetch('/api/calibration').then(r => r.json())
  .then(d => t('calib', JSON.stringify(d, null, 2)));
fetch('/api/info').then(r => r.json())
  .then(d => t('info', JSON.stringify(d, null, 2)));
</script>
"""


def build_app(worker: Worker) -> FastAPI:
    app = FastAPI(title="RealSense D435 web explorer")
    device_info = describe_device(worker.device)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        return StreamingResponse(
            worker.frames(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status() -> JSONResponse:
        return JSONResponse(worker.snapshot_state())

    @app.get("/api/calibration")
    def calibration() -> JSONResponse:
        return JSONResponse(worker.calibration)

    @app.get("/api/info")
    def info() -> JSONResponse:
        return JSONResponse(device_info)

    @app.post("/api/control")
    async def control(payload: dict) -> JSONResponse:
        action = payload.get("action")
        if action not in {"filters", "emitter", "preset", "probe",
                          "snapshot", "pointcloud"}:
            return JSONResponse({"ok": False, "error": "unknown action"},
                                status_code=400)
        worker.submit(action, payload.get("value"))
        return JSONResponse({"ok": True})

    @app.get("/files/{name}")
    def files(name: str):
        # Serve only the flat basename out of the output dir — no traversal.
        path = worker.output_dir / Path(name).name
        if not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8090,
                        help="HTTP port (default 8090; 8000 is left free for rt_200)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--output-dir", type=Path, default=Path("realsense_captures"),
                        help="where snapshots and point clouds are written")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--info", action="store_true",
                        help="dump device capabilities as JSON and exit (no server)")
    parser.add_argument("--verbose", action="store_true",
                        help="with --info, also list every stream profile")
    args = parser.parse_args()

    if len(rs.context().query_devices()) == 0:
        sys.exit("No RealSense device found. Is the D435 plugged into USB 3?")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = Worker(args.output_dir, args.jpeg_quality)

    if args.info:
        print(json.dumps({"device": describe_device(worker.device, args.verbose),
                          "calibration": worker.calibration}, indent=2))
        worker.pipeline.stop()
        return

    worker.start()
    print(f"\nRealSense web explorer on http://{args.host}:{args.port}/")
    try:
        # An MJPEG response never completes on its own, so a graceful shutdown
        # would otherwise block forever on whatever browser tabs are watching.
        # Cap the wait; uvicorn then force-closes the streaming connections.
        uvicorn.run(build_app(worker), host=args.host, port=args.port,
                    log_level="warning", timeout_graceful_shutdown=5)
    finally:
        # Join rather than relying on the daemon flag, so the frame loop gets to
        # run pipeline.stop() and hand the camera back cleanly.
        worker.stop_event.set()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
