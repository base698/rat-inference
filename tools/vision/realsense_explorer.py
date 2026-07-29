#!/usr/bin/env python3
"""Interactive RealSense D435 explorer.

A single-file tour of the D435's capabilities: device/sensor introspection,
synchronized color + depth + infrared streaming, depth-to-color alignment,
post-processing filters, IR emitter control, visual presets, and point cloud
export.

Usage (macOS needs root to claim the USB device):
    sudo .venv/bin/python tools/vision/realsense_explorer.py
    sudo .venv/bin/python tools/vision/realsense_explorer.py --info   # dump capabilities, no GUI

Live view layout (2x2 mosaic):
    [ color         ][ depth colormap ]
    [ infrared left ][ color+depth overlay ]

Keys:
    q / ESC  quit
    f        toggle post-processing filters (spatial + temporal + hole-filling)
    e        toggle IR emitter (watch the dot pattern appear in the IR view)
    v        cycle visual presets (High Accuracy, High Density, ...)
    p        export point cloud of current frame to .ply
    s        save screenshot of the mosaic
Mouse:      hover anywhere to read the depth at that pixel

Hardware background, briefly: the D435 computes depth on-camera by stereo
matching its two IR imagers (85 deg FOV, global shutter). The IR emitter
projects a dot pattern to add texture so matching works on blank surfaces.
The RGB camera is a separate sensor (rolling shutter, different FOV) at a
physical offset, which is why depth must be *aligned* to color before you
can look up the distance of an RGB pixel.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not installed. Run: uv sync --extra realsense")

WIDTH, HEIGHT, FPS = 640, 480, 30


def dump_info(dev: "rs.device", verbose: bool) -> None:
    """Print everything the device will tell us about itself."""
    print("=" * 70)
    for info in (rs.camera_info.name, rs.camera_info.serial_number,
                 rs.camera_info.firmware_version, rs.camera_info.usb_type_descriptor,
                 rs.camera_info.product_line, rs.camera_info.physical_port):
        if dev.supports(info):
            print(f"  {str(info).split('.')[-1]:22s} {dev.get_info(info)}")
    print("=" * 70)

    for sensor in dev.query_sensors():
        print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")

        # Options are the runtime-tunable knobs (exposure, gain, emitter power...)
        if sensor.is_depth_sensor():
            scale = sensor.as_depth_sensor().get_depth_scale()
            print(f"  depth scale: {scale:.4g} m/unit (z16 value * {scale:.4g} = meters)")
            print("  options:")
            for opt in sensor.get_supported_options():
                if sensor.is_option_read_only(opt):
                    continue
                try:
                    rng = sensor.get_option_range(opt)
                    cur = sensor.get_option(opt)
                    print(f"    {str(opt).split('.')[-1]:32s} = {cur:g}  "
                          f"[{rng.min:g}..{rng.max:g} step {rng.step:g}]")
                except RuntimeError:
                    pass
        else:
            # Reading RGB option values over UVC segfaults librealsense's macOS
            # backend (not catchable from Python), so list names only.
            names = [str(o).split(".")[-1] for o in sensor.get_supported_options()
                     if not sensor.is_option_read_only(o)]
            print(f"  options (names only, value reads crash on macOS): {', '.join(names)}")

        # Stream profiles are every (stream, format, resolution, fps) combo
        profiles = sensor.get_stream_profiles()
        print(f"  stream profiles: {len(profiles)}")
        if verbose:
            for p in profiles:
                v = p.as_video_stream_profile()
                print(f"    {p.stream_name():10s} {str(p.format()).split('.')[-1]:6s} "
                      f"{v.width()}x{v.height()} @ {p.fps()}fps")


def print_intrinsics(profile: "rs.pipeline_profile") -> None:
    """Intrinsics/extrinsics — what you need to turn pixels into 3D rays.

    Directly relevant to the rat tracker: fx/fy give you degrees-per-pixel
    for servo aiming, and depth turns a detection box into a range.
    """
    for stream in (rs.stream.depth, rs.stream.color):
        try:
            vsp = profile.get_stream(stream).as_video_stream_profile()
        except RuntimeError:
            continue  # stream not enabled in this session
        i = vsp.get_intrinsics()
        fov_x = 2 * np.degrees(np.arctan2(i.width / 2, i.fx))
        fov_y = 2 * np.degrees(np.arctan2(i.height / 2, i.fy))
        print(f"{vsp.stream_name():8s} intrinsics: fx={i.fx:.1f} fy={i.fy:.1f} "
              f"ppx={i.ppx:.1f} ppy={i.ppy:.1f}  FOV {fov_x:.1f} x {fov_y:.1f} deg  "
              f"distortion={str(i.model).split('.')[-1]}")

    try:
        depth_p = profile.get_stream(rs.stream.depth)
        color_p = profile.get_stream(rs.stream.color)
        ext = depth_p.get_extrinsics_to(color_p)
        t = np.array(ext.translation) * 1000
        print(f"depth->color extrinsics: translation [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm")
    except RuntimeError:
        pass


class Explorer:
    # The macOS backend can't always deliver every stream at once, so probe
    # combinations from richest to plainest and keep the first one that
    # actually produces frames.
    STREAM_SETS = [
        ("depth + color + IR", ("color", "ir")),
        ("depth + color", ("color",)),
        ("depth + IR", ("ir",)),
        ("depth only", ()),
    ]

    def __init__(self) -> None:
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
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
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
        self.mouse_xy = (WIDTH // 2, HEIGHT // 2)
        self.status = ""
        self.status_until = 0.0

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
        self.status, self.status_until = msg, time.time() + 3

    def toggle_emitter(self) -> None:
        self.emitter_on = not self.emitter_on
        self.depth_sensor.set_option(rs.option.emitter_enabled, float(self.emitter_on))
        self.flash(f"IR emitter {'ON' if self.emitter_on else 'OFF'}"
                   " (check dot pattern in IR view and depth quality)")

    def cycle_preset(self) -> None:
        if not self.preset_names:
            self.flash("no visual presets supported")
            return
        self.preset_idx = (self.preset_idx + 1) % len(self.preset_names)
        value, name = self.preset_names[self.preset_idx]
        self.depth_sensor.set_option(rs.option.visual_preset, float(value))
        self.flash(f"visual preset: {name}")

    def export_ply(self, depth_frame: "rs.depth_frame", color_frame: "rs.video_frame") -> None:
        pc = rs.pointcloud()
        pc.map_to(color_frame)  # texture-map points with the color image
        points = pc.calculate(depth_frame)
        out = Path(f"realsense_cloud_{int(time.time())}.ply")
        points.export_to_ply(str(out), color_frame)
        self.flash(f"point cloud saved to {out} (open with MeshLab / Blender / macOS Preview)")

    def run(self) -> None:
        win = "RealSense D435 explorer"
        cv2.namedWindow(win)
        cv2.setMouseCallback(
            win, lambda ev, x, y, flags, param:
            setattr(self, "mouse_xy", (x % WIDTH, y % HEIGHT)))

        fps, frames_seen, t0 = 0.0, 0, time.time()
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                if self.align:
                    frames = self.align.process(frames)
                depth = frames.get_depth_frame()
                if not depth:
                    continue

                if self.filters_on:
                    depth = self.spatial.process(depth)
                    depth = self.temporal.process(depth)
                    depth = self.hole_filling.process(depth).as_depth_frame()

                texture = None  # frame used to texture point cloud exports
                depth_img = np.asanyarray(self.colorizer.colorize(depth).get_data())
                views = []
                base_img = None  # what depth gets blended over in the overlay
                if self.has_color:
                    texture = frames.get_color_frame()
                    base_img = np.asanyarray(texture.get_data())
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
                    overlay = cv2.addWeighted(base_img, 0.5, depth_img, 0.5, 0)
                    views.append((overlay, "overlay"))

                frames_seen += 1
                if time.time() - t0 >= 1.0:
                    fps, frames_seen, t0 = frames_seen / (time.time() - t0), 0, time.time()

                mx, my = self.mouse_xy
                dist = depth.get_distance(mx, my)
                for img, _ in views:
                    cv2.drawMarker(img, (mx, my), (255, 255, 255),
                                   cv2.MARKER_CROSS, 12, 1)
                label = f"{dist:.3f} m" if dist > 0 else "no depth"
                cv2.putText(views[0][0], f"({mx},{my}) {label}", (10, HEIGHT - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                header = (f"{fps:.0f} fps | filters {'ON' if self.filters_on else 'off'} | "
                          f"emitter {'ON' if self.emitter_on else 'off'} | "
                          f"f/e/v/p/s keys, q quits")
                cv2.putText(views[0][0], header, (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if time.time() < self.status_until:
                    cv2.putText(depth_img, self.status, (10, 22),
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
                cv2.imshow(win, mosaic)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("f"):
                    self.filters_on = not self.filters_on
                    self.flash(f"filters {'ON' if self.filters_on else 'OFF'}")
                elif key == ord("e"):
                    self.toggle_emitter()
                elif key == ord("v"):
                    self.cycle_preset()
                elif key == ord("p"):
                    if texture:
                        self.export_ply(depth, texture)
                    else:
                        self.flash("point cloud export needs a color or IR stream")
                elif key == ord("s"):
                    out = Path(f"realsense_shot_{int(time.time())}.png")
                    cv2.imwrite(str(out), mosaic)
                    self.flash(f"saved {out}")
        finally:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--info", action="store_true",
                        help="dump device capabilities and exit (no GUI)")
    parser.add_argument("--verbose", action="store_true",
                        help="with --info, also list every stream profile")
    args = parser.parse_args()

    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        sys.exit("No RealSense device found. Is the D435 plugged into USB 3?")

    # Start streaming *before* introspection: querying stream profiles (and on
    # macOS, most other things) requires the device to be powered, and starting
    # the pipeline is what powers it.
    explorer = None
    for attempt in (1, 2):
        try:
            explorer = Explorer()
            break
        except RuntimeError as e:
            if "power state" not in str(e) and "failed to open" not in str(e).lower():
                raise
            if os.geteuid() != 0:
                sys.exit("\nCould not claim the USB device"
                         " (macOS needs root for this):\n"
                         "  sudo .venv/bin/python tools/vision/realsense_explorer.py")
            if attempt == 1:
                # A previous crash can leave the camera wedged; a hardware
                # reset re-enumerates it on the USB bus, same as replugging.
                # Run it in a subprocess: on a wedged device the reset call
                # itself can segfault, and this keeps that crash contained.
                print("Camera is in a bad state, attempting hardware reset...")
                subprocess.run(
                    [sys.executable, "-c",
                     "import pyrealsense2 as rs;"
                     "rs.context().query_devices()[0].hardware_reset()"],
                    timeout=15, check=False)
                time.sleep(6)
            else:
                sys.exit("Could not open the camera even after a reset — "
                         "unplug and replug it, then retry.")

    # Query intrinsics immediately after start, before anything else touches
    # the sensors — the macOS backend gets progressively wedged by control
    # queries during streaming, so order matters and failure is non-fatal.
    try:
        print_intrinsics(explorer.profile)
    except RuntimeError as e:
        print(f"(intrinsics query failed, continuing: {e})")

    if args.info:
        dump_info(explorer.profile.get_device(), verbose=args.verbose)
        try:
            explorer.pipeline.stop()
        except RuntimeError:
            pass
        return

    print("\nStreaming. Hover the window to read depth; keys: f/e/v/p/s, q quits.")
    explorer.run()


if __name__ == "__main__":
    main()
