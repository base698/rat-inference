# The math in this repository

A self-check guide: starting from high-school algebra, every technique this
codebase uses, grounded in the actual files, functions, and numbers. If you
can follow each section, you know enough to read — and change — that part of
the system. Work top to bottom; each level builds on the previous one.

The one-paragraph system: a camera sees a target, a YOLO model draws a box
around it, geometry converts the box center into servo angles, a statistical
filter ("the belief") smooths noisy detections into a stable target estimate,
and a feedback controller drives two servos toward it — 20 times per second.

---

## 1. Algebra: proportions, linear equations, unit conversion

**Pixels → degrees** (`ratbot/robot/observation.py`, `pixels_to_angle`):

```
angle = pixel_error × (FOV / image_width)
```

The camera sees 60° across 960 pixels (`config.yaml: fov_horizontal`), so one
pixel = 60/960 = **0.0625°**. A can 96 pixels right of the crosshair is 6°
right. That's the whole conversion — a proportion.

**Degrees → raw servo units** (`angle_to_servo_raw`): servos speak "raw"
integers, not degrees. The yaw servo covers its angular range with raw values
1600–3100 (`config.yaml: servos`), so:

```
raw_per_degree = (yaw_max − yaw_min) / yaw_range_degrees = 1500 / 180 ≈ 8.33
```

Chaining both: that 96-pixel error = 6° × 8.33 = **50 raw units**. Every
"error_raw" in the logs is this number.

**Linear interpolation** (`ratbot/robot/aiming.py`, pitch compensation
`points`): given calibration points (pitch₁, offset₁) and (pitch₂, offset₂),
the offset at a pitch in between is the straight line through them:

```
offset = offset₁ + (pitch − pitch₁) × (offset₂ − offset₁)/(pitch₂ − pitch₁)
```

**Clamping** (everywhere, e.g. `ServoBounds`): `max(lo, min(hi, x))` — keep a
value inside hard limits. The last line of defense before any servo write.

*Check yourself:* a detection lands 200 px left of the crosshair. How many
raw yaw units is that? (200 × 0.0625 × 8.33 ≈ 104, to the left.)

---

## 2. Geometry and trigonometry: similar triangles, parallax, angles

**Stereo depth** (`ratbot/vision/stereo_depth.py`): two cameras 52.5 mm apart
(`--baseline-override 52.5`) see the target at slightly different horizontal
pixel positions; the difference is the *disparity*. Similar triangles give:

```
Z_mm = focal_length_px × baseline_mm / disparity_px
```

Closer objects → larger disparity. A can at disparity 20 px with a ~950 px
focal length sits at ≈ 950 × 52.5 / 20 ≈ 2.5 m. When the overlay says
"low texture", the disparity match failed and Z is garbage. The demo uses
depth for the 3-D world view: world tracking runs in *shadow mode*, building
depth-based tracks for display, while the servos are driven by the angular
controller — chosen exactly because it keeps tracking through the moments
when depth drops out.

**Parallax / bore offset** (`aiming.py`, `depth_adjust_px`): the camera sits
82 mm above the barrel, so where the camera should "look" to make the barrel
hit depends on distance. For small angles the pixel shift is:

```
Δy_px ≈ focal_y_px × offset_mm × (1/depth_mm − 1/reference_mm)
```

At the calibrated reference distance the shift is zero; nearer targets need
the crosshair lower. This is why calibration "is only valid at one distance"
without depth compensation.

**Angles and radians**: the world view (`static/index.html`) converts servo
raw → degrees (÷8.33) → radians (×π/180) to rotate the 3-D turret model, and
the world tracker uses `atan` to turn (x, z) positions into yaw angles. If
you remember SOH-CAH-TOA and that 180° = π radians, you have all of it.

---

## 3. Rates of change: velocity, integration, acceleration (pre-calculus)

**Estimating velocity — finite differences** (`ratbot/robot/belief.py`,
`AngularTargetBelief.update`):

```
velocity ≈ (position_now − position_before) / (time_now − time_before)
```

That's the slope of a line through two samples — `vel=(…) raw/s` in the logs.
Noisy inputs make noisy slopes, which is why the estimate is smoothed (§4).

**Integration — velocity back to position** (`VelocityFormController`):

```
commanded_position += velocity × dt        (Euler integration)
```

The controller *decides* a velocity, then accumulates it into a position each
50 ms tick. Summing rate × small-time-slice is the discrete version of an
integral.

**Acceleration limiting** (`_limit_velocity`): velocity may not change by
more than `max_accel × dt` per tick (3500 raw/s² in config) — the discrete
version of bounding the second derivative. One asymmetry, learned from a unit
test that caught the commanded position creeping 4 raw past its target:
*braking is instant*; only speed-ups are rate-limited.

**Why per-second gains matter**: the older controller applied `kp × error`
per *tick*, so its behavior changed with loop rate. The velocity form uses
`kp` in units of (raw/s per raw of error) and multiplies by measured `dt` —
the same physical behavior at 20 Hz or 50 Hz.

---

## 4. Statistics and probability: the belief

This is the heart of the tracker (`ratbot/robot/belief.py`). A detector gives
you noisy, occasionally-missing, occasionally-wrong observations. The
"belief" is a running statistical estimate of where the target really is.

**Exponential moving average — weighted averages with memory**
(`update_alpha: 0.55` in config):

```
belief = α × observation + (1 − α) × belief_before
```

Unroll it and each past observation gets weight α(1−α)ⁿ — a geometric decay.
α near 1 trusts new data (fast, jittery); α near 0 trusts history (smooth,
laggy). The live value 0.55 was tuned on hardware: 0.75 followed box jitter
visibly, lower lagged a moving can.

**The Bayesian reading**: the belief is a *prior*, each detection a
*measurement*, and α is how much you trust measurement vs. prior. An EMA is
exactly a 1-D Kalman filter with constant gain. The full Kalman machinery —
uncertainty that grows without measurements and shrinks with them — lives in
the world tracker (`ratbot/tracking/`), where each stereo measurement carries
a 3×3 *covariance matrix* (`StereoPointMeasurement.covariance_camera`, units
mm²: variance on the diagonal, correlations off it).

**Exponential decay — forgetting** (`miss_decay: 0.82`,
`velocity_decay: 0.90`): each tick without a detection multiplies confidence
by a constant < 1, i.e. `conf(n) = conf₀ × 0.82ⁿ` — geometric/exponential
decay, the same math as radioactive half-life. When confidence falls below
`min_confidence: 0.15` or age exceeds `max_age`, `get_active()` returns
nothing and the turret freezes rather than chasing a ghost. These were tuned
the night the turret "wandered" during detection dropouts: faster forgetting
= freeze, not wander.

**Outlier rejection — the reseed gate** (`reseed_distance_raw: 420`,
`reseed_confirmations: 2`, `reseed_match_distance_raw: 300`): an observation
that jumps > 420 raw from the belief is treated as a suspected outlier and
*held* until a second observation lands within 300 raw of it. That's an
informal hypothesis test: one far sample is probably noise; two consistent
far samples mean the world actually changed, so the belief re-seeds
(snaps) there. Field lesson encoded in `tests/robot/test_belief_reseed.py`:
with the gate too tight (160/120), a can carried smoothly sideways looked
like a stream of outliers and the turret moved in stutters.

**Confidence thresholds as probability filters**: YOLO's per-box confidence
approximates P(this box is really a target). The pipeline uses it three
ways: detection cutoff (`--confidence 0.45`), minimum to keep a belief alive
(0.15), and a *higher* bar (0.55) to accept an outlier jump — extraordinary
claims require better evidence.

**Model quality metrics** (training run `runs/yolo11n-2026-08-03`):
*precision* = of the boxes predicted, how many were right (0.996);
*recall* = of the real cans, how many were found (1.0);
*mAP50* = area under the precision-recall curve at 50% box-overlap (0.995).
Overlap is IoU: intersection area / union area of predicted vs. true box.

---

## 5. Control theory: feedback loops (uses everything above)

**Proportional control** (`AngularBeliefController`, `VelocityFormController`
`kp` terms): each tick, correct by a fraction of the error. With gain k per
tick, the error follows

```
error(n) = error₀ × (1 − k)ⁿ
```

— a geometric sequence (algebra again!). For 0 < k < 1 it converges
monotonically; it *cannot* overshoot by itself. That guarantee is why the
demo controller is P-only.

**Integral and derivative terms**: the classic PID adds ∫error·dt (removes
steady-state offset) and d(error)/dt (damping). A war story is pinned in the
git history: the derivative term differentiates *noise* too. Our error signal
flipped sign every tick (commanded-vs-measured position mismatch at 10 Hz
readback vs 20 Hz control), and kd amplified that flip into full-speed
square-wave commands. kd is now 0; damping instead uses the servo's own
measured velocity (`damping_*` on `Present_Velocity`), which is smooth.

**Latency compensation — time alignment** (`pose_at` in
`ratbot/app/video.py`): a camera frame shows the world ~50–100 ms ago, so an
observation must be computed against the servo pose *at frame time*, not
now:

```
target = pose(t_frame) + k × pixel_error(t_frame),  t_frame ≈ now − 0.05 s
```

The pose history is a 30 Hz ring buffer; `pose_at` walks it backwards for the
newest entry at-or-before the target time. Using the *current* pose instead
injects the turret's own motion into the observation — the bug that made the
turret orbit a stationary can for an entire evening. Both over- and
under-compensating oscillate, in opposite directions; 0.05 s
(`RATBOT_CAMERA_LATENCY_S`) is the empirically stable value.

**The velocity-form loop, end to end** (`VelocityFormController.track_once`):

```
error   = belief_position − commanded_position          (§1 units)
want_v  = kp × error − damping × measured_velocity      (P control)
v       = accel_limit(clip(want_v))                     (§3)
cmd    += v × dt                                        (integration, §3)
goal    = clamp(round(cmd), bounds)                     (§1)
```

plus a slow "reconcile" leak (2 s⁻¹ exponential pull of commanded toward
measured — §4's decay math again) so accumulated command-vs-reality offset
bleeds away without creating a second feedback path.

**One bit of binary**: Feetech servos report velocity in sign-magnitude
form — bit 15 is the sign, bits 0–14 the magnitude
(`TrackingServoController._decode_feetech_signed`). `0x8000 | 120` means
−120, not 32888.

---

## 6. Self-check

You're equipped to work on this repo if you can answer these (all derivable
from the sections above, all with real repo numbers):

1. A detection center is 48 px above the crosshair. How many raw pitch units
   is that, at 45° vertical FOV over 720 px and pitch range 1–500 raw over
   60°? (48 × 45/720 = 3° × 499/60 ≈ 25 raw.)
2. Disparity is 10 px. Roughly how far is the target? (~5 m — and why should
   you distrust it if the overlay says "low texture"?)
3. With α = 0.55, how much of a single observation's influence remains after
   three more updates? ((1−0.55)³ ≈ 9%.)
4. Confidence is 1.0 and `miss_decay` is 0.82. About how many missed ticks
   until the belief deactivates at 0.15? (ln 0.15 / ln 0.82 ≈ 10.)
5. kp = 6 s⁻¹ velocity-form gain, dt = 0.05 s. What per-tick fraction of the
   error is corrected, and why can't it overshoot? (6 × 0.05 = 30%; a
   geometric approach with ratio 0.7 is monotone.)
6. Why must the pose paired with a pixel error come from ~0.05 s ago rather
   than "now"? What failure do you get with each sign of the timing error?

---

## 7. What's next: pluggable controllers and the RL seam

The controller is now a **configuration choice**
(`config.yaml → tracking.controller: angular | velocity | rl`), resolved by a
registry (`ratbot/robot/controllers.py`). Every controller implements the
same loop protocol (`track_once / reset / run / start`), and new ones
register with a decorator — no tracker changes:

```python
@register_controller("my-controller")
def _build(robot, belief, bounds, control_fps, options):
    return MyController(...)
```

The intended next controller is **reinforcement-learning based**, and the
seam for it already exists as `RLControllerStub`:

- **Where it plugs in**: `VelocityFormController._desired_velocities()` is
  the single *decision* hook — inputs in, desired axis velocities out. The
  classical controller computes `kp × error` there; the RL stub calls
  `policy(observation)` instead. Everything after the decision — clipping,
  acceleration limiting, integration into position goals, bounds clamping —
  is shared, inherited safety plumbing. A policy cannot command anything the
  classical controller couldn't.
- **Observation** (10 floats, `build_observation`): yaw/pitch error, measured
  servo velocities (`Present_Velocity`), estimated target velocities from the
  belief (§4), detection confidence and age, and the previous action.
  Natural extensions: bbox width/height and stereo depth (distance cues the
  classical controller ignores).
- **Action**: `[yaw, pitch]` in [−1, 1], scaled to the max velocities — the
  same action space the classical controller effectively occupies.
- **Today's stub policy returns (0, 0)** — selecting `controller: rl` on
  hardware holds position safely. Training it is the project: start with the
  policy as a *residual* on the classical output (`classical + scale × rl`),
  reward small pixel error and small commanded acceleration (smoothness),
  and train against logged episodes / the track-recording store before ever
  closing the loop on hardware. The statistics you need to reason about
  rewards, expectations, and exploration are the same §4 tools, one level up.

Tests pinning the seam: `tests/robot/test_controller_registry.py` (selection,
option plumbing, unknown-name failure, stub-holds-still, a custom policy
actually driving the goal).
