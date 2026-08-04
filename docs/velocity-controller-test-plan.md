# Velocity-form controller — test plan (branch: velocity-control)

## What changed
1. **Hardware motion shaping** (step 1): `configure_motion()` writes the STS3215
   `Acceleration` register (config `tracking.servo_acceleration: 30`) once at
   startup — the servo ramps every position goal itself. `servo_max_speed`
   optionally caps `Goal_Velocity` (0 = untouched).
2. **Velocity-form outer loop** (step 2): `VelocityFormController` outputs a
   velocity (`kp * error`, per-second gains, FPS-independent), acceleration-
   limits it (braking is instant), integrates into a commanded position, and
   writes Goal_Position. Error is vs its own commanded state (first-order
   stable) with a `reconcile_rate` leak toward measured. Readback now 20Hz and
   includes `Present_Velocity` (sign-decoded) for the optional damping terms.
3. Config-switchable: `tracking.velocity_control.enabled: false` reverts to
   the AngularBeliefController — instant A/B via restart.

## On the Jetson
```bash
cd ~/rat-inference
git fetch && git checkout velocity-control
./run.sh build          # rt_200/belief/hardware changed -> image rebuild
ratbot reload
docker logs -f ratbot-demo   # expect: "Motion shaping: yaw Acceleration=30",
                             # "Tracking controller: velocity-form"
```

## Test sequence
1. Stationary can: settles dead, no hunting (match yesterday's baseline).
2. Slow lateral sweep: continuous, no staccato.
3. Fast sweep + reversal: watch for overshoot; "Velocity control:" logs show
   err / vel_cmd / meas_vel every 10 ticks.
4. A/B: flip `velocity_control.enabled: false`, `ratbot restart`, compare feel.

## Tuning knobs (config.yaml, mounted -> restart only)
- `kp_yaw/kp_pitch` (6.0/5.5 ≈ yesterday's feel; raise for snappier)
- `max_accel_raw_per_s2` 3500 (lower = softer launches)
- `servo_acceleration` 30 (lower = smoother/slower hardware ramp; 0 = off)
- `damping_*` uses Present_Velocity whose units are UNVERIFIED — check
  meas_vel magnitudes in the log vs expected raw/s before enabling (start 0.05)
- if 20Hz readback causes bus errors, drop `motor_readback_fps` to 15

## Rollback
`git checkout main && ratbot reload` (or just flip the enabled flag; main
still runs yesterday's frozen demo build).
