# Repository Cleanup Phase 6

Date: 2026-07-16

This cleanup consolidates the vision-side utilities and artifacts under `tools/vision/`.

## What Moved

- Calibration tools moved to `tools/vision/calibration/`.
- Recalibration source images moved to `tools/vision/calibration/images_recal/`.
- Recalibration outputs moved to `tools/vision/calibration/output_recal/`.
- Dataset generation, labeler, and dataset cleanup tools moved to `tools/vision/dataset/`.
- Standalone image/video inference moved to `tools/vision/inference/`.
- Training moved to `tools/vision/training/`.
- `sample-prompts.txt` moved to `tools/vision/sample-prompts.txt`.
- `main.py` moved to `tools/vision/legacy/main.py`.
- Root sample/reference images moved to `tools/vision/assets/`.

## What Was Removed

The old root-level compatibility wrappers were removed now that the canonical commands live under `tools/hardware/...` and `tools/vision/...`.

Before removal, the previous root layout and old tool directories were copied to:

```text
/tmp/rat-inference-root-cleanup-20260716-125144
```

## Runtime Path

The default stereo calibration path used by `rt_200.py` is now:

```text
tools/vision/calibration/output_recal/stereo_calibration.npz
```

Runtime should still use the measured lens-center baseline override:

```bash
--baseline-override 57.5
```
