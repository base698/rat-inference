# Red Bull can dataset (960 demo model)

Single-class (`redbull`) dataset for the stage-demo turret target, built for
the 960×720 rig config. Pipeline:

## 1. Capture from the rig (Jetson)
Bot must be running (`ratbot start`). Then open **http://jetson:8010** —
live clean feed (no overlays, via the new `/raw-frame` endpoint), spacebar or
Capture button, Burst = 10 frames at 1/s. Saves to
`~/rat-inference/datasets/redbull/raw/capture_*.jpg` on the Jetson.
Vary: distance, angle, lighting, backgrounds, partial occlusion, can on side.

## 2. Pull captures to the Mac + auto-label
```bash
tools/vision/dataset/sync_captures.sh
```
(rsyncs captures into `datasets/redbull/raw/`, runs YOLO-World auto-labeler —
proposals land in `raw/.proposals/`.)

## 3. Review (yes/no)
```bash
python3 tools/vision/dataset/review_server.py   # http://localhost:8020
```
`y` accept → `images/{train,val}` + `labels/{train,val}` (every 5th → val).
`n` reject → `rejected/`. Reject any multi-can image with a single box.

## 4. Nano banana background variants (after ~20-30 accepted)
```bash
uv run --no-sync python tools/vision/dataset/augment_backgrounds.py --match "capture_*" --per-image 3
uv run --no-sync python tools/vision/dataset/autolabel_redbull.py --input datasets/redbull/synthetic
python3 tools/vision/dataset/review_server.py --input datasets/redbull/synthetic
```
(Vertex project `ratpack`, gemini-2.5-flash-image; synthetic images are
re-labeled + re-reviewed because the model can nudge the can.)

## 5. Train at 960
```bash
uv run --no-sync python tools/vision/training/train.py \
  --data datasets/redbull/redbull_dataset.yaml \
  --imgsz 960 --model-size n --epochs 100 --batch 8 --device mps
```
(`--device mps` on the Mac; use `--device 0` if training on the Jetson.)

## 6. Deploy to the Jetson
```bash
scp runs/train/*/weights/best.pt base698@jetson:~/rat-inference/runs/redbull/weights/best.pt
```
Then point `~/bin/ratbot` at it: `--model runs/redbull/weights/best.pt
--target-class redbull` (drop the bottle/cup lines), and `ratbot reload`.
