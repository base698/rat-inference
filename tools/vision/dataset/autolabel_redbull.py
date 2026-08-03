#!/usr/bin/env python3
"""Auto-label Red Bull can images with YOLO-World (open-vocabulary).

Reads images from datasets/redbull/raw/ (or --input), writes YOLO-format
label files (class 0 = redbull) next to a .proposals/ mirror so the review
UI can show/accept them. Images with no confident box get an empty label
file — the review UI shows them as "no box" so they can be rejected.

Usage:
  uv run python tools/vision/dataset/autolabel_redbull.py
  uv run python tools/vision/dataset/autolabel_redbull.py --input datasets/redbull/synthetic
"""
import argparse
from pathlib import Path

from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "datasets/redbull/raw"))
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--model", default="yolov8l-worldv2.pt")
    args = ap.parse_args()

    in_dir = Path(args.input)
    prop_dir = in_dir / ".proposals"
    prop_dir.mkdir(exist_ok=True)

    model = YOLOWorld(args.model)
    model.set_classes(["red bull can", "energy drink can", "soda can"])

    images = sorted(
        p for p in in_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    labeled = empty = 0
    for img in images:
        out = prop_dir / (img.stem + ".txt")
        if out.exists():
            continue
        res = model.predict(str(img), conf=args.conf, verbose=False)[0]
        lines = []
        if res.boxes is not None and len(res.boxes):
            # keep the single highest-confidence box — demo scenes have one can
            best = max(res.boxes, key=lambda b: float(b.conf))
            x, y, w, h = best.xywhn[0].tolist()
            lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        if lines:
            labeled += 1
        else:
            empty += 1
        print(f"{img.name}: {'box' if lines else 'NO BOX'}")

    print(f"\nproposals: {labeled} boxed, {empty} empty -> {prop_dir}")


if __name__ == "__main__":
    main()
