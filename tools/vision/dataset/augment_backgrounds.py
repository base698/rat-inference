#!/usr/bin/env python3
"""Nano banana background augmentation for the Red Bull dataset.

Takes accepted images (datasets/redbull/images/train by default — typically
your rig captures) and asks Vertex AI gemini-2.5-flash-image ("nano banana")
to swap the background while keeping the can EXACTLY in place. Outputs land in
datasets/redbull/synthetic/ and must then go through:

  uv run --no-sync python tools/vision/dataset/autolabel_redbull.py --input datasets/redbull/synthetic
  python3 tools/vision/dataset/review_server.py --input datasets/redbull/synthetic

(We re-label rather than copying source labels because the model can shift
the can slightly.)

Auth: same as generate-dataset.py — gcloud application-default login,
GCP project "ratpack" (override with GCP_PROJECT_ID).

Usage:
  uv run --no-sync python tools/vision/dataset/augment_backgrounds.py --per-image 3
  uv run --no-sync python tools/vision/dataset/augment_backgrounds.py --source datasets/redbull/raw --match "capture_*" --per-image 4
"""
import argparse
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ratpack")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MODEL = "gemini-2.5-flash-image"

BACKGROUNDS = [
    "a dim crawlspace with concrete blocks and dirt, night infrared lighting",
    "a wooden deck at night, harsh flashlight illumination",
    "a cluttered garage workbench with tools",
    "a kitchen counter with tile backsplash, warm evening light",
    "outdoor gravel and leaves, overcast daylight",
    "a dark basement floor with pipes in the background",
    "grass and rocks in a backyard at dusk",
    "a concrete patio with scattered leaves, low sun with long shadows",
]

# venue set — the demo runs on a stage / library / office setting
VENUE_BACKGROUNDS = [
    "a modern university library like NC State's Hunt Library: white bookshelves, colorful reading chairs, glass walls, bright even lighting",
    "a university library study area with long wooden tables, rows of books, and warm overhead lighting",
    "a tech-talk stage with a projection screen and wood-slat paneled wall behind, stage lighting from above",
    "an auditorium stage seen from the podium, dark seating area in the background, bright stage lights",
    "a modern office with desks, monitors, and glass meeting rooms in the background",
    "a conference room table with a whiteboard and TV screen behind",
    "a carpeted lecture hall floor with rows of seats in the background, fluorescent lighting",
    "a demo table at a meetup: laptop, projector screen, and wood-paneled wall in the background",
]

REF_PROMPT = (
    "The first image shows a Red Bull can. The second image shows a venue. "
    "Replace the background of the first image with the environment from the "
    "second image, as if the photo was taken there ({bg}). Keep the Red Bull "
    "can from the first image in EXACTLY the same position, size, scale, "
    "angle, and framing. Do not move, resize, redraw, or restyle the can. "
    "Photorealistic output, same resolution as the first image."
)

PROMPT = (
    "Replace the background of this photo with {bg}. Keep the Red Bull can in "
    "EXACTLY the same position, size, scale, angle, and lighting direction as "
    "the original. Do not move, resize, redraw, or restyle the can. "
    "Photorealistic output, same resolution and framing as the input."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "datasets/redbull/images/train"))
    ap.add_argument("--match", default="*", help="glob filter, e.g. 'capture_*'")
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="max source images (0 = all)")
    ap.add_argument("--set", choices=["default", "venue"], default="default",
                    help="background prompt set")
    ap.add_argument("--ref-image", default=None,
                    help="optional venue photo composited as the background environment")
    ap.add_argument("--tag", default=None, help="filename tag (default: set name)")
    args = ap.parse_args()

    import vertexai
    from vertexai.generative_models import GenerativeModel, Part

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL)

    out_dir = ROOT / "datasets/redbull/synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(args.source)
    images = sorted(
        p for p in src_dir.glob(args.match)
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit:
        images = images[: args.limit]
    print(f"{len(images)} source images x {args.per_image} variants -> {out_dir}")

    backgrounds = VENUE_BACKGROUNDS if args.set == "venue" else BACKGROUNDS
    tag = args.tag or ("" if args.set == "default" else args.set + "_")
    ref_part = None
    if args.ref_image:
        ref = Path(args.ref_image)
        ref_mime = "image/png" if ref.suffix.lower() == ".png" else "image/jpeg"
        ref_part = Part.from_data(data=ref.read_bytes(), mime_type=ref_mime)

    made = failed = 0
    for img in images:
        data = img.read_bytes()
        mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
        part = Part.from_data(data=data, mime_type=mime)
        for i in range(args.per_image):
            bg = backgrounds[(made + failed) % len(backgrounds)]
            out = out_dir / f"syn_{tag}{img.stem}_{i}.png"
            if out.exists():
                continue
            if ref_part is not None:
                contents = [part, ref_part, REF_PROMPT.format(bg=bg)]
            else:
                contents = [part, PROMPT.format(bg=bg)]
            for attempt in range(6):
                try:
                    resp = model.generate_content(contents)
                    gen = resp.candidates[0].content.parts[0].inline_data.data
                    out.write_bytes(gen)
                    made += 1
                    print(f"  {out.name}  [{bg[:40]}...]", flush=True)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 5:
                        wait = 2 ** attempt * 5 + random.uniform(0, 3)
                        print(f"  429, backing off {wait:.0f}s ({img.name})", flush=True)
                        time.sleep(wait)
                        continue
                    failed += 1
                    print(f"  FAILED {img.name} variant {i}: {e}", flush=True)
                    break
            time.sleep(3)  # pace requests below the per-minute quota

    print(f"\ndone: {made} generated, {failed} failed")
    print("next: autolabel + review the synthetic dir (see module docstring)")


if __name__ == "__main__":
    main()
