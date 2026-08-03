#!/usr/bin/env python3
"""Render contact sheets of the accepted redbull dataset with boxes drawn.

Output: datasets/redbull/audit/<split>_<n>.jpg — eyeball these to verify
labels. If a box is wrong, delete the image+label pair (or fix with
labeler.py) and re-run.

Usage: python3 tools/vision/dataset/audit_sheets.py
"""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "datasets/redbull"
OUT = DS / "audit"
OUT.mkdir(exist_ok=True)
COLS, CELL, PER = 5, 320, 20

for split in ("train", "val"):
    imgs = sorted((DS / "images" / split).iterdir())
    for s in range(0, len(imgs), PER):
        batch = imgs[s:s + PER]
        rows = (len(batch) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * CELL, rows * (CELL + 26)), "#181818")
        d = ImageDraw.Draw(sheet)
        for i, p in enumerate(batch):
            im = Image.open(p).convert("RGB")
            im.thumbnail((CELL, CELL))
            cx, cy = (i % COLS) * CELL, (i // COLS) * (CELL + 26)
            ox, oy = cx + (CELL - im.width) // 2, cy + (CELL - im.height) // 2
            sheet.paste(im, (ox, oy))
            lbl = DS / "labels" / split / (p.stem + ".txt")
            if lbl.exists() and lbl.read_text().strip():
                x, y, w, h = [float(v) for v in lbl.read_text().split()[1:5]]
                d.rectangle([ox + (x - w / 2) * im.width, oy + (y - h / 2) * im.height,
                             ox + (x + w / 2) * im.width, oy + (y + h / 2) * im.height],
                            outline="#22FF88", width=3)
            else:
                d.text((cx + 6, cy + 6), "NO LABEL!", fill="#FF4444")
            d.text((cx + 6, cy + CELL + 4), f"{split} {p.name[:34]}", fill="#F6C042")
        out = OUT / f"{split}_{s // PER}.jpg"
        sheet.save(out, quality=88)
        print(out)
