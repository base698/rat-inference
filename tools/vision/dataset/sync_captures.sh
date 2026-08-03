#!/usr/bin/env bash
# Pull rig captures from the Jetson into the local redbull raw pool,
# then auto-label them so they're ready for the review UI.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
rsync -av "base698@jetson:~/rat-inference/datasets/redbull/raw/capture_*.jpg" \
  "$HERE/datasets/redbull/raw/" || true
cd "$HERE"
uv run --no-sync python tools/vision/dataset/autolabel_redbull.py
echo
echo "now review: python3 tools/vision/dataset/review_server.py  ->  http://localhost:8020"
