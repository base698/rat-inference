#!/bin/bash

# Get current time in format HhMM (e.g., 6h59)
CURRENT_TIME=$(date +"%kh%M" | sed 's/ //g')

# Create log filename with current time
LOG_FILE="rt_100-2050-08-22-${CURRENT_TIME}.log"

# Run the YOLO command in background with nohup
# -u flag forces unbuffered output so print statements appear immediately
nohup uv run python -u rt_100.py \
    --model runs/yolo11n-2025-08-24/weights/best.pt \
    --keep-images \
    --confidence 0.85 \
    --enable-servo \
    > "$LOG_FILE" 2>&1 &

# Print the process ID and log file name
echo "Started YOLO detection in background"
echo "Process ID: $!"
echo "Log file: $LOG_FILE"
echo "To stop the process, run: kill $!"
