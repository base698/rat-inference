#!/bin/bash

# Get current time in format HhMM (e.g., 6h59)
CURRENT_TIME=$(date +"%kh%M" | sed 's/ //g')

# Create log filename with current time
LOG_FILE="rt_100-2050-08-22-${CURRENT_TIME}.log"

# Run the YOLO command in background with nohup
nohup python rt_100.py \
    --model runs/yolo11n-2025-08-21/weights/best.pt \
    --keep-images \
    --confidence 0.9 \
    --enable-servo \
    > "$LOG_FILE" 2>&1 &

# Print the process ID and log file name
echo "Started YOLO detection in background"
echo "Process ID: $!"
echo "Log file: $LOG_FILE"
echo "To stop the process, run: kill $!"
