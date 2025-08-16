#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p datasets/rat/unsorted

# Process each MP4 file in the source directory
for video in datasets/reolink-videos/*.MP4; do
    # Get the base filename without extension
    basename=$(basename "$video" .mp4)
    
    # Extract one frame per second
    # -i: input file
    # -vf fps=1: video filter to extract 1 frame per second
    # -q:v 2: quality setting (lower is better, 2 is high quality)
    ffmpeg -i "$video" -vf fps=1 -q:v 2 "datasets/rat/unsorted/${basename}_%04d.jpg"
    
    echo "Processed: $video"
done

echo "Frame extraction complete!"
