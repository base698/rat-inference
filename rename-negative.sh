# Rename all PNG files to image1.png, image2.png, etc.
counter=1
for file in datasets/rat/negatives/*.png; do
    [ -e "$file" ] || continue  # Skip if no files match
    mv "$file" "datasets/rat/negatives/image${counter}.png"
    ((counter++))
done
