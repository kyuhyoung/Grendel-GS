#!/bin/bash

# gen_colmap_output.sh
# Usage: ./gen_colmap_output.sh /data/dir_1

# Clear LD_PRELOAD to avoid library loading errors
unset LD_PRELOAD

set -e  # Exit on error
if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_directory>"
    echo "Example: $0 /data/dir_1"
    exit 1
fi

IMAGE_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Directory $IMAGE_DIR does not exist"
    exit 1
fi

echo "BBB"
# Check if directory contains images
IMAGE_COUNT=$(find "$IMAGE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tiff" -o -iname "*.tif" \) | wc -l)
if [ $IMAGE_COUNT -eq 0 ]; then
    echo "Error: No images found in $IMAGE_DIR"
    exit 1
fi

echo "Found $IMAGE_COUNT images in $IMAGE_DIR"

# Get absolute path and create output directory name
ABS_IMAGE_DIR=$(realpath "$IMAGE_DIR")
DATASET_NAME=$(basename "$ABS_IMAGE_DIR")
OUTPUT_DIR="${ABS_IMAGE_DIR}_colmap"

echo "=== Running COLMAP Processing ==="
echo "Input directory: $ABS_IMAGE_DIR"
echo "Dataset name: $DATASET_NAME"
echo "Output directory: $OUTPUT_DIR"

# Create output directory structure for convert.py
mkdir -p "$OUTPUT_DIR/input"

# Copy images to input directory (convert.py expects images in /input)
echo "Step 1: Copying images to input directory..."
cp -r "$ABS_IMAGE_DIR"/* "$OUTPUT_DIR/input/"

echo "Step 2: Running convert.py for full COLMAP processing..."

# Check if convert.py exists
CONVERT_SCRIPT="$SCRIPT_DIR/convert.py"
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "Error: convert.py not found at $CONVERT_SCRIPT"
    exit 1
fi

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel

cd "$SCRIPT_DIR"

# Run convert.py for full COLMAP processing
python convert.py -s "$OUTPUT_DIR" --camera OPENCV

echo "Step 3: Converting binary to text format (if needed)..."
# Convert binary files to text format for better compatibility
if [ -f "$OUTPUT_DIR/sparse/0/cameras.bin" ]; then
    colmap model_converter \
        --input_path "$OUTPUT_DIR/sparse/0" \
        --output_path "$OUTPUT_DIR/sparse/0" \
        --output_type TXT
fi

echo "Step 4: Validating output..."

# Check if essential files were created
REQUIRED_FILES=("cameras.txt" "images.txt" "points3D.txt")
missing_files=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$OUTPUT_DIR/sparse/0/$file" ]; then
        echo "✗ $file not found in $OUTPUT_DIR/sparse/0/"
        missing_files=$((missing_files + 1))
    else
        echo "✓ $file created"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "Warning: Some required files are missing. COLMAP reconstruction may have failed."
fi

# Check if undistorted images were created
if [ -d "$OUTPUT_DIR/images" ] && [ "$(ls -A $OUTPUT_DIR/images)" ]; then
    echo "✓ Undistorted images created in $OUTPUT_DIR/images"
else
    echo "✗ Undistorted images not found"
fi

echo ""
echo "=== Processing Complete ==="
echo "Input images: $ABS_IMAGE_DIR"
echo "COLMAP output: $OUTPUT_DIR"
echo ""
echo "To use with Grendel-GS:"
echo "  python train.py -s $OUTPUT_DIR"
echo ""
echo "Files structure:"
echo "  $OUTPUT_DIR/"
echo "  ├── input/           # Input images"
echo "  ├── images/          # Undistorted images"
echo "  ├── sparse/0/        # COLMAP reconstruction"
echo "  │   ├── cameras.txt"
echo "  │   ├── images.txt"
echo "  │   └── points3D.txt"
echo "  └── distorted/       # Intermediate COLMAP data"
