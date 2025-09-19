#!/bin/bash
# Usage script for Progressive Training with Grendel-GS
# Configure parameters below and run this script

# ====================================================
# USER CONFIGURATION - EDIT THESE PARAMETERS
# ====================================================

# Required parameters
SOURCE_PATH="/data/samsung_dong_mini_5"  # Path to COLMAP reconstruction
OUTPUT_PATH="./output/progressive_test"   # Output directory

# Optional parameters
INITIAL_CAMERAS=3                         # Number of initial cameras
GPU_THRESHOLD=0.9                         # GPU memory threshold (0-1)
ITERATIONS=30000                          # Training iterations
SH_DEGREE=3                              # Spherical harmonics degree
RESOLUTION=1                             # Resolution downscaling factor
BACKEND="gsplat"                         # Rendering backend: default or gsplat

# Flags (set to "true" to enable, "false" to disable)
DETERMINISTIC=false                      # Enable deterministic training
DEBUG=true                               # Enable debug output
#ONLY_ACTUALLY_VISIBLE=false             # Only keep points visible in camera frames
ONLY_ACTUALLY_VISIBLE=true             # Only keep points visible in camera frames

# Advanced options (leave empty if not needed)
DTM_MODULE=""                            # Path to external DTM module
EXTRA_ARGS=""                            # Additional arguments

# ====================================================
# SCRIPT EXECUTION - DO NOT EDIT BELOW THIS LINE
# ====================================================

# Set up logging - redirect all output to log file
LOG_FILE="usage_progressive.log"
rm -f "$LOG_FILE"  # Clear log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_colored $CYAN "======================================"
print_colored $CYAN "Progressive Training for Grendel-GS"
print_colored $CYAN "======================================"
echo ""

# Display configuration
print_colored $GREEN "Current Configuration:"
echo "  Source Path: $SOURCE_PATH"
echo "  Output Path: $OUTPUT_PATH"
echo "  Initial Cameras: $INITIAL_CAMERAS"
echo "  GPU Threshold: $GPU_THRESHOLD"
echo "  Iterations: $ITERATIONS"
echo "  SH Degree: $SH_DEGREE"
echo "  Resolution: $RESOLUTION"
echo "  Backend: $BACKEND"
echo "  Deterministic: $DETERMINISTIC"
echo "  Debug: $DEBUG"
echo "  Only Actually Visible: $ONLY_ACTUALLY_VISIBLE"
if [[ -n "$DTM_MODULE" ]]; then
    echo "  DTM Module: $DTM_MODULE"
fi
if [[ -n "$EXTRA_ARGS" ]]; then
    echo "  Extra Args: $EXTRA_ARGS"
fi
echo ""

# Check if progressive_train.sh exists
if [[ ! -f "./progressive_train.sh" ]]; then
    print_colored $RED "❌ Error: progressive_train.sh not found in current directory"
    print_colored $YELLOW "Please make sure you're running from the Grendel-GS root directory"
    exit 1
fi

# Make progressive_train.sh executable if needed
if [[ ! -x "./progressive_train.sh" ]]; then
    print_colored $YELLOW "⚠️  Making progressive_train.sh executable..."
    chmod +x ./progressive_train.sh
fi

# Install required submodules in editable mode (fast if already installed)
print_colored $YELLOW "⚠️  Installing required submodules in editable mode..."
if pip install -e submodules/diff-gaussian-rasterization -e submodules/gsplat -e submodules/simple-knn; then
    print_colored $GREEN "✓ Submodules installed successfully"
else
    print_colored $RED "❌ Warning: Some submodules may have failed to install"
    print_colored $YELLOW "Continuing anyway..."
fi

# Validate source path
if [[ ! -d "$SOURCE_PATH" ]]; then
    print_colored $RED "❌ Error: Source path does not exist: $SOURCE_PATH"
    print_colored $YELLOW "Please edit SOURCE_PATH in this script"
    exit 1
fi

if [[ ! -d "$SOURCE_PATH/sparse" ]] && [[ ! -f "$SOURCE_PATH/cameras.txt" ]]; then
    print_colored $RED "❌ Error: No COLMAP data found in: $SOURCE_PATH"
    print_colored $YELLOW "Expected: sparse/ directory or cameras.txt file"
    print_colored $YELLOW "Please edit SOURCE_PATH in this script"
    exit 1
fi

# Build command
CMD="./progressive_train.sh"
CMD="$CMD -s \"$SOURCE_PATH\""
CMD="$CMD -o \"$OUTPUT_PATH\""
CMD="$CMD -m $INITIAL_CAMERAS"
CMD="$CMD -t $GPU_THRESHOLD"
CMD="$CMD --iterations $ITERATIONS"
CMD="$CMD --sh-degree $SH_DEGREE"
CMD="$CMD --resolution $RESOLUTION"
CMD="$CMD --backend $BACKEND"

# Add flags
if [[ "$DETERMINISTIC" == "true" ]]; then
    CMD="$CMD --deterministic"
fi

if [[ "$DEBUG" == "true" ]]; then
    CMD="$CMD --debug"
fi

if [[ "$ONLY_ACTUALLY_VISIBLE" == "true" ]]; then
    CMD="$CMD --only-actually-visible"
fi

# Add DTM module if specified
if [[ -n "$DTM_MODULE" ]]; then
    CMD="$CMD --dtm-module \"$DTM_MODULE\""
fi

# Add extra arguments
if [[ -n "$EXTRA_ARGS" ]]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Show command
print_colored $BLUE "Command to execute:"
echo "  $CMD"
echo ""

# Run the command
print_colored $YELLOW "Starting progressive training..."
echo ""

eval "$CMD"
RESULT=$?

echo ""
if [[ $RESULT -eq 0 ]]; then
    print_colored $GREEN "✓ Progressive training completed successfully!"
    print_colored $GREEN "✓ Results saved to: $OUTPUT_PATH"
else
    print_colored $RED "❌ Progressive training failed with exit code $RESULT"
fi

print_colored $CYAN "======================================"

exit $RESULT
