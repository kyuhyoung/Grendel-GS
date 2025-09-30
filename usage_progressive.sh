#!/bin/bash
# Usage script for Progressive Training with Grendel-GS
# Configure parameters below and run this script

# Set environment variables to suppress PyTorch/NCCL warnings
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=ERROR
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS="ignore"

# ====================================================
# USER CONFIGURATION - EDIT THESE PARAMETERS
# ====================================================

# Required parameters
SOURCE_PATH="/data/samsung_dong_mini_5"  # Path to COLMAP reconstruction
OUTPUT_PATH="./output/progressive_test"   # Output directory

# Optional parameters
INITIAL_CAMERAS=2                         # Number of initial cameras
GPU_THRESHOLD=0.9                         # GPU memory threshold (0-1)
ITERATIONS=30000                          # Training iterations
#ITERATIONS_PER_WINDOW=600                # Iterations per sliding window
ITERATIONS_PER_WINDOW=25                # Iterations per sliding window
DENSIFICATION_INTERVAL=20               # Densification every 20 iterations
#DENSIFICATION_INTERVAL=150               # Densification every 20 iterations
DENSIFY_FROM_ITER=10                    # Start densification from iteration 10
SH_DEGREE=3                              # Spherical harmonics degree
RESOLUTION=1                             # Resolution downscaling factor
BACKEND="gsplat"                         # Rendering backend: default or gsplat

# Flags (set to "true" to enable, "false" to disable)
DETERMINISTIC=false                      # Enable deterministic training
DEBUG=true                               # Enable debug output
SHOW_MEMORY_DEBUG_INFO=false          # Show detailed memory debug info (memory, tensor stats)
USE_CHUNK=true                           # Enable chunked SSIM for memory efficiency
#USE_CHUNK=false                           # Enable chunked SSIM for memory efficiency
#ONLY_ACTUALLY_VISIBLE=false             # Only keep points visible in camera frames
ONLY_ACTUALLY_VISIBLE=true             # Only keep points visible in camera frames
TRACK_BY_PROJECTION=true               # Generate tracks by projection instead of using COLMAP tracks
#TRACK_BY_PROJECTION=false               # Generate tracks by projection instead of using COLMAP tracks

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
echo "  Iterations Per Window: $ITERATIONS_PER_WINDOW"
echo "  Densification Interval: $DENSIFICATION_INTERVAL"
echo "  Densify From Iter: $DENSIFY_FROM_ITER"
echo "  SH Degree: $SH_DEGREE"
echo "  Resolution: $RESOLUTION"
echo "  Backend: $BACKEND"
echo "  Deterministic: $DETERMINISTIC"
echo "  Debug: $DEBUG"
echo "  Show Memory Debug Info: $SHOW_MEMORY_DEBUG_INFO"
echo "  Use Chunk: $USE_CHUNK"
echo "  Only Actually Visible: $ONLY_ACTUALLY_VISIBLE"
echo "  Track By Projection: $TRACK_BY_PROJECTION"
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
CMD="$CMD --iterations_per_window $ITERATIONS_PER_WINDOW"
CMD="$CMD --densification_interval $DENSIFICATION_INTERVAL"
CMD="$CMD --densify_from_iter $DENSIFY_FROM_ITER"
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

if [[ "$SHOW_MEMORY_DEBUG_INFO" == "true" ]]; then
    CMD="$CMD --show-memory-debug-info"
fi

if [[ "$USE_CHUNK" == "true" ]]; then
    CMD="$CMD --use_chunk"
fi

if [[ "$ONLY_ACTUALLY_VISIBLE" == "true" ]]; then
    CMD="$CMD --only-actually-visible"
fi

if [[ "$TRACK_BY_PROJECTION" == "true" ]]; then
    CMD="$CMD --track_by_projection"
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
