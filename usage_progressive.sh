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

# Additional NCCL timeout settings for stability
export NCCL_TIMEOUT=1800000  # 30 minutes timeout (in milliseconds)
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU selection - set which GPUs to use (comment out to use all GPUs)
#export CUDA_VISIBLE_DEVICES=0

# ====================================================
# USER CONFIGURATION - EDIT THESE PARAMETERS
# ====================================================

# Required parameters
# Check if dataset name is provided as first argument
echo "DEBUG: First argument \$1 = '$1'"
echo "DEBUG: All arguments: $@"
if [ -n "$1" ]; then
    SOURCE_PATH="/data/$1"  # Use provided dataset name
    echo "DEBUG: Set SOURCE_PATH to: $SOURCE_PATH"
    shift  # Remove dataset name from arguments
else
    #SOURCE_PATH="/data/samsung_dong_mini_5"  # Path to COLMAP reconstruction
    SOURCE_PATH="/data/sillim_ew_mini_100024_20"  # Path to COLMAP reconstruction
    #SOURCE_PATH="/data/Samsung_SN_30"  # Default path to COLMAP reconstruction
    echo "DEBUG: Using default SOURCE_PATH: $SOURCE_PATH"
fi
OUTPUT_PATH="./output/progressive_test"   # Output directory

# Process remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --point_cloud_format)
            POINT_CLOUD_FORMAT="$2"
            shift 2
            ;;
        *)
            # Unknown argument, add to EXTRA_ARGS
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Optional parameters
INITIAL_CAMERAS=2                         # Number of initial cameras
#CAMERA_REMOVAL_MARGIN=0.19                # Margin below densify_memory_limit for camera removal (0.99 - 0.19 = 0.80 = 80%)
ITERATIONS=50                          # Training iterations
###
#ITERATIONS_PER_WINDOW=600                # Iterations per sliding window
#DENSIFICATION_INTERVAL=150               # Densification every 20 iterations
#DENSIFY_FROM_ITER=10                    # Start densification from iteration 10
#CAMERA_REMOVAL_MARGIN=0.25                # Margin below densify_memory_limit for camera removal (0.99 - 0.19 = 0.80 = 80%)
###
#ITERATIONS_PER_WINDOW=125                # Iterations per sliding window
#DENSIFICATION_INTERVAL=50               # Densification every 20 iterations
#DENSIFY_FROM_ITER=10                    # Start densification from iteration 10
#CAMERA_REMOVAL_MARGIN=0.25                # Margin below densify_memory_limit for camera removal (0.99 - 0.19 = 0.80 = 80%)
###
#ITERATIONS_PER_WINDOW=45                # Iterations per sliding window
#DENSIFICATION_INTERVAL=20               # Densification every 20 iterations
#DENSIFY_FROM_ITER=10                    # Start densification from iteration 10
#CAMERA_REMOVAL_MARGIN=0.25                # Margin below densify_memory_limit for camera removal (0.99 - 0.19 = 0.80 = 80%)
###
ITERATIONS_PER_WINDOW=12                # Iterations per sliding window
DENSIFICATION_INTERVAL=100              # Densification every 100 iterations
DENSIFY_FROM_ITER=100                  # Start densification from iteration 300
#CAMERA_REMOVAL_MARGIN=0.1                  # Margin below densify_memory_limit for camera removal (0.3 - 0.1 = 0.2 = 20%)
CAMERA_REMOVAL_MARGIN=0.5625                 # Margin below densify_memory_limit for camera removal (0.99 - 0.25 = 0.74 = 74%)
###

#DENSIFY_MEMORY_LIMIT_PERCENTAGE=0.3     # GPU memory limit for densification (0.3 = 30%)
DENSIFY_MEMORY_LIMIT_PERCENTAGE=0.99     # GPU memory limit for densification (0.3 = 30%)
MAX_WINDOW_SIZE=""                       # Maximum window size (number of cameras). Empty = unlimited
#MAX_WINDOW_SIZE=3                       # Maximum window size (number of cameras). Empty = unlimited

# ====================================================
# ADAPTIVE TRAINING CONFIGURATION
# ====================================================
# Enable adaptive training (auto-stop when converged)
#ENABLE_ADAPTIVE_TRAINING=false           # Enable adaptive training
ENABLE_ADAPTIVE_TRAINING=true           # Enable adaptive training
MIN_ITERATIONS_PER_WINDOW=3             # Minimum iterations before convergence check

#CONVERGENCE_START_ITER=100              # Start convergence checking from this iteration
#CONVERGENCE_LOSS_THRESHOLD=1e-4         # Loss improvement threshold

CONVERGENCE_START_ITER=50              # Start convergence checking from this iteration
CONVERGENCE_LOSS_THRESHOLD=1e-4         # Loss improvement threshold
CONVERGENCE_PATIENCE=60                 # Number of iterations without improvement before convergence
CONVERGENCE_CHECK_INTERVAL=10           # Check convergence every N iterations

# Exponential fitting parameters for dynamic patience
MIN_CAMERA_COUNT=2                      # Minimum camera count for exponential fitting
MAX_PATIENCE_FOR_MIN_CAM=90            # Patience when camera count is MIN_CAMERA_COUNT
MAX_CAMERA_COUNT=30                     # Maximum camera count for exponential fitting
MIN_PATIENCE_FOR_MAX_CAM=35             # Patience when camera count is MAX_CAMERA_COUNT

SH_DEGREE=3                              # Spherical harmonics degree (0 for testing, less memory)
BACKEND="gsplat"                         # Rendering backend: default or gsplat

# Flags (set to "true" to enable, "false" to disable)
DETERMINISTIC=false                      # Enable deterministic training
DEBUG=true                               # Enable debug output
#EXIT_AFTER_FIRST_REMOVAL=true            # Exit after first camera removal (for testing)
EXIT_AFTER_FIRST_REMOVAL=false            # Exit after first camera removal (for testing)
SHOW_MEMORY_DEBUG_INFO=false          # Show detailed memory debug info (memory, tensor stats)
#USE_CHUNK=true                           # Enable chunked SSIM for memory efficiency
USE_CHUNK=false                           # Enable chunked SSIM for memory efficiency
#ONLY_ACTUALLY_VISIBLE=false             # Only keep points visible in camera frames
ONLY_ACTUALLY_VISIBLE=true             # Only keep points visible in camera frames
TRACK_BY_PROJECTION=true               # Generate tracks by projection instead of using COLMAP tracks
#TRACK_BY_PROJECTION=false               # Generate tracks by projection instead of using COLMAP tracks
PRUNE_BY_VISIBILITY=true                # Prune gaussians outside all camera frustums
VISIBILITY_MARGIN=0                     # Margin in pixels for visibility-based pruning (larger = stricter)
NO_OPACITY_RESET=true                  # Disable opacity reset (true to disable, false for vanilla 3DGS behavior)
#NO_OPACITY_RESET=false                  # Disable opacity reset (true to disable, false for vanilla 3DGS behavior)
OPACITY_RESET_INTERVAL=300             # Opacity reset interval in iterations (only used when NO_OPACITY_RESET=false)
#SKIP_HEAVY_VISUALIZATION=false          # Skip heavy visualization files (3d_scene, ortho, nadir) to save time
SKIP_HEAVY_VISUALIZATION=true          # Skip heavy visualization files (3d_scene, ortho, nadir) to save time

# Camera removal strategy
REMOVAL_STRATEGY="fifo"                 # Remove oldest camera first (predictable sliding window)
#REMOVAL_STRATEGY="farthest"             # Remove camera farthest from newly added camera (original)

# F (global center) calculation mode
F_MODE="global"                         # Use global mean F (computed once from all cameras)
#F_MODE="remaining"                      # Use remaining cameras mean F (recomputed each time)

# E camera selection strategy (for Window 2+)
#E_SELECTION_STRATEGY="default"          # Use (yy - F) direction only (original)
#E_SELECTION_STRATEGY="momentum"        # Use recent camera movement momentum (smooth spiral)
#E_SELECTION_STRATEGY="weighted"        # Use weighted R + (yy - F) (balance control)
#E_SELECTION_STRATEGY="tangential"      # Use R + tangential component (mathematical spiral)
#E_SELECTION_STRATEGY="polar"           # Use fixed angle/radius steps (perfect spiral)
#E_SELECTION_STRATEGY="outward_spiral_compact"  # Outward + Spiral + Compact window (3-component score)
E_SELECTION_STRATEGY="balanced_smooth_trajectory"  # Balanced 4-force trajectory (outward + compact + smooth window + smooth camera)

# E selection strategy parameters (only used for certain strategies)
E_WEIGHTED_ALPHA=1.0                    # Weight for R in 'weighted' strategy (0.0-1.0)
E_WEIGHTED_BETA=1.0                     # Weight for (yy - F) in 'weighted' strategy (0.0-1.0)
E_TANGENTIAL_COEFF=0.5                  # Tangential coefficient for 'tangential' strategy
E_POLAR_ANGLE_STEP=30                   # Angle step in degrees for 'polar' strategy
E_POLAR_RADIUS_STEP=1.2                 # Radius multiplier for 'polar' strategy
E_SPIRAL_ALPHA=0.3                      # Distance weight for 'outward_spiral_compact' strategy (낮춤 - window coherence 약하게)
E_SPIRAL_BETA=2.0                       # Diversity weight for 'outward_spiral_compact' strategy (높임 - 나선형 강하게)
E_SPIRAL_GAMMA=0.1                      # Variance penalty for 'outward_spiral_compact' strategy (낮춤 - window shape 덜 중요)
E_OUTWARD_WEIGHT=4.0                    # Outward weight for 'balanced_smooth_trajectory' strategy (increased for global exploration)
E_COMPACT_WEIGHT=3.5                    # Compact weight for 'balanced_smooth_trajectory' strategy (increased to penalize high variance)
E_SMOOTH_WINDOW_WEIGHT=2.8              # Smooth window weight for 'balanced_smooth_trajectory' strategy (maintained for trajectory smoothness)
E_SMOOTH_CAMERA_WEIGHT=0.0              # Smooth camera weight for 'balanced_smooth_trajectory' strategy (disabled)
E_DISTANCE_WEIGHT=1.0                   # Distance weight for 'balanced_smooth_trajectory' strategy (increased to prevent far outlier selection)
E_DIRECTIONAL_WEIGHT=3.0                # Directional alignment weight for 'balanced_smooth_trajectory' strategy (candidate aligns with window movement direction)

# Intersection and filtering options
FOOTPRINT_INTERSECTION_THRESHOLD=0.62    # Minimum intersection area ratio (intersection/window_union) for candidate cameras (0.0-1.0, 0.0=disabled):
#ENABLE_DIRECTION_FILTERING=true        # Filter cameras by forward direction (angle <= 90°)
ENABLE_DIRECTION_FILTERING=false        # Disable filtering, allow all cameras in S (footprint intersection only)

# Camera removal and processing options
#USE_ALL_PROCESSED_CAMERAS=true         # Use all ever-processed cameras when checking for gaussian addition (recommended)
USE_ALL_PROCESSED_CAMERAS=false         # Use only prev window cameras (may add duplicate gaussians)

# Point cloud loading format
POINT_CLOUD_FORMAT="auto"               # Point cloud format: auto, ply, txt, bin
                                        # auto: try bin -> txt (default behavior)
                                        # ply: load points3D.ply directly
                                        # txt: load points3D.txt directly
                                        # bin: load points3D.bin directly

# Resume options
RESUME_FROM_WINDOW=""                   # Resume from specific window (0=initial, 1+=window_XXX, "auto"=auto-detect, ""=disabled)
#RESUME_FROM_WINDOW="auto"                   # Disabled by default - set to "auto" or window number to enable resume
#RESUME_FROM_WINDOW="2"                   # Disabled by default - set to "auto" or window number to enable resume

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

# Auto-configure parameters for adaptive training
if [[ "$ENABLE_ADAPTIVE_TRAINING" == "true" ]]; then
    ITERATIONS_PER_WINDOW=999999999  # Set to very high number for adaptive training
    # Disable save_iterations for adaptive training (only keep camera removal PLY saving)
    SAVE_ITERATIONS=""
    echo "🔄 Adaptive training enabled - setting iterations per window to infinite (999999999)"
    echo "🔄 Adaptive training enabled - disabling save_iterations (only camera removal PLY saving)"
fi

# Display configuration
print_colored $GREEN "Current Configuration:"
echo "  Source Path: $SOURCE_PATH"
echo "  Output Path: $OUTPUT_PATH"
echo "  Initial Cameras: $INITIAL_CAMERAS"
echo "  Camera Removal Margin: $CAMERA_REMOVAL_MARGIN"
echo "  Iterations: $ITERATIONS"
echo "  Iterations Per Window: $ITERATIONS_PER_WINDOW"
echo "  Densification Interval: $DENSIFICATION_INTERVAL"
echo "  Densify From Iter: $DENSIFY_FROM_ITER"
echo "  Densify Memory Limit: $DENSIFY_MEMORY_LIMIT_PERCENTAGE"
echo "  Max Window Size: ${MAX_WINDOW_SIZE:-unlimited}"
echo "  SH Degree: $SH_DEGREE"
echo "  Backend: $BACKEND"
echo "  Deterministic: $DETERMINISTIC"
echo ""
print_colored $GREEN "Adaptive Training Configuration:"
echo "  Enable Adaptive Training: $ENABLE_ADAPTIVE_TRAINING"
echo "  Min Iterations Per Window: $MIN_ITERATIONS_PER_WINDOW"
echo "  Convergence Loss Threshold: $CONVERGENCE_LOSS_THRESHOLD"
echo "  Convergence Patience: $CONVERGENCE_PATIENCE"
echo "  Convergence Check Interval: $CONVERGENCE_CHECK_INTERVAL"
echo "  Debug: $DEBUG"
echo "  Show Memory Debug Info: $SHOW_MEMORY_DEBUG_INFO"
echo "  Use Chunk: $USE_CHUNK"
echo "  Only Actually Visible: $ONLY_ACTUALLY_VISIBLE"
echo "  Track By Projection: $TRACK_BY_PROJECTION"
echo "  Prune By Visibility: $PRUNE_BY_VISIBILITY"
echo "  Visibility Margin: $VISIBILITY_MARGIN"
echo "  No Opacity Reset: $NO_OPACITY_RESET"
if [[ "$NO_OPACITY_RESET" == "false" ]]; then
    echo "  Opacity Reset Interval: $OPACITY_RESET_INTERVAL"
fi
echo "  Removal Strategy: $REMOVAL_STRATEGY"
echo "  Camera Removal Margin: $CAMERA_REMOVAL_MARGIN"
echo "  F Mode: $F_MODE"
echo "  Enable Direction Filtering: $ENABLE_DIRECTION_FILTERING"
echo "  Footprint Intersection Threshold: $FOOTPRINT_INTERSECTION_THRESHOLD"
echo "  Use All Processed Cameras: $USE_ALL_PROCESSED_CAMERAS"
echo "  Point Cloud Format: $POINT_CLOUD_FORMAT"
echo "  E Selection Strategy: $E_SELECTION_STRATEGY"
if [[ "$E_SELECTION_STRATEGY" == "weighted" ]]; then
    echo "    - Alpha (R weight): $E_WEIGHTED_ALPHA"
    echo "    - Beta ((yy-F) weight): $E_WEIGHTED_BETA"
elif [[ "$E_SELECTION_STRATEGY" == "tangential" ]]; then
    echo "    - Tangential coeff: $E_TANGENTIAL_COEFF"
elif [[ "$E_SELECTION_STRATEGY" == "polar" ]]; then
    echo "    - Angle step: $E_POLAR_ANGLE_STEP degrees"
    echo "    - Radius step: $E_POLAR_RADIUS_STEP"
elif [[ "$E_SELECTION_STRATEGY" == "outward_spiral_compact" ]]; then
    echo "    - Alpha (distance weight): $E_SPIRAL_ALPHA"
    echo "    - Beta (diversity weight): $E_SPIRAL_BETA"
    echo "    - Gamma (variance penalty): $E_SPIRAL_GAMMA"
elif [[ "$E_SELECTION_STRATEGY" == "balanced_smooth_trajectory" ]]; then
    echo "    - Outward weight: $E_OUTWARD_WEIGHT"
    echo "    - Compact weight: $E_COMPACT_WEIGHT"
    echo "    - Smooth window weight: $E_SMOOTH_WINDOW_WEIGHT"
    echo "    - Smooth camera weight: $E_SMOOTH_CAMERA_WEIGHT"
    echo "    - Distance weight: $E_DISTANCE_WEIGHT"
    echo "    - Directional weight: $E_DIRECTIONAL_WEIGHT"
fi
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
CMD="$CMD -t $CAMERA_REMOVAL_MARGIN"
CMD="$CMD --iterations $ITERATIONS"
CMD="$CMD --iterations_per_window $ITERATIONS_PER_WINDOW"
CMD="$CMD --densification_interval $DENSIFICATION_INTERVAL"
CMD="$CMD --densify_from_iter $DENSIFY_FROM_ITER"
CMD="$CMD --densify_memory_limit_percentage $DENSIFY_MEMORY_LIMIT_PERCENTAGE"

# Add max_window_size if set
if [[ -n "$MAX_WINDOW_SIZE" ]]; then
    CMD="$CMD --max_window_size $MAX_WINDOW_SIZE"
fi

CMD="$CMD --sh_degree $SH_DEGREE"
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

# Add adaptive training parameters
if [[ "$ENABLE_ADAPTIVE_TRAINING" == "true" ]]; then
    CMD="$CMD --enable_adaptive_training"
    CMD="$CMD --min_iterations_per_window $MIN_ITERATIONS_PER_WINDOW"
    CMD="$CMD --convergence_start_iter $CONVERGENCE_START_ITER"
    CMD="$CMD --convergence_loss_threshold $CONVERGENCE_LOSS_THRESHOLD"
    CMD="$CMD --convergence_patience $CONVERGENCE_PATIENCE"
    CMD="$CMD --convergence_check_interval $CONVERGENCE_CHECK_INTERVAL"
    CMD="$CMD --min_camera_count $MIN_CAMERA_COUNT"
    CMD="$CMD --max_patience_for_min_cam $MAX_PATIENCE_FOR_MIN_CAM"
    CMD="$CMD --max_camera_count $MAX_CAMERA_COUNT"
    CMD="$CMD --min_patience_for_max_cam $MIN_PATIENCE_FOR_MAX_CAM"
fi

if [[ "$PRUNE_BY_VISIBILITY" == "true" ]]; then
    CMD="$CMD --prune_by_visibility"
fi

if [[ -n "$VISIBILITY_MARGIN" ]]; then
    CMD="$CMD --visibility_prune_margin $VISIBILITY_MARGIN"
fi

if [[ -n "$REMOVAL_STRATEGY" ]]; then
    CMD="$CMD --removal_strategy $REMOVAL_STRATEGY"
fi

if [[ -n "$F_MODE" ]]; then
    CMD="$CMD --f_mode $F_MODE"
fi

if [[ -n "$E_SELECTION_STRATEGY" ]]; then
    CMD="$CMD --e_selection_strategy $E_SELECTION_STRATEGY"
fi

if [[ -n "$E_WEIGHTED_ALPHA" ]]; then
    CMD="$CMD --e_weighted_alpha $E_WEIGHTED_ALPHA"
fi

if [[ -n "$E_WEIGHTED_BETA" ]]; then
    CMD="$CMD --e_weighted_beta $E_WEIGHTED_BETA"
fi

if [[ -n "$E_TANGENTIAL_COEFF" ]]; then
    CMD="$CMD --e_tangential_coeff $E_TANGENTIAL_COEFF"
fi

if [[ -n "$E_POLAR_ANGLE_STEP" ]]; then
    CMD="$CMD --e_polar_angle_step $E_POLAR_ANGLE_STEP"
fi

if [[ -n "$E_POLAR_RADIUS_STEP" ]]; then
    CMD="$CMD --e_polar_radius_step $E_POLAR_RADIUS_STEP"
fi

if [[ -n "$E_SPIRAL_ALPHA" ]]; then
    CMD="$CMD --e_spiral_alpha $E_SPIRAL_ALPHA"
fi

if [[ -n "$E_SPIRAL_BETA" ]]; then
    CMD="$CMD --e_spiral_beta $E_SPIRAL_BETA"
fi

if [[ -n "$E_SPIRAL_GAMMA" ]]; then
    CMD="$CMD --e_spiral_gamma $E_SPIRAL_GAMMA"
fi

if [[ -n "$E_OUTWARD_WEIGHT" ]]; then
    CMD="$CMD --e_outward_weight $E_OUTWARD_WEIGHT"
fi

if [[ -n "$E_COMPACT_WEIGHT" ]]; then
    CMD="$CMD --e_compact_weight $E_COMPACT_WEIGHT"
fi

if [[ -n "$E_SMOOTH_WINDOW_WEIGHT" ]]; then
    CMD="$CMD --e_smooth_window_weight $E_SMOOTH_WINDOW_WEIGHT"
fi

if [[ -n "$E_SMOOTH_CAMERA_WEIGHT" ]]; then
    CMD="$CMD --e_smooth_camera_weight $E_SMOOTH_CAMERA_WEIGHT"
fi

if [[ -n "$E_DISTANCE_WEIGHT" ]]; then
    CMD="$CMD --e_distance_weight $E_DISTANCE_WEIGHT"
fi

if [[ -n "$E_DIRECTIONAL_WEIGHT" ]]; then
    CMD="$CMD --e_directional_weight $E_DIRECTIONAL_WEIGHT"
fi

if [[ -n "$CAMERA_REMOVAL_MARGIN" ]]; then
    CMD="$CMD --camera_removal_margin $CAMERA_REMOVAL_MARGIN"
fi

if [[ -n "$FOOTPRINT_INTERSECTION_THRESHOLD" ]]; then
    CMD="$CMD --footprint_intersection_threshold $FOOTPRINT_INTERSECTION_THRESHOLD"
fi

if [[ "$ENABLE_DIRECTION_FILTERING" == "true" ]]; then
    CMD="$CMD --enable_direction_filtering"
fi

if [[ "$USE_ALL_PROCESSED_CAMERAS" == "true" ]]; then
    CMD="$CMD --use_all_processed_cameras"
fi

if [[ "$EXIT_AFTER_FIRST_REMOVAL" == "true" ]]; then
    CMD="$CMD --exit_after_first_removal"
fi

# Add point cloud format if specified
if [[ -n "$POINT_CLOUD_FORMAT" ]] && [[ "$POINT_CLOUD_FORMAT" != "auto" ]]; then
    CMD="$CMD --point_cloud_format $POINT_CLOUD_FORMAT"
fi

# Add DTM module if specified
if [[ -n "$DTM_MODULE" ]]; then
    CMD="$CMD --dtm-module \"$DTM_MODULE\""
fi

# Add resume functionality if specified
if [[ -n "$RESUME_FROM_WINDOW" ]]; then
    CMD="$CMD --resume_from_window \"$RESUME_FROM_WINDOW\""
fi

# Add opacity reset control
if [[ "$NO_OPACITY_RESET" == "true" ]]; then
    CMD="$CMD --opacity_reset_until_iter 0"
else
    # When opacity reset is enabled, set the interval
    CMD="$CMD --opacity_reset_interval $OPACITY_RESET_INTERVAL"
fi

# Add skip heavy visualization flag
if [[ "$SKIP_HEAVY_VISUALIZATION" == "true" ]]; then
    CMD="$CMD --skip_heavy_visualization"
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
