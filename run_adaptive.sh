#!/bin/bash
#
# Adaptive Tile Training Wrapper Script
#
# This script calls train_adaptive.py which manages:
# - Scene extent and tile bbox computation
# - Visible camera calculation
# - OOM-aware tile splitting
# - torchrun invocation
#
# Usage:
#   ./run_adaptive.sh --source /path/to/colmap --output /path/to/output [options]
#

set -euo pipefail

# ============================================
# Logging
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_adaptive.log"
> "${LOG_FILE}"  # Clear log file
exec > >(tee "${LOG_FILE}") 2>&1

echo ""
echo "============================================"
echo "=== Run started at $(date) ==="
echo "============================================"

# ============================================
# Default Configuration
# ============================================
SOURCE_PATH="/data/dabeeo/samsung_dong_mini_30"
OUTPUT_PATH="./output/adaptive_test"
NUM_GPUS=4
GPU_IDS=""  # e.g., "0,1,2,3" or "4,5,6,7"
ITERATIONS=1000
BACKEND="default"
BSZ=1
TILE_CROP_MARGIN=100
SCENE_MARGIN=0.0
NDC_LIMIT=1.0
#EXPLOSIVE_DENSIFICATION=false
EXPLOSIVE_DENSIFICATION=true
# Densification params (will be overridden if EXPLOSIVE_DENSIFICATION=true)
DENSIFY_FROM_ITER=500
DENSIFICATION_INTERVAL=100
DENSIFY_GRAD_THRESHOLD=0.0002

# Debug image saving (for off-center projection verification)
# Set to comma-separated iterations, e.g., "1,100,500,1000" or empty to use defaults
DEBUG_SAVE_ITERS="1"

# ============================================
# Parse arguments
# ============================================
print_usage() {
    echo "Usage: $0 --source PATH --output PATH [options]"
    echo ""
    echo "Required:"
    echo "  --source PATH       Path to COLMAP/source data"
    echo "  --output PATH       Output directory for trained models"
    echo ""
    echo "Optional:"
    echo "  --gpus N            Number of GPUs (default: 4, auto-calculated if --gpu-ids set)"
    echo "  --gpu-ids IDS       Comma-separated GPU IDs, e.g., '0,1,2,3' or '4,5,6,7'"
    echo "  --iterations N      Training iterations per tile (default: 30000)"
    echo "  --backend NAME      Rendering backend: gsplat or default (default: default)"
    echo "  --bsz N             Batch size (default: 4)"
    echo "  --crop-margin N     Pixel margin for camera crops (default: 100)"
    echo "  --scene-margin N    Scene bbox margin ratio (default: 0.0, e.g., 0.1 = 10%)"
    echo "  --ndc-limit N       NDC limit for projection filtering (default: 1.0)"
    echo "  --densify-grad-threshold N  Gradient threshold for densification (default: 0.0002, lower=faster growth)"
    echo "  --explosive-densification   Enable aggressive densification to trigger OOM from gaussian growth"
    echo "  --debug-save-iters ITERS    Comma-separated iterations to save debug GT/rendered images (default: 1,100,500,1000)"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --bsz)
            BSZ="$2"
            shift 2
            ;;
        --crop-margin)
            TILE_CROP_MARGIN="$2"
            shift 2
            ;;
        --scene-margin)
            SCENE_MARGIN="$2"
            shift 2
            ;;
        --ndc-limit)
            NDC_LIMIT="$2"
            shift 2
            ;;
        --densify-grad-threshold)
            DENSIFY_GRAD_THRESHOLD="$2"
            shift 2
            ;;
        --explosive-densification)
            EXPLOSIVE_DENSIFICATION=true
            shift
            ;;
        --debug-save-iters)
            DEBUG_SAVE_ITERS="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments (only if not using defaults)
if [[ -z "$SOURCE_PATH" ]] || [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: --source and --output are required (or set defaults in script)"
    print_usage
    exit 1
fi

if [[ ! -d "$SOURCE_PATH" ]]; then
    echo "Error: Source path does not exist: $SOURCE_PATH"
    exit 1
fi

# ============================================
# Explosive Densification Mode
# ============================================
if [[ "$EXPLOSIVE_DENSIFICATION" == true ]]; then
    echo "[EXPLOSIVE DENSIFICATION MODE]"
    echo "  Overriding densification params for aggressive gaussian growth..."
    DENSIFY_FROM_ITER=100
    DENSIFICATION_INTERVAL=50
    DENSIFY_GRAD_THRESHOLD=0.00005
    ITERATIONS=8000
    echo "  Iterations set to: ${ITERATIONS}"
fi

# ============================================
# Recompile CUDA extensions (incremental - only if source changed)
# ============================================
echo ""
echo "Checking/recompiling CUDA extensions..."
RASTERIZER_DIR="${SCRIPT_DIR}/Grendel-GS/submodules/diff-gaussian-rasterization"
if [[ -d "$RASTERIZER_DIR" ]]; then
    echo "  Found: $RASTERIZER_DIR"

    # Check .so file timestamp before build
    SO_FILE=$(find "$RASTERIZER_DIR" -name "*.so" -type f 2>/dev/null | head -1)
    if [[ -n "$SO_FILE" ]]; then
        SO_TIME_BEFORE=$(stat -c %Y "$SO_FILE" 2>/dev/null || echo "0")
        echo "  Existing .so: $SO_FILE"
        echo "  .so timestamp before: $(date -d @$SO_TIME_BEFORE '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'unknown')"
    else
        SO_TIME_BEFORE="0"
        echo "  No existing .so file found - will compile from scratch"
    fi

    echo "  Running: pip install -e . --no-build-isolation"
    echo "  (If recompiling, you'll see nvcc compilation output...)"
    pushd "$RASTERIZER_DIR" > /dev/null

    # Use --no-build-isolation to ensure proper rebuild detection
    # Show full output including nvcc
    pip install -e . --no-build-isolation -v 2>&1 | while IFS= read -r line; do
        # Only print important lines to reduce noise
        if [[ "$line" == *"nvcc"* ]] || [[ "$line" == *"building"* ]] || \
           [[ "$line" == *"error"* ]] || [[ "$line" == *"Error"* ]] || \
           [[ "$line" == *"Successfully"* ]] || [[ "$line" == *"Running"* ]] || \
           [[ "$line" == *"Compiling"* ]] || [[ "$line" == *".cu"* ]] || \
           [[ "$line" == *".cpp"* ]]; then
            echo "    $line"
        fi
    done
    pip_status=${PIPESTATUS[0]}
    popd > /dev/null

    # Check .so file timestamp after build
    if [[ -n "$SO_FILE" ]] && [[ -f "$SO_FILE" ]]; then
        SO_TIME_AFTER=$(stat -c %Y "$SO_FILE" 2>/dev/null || echo "0")
        echo "  .so timestamp after:  $(date -d @$SO_TIME_AFTER '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'unknown')"
        if [[ "$SO_TIME_AFTER" -gt "$SO_TIME_BEFORE" ]]; then
            echo "  >>> .so file was REBUILT (timestamp changed) <<<"
        else
            echo "  .so file unchanged (using cached build)"
        fi
    fi

    if [[ $pip_status -eq 0 ]]; then
        echo "  diff-gaussian-rasterization: OK"
    else
        echo "  diff-gaussian-rasterization: FAILED (exit code: $pip_status)"
        exit 1
    fi
else
    echo "  Warning: $RASTERIZER_DIR not found, skipping recompilation"
fi

# ============================================
# GPU Configuration
# ============================================
if [[ -n "$GPU_IDS" ]]; then
    # Calculate NUM_GPUS from GPU_IDS (count commas + 1)
    GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
    if [[ "$NUM_GPUS" == "4" ]]; then
        # Default value, override with calculated count
        NUM_GPUS=$GPU_COUNT
    fi
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "Setting CUDA_VISIBLE_DEVICES=${GPU_IDS}"
fi

# ============================================
# Run train_adaptive.py
# ============================================
echo ""
echo "============================================"
echo "Configuration:"
echo "  Source: ${SOURCE_PATH}"
echo "  Output: ${OUTPUT_PATH}"
echo "  GPUs: ${NUM_GPUS}"
if [[ -n "$GPU_IDS" ]]; then
    echo "  GPU IDs: ${GPU_IDS}"
fi
echo "  Iterations: ${ITERATIONS}"
echo "  Backend: ${BACKEND}"
echo "  Batch size: ${BSZ}"
echo "  Crop margin: ${TILE_CROP_MARGIN}"
echo "  Scene margin: ${SCENE_MARGIN}"
echo "  NDC limit: ${NDC_LIMIT}"
echo "  Explosive densification: ${EXPLOSIVE_DENSIFICATION}"
echo "  Densify from iter: ${DENSIFY_FROM_ITER}"
echo "  Densification interval: ${DENSIFICATION_INTERVAL}"
echo "  Densify grad threshold: ${DENSIFY_GRAD_THRESHOLD}"
echo "  Debug save iters: ${DEBUG_SAVE_ITERS:-'default (1,100,500,1000)'}"
echo "============================================"
echo ""

# Export debug save iterations for train_internal.py
if [[ -n "$DEBUG_SAVE_ITERS" ]]; then
    export DEBUG_SAVE_ITERS="${DEBUG_SAVE_ITERS}"
fi

# Build command (use -u for unbuffered output to ensure logs appear immediately)
CMD="python -u ${SCRIPT_DIR}/scripts/train_adaptive.py"
CMD+=" --source_path ${SOURCE_PATH}"
CMD+=" --output_path ${OUTPUT_PATH}"
CMD+=" --num_gpus ${NUM_GPUS}"
CMD+=" --iterations ${ITERATIONS}"
CMD+=" --backend ${BACKEND}"
CMD+=" --bsz ${BSZ}"
CMD+=" --tile_crop_margin ${TILE_CROP_MARGIN}"
CMD+=" --scene_margin ${SCENE_MARGIN}"
CMD+=" --ndc_limit ${NDC_LIMIT}"
CMD+=" --densify_from_iter ${DENSIFY_FROM_ITER}"
CMD+=" --densification_interval ${DENSIFICATION_INTERVAL}"
CMD+=" --densify_grad_threshold ${DENSIFY_GRAD_THRESHOLD}"

echo "Running: ${CMD}"
echo ""

# Run
eval "${CMD}"
exit_code=$?

echo ""
echo "============================================"
echo "=== Finished at $(date) ==="
echo "=== Exit code: ${exit_code} ==="
echo "============================================"

exit ${exit_code}
