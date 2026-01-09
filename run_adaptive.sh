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
RESUME=false

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
    echo "  --resume            Resume from existing state"
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
        --resume)
            RESUME=true
            shift
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
echo "  Resume: ${RESUME}"
echo "============================================"
echo ""

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

if [[ "$RESUME" == true ]]; then
    CMD+=" --resume"
fi

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
