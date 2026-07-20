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
# Always start fresh log per run
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
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
# Auto-detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_IDS="0,1"  # Container maps physical GPUs to 0,1
ITERATIONS=30000
BACKEND="default"
BSZ=1
TILE_CROP_MARGIN=100
SCENE_MARGIN=0.0
NDC_LIMIT=1.0
# 현재 세팅: 품질 런 — 정상 threshold, 표준 30k iter (검증 런들은 2026-07 완료)
# cat3 사이클 재검증이 필요하면 EXPLOSIVE_DENSIFICATION=true + ITERATIONS 축소
EXPLOSIVE_DENSIFICATION=false
#EXPLOSIVE_DENSIFICATION=true
# Densification params (will be overridden if EXPLOSIVE_DENSIFICATION=true)
DENSIFY_FROM_ITER=500
DENSIFICATION_INTERVAL=100
DENSIFY_GRAD_THRESHOLD=0.0002
CHILD_DENSIFY_GRAD_THRESHOLD="0.0002"   # resume 자식 전용 threshold (빈 값 = 부모와 동일)

# Debug image saving (for off-center projection verification)
# Set to comma-separated iterations, e.g., "1,100,500,1000" or empty to use defaults
DEBUG_SAVE_ITERS="1,1000,7000,15000,30000"

# Visual debugging for projection and crop
VISUAL_DEBUG=false
VISUAL_DEBUG_ONLY=false
VISUAL_DEBUG_LEVEL=""  # Empty means level 0 (default)
RENDER_DEBUG=false
RENDER_DEBUG_MIN_POINTS=30
RESUME_VIZ_STOP=false    # true: resume viz 만 저장하고 자식 타일 학습 없이 종료 (검증 모드)
FRESH_START=false        # true: 기존 state 무시하고 처음부터 (기본: state 있으면 이어서 진행)

# OOM timeout settings (seconds)
OOM_ACK_TIMEOUT=60       # Timeout for rank ACK during OOM signaling (1 minute - faster detection)
NCCL_TIMEOUT_OVERRIDE=1200  # NCCL timeout override (20 minutes for large gaussians)
PLY_WAIT_TIMEOUT=3600    # Timeout for waiting PLY files to be saved (60 minutes)

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
    echo "  --child-densify-grad-threshold N  Densify threshold for resume children only (prevents re-explosion)"
    echo "  --debug-save-iters ITERS    Comma-separated iterations to save debug GT/rendered images (default: 1,100,500,1000)"
    echo "  --visual-debug              Enable visual debugging for projection and crop calculations"
    echo "  --visual-debug-only         Run only visual debugging and exit (no training)"
    echo "  --visual-debug-level N      Debug at specific tile level (default: 0, use higher for split tiles)"
    echo "  --oom-ack-timeout N         Timeout for rank ACK during OOM signaling in seconds (default: 900)"
    echo "  --nccl-timeout N            NCCL timeout for distributed operations in seconds (default: 1200)"
    echo "  --ply-wait-timeout N        Timeout for waiting PLY files to be saved in seconds (default: 3600)"
    echo "  --render-debug             Enable render debug dumps/PNGs for low-point cases"
    echo "  --resume-viz-stop           Save resume viz and exit child before training (verification mode)"
    echo "  --fresh                     Ignore existing adaptive_state.json and start over (default: resume)"
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
        --child-densify-grad-threshold)
            CHILD_DENSIFY_GRAD_THRESHOLD="$2"
            shift 2
            ;;
        --explosive-densification)
            EXPLOSIVE_DENSIFICATION=true
            shift
            ;;
        --visual-debug)
            VISUAL_DEBUG=true
            shift
            ;;
        --visual-debug-only)
            VISUAL_DEBUG=true
            VISUAL_DEBUG_ONLY=true
            shift
            ;;
        --visual-debug-level)
            VISUAL_DEBUG_LEVEL="$2"
            shift 2
            ;;
        --debug-save-iters)
            DEBUG_SAVE_ITERS="$2"
            shift 2
            ;;
        --oom-ack-timeout)
            OOM_ACK_TIMEOUT="$2"
            shift 2
            ;;
        --nccl-timeout)
            NCCL_TIMEOUT_OVERRIDE="$2"
            shift 2
            ;;
        --ply-wait-timeout)
            PLY_WAIT_TIMEOUT="$2"
            shift 2
            ;;
        --render-debug)
            RENDER_DEBUG=true
            shift
            ;;
        --resume-viz-stop)
            RESUME_VIZ_STOP=true
            shift
            ;;
        --fresh)
            FRESH_START=true
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
# Explosive Densification Mode
# ============================================
if [[ "$EXPLOSIVE_DENSIFICATION" == true ]]; then
    echo "[EXPLOSIVE DENSIFICATION MODE]"
    echo "  Overriding densification params for aggressive gaussian growth..."
    DENSIFY_FROM_ITER=100
    DENSIFICATION_INTERVAL=50
    DENSIFY_GRAD_THRESHOLD=0.00001  # 적당히 낮은 threshold (0.000001 -> 0.00001)
    # iterations 는 explosive 여부와 무관하게 ITERATIONS 변수/--iterations 가 결정
    echo "  Iterations: ${ITERATIONS}"
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
    # Always override NUM_GPUS with actual GPU count from GPU_IDS
    NUM_GPUS=$GPU_COUNT
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "Setting CUDA_VISIBLE_DEVICES=${GPU_IDS}"
fi

# ============================================
# NCCL Configuration for better OOM handling
# ============================================
export NCCL_TIMEOUT=$NCCL_TIMEOUT_OVERRIDE  # Configurable NCCL timeout for OOM handling
export OOM_ACK_TIMEOUT=$OOM_ACK_TIMEOUT     # Configurable ACK timeout for OOM signaling
export PLY_WAIT_TIMEOUT=$PLY_WAIT_TIMEOUT   # Configurable PLY wait timeout
export NCCL_ASYNC_ERROR_HANDLING=1  # Better async error handling
export TORCH_NCCL_ENABLE_MONITORING=0  # Disable NCCL watchdog to prevent kills during PLY save
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=0  # Disable heartbeat timeout
export TORCH_NCCL_AVOID_RECORD_STREAMS=1  # Avoid NCCL stream recording issues
# Extend torchrun elastic shutdown grace period to allow large PLY writes to finish
export TORCHELASTIC_SHUTDOWN_GRACE_PERIOD=600
export TORCH_ELASTIC_SHUTDOWN_GRACE_PERIOD=600
# Robust OOM PLY save timeouts (increase to allow large PLY writes)
export OOM_PLY_SAVE_TIMEOUT=600
if [[ "${RENDER_DEBUG}" == true ]]; then
    export RENDER_DEBUG_DUMP=1
    export RENDER_DEBUG_PNG=1
    export RENDER_DEBUG_ABORT=1
    export RENDER_DEBUG_MIN_POINTS="${RENDER_DEBUG_MIN_POINTS}"
    echo "  Render debug: ON (min_points=${RENDER_DEBUG_MIN_POINTS})"
else
    unset RENDER_DEBUG_DUMP
    unset RENDER_DEBUG_PNG
    unset RENDER_DEBUG_MIN_POINTS
fi
if [[ "${RESUME_VIZ_STOP}" == true ]]; then
    export RESUME_VIZ_STOP=1
    echo "  Resume viz stop: ON (children exit after resume viz, no training)"
else
    unset RESUME_VIZ_STOP
fi
# Render debug: dump small-point projections and PNGs for loss-zero analysis

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
echo "  OOM ACK timeout: ${OOM_ACK_TIMEOUT}s"
echo "  NCCL timeout: ${NCCL_TIMEOUT_OVERRIDE}s"
echo "============================================"
echo ""

# Export debug save iterations for train_internal.py
if [[ -n "$DEBUG_SAVE_ITERS" ]]; then
    export DEBUG_SAVE_ITERS="${DEBUG_SAVE_ITERS}"
fi
# Disable train_adaptive.py file logging; rely on run_adaptive.log
export TRAIN_ADAPTIVE_NO_LOG=1

# ============================================
# Quality 기반 done (epoch loss 정체 감지 조기 종료)
# ============================================
# iter 상한(ITERATIONS)은 유지하되, densification 종료 후 에폭 평균 손실이
# 정체하면 그 타일은 조기 done. 끄려면 QUALITY_DONE=0 으로 실행.
export QUALITY_DONE="${QUALITY_DONE:-1}"
export QUALITY_DONE_REL_EPS="${QUALITY_DONE_REL_EPS:-0.005}"   # 창 간 상대 개선율 임계 (0.5%)
export QUALITY_DONE_PATIENCE="${QUALITY_DONE_PATIENCE:-2}"     # 연속 정체 판정 횟수
export QUALITY_DONE_WINDOW_EPOCHS="${QUALITY_DONE_WINDOW_EPOCHS:-3}"  # 비교 창(에폭), 최소 500iter 보정됨
echo "  Quality-based done: ${QUALITY_DONE} (rel_eps=${QUALITY_DONE_REL_EPS}, patience=${QUALITY_DONE_PATIENCE})"

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
if [[ -n "${CHILD_DENSIFY_GRAD_THRESHOLD}" ]]; then
    CMD+=" --child_densify_grad_threshold ${CHILD_DENSIFY_GRAD_THRESHOLD}"
fi
if [[ "${FRESH_START}" == true ]]; then
    CMD+=" --fresh_start"
fi

# Add visual debug flag if enabled
if [[ "$VISUAL_DEBUG" == "true" ]]; then
    CMD+=" --visual_debug"
fi

# Add visual debug only flag if enabled
if [[ "$VISUAL_DEBUG_ONLY" == "true" ]]; then
    CMD+=" --visual_debug_only"
fi

# Add visual debug level if specified
if [[ -n "$VISUAL_DEBUG_LEVEL" ]]; then
    CMD+=" --visual_debug_level ${VISUAL_DEBUG_LEVEL}"
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
