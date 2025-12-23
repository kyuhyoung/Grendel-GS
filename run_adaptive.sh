#!/bin/bash
#
# Adaptive Tile Training Wrapper Script
#
# This script manages OOM-aware tile-based training using Grendel-GS.
# When OOM occurs, tiles are split and training continues with smaller tiles.
#
# Usage:
#   ./run_adaptive.sh --source /path/to/colmap --output /path/to/output [options]
#

set -euo pipefail

# ============================================
# Logging - capture ALL output from the start
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_adaptive.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "============================================"
echo "=== Run started at $(date) ==="
echo "============================================"

# ============================================
# Exit codes
# ============================================
EXIT_CODE_SUCCESS=0
EXIT_CODE_OOM=42

# ============================================
# Default Configuration
# ============================================
SOURCE_PATH="/data/dabeeo/samsung_dong_mini_21"
OUTPUT_PATH="/data/output/adaptive_test"
ITERATIONS=30000
NUM_GPUS=4
BACKEND="default"
BSZ=4
TILE_CROP_MARGIN=100

# Initial bounding box (will be computed from scene if not provided)
INITIAL_BBOX=""

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

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
    echo "  --gpus N            Number of GPUs (default: 4)"
    echo "  --iterations N      Training iterations per tile (default: 30000)"
    echo "  --backend NAME      Rendering backend: gsplat or default (default: gsplat)"
    echo "  --bsz N             Batch size (default: 4)"
    echo "  --bbox X1,Y1,Z1,X2,Y2,Z2  Initial bounding box (computed if not provided)"
    echo "  --crop-margin N     Pixel margin for camera crops (default: 100)"
    echo "  --resume            Resume from existing state"
    echo ""
}

RESUME=false

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
        --bbox)
            INITIAL_BBOX="$2"
            shift 2
            ;;
        --crop-margin)
            TILE_CROP_MARGIN="$2"
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

# Validate paths exist
if [[ ! -d "$SOURCE_PATH" ]]; then
    echo "Error: Source path does not exist: $SOURCE_PATH"
    exit 1
fi

# Clean up previous run if not resuming
if [[ "$RESUME" != true ]]; then
    rm -rf "${OUTPUT_PATH}"
fi

# ============================================
# Setup paths
# ============================================
GRENDEL_DIR="${SCRIPT_DIR}/Grendel-GS"
STATE_FILE="${OUTPUT_PATH}/adaptive_tile_state.json"
TILES_DIR="${OUTPUT_PATH}/tiles"
QUEUE_FILE="${OUTPUT_PATH}/tile_queue.json"

log "Configuration:"
log "  Source: ${SOURCE_PATH}"
log "  Output: ${OUTPUT_PATH}"
log "  GPUs: ${NUM_GPUS}"
log "  Iterations: ${ITERATIONS}"
log "  Backend: ${BACKEND}"
log "  Batch size: ${BSZ}"

# ============================================
# Tile Queue Management (using simple JSON file)
# ============================================

# Initialize queue with a single tile
init_queue() {
    local bbox="$1"
    local tile_id="$2"

    cat > "${QUEUE_FILE}" << EOF
{
    "tiles": [
        {"tile_id": "${tile_id}", "bbox": "${bbox}", "status": "pending"}
    ],
    "completed": [],
    "failed": []
}
EOF
    log "Initialized tile queue with tile ${tile_id}: ${bbox}"
}

# Get next pending tile from queue
get_next_tile() {
    if [[ ! -f "${QUEUE_FILE}" ]]; then
        echo ""
        return
    fi

    # Extract first pending tile
    python3 << EOF
import json
import sys

with open("${QUEUE_FILE}", "r") as f:
    queue = json.load(f)

pending = [t for t in queue["tiles"] if t["status"] == "pending"]
if pending:
    tile = pending[0]
    print(f"{tile['tile_id']}|{tile['bbox']}")
else:
    print("")
EOF
}

# Mark tile as in_progress
mark_tile_in_progress() {
    local tile_id="$1"

    python3 << EOF
import json

with open("${QUEUE_FILE}", "r") as f:
    queue = json.load(f)

for tile in queue["tiles"]:
    if tile["tile_id"] == "${tile_id}":
        tile["status"] = "in_progress"
        break

with open("${QUEUE_FILE}", "w") as f:
    json.dump(queue, f, indent=2)
EOF
}

# Mark tile as completed
mark_tile_completed() {
    local tile_id="$1"

    python3 << EOF
import json

with open("${QUEUE_FILE}", "r") as f:
    queue = json.load(f)

for tile in queue["tiles"]:
    if tile["tile_id"] == "${tile_id}":
        tile["status"] = "completed"
        queue["completed"].append(tile_id)
        break

with open("${QUEUE_FILE}", "w") as f:
    json.dump(queue, f, indent=2)
EOF
    log "Tile ${tile_id} completed"
}

# Add new tiles to queue (after OOM split)
add_tiles_to_queue() {
    local tile_a_id="$1"
    local tile_a_bbox="$2"
    local tile_b_id="$3"
    local tile_b_bbox="$4"
    local original_tile_id="$5"

    python3 << EOF
import json

with open("${QUEUE_FILE}", "r") as f:
    queue = json.load(f)

# Mark original as split
for tile in queue["tiles"]:
    if tile["tile_id"] == "${original_tile_id}":
        tile["status"] = "split"
        break

# Add new tiles
queue["tiles"].append({
    "tile_id": "${tile_a_id}",
    "bbox": "${tile_a_bbox}",
    "status": "pending",
    "parent": "${original_tile_id}"
})
queue["tiles"].append({
    "tile_id": "${tile_b_id}",
    "bbox": "${tile_b_bbox}",
    "status": "pending",
    "parent": "${original_tile_id}"
})

with open("${QUEUE_FILE}", "w") as f:
    json.dump(queue, f, indent=2)
EOF
    log "Added split tiles: ${tile_a_id}, ${tile_b_id}"
}

# ============================================
# Compute initial bounding box from point cloud
# ============================================
compute_initial_bbox() {
    echo "[compute_initial_bbox] Computing initial bounding box from point cloud..." >&2

    local ply_path=""
    if [[ -f "${SOURCE_PATH}/sparse/0/points3D.ply" ]]; then
        ply_path="${SOURCE_PATH}/sparse/0/points3D.ply"
    elif [[ -f "${SOURCE_PATH}/points3D.ply" ]]; then
        ply_path="${SOURCE_PATH}/points3D.ply"
    elif [[ -f "${SOURCE_PATH}/sparse/0/points3D.txt" ]]; then
        # Compute from text file
        python3 << EOF
import numpy as np

points = []
with open("${SOURCE_PATH}/sparse/0/points3D.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) >= 4:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points.append([x, y, z])

if points:
    points = np.array(points)
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
    # Add 10% margin
    margin = 0.1
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    x_min -= dx * margin
    y_min -= dy * margin
    z_min -= dz * margin
    x_max += dx * margin
    y_max += dy * margin
    z_max += dz * margin
    print(f"{x_min},{y_min},{z_min},{x_max},{y_max},{z_max}")
else:
    print("")
EOF
        return
    fi

    if [[ -n "$ply_path" ]]; then
        python3 << EOF
import numpy as np
from plyfile import PlyData

ply = PlyData.read("${ply_path}")
vertex = ply['vertex']
x = np.array(vertex['x'])
y = np.array(vertex['y'])
z = np.array(vertex['z'])

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
z_min, z_max = z.min(), z.max()

# Add 10% margin
margin = 0.1
dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
x_min -= dx * margin
y_min -= dy * margin
z_min -= dz * margin
x_max += dx * margin
y_max += dy * margin
z_max += dz * margin

print(f"{x_min},{y_min},{z_min},{x_max},{y_max},{z_max}")
EOF
    else
        echo "[compute_initial_bbox] Warning: Could not find point cloud file, using default bbox" >&2
        echo "-100,-100,-100,100,100,100"
    fi
}

# ============================================
# Run training for a single tile
# ============================================
run_tile_training() {
    local tile_id="$1"
    local tile_bbox="$2"

    local tile_model_path="${OUTPUT_PATH}/models/${tile_id}"
    local tile_log_folder="${OUTPUT_PATH}/logs/${tile_id}"

    mkdir -p "${tile_model_path}"
    mkdir -p "${tile_log_folder}"
    mkdir -p "${TILES_DIR}"

    log "Starting training for tile ${tile_id}"
    log "  BBox: ${tile_bbox}"
    log "  Model path: ${tile_model_path}"

    # Build the training command
    local cmd="torchrun --nproc_per_node=${NUM_GPUS} ${GRENDEL_DIR}/train.py"
    cmd+=" --source_path ${SOURCE_PATH}"
    cmd+=" --model_path ${tile_model_path}"
    cmd+=" --log_folder ${tile_log_folder}"
    cmd+=" --iterations ${ITERATIONS}"
    cmd+=" --backend ${BACKEND}"
    cmd+=" --bsz ${BSZ}"
    cmd+=" --adaptive_tile_enabled"
    cmd+=" --tile_bbox=\"${tile_bbox}\""
    cmd+=" --tile_id ${tile_id}"
    cmd+=" --tile_output_dir ${TILES_DIR}"
    cmd+=" --tile_crop_margin ${TILE_CROP_MARGIN}"
    cmd+=" --tile_state_file ${STATE_FILE}"

    log "Command: ${cmd}"

    # Run training and capture exit code
    set +e
    eval "${cmd}"
    local exit_code=$?
    set -e

    log "Training exited with code: ${exit_code}"
    return ${exit_code}
}

# ============================================
# Handle OOM state file
# ============================================
process_oom_state() {
    if [[ ! -f "${STATE_FILE}" ]]; then
        log "Error: OOM state file not found: ${STATE_FILE}"
        return 1
    fi

    log "Processing OOM state file..."

    # Read state and add tiles to queue
    python3 << EOF
import json

with open("${STATE_FILE}", "r") as f:
    state = json.load(f)

if not state.get("oom_occurred"):
    print("NO_OOM")
    exit(0)

tile_a = state["tile_a"]
tile_b = state["tile_b"]
original_id = state["original_tile_id"]

# Output for bash to parse
print(f"OOM|{original_id}|{tile_a['tile_id']}|{tile_a['bbox']}|{tile_b['tile_id']}|{tile_b['bbox']}")
EOF
}

# ============================================
# Main loop
# ============================================
main() {
    # Setup output directories
    mkdir -p "${OUTPUT_PATH}"
    mkdir -p "${TILES_DIR}"
    mkdir -p "${OUTPUT_PATH}/models"
    mkdir -p "${OUTPUT_PATH}/logs"

    # Initialize or resume
    if [[ "$RESUME" == true ]] && [[ -f "${QUEUE_FILE}" ]]; then
        log "Resuming from existing queue..."
    else
        # Compute or use provided initial bbox
        if [[ -z "${INITIAL_BBOX}" ]]; then
            INITIAL_BBOX=$(compute_initial_bbox)
        fi

        if [[ -z "${INITIAL_BBOX}" ]]; then
            log "Error: Could not determine initial bounding box"
            exit 1
        fi

        log "Initial bounding box: ${INITIAL_BBOX}"
        init_queue "${INITIAL_BBOX}" "tile_0000"
    fi

    # Main training loop
    local tile_count=0
    local max_tiles=1000  # Safety limit

    while [[ ${tile_count} -lt ${max_tiles} ]]; do
        # Get next tile
        local tile_info
        tile_info=$(get_next_tile)

        if [[ -z "${tile_info}" ]]; then
            log "No more pending tiles. Training complete!"
            break
        fi

        # Parse tile info
        local tile_id
        local tile_bbox
        tile_id=$(echo "${tile_info}" | cut -d'|' -f1)
        tile_bbox=$(echo "${tile_info}" | cut -d'|' -f2)

        log ""
        log "=========================================="
        log "Processing tile ${tile_id} (${tile_count} tiles processed so far)"
        log "=========================================="

        mark_tile_in_progress "${tile_id}"

        # Run training
        set +e
        run_tile_training "${tile_id}" "${tile_bbox}"
        local exit_code=$?
        set -e

        if [[ ${exit_code} -eq ${EXIT_CODE_SUCCESS} ]]; then
            # Training completed successfully
            mark_tile_completed "${tile_id}"
            tile_count=$((tile_count + 1))

        elif [[ ${exit_code} -eq ${EXIT_CODE_OOM} ]]; then
            # OOM occurred, process state and add split tiles
            log "OOM detected, processing split tiles..."

            local oom_info
            oom_info=$(process_oom_state)

            if [[ "${oom_info}" == "NO_OOM" ]]; then
                log "Error: OOM exit code but no OOM state"
                exit 1
            fi

            # Parse OOM info: OOM|original_id|tile_a_id|tile_a_bbox|tile_b_id|tile_b_bbox
            local original_id tile_a_id tile_a_bbox tile_b_id tile_b_bbox
            original_id=$(echo "${oom_info}" | cut -d'|' -f2)
            tile_a_id=$(echo "${oom_info}" | cut -d'|' -f3)
            tile_a_bbox=$(echo "${oom_info}" | cut -d'|' -f4)
            tile_b_id=$(echo "${oom_info}" | cut -d'|' -f5)
            tile_b_bbox=$(echo "${oom_info}" | cut -d'|' -f6)

            add_tiles_to_queue "${tile_a_id}" "${tile_a_bbox}" "${tile_b_id}" "${tile_b_bbox}" "${original_id}"

            # Generate debug images showing the OOM tile with X mark
            local debug_dir="${OUTPUT_PATH}/debug_images"
            mkdir -p "${debug_dir}"
            log "Generating OOM debug images..."
            python3 "${GRENDEL_DIR}/scripts/generate_oom_debug_image.py" \
                --state_file "${STATE_FILE}" \
                --source_path "${SOURCE_PATH}" \
                --output_dir "${debug_dir}" \
                --max_cameras 5 || log "Warning: Debug image generation failed (non-fatal)"

            # Clear OOM state
            rm -f "${STATE_FILE}"

            log "Split tile ${original_id} into ${tile_a_id} and ${tile_b_id}"
            log "Debug images saved to: ${debug_dir}"

        else
            # Other error
            log "Error: Training failed with exit code ${exit_code}"
            exit ${exit_code}
        fi
    done

    if [[ ${tile_count} -ge ${max_tiles} ]]; then
        log "Warning: Reached maximum tile count limit (${max_tiles})"
    fi

    log ""
    log "============================================"
    log "=== Adaptive Tile Training Complete ==="
    log "=== Processed ${tile_count} tiles ==="
    log "=== Finished at $(date) ==="
    log "============================================"

    # Optionally merge all tile PLYs
    log ""
    log "Tile PLY files are saved in: ${TILES_DIR}"
    log "You can merge them using: python merge_tiles.py ${TILES_DIR} output.ply"
}

# Run main
main
