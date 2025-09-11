#!/bin/bash
# Minimal deterministic test script - for testing only

# Clear existing log and redirect output to both console and log file
> test_deterministic_mini.log
exec > >(tee -a test_deterministic_mini.log) 2>&1

echo "===== Minimal Deterministic Test Script ====="
echo "Testing reproducibility with minimal settings"
echo ""

# Set deterministic environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_LAUNCH_BLOCKING=1
export PYTHONHASHSEED=0

# Scene configuration - use smallest possible dataset
SCENE=samsung_dong_mini_5
DIR_DATA=/data/$SCENE
OUTPUT_BASE=./output/${SCENE}_deterministic_mini_test

# Minimal test parameters
N_GAUSSIANS=10000   # Very small
ITERATIONS=50       # Very short
DENSIFY_INTERVAL=100
SH_DEGREE=0

echo "Configuration:"
echo "  Scene: $SCENE"
echo "  Max Gaussians: $N_GAUSSIANS"
echo "  Iterations: $ITERATIONS"
echo "  Output: $OUTPUT_BASE"
echo ""

# Function to run deterministic test
run_test() {
    local run_id=$1
    local output_dir="${OUTPUT_BASE}_run_${run_id}"
    
    echo "Starting Run #${run_id}..."
    echo "Output: $output_dir"
    
    # Run training with single GPU and minimal settings
    torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py \
        --bsz 1 \
        -s $DIR_DATA \
        --model_path $output_dir \
        --preload_dataset_to_gpu_threshold 0 \
        --densification_interval $DENSIFY_INTERVAL \
        --n_g_per_proc $N_GAUSSIANS \
        --sh_degree $SH_DEGREE \
        --deterministic \
        --iterations $ITERATIONS \
        --densify_memory_limit_percentage 0.3 \
        --test_iterations $ITERATIONS \
        --save_iterations $ITERATIONS \
        --checkpoint_iterations $ITERATIONS \
        --lambda_dssim 0.0 \
        --num_train_cameras 2 \
        --densify_until_iter 30 \
        --quiet
    
    echo "Run #${run_id} completed"
    echo ""
}

# Run two tests
echo "===== Starting minimal deterministic tests ====="
run_test 1
run_test 2

echo "===== Comparing results ====="

# Find PLY files from both runs
PLY1=$(find ${OUTPUT_BASE}_run_1 -name "point_cloud_*.ply" 2>/dev/null | head -n 1)
PLY2=$(find ${OUTPUT_BASE}_run_2 -name "point_cloud_*.ply" 2>/dev/null | head -n 1)

# If no point_cloud PLY found, try input.ply
if [[ -z "$PLY1" ]]; then
    PLY1="${OUTPUT_BASE}_run_1/input.ply"
fi
if [[ -z "$PLY2" ]]; then
    PLY2="${OUTPUT_BASE}_run_2/input.ply"
fi

if [[ -f "$PLY1" && -f "$PLY2" ]]; then
    echo "Comparing PLY files:"
    echo "  Run 1: $PLY1"
    echo "  Run 2: $PLY2"
    echo ""
    
    # Use the compare_ply.py directly for simpler output
    python compare_ply.py "$PLY1" "$PLY2"
else
    echo "Error: Could not find PLY files to compare"
    echo "  PLY1: $PLY1"
    echo "  PLY2: $PLY2"
fi

echo ""
echo "===== Test completed ====="