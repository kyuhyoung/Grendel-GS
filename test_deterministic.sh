#!/bin/bash
# Deterministic test script with memory optimization

# Clear existing log and redirect output to both console and log file
> test_deterministic.log
exec > >(tee -a test_deterministic.log) 2>&1

echo "===== Deterministic Test Script ====="
echo "Testing reproducibility of 3DGS training"
echo ""

# Set deterministic environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_LAUNCH_BLOCKING=1
export PYTHONHASHSEED=0

# Scene configuration
SCENE=samsung_dong_mini_5
DIR_DATA=/data/$SCENE
OUTPUT_BASE=./output/${SCENE}_deterministic_test

# Test parameters for memory efficiency
N_GAUSSIANS=100000  # Further reduced for single GPU
ITERATIONS=35       # Test after multiple densifications (at iter 10, 20, 30)
DENSIFY_INTERVAL=10 # Densify every 10 iterations
SH_DEGREE=0         # Minimal SH
IMAGE_RESOLUTION=1  # Add resolution downscaling if available

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
    
    # Clear GPU memory before starting
    nvidia-smi --gpu-reset -i 0 2>/dev/null || true
    
    # Run training with single GPU
    torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py \
        --bsz 1 \
        -s $DIR_DATA \
        --model_path $output_dir \
        --preload_dataset_to_gpu_threshold 0 \
        --densification_interval $DENSIFY_INTERVAL \
        --densify_from_iter 5 \
        --n_g_per_proc $N_GAUSSIANS \
        --sh_degree $SH_DEGREE \
        --deterministic \
        --backend gsplat \
        --iterations $ITERATIONS \
        --densify_memory_limit_percentage 0.5 \
        --test_iterations $ITERATIONS \
        --save_iterations $ITERATIONS \
        --checkpoint_iterations $ITERATIONS \
        --lambda_dssim 0.0 \
        --use_chunk
    
    echo "Run #${run_id} completed"
    echo ""
}

# Run two tests
echo "===== Starting deterministic tests ====="
run_test 1
run_test 2

echo "===== Comparing results ====="

# Find PLY files from both runs (specifically the trained models at final iteration)
PLY1=$(find ${OUTPUT_BASE}_run_1 -name "*_i_$(printf "%05d" ${ITERATIONS})_*.ply" | head -n 1)
PLY2=$(find ${OUTPUT_BASE}_run_2 -name "*_i_$(printf "%05d" ${ITERATIONS})_*.ply" | head -n 1)

# Fallback to any non-input PLY file if specific iteration file not found
if [[ ! -f "$PLY1" ]]; then
    PLY1=$(find ${OUTPUT_BASE}_run_1 -name "*.ply" | grep -v "input.ply" | grep -v "iteration_000110" | head -n 1)
fi
if [[ ! -f "$PLY2" ]]; then
    PLY2=$(find ${OUTPUT_BASE}_run_2 -name "*.ply" | grep -v "input.ply" | grep -v "iteration_000110" | head -n 1)
fi

if [[ -f "$PLY1" && -f "$PLY2" ]]; then
    echo "Comparing PLY files:"
    echo "  Run 1: $PLY1"
    echo "  Run 2: $PLY2"
    echo ""
    
    # Use the compare_ply.sh script
    if [[ -f "./compare_ply.sh" ]]; then
        ./compare_ply.sh "$PLY1" "$PLY2"
    else
        echo "Warning: compare_ply.sh not found"
    fi
else
    echo "Error: Could not find PLY files to compare"
    echo "  PLY1: $PLY1"
    echo "  PLY2: $PLY2"
fi

echo ""
echo "===== Test completed ====="