#!/bin/bash
# Clear existing usage.log and start fresh
> usage.log
exec > >(tee -a usage.log) 2>&1

# These environment variables can be enabled for extra determinism
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# export CUDA_LAUNCH_BLOCKING=1

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

#DIR_DATA=/data/samsung_dong
#DIR_DATA=/data/samsung_dong_mini_30
SCENE=samsung_dong_mini_5
#SCENE=sillim_ew_mini_30
#SCENE=sillim_ew_mini_100024_20
#SCENE=samsung_dong_mini_30
DIR_DATA=/data/$SCENE

# gsplat and simple-knn are now installed during Docker build
#pip install submodules/gsplat submodules/simple-knn
#pip install submodules/simple-knn
./scripts/install/fix_gsplat_cuda.sh
if pip install submodules/diff-gaussian-rasterization submodules/gsplat submodules/simple-knn; then
    #torchrun --standalone --nnodes=1 --nproc-per-node=$GPU_COUNT train.py --bsz 1 -s $DIR_DATA --model_path ./output/$SCENE --preload_dataset_to_gpu_threshold 2 --densification_interval 100 --backend gsplat --n_g_per_proc 12000000 --sh_degree 3
    # Disable preload to GPU (set threshold to 0) to avoid GPU memory issues
    # Add --deterministic flag for reproducible results
    #torchrun --standalone --nnodes=1 --nproc-per-node=$GPU_COUNT train.py --bsz 1 -s $DIR_DATA --model_path ./output/$SCENE --preload_dataset_to_gpu_threshold 0 --densification_interval 100 --n_g_per_proc 12000000 --sh_degree 0 --deterministic
    # Single GPU with memory optimization for deterministic testing
    GPU_COUNT=1 torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py --bsz 1 -s $DIR_DATA --model_path ./output/$SCENE --preload_dataset_to_gpu_threshold 0 --densification_interval 100 --n_g_per_proc 1000000 --sh_degree 0 --deterministic --iterations 1000 --densify_memory_limit_percentage 0.8
fi
