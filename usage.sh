#!/bin/bash
# Clear existing usage.log and start fresh
> usage.log
exec > >(tee -a usage.log) 2>&1

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

#DIR_DATA=/data/samsung_dong
#DIR_DATA=/data/samsung_dong_mini_30
#SCENE=samsung_dong_mini_5
#SCENE=sillim_ew_mini_30
SCENE=sillim_ew_mini_100024_20
#SCENE=samsung_dong_mini_30
DIR_DATA=/data/$SCENE

# gsplat and simple-knn are now installed during Docker build
#pip install submodules/gsplat submodules/simple-knn
#pip install submodules/simple-knn
if pip install submodules/diff-gaussian-rasterization; then
    #torchrun --standalone --nnodes=1 --nproc-per-node=$GPU_COUNT train.py --bsz 1 -s $DIR_DATA --model_path ./output/$SCENE --preload_dataset_to_gpu_threshold 2 --densification_interval 100 --backend gsplat --n_g_per_proc 1200000 --sh_degree 0
    torchrun --standalone --nnodes=1 --nproc-per-node=$GPU_COUNT train.py --bsz 1 -s $DIR_DATA --model_path ./output/$SCENE --preload_dataset_to_gpu_threshold 2 --densification_interval 100 --n_g_per_proc 1200000 --sh_degree 0
fi
