GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

#DIR_DATA=/data/samsung_dong
#DIR_DATA=/data/samsung_dong_mini_30
SCENE=samsung_dong_mini_5
DIR_DATA=/data/$SCENE

#if pip install submodules/diff-gaussian-rasterization submodules/simple-knn submodules/gsplat; then
: << END
pip install submodules/simple-knn submodules/gsplat
END
if pip install submodules/diff-gaussian-rasterization; then
    #torchrun --standalone --nnodes=1 --nproc-per-node=3 train.py --bsz 4 -s $DIR_DATA --eval
    torchrun --standalone --nnodes=1 --nproc-per-node=$GPU_COUNT train.py --bsz 1 -s $DIR_DATA --model_path ./output/$SCENE --preload_dataset_to_gpu_threshold 2 --densification_interval 100 --backend gsplat --n_g_per_proc 400000
fi
