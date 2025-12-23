# Out-of-Core Gaussian Splatting
대규모 Gaussian Splat 장면 최적화를 위해 장면을 타일화 하고 가우시안 파라메터와 옵티마이저 상태를 OoC로 스트리밍 하여 학습하게 하는 프로젝트 입니다.

## 설치 방법

### Conda 환경 설정 및 Grendel-GS 빌드
Quadro RTX 6000 환경 기준으로 작동한 방법입니다.
```
conda create -n gs python==3.8
conda activate gs

conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
python -m pip install --upgrade pip
python -m pip install \
  torch==1.12.1+cu116 \
  torchvision==0.13.1+cu116 \
  torchaudio==0.12.1 \
  --index-url https://download.pytorch.org/whl/cu116

python -m pip install plyfile opencv-python tqdm psutil pillow matplotlib pandas
python -m pip install "numpy<2.0"

sudo apt update
sudo apt install g++-10 gcc-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10   

python -m pip install ./Grendel-GS/submodules/diff-gaussian-rasterization/ --no-build-isolation
python -m pip install ./Grendel-GS/submodules/simple-knn/ --no-build-isolation
python -m pip install ./Grendel-GS/submodules/gsplat/ --no-build-isolation
```

## 사용 방법

### 장면 타일 생성
```
python scripts/create_gaussians_and_tiles.py \
--input_ply data/Samsung_SN_82/sparse/0/points3D.ply \
--output_dir data/Samsung_SN_82/tiled_scene \
--grid_size 16 16 1 \
--overlap 1.0
```

### 가시성 데이터 계산
```
python scripts/precompute_tile_visibility.py \
    --tiles data/Samsung_SN_82/tiled_scene \
    --colmap data/Samsung_SN_82/sparse/0
```

### 학습 
```
torchrun --nproc_per_node=8 --master_port 29505 scripts/train_streaming_grendel.py \
--images_path data/Samsung_SN_82/images \
--colmap_path data/Samsung_SN_82/sparse/0 \
--tiles_path data/Samsung_SN_82/tiled_scene \
--output_path output/samsung_sn_82_run0 \
--resolution 1 \
--cache_size 240 \
--max_cached_images 82 \
--densification_interval 100 \
--densify_from_iter 15000 \
--iterations 15000 \
--save_iterations 10000 15000 \
--view_iter 1 \
--preload_images
```

기존에 저장된 학습에 이어서 학습하려면 아래 옵션을 사용하면 됩니다.
```
--resume_from output/samsung_sn_82_run0/iteration_1000
```

### 학습된 타일들을 ply로 변환
```
python scripts/export_tiles_to_ply.py \
--tiles-root output/samsung_sn_82_run0/iteration_3000 \
--output-ply output/samsung_sn_82_run0/iteration_3000/merged.ply
```