#!/usr/bin/env bash
# Ceres Solver 를 CUDA 켜서 소스 빌드(→ ../ceres_cuda) 하고, COLMAP main 을 그 Ceres 로 다시 빌드(→ ../colmap_main).
# 이유: apt 의 libceres-dev 는 CUDA 없이 빌드되어 COLMAP 번들조정이 GPU 옵션에도 CPU 로 떨어짐 (실측 경고).
# 사용: nohup bash run_build_ceres_cuda.sh > /dev/null 2>&1 &     로그: run_build_ceres_cuda.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run_build_ceres_cuda.log"; : > "$LOG"
exec > >(stdbuf -o0 tee "$LOG") 2>&1
CERES_SRC="$HERE/../ceres_src"; CERES_PREFIX="$HERE/../ceres_cuda"
COLMAP_SRC="$HERE/../colmap_main_src"; COLMAP_PREFIX="$HERE/../colmap_main"
JOBS="${JOBS:-40}"; ARCH="${ARCH:-100}"
step(){ echo; echo "=== [$(date '+%m/%d %H:%M:%S')] $* ==="; }
die(){ echo "!!! 실패: $*"; exit 1; }

step "1) Ceres 소스 (2.2.0)"
[ -d "$CERES_SRC/.git" ] || git clone --depth 1 --branch 2.2.0 https://github.com/ceres-solver/ceres-solver.git "$CERES_SRC" || die clone
cd "$CERES_SRC" && git describe --tags --always

step "2) Ceres cmake (USE_CUDA=ON, arch $ARCH)"
rm -rf build && mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CERES_PREFIX" \
  -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCERES_CUDA_ARCH_OVERRIDE="$ARCH" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DSUITESPARSE=ON -DEIGENSPARSE=ON 2>&1 | grep -E 'CUDA|cuDSS|SuiteSparse|Error|error|Configuring' | tail -8
[ ${PIPESTATUS[0]} -eq 0 ] || die "ceres cmake"

step "3) Ceres 빌드·설치 (-j$JOBS)"
ninja -j"$JOBS" 2>&1 | grep -E 'error:|FAILED' | head -5; [ ${PIPESTATUS[0]} -eq 0 ] || die "ceres ninja"
ninja install >/dev/null || die "ceres install"
ls "$CERES_PREFIX"/lib/cmake/Ceres/CeresConfig.cmake >/dev/null || die "CeresConfig 없음"
grep -E 'CERES_USE_CUDA|CERES_NO_CUDA' "$CERES_PREFIX"/include/ceres/internal/config.h | head -2

step "4) COLMAP main 재빌드 (Ceres_DIR=$CERES_PREFIX)"
cd "$COLMAP_SRC" && rm -rf build && mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF -DCMAKE_INSTALL_PREFIX="$COLMAP_PREFIX" \
  -DCeres_DIR="$CERES_PREFIX/lib/cmake/Ceres" -DCMAKE_PREFIX_PATH="$CERES_PREFIX" 2>&1 | grep -E 'Ceres|CUDA|Error|error|Configuring' | tail -6
[ ${PIPESTATUS[0]} -eq 0 ] || die "colmap cmake"
ninja -j"$JOBS" 2>&1 | grep -E 'error:|FAILED' | head -5; [ ${PIPESTATUS[0]} -eq 0 ] || die "colmap ninja"
ninja install 2>&1 | grep -E 'bin/colmap' | tail -1

step "5) 확인"
ldd "$COLMAP_PREFIX/bin/colmap" | grep -iE 'ceres' ; "$COLMAP_PREFIX/bin/colmap" -h 2>&1 | head -2
echo "=== 완료 $(date) ==="
