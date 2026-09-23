#!/usr/bin/env bash
# cuDSS(NVIDIA 희소 직접 솔버) + Ceres main(cuDSS 지원) + COLMAP main 재빌드 → 번들조정의 희소 솔버까지 GPU.
# 배경: Ceres 2.2.0 은 cuDSS 옵션이 없고, COLMAP 은 cuDSS 없으면 "Falling back to CPU-based sparse solvers".
# 사용: nohup bash run_build_ceres_cudss.sh > /dev/null 2>&1 &     로그: run_build_ceres_cudss.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run_build_ceres_cudss.log"; : > "$LOG"
exec > >(stdbuf -o0 tee "$LOG") 2>&1
CUDSS_VER=0.8.0.10; CUDSS_TAR="libcudss-linux-x86_64-${CUDSS_VER}_cuda13-archive.tar.xz"
CUDSS_DIR="$HERE/../cudss"; CERES_SRC="$HERE/../ceres_main_src"; CERES_PREFIX="$HERE/../ceres_cuda"
COLMAP_SRC="$HERE/../colmap_main_src"; COLMAP_PREFIX="$HERE/../colmap_main"
JOBS="${JOBS:-40}"; ARCH="${ARCH:-100}"
step(){ echo; echo "=== [$(date '+%m/%d %H:%M:%S')] $* ==="; }
die(){ echo "!!! 실패: $*"; exit 1; }

ABSL_SRC="$HERE/../abseil_src"; ABSL_PREFIX="$HERE/../absl"
step "0) Abseil (Ceres main 요구) 소스 빌드 → $ABSL_PREFIX"
if [ ! -f "$ABSL_PREFIX/lib/cmake/absl/abslConfig.cmake" ]; then
  [ -d "$ABSL_SRC/.git" ] || git clone -q --depth 1 --branch lts_2025_01_27 https://github.com/abseil/abseil-cpp.git "$ABSL_SRC" || die "absl clone"
  cd "$ABSL_SRC" && rm -rf build && mkdir build && cd build
  cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ABSL_PREFIX" -DCMAKE_CXX_STANDARD=17 \
    -DABSL_PROPAGATE_CXX_STD=ON -DABSL_BUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON >/dev/null || die "absl cmake"
  ninja -j"$JOBS" 2>&1 | grep -E 'error:|FAILED' | head -3; [ ${PIPESTATUS[0]} -eq 0 ] || die "absl ninja"
  ninja install >/dev/null || die "absl install"
fi
ls "$ABSL_PREFIX/lib/cmake/absl/abslConfig.cmake" >/dev/null && echo "absl OK"

step "1) cuDSS $CUDSS_VER 다운로드·풀기 → $CUDSS_DIR"
mkdir -p "$CUDSS_DIR" && cd "$CUDSS_DIR"
[ -f "$CUDSS_TAR" ] || curl -sSL --max-time 600 -o "$CUDSS_TAR" "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/$CUDSS_TAR" || die "cudss download"
tar -xJf "$CUDSS_TAR" --strip-components=1 || die "cudss untar"
CUDSS_CMAKE=$(find "$CUDSS_DIR" -name 'cudss-config.cmake' | head -1); [ -n "$CUDSS_CMAKE" ] || die "cudss-config.cmake 없음"
CUDSS_CMAKE_DIR=$(dirname "$CUDSS_CMAKE"); echo "cudss cmake: $CUDSS_CMAKE_DIR"; ls "$CUDSS_DIR/lib" | head -3

step "2) Ceres main cmake (USE_CUDA + cuDSS, arch $ARCH)"
cd "$CERES_SRC" && git log --oneline -1
grep -q 'set(CMAKE_CUDA_ARCHITECTURES "50;60;70;80")' CMakeLists.txt && sed -i 's|set(CMAKE_CUDA_ARCHITECTURES "50;60;70;80")|set(CMAKE_CUDA_ARCHITECTURES "'"$ARCH"'")|' CMakeLists.txt && echo "arch 고정줄 패치"
rm -rf build && mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CERES_PREFIX" \
  -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -Dcudss_DIR="$CUDSS_CMAKE_DIR" -Dabsl_DIR="$ABSL_PREFIX/lib/cmake/absl" -DCMAKE_PREFIX_PATH="$CUDSS_DIR;$ABSL_PREFIX" \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DSUITESPARSE=ON -DEIGENSPARSE=ON 2>&1 | grep -iE 'cudss|CUDA|Error|error|Configuring' | tail -10
[ ${PIPESTATUS[0]} -eq 0 ] || die "ceres cmake"
grep -iE 'CUDSS' CMakeCache.txt | grep -iE 'FOUND|_DIR' | head -3

step "3) Ceres 빌드·설치 (-j$JOBS) → $CERES_PREFIX (2.2.0 판 덮어씀)"
ninja -j"$JOBS" 2>&1 | grep -E 'error:|FAILED' | head -5; [ ${PIPESTATUS[0]} -eq 0 ] || die "ceres ninja"
rm -rf "$CERES_PREFIX" && ninja install >/dev/null || die "ceres install"
grep -E 'CERES_NO_CUDA|CERES_NO_CUDSS|CERES_USE_CUDSS' "$CERES_PREFIX"/include/ceres/internal/config.h | head -3

step "4) COLMAP main 재빌드"
cd "$COLMAP_SRC" && rm -rf build && mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF -DCMAKE_INSTALL_PREFIX="$COLMAP_PREFIX" \
  -DCeres_DIR="$CERES_PREFIX/lib/cmake/Ceres" -Dcudss_DIR="$CUDSS_CMAKE_DIR" -Dabsl_DIR="$ABSL_PREFIX/lib/cmake/absl" -DCMAKE_PREFIX_PATH="$CERES_PREFIX;$CUDSS_DIR;$ABSL_PREFIX" 2>&1 | grep -iE 'Ceres|cudss|Error|error|Configuring' | tail -6
[ ${PIPESTATUS[0]} -eq 0 ] || die "colmap cmake"
ninja -j"$JOBS" 2>&1 | grep -E 'error:|FAILED' | head -5; [ ${PIPESTATUS[0]} -eq 0 ] || die "colmap ninja"
ninja install 2>&1 | grep -E 'bin/colmap' | tail -1

step "5) 확인: GPU BA 스모크 (장안문 모델 3회 반복)"
export LD_LIBRARY_PATH="$CUDSS_DIR/lib:$ABSL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
ldd "$COLMAP_PREFIX/bin/colmap" | grep -iE 'cudss|ceres' | awk '{print "  "$1, $3}'
T="$HERE/../pycolmap_build_tmp/ba_test2"; rm -rf "$T"; mkdir -p "$T"
CUDA_VISIBLE_DEVICES=2 timeout 900 "$COLMAP_PREFIX/bin/colmap" bundle_adjuster \
  --input_path /NHNHOME/WORKSPACE/26molit001_dbo/kevin/work/dabeeo/suwon_hwasung/data/hwaseong_drone_rtk/colmap_glomap/janganmun/sparse/0 \
  --output_path "$T" --BundleAdjustmentCeres.use_gpu 1 --BundleAdjustmentCeres.max_num_iterations 3 2>&1 \
  | grep -iE 'Falling back|cuDSS|Iterations|Elapsed' | head -5
echo "=== 완료 $(date) ==="
