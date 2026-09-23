#!/usr/bin/env bash
# COLMAP CUDA 빌드 (B200 sm_100) + pycolmap CUDA 판을 venv 에 설치.
# 전제: gcsudo(=/engrid/ensh/gpubin/ctn_gcsudo, alias 라 스크립트에선 절대경로) apt-get 으로 의존성 설치 완료 (ceres/eigen/glew/freeimage/metis/flann/cgal/qt5 등)
# 사용: nohup bash run_build_colmap_b200.sh > /dev/null 2>&1 &   (로그: run_build_colmap_b200.log)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run_build_colmap_b200.log"; : > "$LOG"
exec > >(stdbuf -o0 tee "$LOG") 2>&1
export PYTHONUNBUFFERED=1

VENV="${VENV:-$HERE/../venv_ogs_b200}"
SRC="${SRC:-$HERE/../colmap_src}"
VER="${VER:-3.11.1}"
JOBS="${JOBS:-48}"
ARCH="${ARCH:-100}"           # B200 = sm_100

step(){ echo; echo "=== [$(date '+%m/%d %H:%M:%S')] $* ==="; }
die(){ echo "!!! 실패: $*"; exit 1; }

step "0) 환경"; nvcc --version | tail -1; cmake --version | head -1; gcc --version | head -1

step "1) 소스 ($VER)"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$VER" https://github.com/colmap/colmap.git "$SRC" || die "clone"
fi
cd "$SRC" && git describe --tags --always

step "2) cmake 구성 (CUDA arch $ARCH, GUI 끔)"
rm -rf build && mkdir build && cd build
cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGUI_ENABLED=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local || die "cmake"

step "3) 빌드 (-j$JOBS)"
ninja -j"$JOBS" || die "ninja"

step "4) 설치 → /usr/local (gcsudo)"
gcsudo ninja install 2>&1 | grep -vE 'execmd' | tail -3
which colmap && colmap -h 2>&1 | head -3

step "5) pycolmap CUDA 판 → venv"
source "$VENV/bin/activate"
pip uninstall -y -q pycolmap 2>/dev/null
pip install -q scikit-build-core pybind11 || die "build backend"
# /tmp 가 noexec 라 스텁 생성이 .so 를 못 올림 → TMPDIR 을 실행 가능한 곳으로, 스텁은 끔(ruff 도 없음)
BT="$HERE/../pycolmap_build_tmp"; mkdir -p "$BT"
cd "$SRC" && TMPDIR="$BT" CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=$ARCH -DGUI_ENABLED=OFF -DGENERATE_STUBS=OFF" \
  CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" pip install --no-build-isolation --no-cache-dir ./pycolmap 2>&1 | grep -E 'Successfully|CMake build failed' | tail -2
rm -rf "$BT"
python -c "import pycolmap;print('pycolmap',pycolmap.__version__)" || die "pycolmap import"

step "6) GPU 스모크: SIFT 추출 1장"
python - <<'PY'
import pycolmap, tempfile, os, glob
img=sorted(glob.glob('/NHNHOME/WORKSPACE/26molit001_dbo/kevin/work/dabeeo/suwon_hwasung/data/hwaseong_drone_rtk/rgb/paldalmun/*/*_D.JPG'))[:1]
d=tempfile.mkdtemp(); os.symlink(img[0], f"{d}/a.JPG")
eo=pycolmap.FeatureExtractionOptions(); eo.use_gpu=True
pycolmap.extract_features(f"{d}/db.db", d, extraction_options=eo)
db=pycolmap.Database(f"{d}/db.db"); print("GPU SIFT OK, keypoints:", db.num_keypoints)
PY
echo; echo "=== 빌드 완료 $(date) ==="
