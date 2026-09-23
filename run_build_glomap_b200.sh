#!/usr/bin/env bash
# GLOMAP(global SfM) — 별도 저장소는 deprecated 되어 COLMAP main 에 `colmap global_mapper` 로 흡수됨.
# 그래서 COLMAP main 브랜치를 B200(sm_100)용으로 빌드해 ../colmap_main 에 설치한다 (/usr/local 의 3.11.1 은 유지).
# 사용: nohup bash run_build_glomap_b200.sh > /dev/null 2>&1 &   (로그: run_build_glomap_b200.log)
#   결과: ../colmap_main/bin/colmap  (global_mapper 포함)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run_build_glomap_b200.log"; : > "$LOG"
exec > >(stdbuf -o0 tee "$LOG") 2>&1

SRC="${SRC:-$HERE/../colmap_main_src}"
PREFIX="${PREFIX:-$HERE/../colmap_main}"
JOBS="${JOBS:-48}"
ARCH="${ARCH:-100}"

step(){ echo; echo "=== [$(date '+%m/%d %H:%M:%S')] $* ==="; }
die(){ echo "!!! 실패: $*"; exit 1; }

step "1) 소스 (colmap main)"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch main https://github.com/colmap/colmap.git "$SRC" || die "clone"
fi
cd "$SRC" && git log --oneline -1
grep -q '"global_mapper"' src/colmap/exe/colmap.cc || die "이 소스에 global_mapper 없음"

step "2) cmake (CUDA arch $ARCH, GUI 끔, prefix $PREFIX)"
rm -rf build && mkdir build && cd build
cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" 2>&1 | grep -E 'error|Error|Configuring|Generating' | tail -4
[ ${PIPESTATUS[0]} -eq 0 ] || die "cmake"

step "3) 빌드 (-j$JOBS)"
ninja -j"$JOBS" 2>&1 | grep -E 'error:|FAILED' | head -5
[ ${PIPESTATUS[0]} -eq 0 ] || die "ninja"

step "4) 설치 → $PREFIX (사용자 경로, root 불필요)"
ninja install 2>&1 | grep -E 'Installing.*bin/colmap' | tail -1
"$PREFIX/bin/colmap" -h 2>&1 | head -2
"$PREFIX/bin/colmap" global_mapper -h 2>&1 | head -3 || die "global_mapper 실행"
echo; echo "=== 빌드 완료 $(date) ==="
