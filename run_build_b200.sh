#!/usr/bin/env bash
# B200(Blackwell, sm_100) native 빌드 — docker 없는 머신용.
# docker_file/Dockerfile_ogs_b200 의 RUN 들을 셸로 옮긴 것.
# 이미 NGC 계열 컨테이너(torch 2.10a / CUDA 13.1) 안이고 root 가 아니므로
# apt 대신 venv(--system-site-packages) 를 쓴다. NGC torch/torchvision 은 건드리지 않는다.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run_build_b200.log"
: > "$LOG"                      # 실행마다 새로 생성 (append 아님)
exec > >(stdbuf -o0 tee "$LOG") 2>&1
export PYTHONUNBUFFERED=1

VENV="${VENV:-$HERE/../venv_ogs_b200}"
SUB="$HERE/Grendel-GS/submodules"      # 주의: graphdeco clone 이 아니라 이 저장소의 분산판
export TORCH_CUDA_ARCH_LIST="10.0+PTX" # 없으면 커널이 B200 용으로 안 만들어짐
export MAX_JOBS="${MAX_JOBS:-16}"

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
die()  { echo "!!! 실패: $*"; exit 1; }

step "0) 환경"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
echo "SUB=$SUB"; [ -d "$SUB/simple-knn" ] || die "$SUB 없음 (중첩 경로 확인)"

step "1) venv (--system-site-packages)"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV" || die "venv 생성"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'cap',torch.cuda.get_device_capability(0))"
pip install --no-cache-dir -q -U pip setuptools wheel || die "pip/setuptools"
# torchrun 래퍼: 시스템 torchrun(/usr/local/bin) 은 /usr/bin/python 으로 자식을 띄워 venv 패키지를 못 본다 (실측: ModuleNotFoundError)
cat > "$VENV/bin/torchrun" <<EOT
#!$VENV/bin/python
import sys
from torch.distributed.run import main
if __name__ == "__main__":
    sys.exit(main())
EOT
chmod +x "$VENV/bin/torchrun"

step "2) 파이썬 의존성 (torch/torchvision 제외)"
pip install --no-cache-dir \
  plyfile tqdm opencv-python-headless imageio imageio-ffmpeg \
  scikit-image scipy scikit-learn pandas matplotlib psutil \
  lpips tifffile einops omegaconf configargparse rich \
  tensorboard wandb h5py trimesh networkx pyyaml ninja mlflow || die "deps"

step "3) 패치 A — <cstdint> (2023 코드 + CUDA13/GCC13 함정)"
n=0
while IFS= read -r f; do
  grep -qE '#include <cstdint>|#include <stdint.h>' "$f" && continue
  sed -i '1i #include <cstdint>' "$f"; n=$((n+1)); echo "  patched: ${f#$HERE/}"
done < <(grep -rlE 'uint32_t|uint64_t|uintptr_t|int64_t' "$SUB" \
           --include=*.h --include=*.cu --include=*.cuh --include=*.cpp)
echo "cstdint 패치 완료 ($n 파일)"

step "4) simple-knn"
pip install --no-build-isolation --no-cache-dir "$SUB/simple-knn" || die "simple-knn"

step "5) diff-gaussian-rasterization (Grendel 분산판)"
pip install --no-build-isolation --no-cache-dir "$SUB/diff-gaussian-rasterization" || die "diff-gaussian-rasterization"

step "6) fused-ssim (필수 — 벤더링 소스, PyPI 아님)"
pip install --no-build-isolation --no-cache-dir "$SUB/fused-ssim" || die "fused-ssim"
python -c "import fused_ssim; print('fused_ssim OK')" || die "fused_ssim import"

step "6b) gsplat (선택 — 정사영상만 영향)"
pip install --no-build-isolation --no-cache-dir gsplat==1.4.0 || echo "[warn] gsplat 실패 — 정사영상만 영향, 학습은 가능"

step "7) 스모크 (import + 실제 커널 실행)"
python - <<'PY'
import torch, diff_gaussian_rasterization as dgr, simple_knn._C as knn, plyfile, fused_ssim
print('torch', torch.__version__, torch.version.cuda, 'cap', torch.cuda.get_device_capability(0))
x = torch.rand(1000, 3, device='cuda')
d = knn.distCUDA2(x); torch.cuda.synchronize()
print('simple-knn kernel OK', tuple(d.shape), float(d.mean()))
print('rasterizer OK', dgr.__file__)
print('fused_ssim OK', fused_ssim.__file__)
PY
rc=$?
echo; [ $rc -eq 0 ] && echo "=== 빌드 성공 ===  source $VENV/bin/activate" || echo "=== 스모크 실패 (rc=$rc) ==="
echo "로그: $LOG"
exit $rc
