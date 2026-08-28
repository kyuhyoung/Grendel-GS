#!/bin/bash
#
# ogs 도커 이미지 재빌드.
# 2026-08-18: 8/17 21:34 마지막 런 이후 이미지가 데몬에서 사라졌다(현재 5개만
# 남아 있고 ogs 없음). 실험을 다시 돌리려면 이게 먼저다.
#
# Usage: nohup bash run_build_ogs.sh > /dev/null 2>&1 &   (로그: run_build_ogs.log)
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_build_ogs.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

echo "=== ogs 빌드 시작 $(date) ==="
docker build --progress=plain -f "${SCRIPT_DIR}/docker_file/Dockerfile_ogs" -t ogs "${SCRIPT_DIR}"
rc=$?
echo "=== ogs 빌드 종료 rc=${rc} $(date) ==="
if [ ${rc} -eq 0 ]; then
    docker images ogs
    echo "--- 스모크: 임포트 확인 ---"
    docker run --rm ogs bash -c "source activate citygs-x && python -c \"
import torch, plyfile, numpy
print('torch', torch.__version__, 'cuda', torch.version.cuda)
import diff_gaussian_rasterization, simple_knn
print('rasterizer/simple-knn OK')
\""
    echo "--- 스모크 rc=$? ---"
fi
exit ${rc}
