#!/bin/bash
#
# §4.4 oracle 런 (호스트에서 실행).
# 원조 Grendel-GS 방식: 타일 분할 없이 씬 통째로, GPU 8장 전부.
# train.py 를 직접 호출 (래퍼 없음, adaptive 플래그 없음 = 순정 경로).
# 목적: 기준 기록 (D=시간, Z=최종 loss, Y=가우시안 수) 확보.
# OOM 으로 죽어도 그 자체가 결과 — 죽은 시각/iteration 을 기록한다.
#
# Usage: bash run_oracle.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_oracle.log"
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

OUT="./output/oracle_8gpu"
docker_name=ogs
dir_cur=/workspace/ada_grendel
dir_data=/media2/data/dataset_stereo/non-sat

echo "============================================"
echo "=== ORACLE RUN: single-tile Grendel-GS, 8 GPU ==="
echo "  output : ${OUT}"
echo "  설정   : vanilla densify (interval 100, thr 0.0002), 30k iter, bsz 1"
echo "  시작   : $(date)"
echo "============================================"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

# 순정 경로 확인용: adaptive 플래그 없음 → OOM 핸들러·타일 로직 미탑재.
# MEDIAN_SPLIT/DENSIFY_INSIDE_BBOX 도 tile_bbox 없이는 발화하지 않음.
INNER="source activate citygs-x && cd ${dir_cur} && \
  torchrun --nproc_per_node=8 Grendel-GS/train.py \
    --source_path /data/dabeeo/samsung_dong_mini_30 \
    --model_path ${OUT} \
    --iterations 30000 \
    --bsz 1 \
    --backend default \
    --test_iterations 999999999"

docker run --rm -i --shm-size=64g --gpus all \
    --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -w "${dir_cur}" \
    -v "${dir_data}":/data \
    -v "${SCRIPT_DIR}":"${dir_cur}" \
    -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
    "${docker_name}" bash -c "${INNER}"
rc=$?

echo ""
echo "============================================"
echo "=== ORACLE 종료 $(date) / exit code: ${rc} ==="
echo "  (0 = 30k 완주 / 그 외 = 중도 사망, 로그 마지막 iteration 확인)"
echo "============================================"
exit ${rc}
