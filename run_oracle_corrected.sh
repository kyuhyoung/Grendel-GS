#!/bin/bash
#
# 해상도 보정 oracle (§4.4 [TODO]): 8 GPU 가 전부 빌 때까지 대기 후 자동 발사.
# vanilla densify 임계값 2e-4 는 화면상대(NDC) 단위라 1.96억 px 뷰에서 신호가
# 임계 아래로 깔림 → 해상도 비례 보정 2e-4 × 2428/17310 ≈ 3e-5 로 재실행.
# 나머지는 8/12 oracle 런과 동일 (순정 train.py, 30k, bsz 1).
#
# Usage: nohup bash run_oracle_corrected.sh > /dev/null 2>&1 &  (로그: run_oracle_corrected.log)
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_oracle_corrected.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

echo "=== 보정 oracle 대기 시작 $(date) ==="
echo "[1/2] GPU 8장 전부 비기를 대기 (10분 연속 idle 확인; 최대 48h)"

DEADLINE=$(( $(date +%s) + 48*3600 ))
IDLE_STREAK=0
while true; do
    BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
    if [ "${BUSY}" -eq 0 ]; then
        IDLE_STREAK=$((IDLE_STREAK+1))
        # 60초 x 10 = 10분 연속 idle 이면 진짜 빈 것으로 판정 (잡 사이 틈 오탐 방지)
        [ "${IDLE_STREAK}" -ge 10 ] && { echo "[1/2] 8장 idle 확정 $(date)"; break; }
    else
        IDLE_STREAK=0
    fi
    if [ "$(date +%s)" -gt "${DEADLINE}" ]; then
        echo "[중단] 48시간 내 GPU 미확보. 수동 재시도 필요."
        exit 1
    fi
    sleep 60
done

OUT="./output/oracle_8gpu_thr3e-5"
docker_name=ogs
dir_cur=/workspace/ada_grendel
dir_data=/media2/data/dataset_stereo/non-sat

echo "[2/2] 보정 oracle 발사 $(date)  (thr=3e-5, output=${OUT})"
INNER="source activate citygs-x && cd ${dir_cur} && \
  torchrun --nproc_per_node=8 Grendel-GS/train.py \
    --source_path /data/dabeeo/samsung_dong_mini_30 \
    --model_path ${OUT} \
    --iterations 30000 \
    --bsz 1 \
    --backend default \
    --densify_grad_threshold 3e-5 \
    --test_iterations 999999999"

docker run --rm -i --shm-size=64g --gpus all \
    --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -w "${dir_cur}" \
    -v "${dir_data}":/data -v "${SCRIPT_DIR}":"${dir_cur}" \
    -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
    "${docker_name}" bash -c "${INNER}"
rc=$?
echo "=== 보정 oracle 종료 rc=${rc} $(date) ==="
exit ${rc}
