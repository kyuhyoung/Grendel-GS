#!/bin/bash
#
# ours 화질 재채점 (야간 체인 종료 후 자동 발사).
#
# 2026-08-18 2차 재측정에서 ours 만 실패: 뷰 3에서 GPU 9GB 연속 할당 실패
# (조각화). 처방 = max_split_size(run_eval.sh 에 반영) + 랭크 8개로 랭크당
# 가우시안·버퍼 절반. 스윕은 벽시계 측정이라 끝날 때까지 기다린다.
#
# Usage: nohup bash run_eval_ours_retry.sh > /dev/null 2>&1 &  (로그: run_eval_ours_retry.log)
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_eval_ours_retry.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

echo "=== ours 재채점 대기 시작 $(date) ==="
echo "[1/2] 야간 체인(스윕) 종료 대기 (run_tonight.log 의 요약 마커)"

DEADLINE=$(( $(date +%s) + 12*3600 ))
while ! grep -q "야간 체인 요약" "${SCRIPT_DIR}/run_tonight.log" 2>/dev/null; do
    if [ "$(date +%s)" -gt "${DEADLINE}" ]; then
        echo "[중단] 12시간 내 체인 미종료 — 재채점 강행하지 않음. run_tonight.log 확인."
        exit 1
    fi
    sleep 120
done
echo "[1/2] 체인 종료 감지 $(date)"

# GPU 정리 대기
for i in $(seq 1 30); do
    BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
    [ "${BUSY}" -eq 0 ] && break
    sleep 20
done
echo "  GPU: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

echo "[2/2] ours 재채점 시작 $(date)  (8 GPU, max_split_size)"
DEVICES="0,1,2,3,4,5,6,7" NPROC=8 ARMS="ours" bash "${SCRIPT_DIR}/run_eval.sh"
rc=$?
echo "=== ours 재채점 종료 rc=${rc} $(date) ==="
exit ${rc}
