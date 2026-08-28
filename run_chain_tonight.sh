#!/bin/bash
#
# 오늘 밤 이어달리기 (호스트에서 nohup 으로 실행 후 퇴근).
#   1) 지금 도는 midpoint3 (면적 재경기, GPU 5-7) 완주를 기다림
#   2) 끝나면 GPU 가 빈 것을 확인하고 oracle (8 GPU) 자동 시작
#
# Usage:  nohup bash run_chain_tonight.sh > /dev/null 2>&1 &
#         (자체 로그: run_chain_tonight.log)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_chain_tonight.log"
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

echo "=== chain 시작 $(date) ==="
echo "[1/2] midpoint3 완주 대기 (run_smoke3_midpoint.log 의 종료 마커 감시)"

# 최대 12시간 대기 (그 이상이면 뭔가 잘못된 것 — oracle 을 강행하지 않고 멈춤)
DEADLINE=$(( $(date +%s) + 12*3600 ))
while true; do
    if grep -q "=== 종료" "${SCRIPT_DIR}/run_smoke3_midpoint.log" 2>/dev/null; then
        echo "[1/2] midpoint3 종료 감지: $(date)"
        break
    fi
    if [ "$(date +%s)" -gt "${DEADLINE}" ]; then
        echo "[중단] 12시간 내 midpoint3 가 안 끝남 — oracle 강행하지 않고 종료. 확인 필요."
        exit 1
    fi
    sleep 60
done

# GPU 가 실제로 비었는지 확인 (컨테이너 정리 지연 대비 최대 10분)
for i in $(seq 1 60); do
    BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
    [ "${BUSY}" -eq 0 ] && break
    sleep 10
done
echo "[1/2] GPU 상태:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "[2/2] oracle 시작: $(date)"
bash "${SCRIPT_DIR}/run_oracle.sh"
rc=$?
echo "=== chain 종료 $(date) / oracle exit: ${rc} ==="
exit ${rc}
