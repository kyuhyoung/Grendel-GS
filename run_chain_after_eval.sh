#!/bin/bash
#
# 자동 이어달리기: PSNR 측정 종료 대기 → 계단 스윕(§4.5) 발사.
# 2026-08-17 교훈: 다음 단계를 "내가 알림 받고 실행"에 매달면 알림이 늦을 때
# 통째로 멈춘다(ours 완주 후 26시간 공백). 컴퓨터가 스스로 잇게 한다.
#
# Usage: nohup bash run_chain_after_eval.sh > /dev/null 2>&1 &
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_chain_after_eval.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

echo "=== chain 시작 $(date) ==="
echo "[1/2] PSNR 측정 3종 완료 대기 (eval_*/metrics.json)"

# 2026-08-18: 예전엔 metrics.json "존재"만 봤다. 그래서 30장 중 2장만 채점된
# eval_ours 를 완료로 읽고 스윕을 발사했고, §4.4 비교가 통째로 무효가 됐다.
# 파일이 아니라 내용(장수)을 본다.
EXPECT_N=${EXPECT_N:-30}
complete_arm () {   # $1=eval 폴더명. 완료면 0, 아니면 1
    python3 -c "
import json, sys
try:
    d = json.load(open('${SCRIPT_DIR}/output/$1/metrics.json'))
except Exception:
    sys.exit(1)
sys.exit(0 if d.get('n') == ${EXPECT_N} and d.get('mean_psnr') is not None else 1)
" 2>/dev/null
}

DEADLINE=$(( $(date +%s) + 6*3600 ))
while true; do
    N=0
    for d in eval_ours eval_fmedian eval_fmidpoint; do
        complete_arm "${d}" && N=$((N+1))
    done
    [ "${N}" -eq 3 ] && { echo "[1/2] 3종 완료 감지 (각 ${EXPECT_N}장) $(date)"; break; }
    if [ "$(date +%s)" -gt "${DEADLINE}" ]; then
        echo "[중단] 6시간 내 미완 — 스윕 강행하지 않음 (완결 ${N}/3)."
        for d in eval_ours eval_fmedian eval_fmidpoint; do
            complete_arm "${d}" || echo "  미완: ${d}"
        done
        exit 1
    fi
    sleep 60
done

# GPU 가 비었는지 확인 (측정 컨테이너 정리 대기)
for i in $(seq 1 30); do
    BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
    [ "${BUSY}" -eq 0 ] && break
    sleep 20
done
echo "[1/2] GPU 상태:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "[2/2] 계단 스윕 시작 $(date)"
bash "${SCRIPT_DIR}/run_sweep.sh"
echo "=== chain 종료 $(date) / sweep rc=$? ==="
