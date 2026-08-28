#!/bin/bash
#
# 2026-08-18 야간 체인: ogs 이미지 빌드 완료 대기 → 화질 재측정 → 계단 스윕 재실행.
#
# 왜 순차인가: 스윕은 벽시계가 측정값이라 머신을 단독 점유해야 한다. 채점은
# 시간에 민감하지 않으므로 먼저 돌린다. 채점이 실패해도 스윕은 돌린다 —
# 둘은 서로 의존하지 않고, 밤 시간을 통째로 버릴 이유가 없다.
#
# Usage: nohup bash run_tonight.sh > /dev/null 2>&1 &   (로그: run_tonight.log)
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_tonight.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

echo "============================================"
echo "=== 야간 체인 시작 $(date) ==="
echo "============================================"

# ── [1/3] ogs 이미지 대기 ──────────────────────────────────────────────
echo ""
echo "[1/3] ogs 이미지 빌드 완료 대기 $(date)"
DEADLINE=$(( $(date +%s) + 3*3600 ))
while ! docker image inspect ogs >/dev/null 2>&1; do
    if [ "$(date +%s)" -gt "${DEADLINE}" ]; then
        echo "[중단] 3시간 내 ogs 이미지 안 생김. run_build_ogs.log 확인 필요."
        exit 1
    fi
    sleep 30
done
echo "[1/3] ogs 이미지 확인 $(date)"
docker images ogs

# ── [2/3] 화질 재측정 ──────────────────────────────────────────────────
echo ""
echo "[2/3] 화질 재측정 시작 $(date)"
bash "${SCRIPT_DIR}/run_eval.sh"
EVAL_RC=$?
echo "[2/3] 화질 재측정 종료 rc=${EVAL_RC} $(date)"
[ ${EVAL_RC} -ne 0 ] && echo "  (채점 실패/미완 — 스윕은 그대로 진행. run_eval.log 확인)"

# GPU 가 비었는지 확인 (채점 컨테이너 정리 대기)
for i in $(seq 1 30); do
    BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
    [ "${BUSY}" -eq 0 ] && break
    sleep 20
done
echo "  GPU 상태: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

# ── [3/3] 계단 스윕 (스케줄 정렬판) ────────────────────────────────────
echo ""
echo "[3/3] 계단 스윕 재실행 시작 $(date)  (STEPS=4 2, TAG=matched)"
STEPS="4 2" TAG="matched" bash "${SCRIPT_DIR}/run_sweep.sh"
SWEEP_RC=$?
echo "[3/3] 계단 스윕 종료 rc=${SWEEP_RC} $(date)"

# ── 요약 ───────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "=== 야간 체인 요약 $(date) ==="
echo "  채점 rc=${EVAL_RC} / 스윕 rc=${SWEEP_RC}"
echo ""
echo "--- 화질 ---"
for ARM in ours fmedian fmidpoint; do
    M="${SCRIPT_DIR}/output/eval_${ARM}/metrics.json"
    [ -f "${M}" ] && python3 -c "
import json
d=json.load(open('${M}'))
print('  %-10s n=%s/%s psnr=%s l1=%s' % ('${ARM}', d.get('n'), d.get('n_expected'), d.get('mean_psnr'), d.get('mean_l1')))
" || echo "  ${ARM}: 없음"
done
echo ""
echo "--- 스윕 계단별 시간 ---"
grep -aE "^=== 1/[0-9]+ (ORIGINAL|OURS).*(시작|종료)" "${SCRIPT_DIR}/run_sweep.log" 2>/dev/null
echo ""
echo "--- 스윕 계단별 가우시안 수 ---"
for STEP in 4 2; do
    for f in "${SCRIPT_DIR}"/output/sweep_matched_orig_${STEP}/point_cloud/iteration_*/point_cloud.ply; do
        [ -f "$f" ] && echo "  orig_${STEP}: $(grep -aoE 'element vertex [0-9]+' "$f" | head -1)"
    done
    for f in "${SCRIPT_DIR}"/output/sweep_matched_ours_${STEP}/ply/*_completed_*.ply; do
        [ -f "$f" ] && echo "  ours_${STEP}: $(basename "$f")"
    done
done
echo "============================================"
