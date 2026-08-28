#!/bin/bash
#
# B-2: 1/4 계단 반복 실험 (오차막대용). 기존 TAG=matched 가 1회차 —
# rep2, rep3 를 추가해 n=3. 각 회차 = original + ours (정렬 스케줄, run_sweep.sh 재사용).
#
# Usage: nohup bash run_repeats.sh > /dev/null 2>&1 &   (로그: run_repeats.log)
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_repeats.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

echo "=== 반복 실험 시작 $(date) ==="
for i in 2 3; do
    echo ""
    echo "[rep${i}] 시작 $(date)"
    STEPS="4" TAG="matched_rep${i}" DEVICES="5,6" bash "${SCRIPT_DIR}/run_sweep.sh"
    echo "[rep${i}] 종료 rc=$? $(date)"
    # run_sweep.sh 가 자기 로그를 새로 쓰므로 회차별 보존
    cp -p "${SCRIPT_DIR}/run_sweep.log" "${SCRIPT_DIR}/logs_keep/run_sweep_rep${i}.log"
done

echo ""
echo "=== 반복 요약 $(date) ==="
for TAG in matched matched_rep2 matched_rep3; do
    for ARM in orig ours; do
        D="${SCRIPT_DIR}/output/sweep_${TAG}_${ARM}_4"
        if [ "$ARM" = "orig" ]; then
            P=$(ls ${D}/point_cloud/iteration_10000/point_cloud.ply 2>/dev/null)
        else
            P=$(ls ${D}/ply/*_completed_*.ply 2>/dev/null | head -1)
        fi
        [ -n "${P:-}" ] && echo "  ${TAG}/${ARM}: $(grep -aoE 'element vertex [0-9]+' "$P" | head -1)"
    done
done
echo "=== 반복 실험 완료 ==="
