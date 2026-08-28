#!/bin/bash
#
# §4.3/§4.4 최종 재측정 스위트 (호스트에서 실행). 최적화 전부 반영된 최종
# 코드(4751d17)로 3개 런을 순차 실행 — 시간이 측정값이므로 절대 겹치지 않음.
#   1) §4.3 면적 팔:   MEDIAN_SPLIT=0, 3 GPU  (~8h)
#   2) §4.3 median 팔: 기본,           3 GPU  (~6h)
#   3) §4.4 ours:      run_ours.sh,    2 GPU  (~2.5일)
# oracle 은 기존 기록 재사용 (우리 코드 오버헤드와 무관).
#
# Usage: nohup bash run_suite.sh > /dev/null 2>&1 &   (로그: run_suite.log)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_suite.log"
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

echo "=== SUITE 시작 $(date)  (코드: $(git -C ${SCRIPT_DIR} rev-parse --short HEAD)) ==="

echo ""
echo "[1/3] §4.3 면적 팔 (3 GPU) 시작 $(date)"
MEDIAN_SPLIT=0 OUT=./output/final_midpoint bash "${SCRIPT_DIR}/run_smoke3.sh"
echo "[1/3] 종료 rc=$? $(date)"

echo ""
echo "[2/3] §4.3 median 팔 (3 GPU) 시작 $(date)"
OUT=./output/final_median bash "${SCRIPT_DIR}/run_smoke3.sh"
echo "[2/3] 종료 rc=$? $(date)"

echo ""
echo "[3/3] §4.4 ours (2 GPU 품질) 시작 $(date)"
bash "${SCRIPT_DIR}/run_ours.sh"
echo "[3/3] 종료 rc=$? $(date)"

echo ""
echo "=== SUITE 완료 $(date) ==="
