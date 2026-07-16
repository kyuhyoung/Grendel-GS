#!/bin/bash
#
# Merge completed tiles' final PLYs into one scene PLY (no training).
# 학습 재실행 없이 adaptive_state.json + models/ 의 완료 타일 PLY 만으로
# merged/scene_point_cloud.ply 를 (재)생성.
# plyfile 이 필요하므로 도커 컨테이너 안에서 실행 권장.
#
# Usage: ./merge_scene.sh [OUTPUT_PATH] [FILTER_MODE]
#   OUTPUT_PATH: 학습 output 경로 (default: ./output/adaptive_test)
#   FILTER_MODE: bbox(경계 중복 제거) | none(단순 concat) (default: bbox)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/merge_scene.log"

# 매 실행마다 로그 새로 생성 (append 아님)
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "============================================"
echo "=== merge_scene.sh started at $(date) ==="
echo "============================================"

OUTPUT_PATH="${1:-./output/adaptive_test}"
FILTER_MODE="${2:-bbox}"

echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "FILTER_MODE: ${FILTER_MODE}"
echo ""

cd "${SCRIPT_DIR}"

# python -u 로 라인별 flush 보장
python -u - <<PYEOF
import sys
sys.path.insert(0, "scripts")
from train_adaptive import merge_final_scene

result = merge_final_scene("${OUTPUT_PATH}", filter_mode="${FILTER_MODE}")
if result is None:
    print("[merge_scene.sh] FAIL: merge did not produce output", flush=True)
    sys.exit(1)
print(f"[merge_scene.sh] OK -> {result}", flush=True)
PYEOF

exit_code=$?
echo ""
echo "============================================"
echo "=== Finished at $(date) ==="
echo "=== Exit code: ${exit_code} ==="
echo "============================================"
exit ${exit_code}
