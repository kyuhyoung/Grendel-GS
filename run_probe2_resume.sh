#!/bin/bash
#
# probe600 이어하기 (컨테이너 안에서 실행).
#
# 배경: 7/23 16:44 시작한 run_probe2.sh 런을 7/24 10:29 에 중단하고 여기서 이어받는다.
#   중단 시점 진도: 완료 21 / split 25 / failed 2 / pending 3 (씬 면적의 24% 완료, 62.5% 미착수)
#   이어받는 이유 = (1) fused SSIM 을 남은 69% 구간에 적용, (2) failed 2개를 수정된
#   visibility check 코드로 재시도. 중단 직전 로그는 run_adaptive_probe600_part1.log 에 백업됨.
#
# resume 동작 (--fresh 없음 = 기본 resume):
#   완료 21개  → 최종 PLY 확인 후 그대로 유지 (재학습 안 함)
#   failed 2개 → retry_count<2 이므로 pending 으로 강등해 재시도
#   pending 3개 → 학습 (tile_0002 는 씬의 50%)
#
# densify 설정은 1차 구간과 동일하게 유지해야 타일 분할 기준이 일관된다.
#
# Usage:  bash run_probe2_resume.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── run_probe2.sh 와 동일한 densification 설정 (변경 금지) ──
export DENSIFICATION_INTERVAL=25
export DENSIFY_GRAD_THRESHOLD=0.0001
export CHILD_DENSIFY_GRAD_THRESHOLD=0.0001

# ── 이번 구간부터 적용되는 것 ──
export FUSED_SSIM=1          # 융합 SSIM 커널 (수치 동일, 미설치/오류 시 자동 폴백)

STATE="${SCRIPT_DIR}/output/probe600/adaptive_state.json"
if [ ! -f "${STATE}" ]; then
    echo "ERROR: ${STATE} 없음 — 이어받을 상태가 없습니다. run_probe2.sh 로 새로 시작하세요."
    exit 1
fi

echo "============================================"
echo "=== DENSIFY PROBE 2 — RESUME ==="
echo "  state = ${STATE}"
echo "  densification_interval = ${DENSIFICATION_INTERVAL}"
echo "  densify_grad_threshold = ${DENSIFY_GRAD_THRESHOLD}"
echo "  epoch-cap = 600, output = ./output/probe600"
echo "  FUSED_SSIM = ${FUSED_SSIM} (이번 구간부터 적용)"
echo "  (--fresh 없음 = 완료 타일 유지하고 이어서 진행)"
echo "============================================"

cd "${SCRIPT_DIR}"
exec bash run_adaptive.sh --output ./output/probe600 --epoch-cap 600
