#!/bin/bash
#
# 표적 프로브: tile_0092 단독, epoch-cap 600 (컨테이너 안에서 실행).
# 질문: "converged 로 멈춘 tile_0092(floor 0.0654)에 런웨이/정련을 더 주면 floor 가 내려가나?"
#
# 설계 (probe1 과의 비교를 위해 변수 통제):
#   - 시작점 동일: probe1 과 같은 부모 Cat3 저장분(output/probe/ply/tile_0092_...merged.ply)에서 resume
#   - densify 이벤트 수 유사: probe1 = interval 50 × 창 4320 ≈ 86회
#                             이번   = interval 100 × 창 9000 = 90회
#     (interval 50 을 유지하면 창이 2배라 가우시안 2배 → Cat3 분할로 프로브 무산 위험)
#   - epoch-cap 600 → 18000 iters, refinement 9000~18000 (probe1 은 4320~8640)
# 비교 기준: lronly 0.0718 / probe1 0.0654 (converged ep274)
#
# 사전 조건: output/probe0092/adaptive_state.json (tile_0092 만 pending) 이 준비돼 있어야 함.
# Usage:  bash run_probe0092.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DENSIFICATION_INTERVAL=100
export DENSIFY_GRAD_THRESHOLD=0.00013
export CHILD_DENSIFY_GRAD_THRESHOLD=0.00013

if [ ! -f "${SCRIPT_DIR}/output/probe0092/adaptive_state.json" ]; then
    echo "ERROR: output/probe0092/adaptive_state.json 이 없습니다 (tile_0092-only 상태 파일 필요)"
    exit 1
fi

echo "============================================"
echo "=== TARGETED PROBE: tile_0092 / cap 600 ==="
echo "  densification_interval = ${DENSIFICATION_INTERVAL}"
echo "  densify_grad_threshold = ${DENSIFY_GRAD_THRESHOLD}"
echo "  epoch-cap = 600 (18000 iters), resume from probe1 parent ply"
echo "  output = ./output/probe0092"
echo "============================================"

cd "${SCRIPT_DIR}"
# --fresh 없음: adaptive_state.json 을 읽어 tile_0092 만 학습
exec bash run_adaptive.sh --output ./output/probe0092 --epoch-cap 600
