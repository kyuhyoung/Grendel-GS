#!/bin/bash
#
# Densification 프로브 2차 (컨테이너 안에서 실행).
# 1차 프로브 결과: aggressive densify 로 loss 7~9% 개선, 단 15/16 타일이 epoch-cap(300)에 걸려 iter_end 종료.
# 2차 목표: (a) cap 600 으로 converged 비율을 올리고, (b) densify 를 더 밀어붙여 loss floor 자체를 낮춘다.
# 비교 기준:
#   lronly  : tile_0018=0.0765 / tile_0092=0.0718
#   probe 1 : tile_0018=0.0706(iter_end) / tile_0092=0.0654(converged ep274)
#
# Usage:  bash run_probe2.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1차보다 더 aggressive 한 densification ──
export DENSIFICATION_INTERVAL=25          # 1차 50 → 25
export DENSIFY_GRAD_THRESHOLD=0.0001      # 1차 0.00013 → 0.0001
export CHILD_DENSIFY_GRAD_THRESHOLD=0.0001

echo "============================================"
echo "=== DENSIFY PROBE 2 (cap 600 + interval 25) ==="
echo "  densification_interval = ${DENSIFICATION_INTERVAL} (1차 50, 기본 100)"
echo "  densify_grad_threshold = ${DENSIFY_GRAD_THRESHOLD} (1차 0.00013, 기본 0.0002)"
echo "  epoch-cap = 600 (1차 300)"
echo "  output = ./output/probe600"
echo "  (schedule-align ON: densify/reset/LR 이 cap 600 에 맞춰 자동 연장)"
echo "============================================"

cd "${SCRIPT_DIR}"
exec bash run_adaptive.sh --output ./output/probe600 --fresh --epoch-cap 600
