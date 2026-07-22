#!/bin/bash
#
# Densification 프로브 (컨테이너 안에서 실행).
# "가우시안 예산을 늘리면 loss 가 얼마나 내려가나"를 측정하기 위한 실험용 런.
# run_adaptive.sh 를 aggressive densification + 별도 output(./output/probe) 으로 호출.
# 스케줄 정렬(기본 ON)은 그대로 유지 — 정련 단계 + 더 촘촘한 densify 조합 측정.
#
# Usage:  bash run_probe.sh
# 정상 run 과 비교: lronly 의 30뷰 타일 tile_0018=0.0765 / tile_0092=0.0718
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── aggressive densification (환경변수로 run_adaptive.sh 기본값 덮어쓰기) ──
export DENSIFICATION_INTERVAL=50          # 100 → 50 (2배 촘촘)
export DENSIFY_GRAD_THRESHOLD=0.00013     # 0.0002 → 0.00013 (더 잘 쪼갬)
export CHILD_DENSIFY_GRAD_THRESHOLD=0.00013

echo "============================================"
echo "=== DENSIFY PROBE ==="
echo "  densification_interval = ${DENSIFICATION_INTERVAL} (기본 100)"
echo "  densify_grad_threshold = ${DENSIFY_GRAD_THRESHOLD} (기본 0.0002)"
echo "  output = ./output/probe"
echo "  (schedule-align 기본 ON, epoch-cap 300 유지)"
echo "============================================"

cd "${SCRIPT_DIR}"
exec bash run_adaptive.sh --output ./output/probe --fresh
