#!/bin/bash
#
# 최종 씬 생성 런 (컨테이너 안에서 실행). 완전히 새로 시작(--fresh).
#
# 목적: probe1(output/probe, 34.7M 가우시안)보다 선명한 씬을 만든다.
#   방법 = 타일을 더 잘게 쪼개 같은 땅에 가우시안을 더 많이 박는다.
#   probe600 실측 근거: 면적당 밀도 8.57 → 74.84 (약 8.7배), 타일 평균 면적 1/5.
#
# 설정은 probe600 과 동일(검증된 조합). 다른 점은 코드가 전부 고쳐진 상태라는 것:
#   - 검은 타일 게이트 수정 (converged 가 opacity reset 직후에 발화하던 off-by-grace)
#   - robust-oom state fallback (정상 OOM 이 failed 로 오분류돼 타일이 통째로 날아가던 레이스)
#   - crop 카메라 visibility check 좌표계 수정
#   - fused SSIM (수치 동일, SSIM 커널 10배)
#   - 타일 ID %08d (레벨마다 2배로 커져 자릿수가 섞이던 문제)
#
# 예상: 2.5~3일, 최종 3억 개 규모 / 66~80GB.
#
# Usage:  bash run_final.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── densification: vanilla 3DGS 기본값 ──
# 2026-07-24 되돌림. 이전 값(interval 25, threshold 0.0001)은 probe 실험에서 Cat3 OOM 유발+
# loss 인위적 저하를 노린 "explosive 급" 설정이었는데, 그걸 최종 품질 런에 그대로 복사한 실수였다.
# interval 25(기본 100의 4배) + threshold 0.0001(기본의 2배)로 가우시안이 폭증→과분할→소형
# 타일 floater 를 유발했다. Grendel-GS/arguments 기본값(=vanilla 3DGS)으로 복귀.
export DENSIFICATION_INTERVAL=100
export DENSIFY_GRAD_THRESHOLD=0.0002
export CHILD_DENSIFY_GRAD_THRESHOLD=0.0002

# ── fused SSIM (import 실패/오류 시 자동 폴백) ──
export FUSED_SSIM=1

OUT="./output/final"

echo "============================================"
echo "=== FINAL SCENE RUN (fresh) ==="
echo "  densification_interval = ${DENSIFICATION_INTERVAL}"
echo "  densify_grad_threshold = ${DENSIFY_GRAD_THRESHOLD}"
echo "  epoch-cap = 0 (제거 — converge(best+patience) 또는 iterations 30000 이 종료 결정)"
echo "  FUSED_SSIM = ${FUSED_SSIM}"
echo "  output = ${OUT}  (기존 output/probe600, output/probe 는 보존됨)"
echo "  tile id = %08d, vis counter = %03d"
echo "============================================"

# 2026-07-24: epoch-cap 제거. 이유 = iter_end 로 끝난 타일들이 best_epoch 가 cap 끝(599/600)에
# 붙어 있어 "아직 loss 내려가던 중 잘림"(미적합)이 확인됨. 과적합 시작점은 타일당 카메라가 적어
# held-out 검증뷰를 못 빼 측정 불가 → 불확실한 과적합보다 확실한 미적합을 피하기로 결정.
# 안전장치: (1) LR 이 30000 에 정렬돼 후반엔 near-zero 로 자연 제동, (2) --iterations 30000 최후 상한,
# (3) converge(best+patience) 가 대부분 그 전에 종료. epoch_cap=0 → effective_tile_iterations 비활성.
cd "${SCRIPT_DIR}"
exec bash run_adaptive.sh --output "${OUT}" --epoch-cap 0 --fresh
