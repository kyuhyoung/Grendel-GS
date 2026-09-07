#!/usr/bin/env bash
# B200 스모크 (docker 없는 컨테이너 안, venv 사용). run_smoke_fused.sh 의 ON arm 을 venv 로 옮긴 것.
# 확인 항목: ① [fused-ssim] enabled 로그 ② output/smoke_b200/adaptive_state.json 의 completed 타일 ③ merged/ 생성
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run_smoke_b200.log"; : > "$LOG"
exec > >(stdbuf -o0 tee "$LOG") 2>&1
export PYTHONUNBUFFERED=1

VENV="${VENV:-$HERE/../venv_ogs_b200}"
SRC="${SRC:-/NHNHOME/WORKSPACE/26molit001_dbo/kevin/data/dabeeo/samsung_dong_mini_30}"
OUT="${OUT:-./output/smoke_b200}"
GPUS="${GPUS:-0,1}"
ITERS="${ITERS:-2000}"
DINT="${DINT:-100}"            # 분할 경로 강제 시험: DINT=25 ITERS=6000 처럼 공격적으로
export FUSED_SSIM=1 DENSIFICATION_INTERVAL=$DINT DENSIFY_GRAD_THRESHOLD=0.0002 CHILD_DENSIFY_GRAD_THRESHOLD=0.0002

source "$VENV/bin/activate" || { echo "!!! venv 없음: $VENV (run_build_b200.sh 먼저)"; exit 1; }
[ -d "$SRC/sparse" ] || { echo "!!! 데이터 없음: $SRC (run_prep_data_b200.sh 먼저)"; exit 1; }
echo "=== B200 스모크 $(date)  GPUS=$GPUS ITERS=$ITERS DINT=$DINT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
T0=$(date +%s)
bash "$HERE/run_adaptive.sh" --source "$SRC" --output "$OUT" --gpu-ids "$GPUS" --iterations "$ITERS" --epoch-cap 0 --fresh
rc=$?
echo "=== rc=$rc  벽시계 $(( $(date +%s) - T0 ))s"
echo "--- ① fused-ssim:"; grep -m3 -rh 'fused-ssim' "$OUT" "$LOG" 2>/dev/null | head -3
echo "--- ② adaptive_state.json:"; python - "$OUT" <<'PY'
import json,sys,os,collections
p=os.path.join(sys.argv[1],'adaptive_state.json')
if not os.path.exists(p): print('없음'); sys.exit()
s=json.load(open(p)); tiles=s.get('tiles',s)
if isinstance(tiles,dict): tiles=list(tiles.values())
c=collections.Counter(t.get('status','?') for t in tiles if isinstance(t,dict))
print('타일 수',len(tiles),dict(c)); print('분할 발생' if c.get('split') else '분할 없음')
PY
echo "--- ③ merged/:"; ls "$OUT/merged" 2>/dev/null | head -3 || echo "없음"
exit $rc
