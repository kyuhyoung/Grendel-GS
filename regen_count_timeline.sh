#!/bin/bash
#
# Regenerate count timeline PNG from existing jsonl files (no training).
# 학습 다시 돌리지 않고 jsonl 만 가지고 PNG 만 새로 그릴 때 사용.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/regen_count_timeline.log"

# 매 실행마다 로그 새로 생성 (append 아님)
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "============================================"
echo "=== regen_count_timeline.sh started at $(date) ==="
echo "============================================"

OUTPUT_PATH="${1:-./output/adaptive_test}"
TILE_ID="${2:-tile_0000}"

echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "TILE_ID:     ${TILE_ID}"
echo ""

cd "${SCRIPT_DIR}"

# python -u 로 라인별 flush 보장
python -u - <<PYEOF
import sys
sys.path.insert(0, "Grendel-GS")
from pathlib import Path
from utils.oom_viz import save_count_timeline_viz

output_path = Path("${OUTPUT_PATH}")
tile_id = "${TILE_ID}"
d = output_path / "visualizations" / "count_timeline"
print(f"[regen] count_dir={d}", flush=True)

files = sorted(d.glob(f"{tile_id}_rank*.jsonl"))
print(f"[regen] found {len(files)} jsonl: {[f.name for f in files]}", flush=True)
if not files:
    print(f"[regen] FAIL: no jsonl found in {d}", flush=True)
    sys.exit(1)

out = d / f"{tile_id}_timeline.png"
result = save_count_timeline_viz(
    [str(f) for f in files],
    tile_id=tile_id,
    out_path=out,
)
print(f"[regen] result={result}", flush=True)
print(f"[regen] exists={out.exists()}", flush=True)
if not out.exists():
    print(f"[regen] FAIL: PNG not created", flush=True)
    sys.exit(1)
print(f"[regen] OK -> {out}", flush=True)
PYEOF

exit_code=$?
echo ""
echo "============================================"
echo "=== Finished at $(date) ==="
echo "=== Exit code: ${exit_code} ==="
echo "============================================"
exit ${exit_code}
