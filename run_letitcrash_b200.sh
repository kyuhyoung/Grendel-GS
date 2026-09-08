#!/usr/bin/env bash
#
# Let It Crash 실전 런 — B200 venv 판 (docker 없는 컨테이너용). run_letitcrash.sh 의 이식.
#   vanilla densify (interval 100, thr 0.0002), 30k iter, epoch-cap 0,
#   median 분할 + densify-mask, 인프라 재시도 최대 10회, MLflow RUNNING 선등록 + 완주 리포트.
#
# GPU 는 시작 시 자동 판정: used < 500MiB 인 카드만 사용 (타 사용자와 공유 금지 — 이웃 할당이
# 가짜 Cat1 을 만든다). 빈 카드가 6장 미만이면 그 수로 시작, 2장 미만이면 거부.
#
# Usage: nohup bash run_letitcrash_b200.sh > /dev/null 2>&1 &     (로그: run_letitcrash_b200.log)
#        bash run_letitcrash_b200.sh --dry-run                      (GPU 판정·포트·설정만 출력)
#   env: MAX_GPUS(기본 6) MASTER_PORT(기본 29601) OUT(기본 ./output/lic_b200) SRC(데이터 경로)
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_letitcrash_b200.log"
: > "${LOG_FILE}"                                   # 실행마다 새로 생성 (덮어쓰기)
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1
export PYTHONUNBUFFERED=1
cd "${SCRIPT_DIR}"

DRY_RUN=false; [ "${1:-}" = "--dry-run" ] && DRY_RUN=true

VENV="${VENV:-${SCRIPT_DIR}/../venv_ogs_b200}"
SRC="${SRC:-/NHNHOME/WORKSPACE/26molit001_dbo/kevin/data/dabeeo/samsung_dong_mini_30}"
OUT="${OUT:-./output/lic_b200}"
MAX_GPUS="${MAX_GPUS:-6}"
MASTER_PORT="${MASTER_PORT:-29601}"
MAX_ATTEMPTS=10
DATASET_MD5=695bb32c3392c57a9052fcf69e00c554        # dvc: samsung-dong-aerial-30.tar
ENV_TAG="venv_ogs_b200:torch2.10-cu13.1"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://10.5.0.52:5000}"
MLFLOW_EXP="${MLFLOW_EXPERIMENT:-lab2_aer-samsung_ada-3dgs}"
RUN_NAME="lic_b200_$(date +%m%d)"

# ---------- 환경 ----------
source "${VENV}/bin/activate" || { echo "!!! venv 없음: ${VENV} (run_build_b200.sh 먼저)"; exit 1; }
[ -d "${SRC}/sparse" ] || { echo "!!! 데이터 없음: ${SRC} (run_prep_data_b200.sh 먼저)"; exit 1; }
[ -x "${VENV}/bin/torchrun" ] || { echo "!!! venv torchrun 래퍼 없음 — 시스템 torchrun 은 venv 패키지를 못 봄"; exit 1; }

# ---------- GPU 자동 판정: used < 500MiB 만 ----------
mapfile -t FREE_IDS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                         | awk -F', *' '$2 < 500 {print $1}')
echo "GPU 상태:"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
N_FREE=${#FREE_IDS[@]}
if [ "${N_FREE}" -lt 2 ]; then
    echo "!!! 빈 GPU ${N_FREE}장 (<2) — 시작 거부. 타 사용자와 공유하는 카드는 OOM 신호를 오염시키므로 쓰지 않는다."
    exit 2
fi
N_USE=$(( N_FREE < MAX_GPUS ? N_FREE : MAX_GPUS ))
GPU_IDS=$(IFS=,; echo "${FREE_IDS[*]:0:${N_USE}}")
[ "${N_USE}" -lt "${MAX_GPUS}" ] && echo "[주의] 빈 카드 ${N_FREE}장 — 목표 ${MAX_GPUS}장 대신 ${N_USE}장으로 시작"

# ---------- MASTER_PORT: 비어 있는지 확인 후 고정 ----------
if ss -ltn 2>/dev/null | grep -qE ":${MASTER_PORT} "; then
    echo "!!! MASTER_PORT ${MASTER_PORT} 사용 중 — MASTER_PORT=<다른 값> 으로 재실행"; exit 3
fi
export PET_MASTER_PORT="${MASTER_PORT}"          # torchrun --master_port 의 env 경로 (실측 확인)

echo "============================================"
echo "=== LET IT CRASH RUN (B200 venv): ${N_USE} GPU (물리 ${GPU_IDS}) ==="
echo "  output : ${OUT}"
echo "  env    : ${ENV_TAG}  ($(python -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda)'))"
echo "  port   : ${MASTER_PORT}"
echo "  설정   : vanilla densify, 30k iter, epoch-cap 0, median split + densify-mask, fused SSIM"
echo "  disk   : $(df -h "${SCRIPT_DIR}" | awk 'NR==2{print $4" free"}')"
echo "  시작   : $(date)"
echo "============================================"
if ${DRY_RUN}; then echo "(dry-run — 여기서 종료)"; exit 0; fi

# ---------- MLflow RUNNING 선등록 (1회) ----------
mkdir -p "${OUT}"
if [ ! -s "${OUT}/.mlflow_run_id" ]; then
    MLFLOW_DISABLE_AGENT_HINT=1 python - "${MLFLOW_URI}" "${MLFLOW_EXP}" "${RUN_NAME}" "${OUT}" "${N_USE}" <<'PY' \
        || echo "[mlflow] 선등록 실패 — 리포터가 완주 시 새 런으로 기록 (학습은 계속)"
import sys, mlflow
uri, exp_name, run_name, out, n = sys.argv[1:6]
mlflow.set_tracking_uri(uri)
c = mlflow.MlflowClient()
exp = c.get_experiment_by_name(exp_name) or c.get_experiment(c.create_experiment(exp_name))
r = c.create_run(exp.experiment_id, run_name=run_name)
for k, v in {"team":"lab2","task":"ada-3dgs","job_type":"reconstruction","owner":"kevin",
             "stage":"dev","live_status":f"training on B200 ({n}x183GB)"}.items():
    c.set_tag(r.info.run_id, k, v)
open(f"{out}/.mlflow_run_id","w").write(r.info.run_id)
print(f"[mlflow] RUNNING 런 선등록: {run_name} ({r.info.run_id})")
PY
else
    echo "[mlflow] 기존 run_id 유지: $(cat "${OUT}/.mlflow_run_id")"
fi

# ---------- 학습 (재시도 루프) ----------
T0=$(date +%s)
attempt=0
FRESH="--fresh"
rc=1
while [ ${attempt} -lt ${MAX_ATTEMPTS} ]; do
    attempt=$((attempt+1))
    echo ""
    echo "[lic] attempt ${attempt}/${MAX_ATTEMPTS} ($(date)) ${FRESH:+(fresh)}"

    FUSED_SSIM=1 DENSIFICATION_INTERVAL=100 DENSIFY_GRAD_THRESHOLD=0.0002 CHILD_DENSIFY_GRAD_THRESHOLD=0.0002 \
    bash "${SCRIPT_DIR}/run_adaptive.sh" --source "${SRC}" --output "${OUT}" --gpu-ids "${GPU_IDS}" \
        --iterations 30000 --epoch-cap 0 ${FRESH}
    rc=$?

    if [ ${rc} -eq 0 ]; then
        echo "[lic] 완주 (attempt ${attempt}) $(date)"
        break
    fi
    echo "[lic] 비정상 종료 rc=${rc} — 60초 후 resume"
    FRESH=""            # 재시도부터는 이어서 (state json resume)
    sleep 60
done

T1=$(date +%s)
HOURS=$(awk "BEGIN{printf \"%.3f\", (${T1}-${T0})/3600}")
echo ""
echo "============================================"
echo "=== LIC 종료 $(date) / exit: ${rc} / attempts: ${attempt} / ${HOURS}h / ${N_USE} GPU ==="
echo "============================================"

# ---------- 완주 시 MLflow 리포트 (.mlflow_run_id 있으면 그 런을 FINISHED 로 닫음) ----------
if [ ${rc} -eq 0 ]; then
    echo "[lic] MLflow 리포트..."
    MLFLOW_DISABLE_AGENT_HINT=1 python "${SCRIPT_DIR}/scripts/mlflow_report.py" \
        --run-dir "${OUT#./}" --scene samsung-dong-aerial-30 \
        --gpus "${N_USE}" --wallclock-hours "${HOURS}" \
        --dataset-md5 "${DATASET_MD5}" \
        --run-name "${RUN_NAME}" \
        --docker-tag "${ENV_TAG}" \
        --note "B200 real run; fused; attempts=${attempt}; gpus=${GPU_IDS}; eval metrics to follow" \
        2>&1 | tee "${SCRIPT_DIR}/run_letitcrash_b200_mlflow.log"
fi
exit ${rc}
