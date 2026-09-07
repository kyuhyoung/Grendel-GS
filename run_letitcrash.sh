#!/bin/bash
#
# Let It Crash 실전 런 (MLflow 연동판). run_ours.sh 와 동일 설정에
# 종료 시 scripts/mlflow_report.py 자동 호출을 더한 것.
#   vanilla densify (interval 100, thr 0.0002), 30k iter, epoch-cap 0,
#   median 분할 + densify-mask, 인프라 재시도 최대 10회.
#
# Usage: nohup bash run_letitcrash.sh > /dev/null 2>&1 &   (로그: run_letitcrash.log)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_letitcrash.log"
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

DEVICES="${DEVICES:-5,6}"        # 2 GPU
OUT="./output/lic_2gpu"
docker_name=ogs:v2-44a2c86       # 고정 태그 — fused-ssim 포함 (a7c48d6)
dir_cur=/workspace/ada_grendel
dir_data=/media2/data/dataset_stereo/non-sat
MAX_ATTEMPTS=10
DATASET_MD5=695bb32c3392c57a9052fcf69e00c554   # dvc: samsung-dong-aerial-30.tar

T0=$(date +%s)
echo "============================================"
echo "=== LET IT CRASH RUN: 2 GPU (물리 ${DEVICES}) ==="
echo "  output : ${OUT}"
echo "  image  : ${docker_name}"
echo "  설정   : vanilla densify, 30k iter, epoch-cap 0, median split + densify-mask"
echo "  시작   : $(date)"
echo "============================================"

attempt=0
FRESH="--fresh"
while [ ${attempt} -lt ${MAX_ATTEMPTS} ]; do
    attempt=$((attempt+1))
    echo ""
    echo "[lic] attempt ${attempt}/${MAX_ATTEMPTS} ($(date)) ${FRESH:+(fresh)}"

    INNER="source activate citygs-x && cd ${dir_cur} && \
      FUSED_SSIM=1 DENSIFICATION_INTERVAL=100 DENSIFY_GRAD_THRESHOLD=0.0002 CHILD_DENSIFY_GRAD_THRESHOLD=0.0002 \
      bash run_adaptive.sh --output ${OUT} --gpu-ids 0,1 \
        --iterations 30000 --epoch-cap 0 ${FRESH}"

    docker run --rm -i --shm-size=64g --gpus "\"device=${DEVICES}\"" \
        --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -w "${dir_cur}" \
        -v "${dir_data}":/data \
        -v "${SCRIPT_DIR}":"${dir_cur}" \
        -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
        "${docker_name}" bash -c "${INNER}"
    rc=$?

    if [ ${rc} -eq 0 ]; then
        echo "[lic] 완주 (attempt ${attempt}) $(date)"
        break
    fi
    echo "[lic] 비정상 종료 rc=${rc} — 60초 후 새 컨테이너로 resume"
    FRESH=""            # 재시도부터는 이어서 (state json resume)
    sleep 60
done

T1=$(date +%s)
HOURS=$(awk "BEGIN{printf \"%.3f\", (${T1}-${T0})/3600}")
echo ""
echo "============================================"
echo "=== LIC 종료 $(date) / exit: ${rc} / attempts: ${attempt} / ${HOURS}h ==="
echo "============================================"

# 완주했으면 MLflow 에 자동 리포트 (호스트 python, 화질 지표는 eval 후 별도 추가)
if [ ${rc} -eq 0 ]; then
    echo "[lic] MLflow 리포트..."
    python3 "${SCRIPT_DIR}/scripts/mlflow_report.py" \
        --run-dir "${OUT#./}" --scene samsung-dong-aerial-30 \
        --gpus 2 --wallclock-hours "${HOURS}" \
        --dataset-md5 "${DATASET_MD5}" \
        --run-name "lic_2gpu_$(date +%m%d)" \
        --note "live run (launcher-integrated report); attempts=${attempt}; eval metrics to follow" \
        2>&1 | tee "${SCRIPT_DIR}/run_letitcrash_mlflow.log"
fi
exit ${rc}
