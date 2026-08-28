#!/bin/bash
#
# §4.4 ours 런 (호스트에서 실행). oracle 대비 도전 기록.
# Split on Failure, GPU 2장, 품질 설정 (run_final.sh 와 동일 철학):
#   vanilla densify (interval 100, thr 0.0002), 30k iter, epoch-cap 0,
#   median 분할 + densify-mask (코드 기본값), 인프라 재시도 포함.
#
# 시간이 측정값이므로 머신 단독 점유 상태에서 시작할 것 (oracle 종료 후).
# 2~2.5일 예상. 인프라 사고(exit 77 등)로 컨테이너가 죽으면 새 컨테이너로
# resume 재시도 (최대 10회) — run_forever.sh 철학의 내장판.
#
# Usage: nohup bash run_ours.sh > /dev/null 2>&1 &     (로그: run_ours.log)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_ours.log"
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

DEVICES="${DEVICES:-5,6}"        # 2 GPU
OUT="./output/ours_2gpu"
docker_name=ogs
dir_cur=/workspace/ada_grendel
dir_data=/media2/data/dataset_stereo/non-sat
MAX_ATTEMPTS=10

echo "============================================"
echo "=== OURS RUN: Split on Failure, 2 GPU (물리 ${DEVICES}) ==="
echo "  output : ${OUT}"
echo "  설정   : vanilla densify, 30k iter, epoch-cap 0, median split + densify-mask"
echo "  시작   : $(date)"
echo "============================================"

attempt=0
FRESH="--fresh"
while [ ${attempt} -lt ${MAX_ATTEMPTS} ]; do
    attempt=$((attempt+1))
    echo ""
    echo "[ours] attempt ${attempt}/${MAX_ATTEMPTS} ($(date)) ${FRESH:+(fresh)}"

    INNER="source activate citygs-x && cd ${dir_cur} && \
      DENSIFICATION_INTERVAL=100 DENSIFY_GRAD_THRESHOLD=0.0002 CHILD_DENSIFY_GRAD_THRESHOLD=0.0002 \
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
        echo "[ours] 완주 (attempt ${attempt}) $(date)"
        break
    fi
    echo "[ours] 비정상 종료 rc=${rc} — 60초 후 새 컨테이너로 resume"
    FRESH=""            # 재시도부터는 이어서 (state json resume)
    sleep 60
done

echo ""
echo "============================================"
echo "=== OURS 종료 $(date) / exit: ${rc} / attempts: ${attempt} ==="
echo "============================================"
exit ${rc}
