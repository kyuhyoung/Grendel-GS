#!/bin/bash
#
# B-1: Mill-19 rubble 벤치마크 — ours (Split on Failure), 2 GPU (물리 7,6).
# 씬: 1,657장 4608×3456, SfM 2.1M (Mega-NeRF 포즈 → COLMAP 재삼각측량, V2 컨벤션).
# 설정: 품질 런 철학 그대로 (vanilla densify, 30k, epoch-cap 0) + 인프라 재시도.
#
# Usage: nohup bash run_rubble.sh > /dev/null 2>&1 &   (로그: run_rubble.log)
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_rubble.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

DEVICES="${DEVICES:-7,6}"
OUT="./output/rubble_ours"
docker_name=ogs
dir_cur=/workspace/ada_grendel
MAX_ATTEMPTS=10

echo "=== RUBBLE OURS: 2 GPU (물리 ${DEVICES}) 시작 $(date) ==="
attempt=0; FRESH="--fresh"
while [ ${attempt} -lt ${MAX_ATTEMPTS} ]; do
    attempt=$((attempt+1))
    echo "[rubble] attempt ${attempt}/${MAX_ATTEMPTS} ($(date)) ${FRESH:+(fresh)}"
    INNER="source activate citygs-x && cd ${dir_cur} && \
      DENSIFICATION_INTERVAL=100 DENSIFY_GRAD_THRESHOLD=0.0002 CHILD_DENSIFY_GRAD_THRESHOLD=0.0002 \
      bash run_adaptive.sh --source /b/rubble_scene --output ${OUT} --gpu-ids 0,1 \
        --iterations 30000 --epoch-cap 0 ${FRESH}"
    docker run --rm -i --shm-size=64g --gpus "\"device=${DEVICES}\"" \
        --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -w "${dir_cur}" \
        -v "${SCRIPT_DIR}/benchmarks":/b \
        -v "${SCRIPT_DIR}":"${dir_cur}" \
        -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
        "${docker_name}" bash -c "${INNER}"
    rc=$?
    [ ${rc} -eq 0 ] && { echo "[rubble] 완주 (attempt ${attempt}) $(date)"; break; }
    echo "[rubble] 비정상 종료 rc=${rc} — 60초 후 resume"
    FRESH=""; sleep 60
done
echo "=== RUBBLE OURS 종료 $(date) rc=${rc} ==="
exit ${rc}
