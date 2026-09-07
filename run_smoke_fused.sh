#!/bin/bash
# fused SSIM A/B 스모크: 같은 설정 2000 iter, ON(GPU 0,1) / OFF(GPU 2,3) 동시 실행
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_smoke_fused.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

dir_cur=/workspace/ada_grendel
dir_data=/media2/data/dataset_stereo/non-sat
IMG=ogs:v2-44a2c86

run_arm() {  # $1=arm $2=devices $3=fused(0/1)
  local OUT="./output/smoke_fused_$1"
  docker run --rm -i --shm-size=64g --gpus "\"device=$2\"" \
    --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -w "${dir_cur}" -v "${dir_data}":/data -v "${SCRIPT_DIR}":"${dir_cur}" \
    -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
    -e FUSED_SSIM=$3 "${IMG}" bash -c "
      source activate citygs-x && cd ${dir_cur} &&
      DENSIFICATION_INTERVAL=100 DENSIFY_GRAD_THRESHOLD=0.0002 CHILD_DENSIFY_GRAD_THRESHOLD=0.0002 \
      bash run_adaptive.sh --output ${OUT} --gpu-ids 0,1 --iterations 2000 --epoch-cap 0 --fresh" \
    > "${SCRIPT_DIR}/run_smoke_fused_$1.log" 2>&1
  echo "[$1] rc=$?"
}

echo "=== fused A/B 스모크 시작 $(date) ==="
run_arm on  "0,1" 1 &
P_ON=$!
run_arm off "2,3" 0 &
P_OFF=$!
wait $P_ON $P_OFF
echo "=== 종료 $(date) ==="
