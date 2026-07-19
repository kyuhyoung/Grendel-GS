#!/bin/bash
#
# 무인 감독 실행기 (호스트에서 실행).
# run_adaptive.sh 를 매번 "새 컨테이너"로 실행하고, 비정상 종료(크래시,
# 컨테이너 NVML 상실, 호스트 문제 등)면 잠시 후 자동 재실행한다.
# 재실행 시 wrapper 의 resume 이 완료 타일을 건너뛰고 이어서 진행한다.
# wrapper 가 정상 종료(exit 0)하면 결과(ALL/PARTIAL)를 보고하고 멈춘다.
#
# Usage: ./run_forever.sh [run_adaptive.sh 인자들...]
#   예) ./run_forever.sh            # 기존 state 있으면 이어서
#       ./run_forever.sh --fresh    # 처음부터 (재시도부터는 자동으로 resume)
#
# 중지: Ctrl-C (컨테이너까지 정리됨)
#

set -uo pipefail  # -e 는 쓰지 않음: 자식 실패를 직접 처리해야 하므로

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_forever.log"

# 매 실행마다 로그 새로 생성 (append 아님)
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
exec > >(tee "${LOG_FILE}") 2>&1

IMAGE="ogs"
CONTAINER_NAME="ada_grendel_train"
DATA_DIR="/media2/data/dataset_stereo/non-sat"
MAX_ATTEMPTS=50
COOLDOWN_SEC=60

echo "============================================"
echo "=== run_forever.sh started at $(date) ==="
echo "=== args: $* ==="
echo "============================================"

cleanup() {
    echo ""
    echo "[run_forever] 중지 요청 — 컨테이너 정리"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
    exit 130
}
trap cleanup INT TERM

args=("$@")
attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
    echo ""
    echo "############################################"
    echo "### Attempt ${attempt}/${MAX_ATTEMPTS} at $(date)"
    echo "############################################"

    # 같은 이름의 잔여 컨테이너 정리 (동시 이중 실행 방지 겸용)
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

    docker run --rm --name "${CONTAINER_NAME}" --shm-size=64g \
        --gpus '"device=6,7"' \
        -v "${SCRIPT_DIR}:/workspace/ada_grendel" \
        -v "${DATA_DIR}:/data" \
        -w /workspace/ada_grendel \
        "${IMAGE}" conda run --no-capture-output -n citygs-x \
        bash run_adaptive.sh "${args[@]}"
    rc=$?

    if [[ ${rc} -eq 0 ]]; then
        # wrapper 가 자연 종료함 — 결과 판정 (재시도 대상 아님)
        if grep -q 'ALL TILES COMPLETED' "${SCRIPT_DIR}/run_adaptive.log" 2>/dev/null; then
            echo ""
            echo "[run_forever] ✅ 전 타일 완료 — 종료 (attempt ${attempt})"
            exit 0
        else
            echo ""
            echo "[run_forever] ⚠ wrapper 정상 종료했으나 일부 타일 미완료 (PARTIAL)."
            echo "  재시도해도 결과가 같으므로 멈춥니다. run_adaptive.log 확인 필요."
            exit 1
        fi
    fi

    echo ""
    echo "[run_forever] 비정상 종료 (exit=${rc}) — ${COOLDOWN_SEC}s 후 새 컨테이너로 재시도"

    # 첫 시도에만 --fresh 허용: 재시도가 처음부터 다시 돌지 않도록 제거
    new_args=()
    for a in "${args[@]}"; do
        [[ "$a" == "--fresh" ]] || new_args+=("$a")
    done
    args=("${new_args[@]}")

    sleep "${COOLDOWN_SEC}"
    (( attempt++ ))
done

echo ""
echo "[run_forever] ❌ ${MAX_ATTEMPTS}회 재시도 소진 — 반복 크래시. 로그 확인 필요."
exit 1
