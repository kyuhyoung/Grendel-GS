#!/bin/bash
#
# 3 GPU (홀수 world_size) 스모크 테스트
#
# 배경: 파이프라인은 2 GPU 로만 실전 검증됨. 코드를 읽어본 결과 홀짝 의존은
#   없어 보이지만(짝수 강제 assert 는 local_sampling=False 로 비활성, 2/4/8
#   하드코딩 테이블은 도달 불가 경로), 실제로 돌려본 적이 없어 확인한다.
#
# 확인 대상 (2 rank 에서만 검증된 것들):
#   1. init_distributed / IN_NODE_GROUP 구성이 WORLD_SIZE=3 에서 정상인가
#   2. DivisionStrategyFinal 이 이미지를 3 밴드로 나누는가 (division_pos 4개)
#      → 2026-08-07 1차 실행에서 통과 확인 (5760 / 5776 / 5774 밴드)
#   3. Cat3 OOM 신호 ACK 가 3 rank 에서 데드락 없이 도는가
#   4. rank0/1/2 PLY 3개 저장 → merge_ply_files(num_ranks=3) 병합
#   5. 병합 PLY 로 자식 타일 resume 이 되는가
#
# OOM 을 빨리 유발해야 3~5 를 볼 수 있으므로 --explosive-densification 사용
# (densify_from 100 / interval 50 / threshold 0.00001).
#
# 실행 위치는 자동 판별한다:
#   - 호스트에서 실행  → docker 를 GPU 3개로 띄우고 그 안에서 학습
#   - 컨테이너에서 실행 → run_adaptive.sh 를 바로 실행 (컨테이너에 GPU 3개 필요)
#
# Usage:  bash run_smoke3.sh
#
set -uo pipefail    # -e 제외: 진단용 명령 하나가 실패해도 죽지 않게

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 분할 방식 ──
# MEDIAN_SPLIT=1 (기본): Cat3 를 가우시안 median 으로 분할
# MEDIAN_SPLIT=0        : 기존 기하 중점 분할 (A/B 비교용 기준선)
# 출력·로그를 방식별로 분리한다. 2026-08-07 중점 분할 기준선이
# output/smoke_3gpu + run_smoke3.log 에 남아 있으므로 그건 건드리지 않는다.
export MEDIAN_SPLIT="${MEDIAN_SPLIT:-1}"
export MEDIAN_SPLIT_MAX_RATIO="${MEDIAN_SPLIT_MAX_RATIO:-0.65}"
if [[ "${MEDIAN_SPLIT}" == "0" ]]; then
    MODE="midpoint"
else
    MODE="median"
fi

OUT="${OUT:-./output/smoke_3gpu_${MODE}}"
LOG_FILE="${SCRIPT_DIR}/run_smoke3_${MODE}.log"
rm -f "${LOG_FILE}"
: > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

die() { echo ""; echo "[중단] $*"; exit 1; }

# ── 설정 ──
DEVICES="${DEVICES:-5,6,7}"   # 물리 GPU (GPU4 는 타 계정 사용 중이라 기본값에서 회피)
ITERS=3000
docker_name=ogs
dir_cur=/workspace/${PWD##*/}
dir_data=/media2/data/dataset_stereo/non-sat

# ── 최종 런 로그 보호 ──
# run_adaptive.sh 는 매 실행마다 run_adaptive.log 를 rm 한다. 그 파일이 7/27
# 최종 66M 런의 유일한 기록이므로 먼저 이름을 바꿔 보존한다.
if [[ -f "${SCRIPT_DIR}/run_adaptive.log" && ! -f "${SCRIPT_DIR}/run_adaptive_final_run.log" ]]; then
    cp -a "${SCRIPT_DIR}/run_adaptive.log" "${SCRIPT_DIR}/run_adaptive_final_run.log"
    echo "[backup] run_adaptive.log -> run_adaptive_final_run.log"
fi

# ── 학습 명령 (컨테이너 안에서 실행될 내용) ──
TRAIN_CMD="bash run_adaptive.sh \
  --output ${OUT} \
  --gpu-ids 0,1,2 \
  --iterations ${ITERS} \
  --epoch-cap 0 \
  --explosive-densification \
  --fresh"

echo "============================================"
echo "=== 3 GPU SMOKE TEST ==="
echo "  분할 방식     : ${MODE}  (MEDIAN_SPLIT=${MEDIAN_SPLIT}, clamp=${MEDIAN_SPLIT_MAX_RATIO})"
echo "  world_size    : 3"
echo "  output        : ${OUT}"
echo "  log           : ${LOG_FILE}"
echo "  iterations    : ${ITERS}"
echo "  densification : explosive (OOM 조기 유발용)"
echo "  시작          : $(date)"

# ============================================
# 컨테이너 안에서 실행된 경우
# ============================================
if [[ -f /.dockerenv ]]; then
    echo "  실행 위치     : 컨테이너 안 (docker 단계 건너뜀)"
    echo "============================================"
    echo ""

    # 이 컨테이너에 GPU 가 3개 이상 보이는지 확인.
    # 오래 떠 있는 컨테이너는 nvidia-smi(NVML)가 깨져 있는 경우가 있어 torch 로 센다.
    NGPU="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
    echo "[GPU] 이 컨테이너에서 보이는 GPU: ${NGPU}개"
    if [[ "${NGPU}" -lt 3 ]]; then
        die "GPU 가 ${NGPU}개뿐이라 world_size=3 테스트 불가.
       이 컨테이너는 --gpus '\"device=6,7\"' 로 떠 있는 것으로 보입니다.
       호스트 셸에서 아래처럼 GPU 3개짜리로 새로 띄우거나,
           DEVICES=5,6,7 bash run_smoke3.sh
       호스트에서 이 스크립트를 그대로 실행하십시오 (docker 를 알아서 띄웁니다)."
    fi

    cd "${SCRIPT_DIR}"
    export FUSED_SSIM=1
    eval "${TRAIN_CMD}"
    rc=$?

# ============================================
# 호스트에서 실행된 경우
# ============================================
else
    echo "  실행 위치     : 호스트 (docker 를 띄움)"
    echo "  물리 GPU      : ${DEVICES}  → 컨테이너 내부 0,1,2"
    echo "============================================"
    echo ""

    command -v docker >/dev/null || die "docker 를 찾을 수 없습니다."
    echo "[GPU 상태]"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
    echo ""

    # conda env 활성화 필수: 비대화형 bash 는 ~/.bashrc 를 읽지 않아 torch 를 못 찾는다
    # (대화형 셸에서는 .bashrc 가 자동으로 처리하던 부분)
    INNER_CMD="source activate citygs-x && export FUSED_SSIM=1 && ${TRAIN_CMD}"

    docker run --rm -i --shm-size=64g --gpus "\"device=${DEVICES}\"" \
        --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -e MEDIAN_SPLIT="${MEDIAN_SPLIT}" \
        -e MEDIAN_SPLIT_MAX_RATIO="${MEDIAN_SPLIT_MAX_RATIO}" \
        -w "${dir_cur}" \
        -v "${dir_data}":/data \
        -v "$PWD":"${dir_cur}" \
        -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
        "${docker_name}" bash -c "cd ${dir_cur} && ${INNER_CMD}"
    rc=$?
fi

echo ""
echo "============================================"
echo "=== 종료 $(date) / exit code: ${rc} ==="
echo "============================================"
exit ${rc}
