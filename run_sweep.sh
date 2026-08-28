#!/bin/bash
#
# §4.5 계단 스윕 (호스트에서 실행) — "시간 손해가 크지 않다"의 직접 증거.
#
# 같은 2 GPU 에서 씬 규모를 계단식으로 올리며 original(단일 타일)과
# ours(Split on Failure)를 나란히 돌린다:
#   · original 이 완주하는 구간 → 직접 시간 비교 = 오버헤드 X%
#   · original 이 OOM 으로 죽는 구간 → 결과 유무 자체가 결론
# 계단은 이미지 해상도로 만든다 (1/4 → 1/2 → 1/1).
#
# 시간이 측정값이므로 머신 단독 점유 상태에서 실행할 것.
# Usage: nohup bash run_sweep.sh > /dev/null 2>&1 &   (로그: run_sweep.log)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_sweep.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

DEVICES="${DEVICES:-5,6}"          # 2 GPU 고정 (실험의 전제)
# 2026-08-17: 스윕의 목적은 "original 이 어디서 죽고, 죽기 전까지 우리가 몇 % 느린가"
# — 최종 품질이 아니라 절벽 위치와 오버헤드 비율이다. 30k 는 과하다.
# 10k 로 잡으면 계단당 시간이 1/3 이고 두 방식에 동일 적용이라 비율은 보존된다.
ITERS="${ITERS:-10000}"

# 2026-08-18: 1차 스윕(8/17)의 시간 비교는 무효였다. run_adaptive 는 densify/opacity
# 스케줄을 학습량에 비례 축소(aligned_schedule)하는데 vanilla train.py 는 30k 기준
# 상수를 그대로 써서, ITERS=10000 에서 두 팔이 이렇게 갈렸다:
#     original: densify 500~15000(=전 구간)  → 1/4 계단 16.3M 가우시안
#     ours    : densify 167~5000(=절반)      → 1/4 계단  6.8M 가우시안
# 우리가 25% 빨랐던 건 오버헤드가 아니라 모델을 절반 이하로 만들었기 때문이다.
# → original 에도 같은 스케줄을 명시해 준다. 계산식은 train_adaptive.aligned_schedule
#   과 동일(frac=ITERS/30000 을 500/15000/3000 에 곱함).
read -r SD_FROM SD_UNTIL OP_RESET <<< "$(python3 -c "
frac = ${ITERS} / 30000.0
until_ = int(round(15000 * frac))
from_ = max(int(round(500 * frac)), 100)
from_ = min(from_, max(until_ - 100, 1))
reset = max(int(round(3000 * frac)), 500)
print(from_, until_, reset)
")"
# 저장 횟수도 맞춘다 — ours 는 끝에 한 번 저장하는데 original 기본값은 7000 에도
# 저장한다(1/4 계단에서 4GB PLY). 벽시계 비교라 이것도 차이가 된다.
echo "[스케줄 정렬] ITERS=${ITERS} → densify ${SD_FROM}~${SD_UNTIL}, opacity_reset ${OP_RESET}, 저장 ${ITERS} 1회"

STEPS="${STEPS:-4 2}"          # 1/1 은 1차 스윕 결과가 유효(original 이 iteration 1 도
                               # 못 돌고 죽음 → densify 스케줄과 무관). 필요하면 STEPS="4 2 1"
TAG="${TAG:-matched}"          # 1차 스윕 결과(output/sweep_*)를 덮지 않도록 분리
docker_name=ogs
dir_cur=/workspace/ada_grendel
dir_data=/media2/data/dataset_stereo/non-sat

run_in_docker () {   # $1=설명, $2=inner command
    echo ""
    echo "=== $1 시작 $(date) ==="
    docker run --rm -i --shm-size=64g --gpus "\"device=${DEVICES}\"" \
        --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -w "${dir_cur}" -v "${dir_data}":/data -v "${SCRIPT_DIR}":"${dir_cur}" \
        -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
        "${docker_name}" bash -c "source activate citygs-x && $2"
    echo "=== $1 종료 rc=$? $(date) ==="
}

for STEP in ${STEPS}; do
    if [ "${STEP}" = "1" ]; then
        SRC="/data/dabeeo/samsung_dong_mini_30"
    else
        SRC="${dir_cur}/data_scale_${STEP}"
    fi

    # ── original: 단일 타일, 타일링 인자 없음 (순정 Grendel 경로) ──
    run_in_docker "1/${STEP} ORIGINAL (2 GPU, single-tile)" \
      "cd ${dir_cur} && torchrun --nproc_per_node=2 Grendel-GS/train.py \
         --source_path ${SRC} --model_path ./output/sweep_${TAG}_orig_${STEP} \
         --iterations ${ITERS} --bsz 1 --backend default --test_iterations 999999999 \
         --save_iterations ${ITERS} \
         --densify_from_iter ${SD_FROM} --densify_until_iter ${SD_UNTIL} \
         --opacity_reset_interval ${OP_RESET} --position_lr_max_steps ${ITERS}"

    # ── ours: Split on Failure ──
    run_in_docker "1/${STEP} OURS (2 GPU, split-on-failure)" \
      "cd ${dir_cur} && DENSIFICATION_INTERVAL=100 DENSIFY_GRAD_THRESHOLD=0.0002 \
         CHILD_DENSIFY_GRAD_THRESHOLD=0.0002 \
         bash run_adaptive.sh --source ${SRC} --output ./output/sweep_${TAG}_ours_${STEP} \
           --gpu-ids 0,1 --iterations ${ITERS} --epoch-cap 0 --fresh"
done

echo ""
echo "=== SWEEP 완료 $(date) ==="
