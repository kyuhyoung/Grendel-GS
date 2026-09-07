#!/bin/bash
#
# §4.4 화질 재측정 (호스트에서 실행). 세 팔을 같은 조건으로 렌더 → L1/PSNR.
#
# 2026-08-18 재작성 이유: 이전 측정은 임시 명령으로 돌려서 재현이 안 됐고,
# eval_ours 는 30장 중 2장만 채점된 채 "완료"로 흘러갔다. 원인은 랭크 4개가
# 각각 71.5M 가우시안 PLY 를 float64 로 펼쳐 랭크당 51GB → 호스트 램 초과 →
# 커널이 랭크 0~2 를 죽임(로그 0바이트). gaussian_model.load_raw_ply 를
# float32 + 원본 즉시 해제로 고쳐 랭크당 ~34GB 로 낮췄다.
#
# Usage: nohup bash run_eval.sh > /dev/null 2>&1 &     (로그: run_eval.log)
#        ARMS="ours" bash run_eval.sh                  (특정 팔만)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_eval.log"
rm -f "${LOG_FILE}"; : > "${LOG_FILE}"
exec > >(stdbuf -oL -eL tee "${LOG_FILE}") 2>&1

DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"  # 4랭크는 호스트 램 초과로 랭크 사망 2회 재발 (8/18, 9/4) — 8랭크가 검증된 구성
NPROC="${NPROC:-8}"
EXPECT="${EXPECT:-30}"            # data_eval_half/images 의 tif 장수
ARMS="${ARMS:-ours fmedian fmidpoint}"
docker_name=ogs
dir_cur=/w
dir_data=/media2/data/dataset_stereo/non-sat

# 팔 이름 -> 채점할 병합 PLY (컨테이너 경로)
ply_of () {
    case "$1" in
        ours)      echo "/w/output/ours_2gpu/merged/scene_point_cloud.ply" ;;
        lic)       echo "/w/output/lic_2gpu/merged/scene_point_cloud.ply" ;;
        fmedian)   echo "/w/output/final_median/merged/scene_point_cloud.ply" ;;
        fmidpoint) echo "/w/output/final_midpoint/merged/scene_point_cloud.ply" ;;
        oracle3e5) echo "/w/output/oracle_8gpu_thr3e-5/point_cloud/iteration_30000/point_cloud.ply" ;;
        *)         echo "" ;;
    esac
}

echo "============================================"
echo "=== EVAL 시작 $(date) (코드: $(git -C "${SCRIPT_DIR}" rev-parse --short HEAD)) ==="
echo "  팔     : ${ARMS}"
echo "  GPU    : ${DEVICES} (rank ${NPROC})"
echo "  기대   : ${EXPECT} 장"
echo "============================================"

RC_ALL=0
for ARM in ${ARMS}; do
    OUT="${SCRIPT_DIR}/output/eval_${ARM}"
    PLY="$(ply_of "${ARM}")"
    if [ -z "${PLY}" ]; then echo "[${ARM}] 알 수 없는 팔 — 건너뜀"; RC_ALL=1; continue; fi

    HOST_PLY="${SCRIPT_DIR}${PLY#/w}"
    if [ ! -e "${HOST_PLY}" ]; then
        echo "[${ARM}] PLY 없음: ${HOST_PLY} — 건너뜀"; RC_ALL=1; continue
    fi

    echo ""
    echo "=== [${ARM}] 준비 $(date) ==="
    echo "  PLY: ${HOST_PLY} ($(du -h "${HOST_PLY}" | cut -f1))"

    # 이전 렌더는 지운다 — render.py 는 이미 있는 png 를 건너뛰므로
    # 반쪽 결과가 남아 있으면 그대로 재사용돼 버린다.
    # 2026-08-18: output/ 파일들은 도커가 root 로 만든다. 호스트(kevin)의 rm 은
    # 전부 Permission denied 로 실패했고(1차 재측정 사고), 어제 파일이 그대로
    # 재사용됐다. 청소·준비도 도커(root) 안에서 한다.
    docker run --rm -i \
        -v "${SCRIPT_DIR}":"${dir_cur}" \
        "${docker_name}" bash -c "
        rm -rf ${dir_cur}/output/eval_${ARM}/train \
               ${dir_cur}/output/eval_${ARM}/metrics.json \
               ${dir_cur}/output/eval_${ARM}/render_ws=*.log
        mkdir -p ${dir_cur}/output/eval_${ARM}/point_cloud/iteration_1
        ln -sfn ${PLY} ${dir_cur}/output/eval_${ARM}/point_cloud/iteration_1/point_cloud.ply
        echo \"Namespace(eval=False, images='images', model_path='output/eval_${ARM}', sh_degree=3, source_path='/w/data_eval_half', white_background=False)\" \
            > ${dir_cur}/output/eval_${ARM}/cfg_args
        "
    if [ $? -ne 0 ]; then echo "[${ARM}] 준비(청소) 실패 — 건너뜀"; RC_ALL=1; continue; fi

    # 2026-08-18: ours(71.5M 가우시안)는 뷰 3에서 9GB 연속 할당 실패 — 카드에
    # 8.3GB 가 남아 있어도 조각나 있으면 못 잡는다. max_split_size 로 캐시 조각을
    # 반환하게 한다. 그래도 부족하면 NPROC=8 DEVICES=0..7 로 랭크당 부담을 반으로.
    INNER="source activate citygs-x && cd ${dir_cur}/Grendel-GS && \
      PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
      torchrun --nproc_per_node=${NPROC} render.py \
        --model_path ../output/eval_${ARM} \
        --source_path ${dir_cur}/data_eval_half \
        --iteration 1 --skip_test --backend default"

    echo "=== [${ARM}] 렌더 시작 $(date) ==="
    docker run --rm -i --shm-size=64g --gpus "\"device=${DEVICES}\"" \
        --net=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -w "${dir_cur}" \
        -v "${dir_data}":/data \
        -v "${SCRIPT_DIR}":"${dir_cur}" \
        -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
        "${docker_name}" bash -c "${INNER}"
    rc=$?
    echo "=== [${ARM}] 렌더 종료 rc=${rc} $(date) ==="

    RENDER_DIR="$(ls -d "${OUT}"/train/ours_* 2>/dev/null | head -1)"
    if [ -z "${RENDER_DIR}" ]; then
        echo "[${ARM}] 렌더 결과 폴더 없음 — 채점 불가"; RC_ALL=1; continue
    fi
    echo "  렌더된 장수: $(ls "${RENDER_DIR}/renders" 2>/dev/null | wc -l)"

    echo "=== [${ARM}] 채점 시작 $(date) ==="
    docker run --rm -i --gpus "\"device=${DEVICES}\"" \
        --net=host -w "${dir_cur}" \
        -v "${SCRIPT_DIR}":"${dir_cur}" \
        -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro \
        "${docker_name}" bash -c "source activate citygs-x && \
          python -u ${dir_cur}/scripts/eval_metrics.py \
            ${dir_cur}/output/eval_${ARM}/train/$(basename "${RENDER_DIR}") \
            --expect ${EXPECT} --out ${dir_cur}/output/eval_${ARM}/metrics.json"
    rc=$?
    echo "=== [${ARM}] 채점 종료 rc=${rc} $(date) ==="
    [ ${rc} -ne 0 ] && RC_ALL=1
done

echo ""
echo "============================================"
echo "=== EVAL 요약 $(date) ==="
for ARM in ${ARMS}; do
    M="${SCRIPT_DIR}/output/eval_${ARM}/metrics.json"
    if [ -f "${M}" ]; then
        python3 -c "
import json,sys
d=json.load(open('${M}'))
print('  %-10s n=%s/%s complete=%s psnr=%s l1=%s' % ('${ARM}', d.get('n'), d.get('n_expected'), d.get('complete'), d.get('mean_psnr'), d.get('mean_l1')))
"
    else
        echo "  ${ARM}: metrics.json 없음"
    fi
done
echo "=== EVAL 전체 rc=${RC_ALL} ==="
echo "============================================"
exit ${RC_ALL}
