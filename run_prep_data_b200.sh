#!/usr/bin/env bash
# B200 머신: 3090 서버에서 받은 samsung-dong-aerial-30.tar 를 md5 검증 후 풀기.
# 사용: bash run_prep_data_b200.sh [TAR 경로]   (기본: $DATA_ROOT/samsung-dong-aerial-30.tar)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/run_prep_data_b200.log"; : > "$LOG"
exec > >(stdbuf -o0 tee "$LOG") 2>&1

DATA_ROOT="${DATA_ROOT:-/NHNHOME/WORKSPACE/26molit001_dbo/kevin/data/dabeeo}"   # /data 는 root 없이 못 만듦
TAR="${1:-$DATA_ROOT/samsung-dong-aerial-30.tar}"
EXPECT_MD5=695bb32c3392c57a9052fcf69e00c554
DST="$DATA_ROOT/samsung_dong_mini_30"

echo "=== [$(date +%H:%M:%S)] tar: $TAR"
[ -f "$TAR" ] || { echo "!!! tar 없음: $TAR  (3090: /media2/4tb/kevin/data_registry/aer-samsung/samsung-dong-aerial-30.tar 을 여기로 전송)"; exit 1; }
ls -l "$TAR"
echo "=== [$(date +%H:%M:%S)] md5 검증 (28GB, 수 분)"
got=$(md5sum "$TAR" | cut -d' ' -f1); echo "got=$got expect=$EXPECT_MD5"
[ "$got" = "$EXPECT_MD5" ] || { echo "!!! md5 불일치 — 전송 손상"; exit 2; }
echo "=== [$(date +%H:%M:%S)] 풀기 → $DATA_ROOT"
mkdir -p "$DATA_ROOT"; tar -xf "$TAR" -C "$DATA_ROOT" || { echo "!!! tar 풀기 실패"; exit 3; }
echo "=== [$(date +%H:%M:%S)] 구조 확인"
ls "$DST" && ls "$DST/sparse" && echo "images: $(ls "$DST/images" | wc -l) 장" || { echo "!!! $DST/{images,sparse} 구조 아님 — tar 내부 경로 확인:"; tar -tf "$TAR" | head -5; exit 4; }
[ -f "$DST/README"* ] && head -20 "$DST"/README* 2>/dev/null
echo "=== 완료: --source $DST"
