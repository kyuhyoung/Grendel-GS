#!/usr/bin/env python3
"""런 로그의 벽시계를 내역으로 분해 — §4.4 오버헤드 X% 의 영수증.

숫자는 절대 빼지 않는다: 총 시간은 그대로, 이 표는 '어디에 쓰였나'만 보인다.
  유용 학습(완주 타일) / Cat3(회수됨) / Cat1·2 폐기 / 인프라 재시도 /
  콜드스타트(프로세스 기동~학습 시작) / 저장·병합 / 래퍼 공백

Usage: python scripts/decompose_overhead.py <run_log> [<run_log> ...]
"""
import re
import sys
from datetime import datetime, timedelta


def parse_ts(s):
    return datetime.strptime(s, "%H:%M:%S")


def decompose(path):
    txt = open(path, errors="ignore").read()
    # 타임스탬프 형식 [DD/MM HH:MM:SS] — 날짜 넘김 처리 위해 일자도 취함
    TS = re.compile(r"\[(\d\d)/(\d\d) (\d\d:\d\d:\d\d)\]")

    def seg_ts(seg):
        return [(int(d), t) for d, m, t in TS.findall(seg)]

    def span(seg):
        ts = TS.findall(seg)
        if len(ts) < 2:
            return 0.0
        (d0, m0, t0), (d1, m1, t1) = ts[0], ts[-1]
        a = parse_ts(t0)
        b = parse_ts(t1) + timedelta(days=(int(d1) - int(d0)) % 28)
        return max(0.0, (b - a).total_seconds())

    starts = [(m.group(1), m.start())
              for m in re.finditer(r"\[Tile (tile_\d+)\] Running torchrun", txt)]
    rows = []
    for i, (tid, pos) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(txt)
        seg = txt[pos:end]
        dur = span(seg)
        # 콜드스타트: 기동 ~ 첫 Training progress
        m = re.search(r"Training progress", seg)
        cold = span(seg[:m.start()]) if m else 0.0
        # 저장·병합: save_ply 시작 ~ merge 완료 구간
        sm = 0.0
        m1 = re.search(r"\[save_ply\]", seg)
        m2 = None
        for m2 in re.finditer(r"\[merge_ply\] Merged", seg):
            pass
        if m1 and m2:
            sm = span(seg[m1.start():m2.end() + 200])
        if "Completed successfully" in seg:
            kind = "완주(유용)"
        elif "oom_category=3" in seg:
            kind = "Cat3(회수)"
        elif "no GPUs found" in seg or "INFRA RETRY" in seg:
            kind = "인프라"
        elif re.search(r"oom_category=[12]", seg):
            kind = "Cat1/2(폐기)"
        else:
            kind = "기타"
        rows.append((tid, kind, dur, cold, sm))

    total_serial = sum(r[2] for r in rows)
    agg = {}
    for _, k, d, c, s in rows:
        a = agg.setdefault(k, [0, 0.0, 0.0, 0.0])
        a[0] += 1
        a[1] += d
        a[2] += c
        a[3] += s

    # 전체 벽시계 (로그 처음~끝)
    wall = span(txt)

    print(f"\n=== {path} ===")
    print(f"벽시계 {wall/3600:.2f}h | 타일 시도 구간 합(직렬) {total_serial/3600:.2f}h "
          f"| 래퍼 공백 {max(0,(wall-total_serial))/3600:.2f}h")
    print(f"{'구분':12} {'횟수':>4} {'시간':>8} {'비중':>6} {'콜드스타트':>9} {'저장·병합':>9}")
    for k in ("완주(유용)", "Cat3(회수)", "Cat1/2(폐기)", "인프라", "기타"):
        if k not in agg:
            continue
        n, d, c, s = agg[k]
        print(f"{k:12} {n:4d} {d/3600:7.2f}h {100*d/total_serial:5.1f}% "
              f"{c/3600:8.2f}h {s/3600:8.2f}h")
    tc = sum(a[2] for a in agg.values())
    tsm = sum(a[3] for a in agg.values())
    print(f"{'합계 부대비용':12} {'':4} {'':8} {'':6} {tc/3600:8.2f}h {tsm/3600:8.2f}h")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["run_smoke3_midpoint.log"]:
        decompose(p)
