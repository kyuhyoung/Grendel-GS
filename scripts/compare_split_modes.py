#!/usr/bin/env python3
"""중점 분할 vs median 분할 비교.

2026-08-07 median 분할 도입의 실제 이득을 재는 스크립트.
균형(자식 가우시안 비율)은 이미 확인됐고, 이게 재는 건 그 결과물이다:
  - 분할 단계/타일 수가 실제로 줄었나
  - skipped(= 최종 씬의 구멍)가 안 생겼나
  - Cat2 재시도로 날린 시간 (3번 작업의 부산물 → 2번 판단 근거)

Usage:
  python scripts/compare_split_modes.py output/smoke_3gpu_midpoint output/smoke_3gpu_median
  python scripts/compare_split_modes.py <dir> [<dir> ...]   (1개만 줘도 됨)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path


def tile_level(tid, tiles):
    """타일 ID 는 부모*2+1/+2 로 만들어지는 이진트리. ID 에서 깊이를 역산한다."""
    n = int(tid.split("_")[-1])
    lvl = 0
    while n > 0:
        n = (n - 1) // 2
        lvl += 1
    return lvl


def analyze(out_dir: Path):
    state_path = out_dir / "adaptive_state.json"
    if not state_path.exists():
        return None
    tiles = json.load(open(state_path)).get("tiles", {})
    if not tiles:
        return None

    status = Counter(v.get("status") for v in tiles.values())
    leaves = {k: v for k, v in tiles.items() if v.get("status") != "split"}
    levels = [tile_level(k, tiles) for k in leaves]
    cats = Counter(v.get("oom_category") for v in tiles.values()
                   if v.get("oom_category") is not None)
    retries = sum(int(v.get("retry_count") or 0) for v in tiles.values())

    losses = [v["quality"]["final_epoch_loss"] for v in tiles.values()
              if (v.get("quality") or {}).get("final_epoch_loss") is not None]

    return {
        "dir": out_dir.name,
        "total": len(tiles),
        "status": dict(status),
        "skipped": status.get("skipped", 0) + status.get("failed", 0),
        "max_level": max(levels) if levels else 0,
        "leaves": len(leaves),
        "cats": dict(cats),
        "retries": retries,
        "losses": losses,
        "done": status.get("pending", 0) == 0 and status.get("in_progress", 0) == 0,
    }


def split_balance(log_path: Path):
    """로그에서 Cat3 자식 병합 총계를 뽑아 자식 간 불균형을 계산."""
    if not log_path.exists():
        return []
    txt = log_path.read_text(errors="ignore")
    merged = re.findall(r"\[merge_ply\] Merged ([\d,]+) gaussians -> .*?(tile_\d+)_L\d+_oom_iter(\d+)_merged\.ply", txt)
    by_iter = {}
    for count, tid, it in merged:
        by_iter.setdefault(it, []).append(int(count.replace(",", "")))
    out = []
    for it, counts in sorted(by_iter.items(), key=lambda kv: int(kv[0])):
        if len(counts) == 2:
            a, b = sorted(counts)
            out.append((int(it), a, b, b / a if a else float("inf")))
    return out


def main(dirs):
    results = [r for r in (analyze(Path(d)) for d in dirs) if r]
    if not results:
        print("분석할 adaptive_state.json 을 못 찾음")
        return

    print("=" * 74)
    print(f"{'':22}" + "".join(f"{r['dir']:>26}" for r in results))
    print("=" * 74)

    def row(label, fn):
        print(f"{label:22}" + "".join(f"{fn(r):>26}" for r in results))

    row("완주 여부", lambda r: "완주" if r["done"] else "진행중/중단")
    row("전체 타일", lambda r: f"{r['total']:,}")
    row("leaf 타일", lambda r: f"{r['leaves']:,}")
    row("분할된 타일", lambda r: f"{r['status'].get('split', 0):,}")
    row("최대 분할 깊이", lambda r: r["max_level"])
    row("skipped/failed", lambda r: f"{r['skipped']}  <-- 씬의 구멍")
    row("Cat1 / Cat2 / Cat3", lambda r: "  ".join(str(r["cats"].get(c, 0)) for c in (1, 2, 3)))
    row("Cat2 재시도 누계", lambda r: r["retries"])
    row("타일 loss 중앙값", lambda r: f"{sorted(r['losses'])[len(r['losses'])//2]:.4f}" if r["losses"] else "-")

    for d in dirs:
        p = Path(d)
        # run_smoke3_<mode>.log 또는 run_smoke3.log
        for cand in (p.parent.parent / f"run_smoke3_{p.name.split('_')[-1]}.log",
                     p.parent.parent / "run_smoke3.log"):
            bal = split_balance(cand)
            if bal:
                print(f"\n[{p.name}] Cat3 자식 균형  ({cand.name})")
                for it, a, b, ratio in bal:
                    tot = a + b
                    print(f"  iter {it:>6}:  {a:>10,} / {b:>10,}   "
                          f"= {100*a/tot:4.1f}% / {100*b/tot:4.1f}%   불균형 {ratio:.3f}배")
                break


if __name__ == "__main__":
    main(sys.argv[1:] or ["output/smoke_3gpu", "output/smoke_3gpu_median"])
