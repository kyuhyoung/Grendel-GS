#!/usr/bin/env python3
"""
렌더 결과(gt/ vs renders/) 화질 지표 계산.

2026-08-18 교훈: 이전 측정은 metrics.json 이 "생겼다"는 것만 확인하고 넘어가서,
30장 중 2장만 채점된 파일이 완료로 취급됐다(eval_ours: 랭크 0~2 가 호스트 램
부족으로 커널에 죽음). 그래서 여기서 기대 장수를 강제하고, 모자라면 0 이 아닌
코드로 죽는다 — 반쪽 결과가 조용히 다음 단계로 흘러가지 않게.

Usage: python scripts/eval_metrics.py <render_dir> --expect 30 --out metrics.json
  <render_dir> 안에 gt/ 와 renders/ 가 있어야 한다.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 초대형 항공 이미지


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render_dir")
    ap.add_argument("--expect", type=int, default=0, help="기대 장수 (0=검사 안 함)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt_dir = os.path.join(args.render_dir, "gt")
    rd_dir = os.path.join(args.render_dir, "renders")
    for d in (gt_dir, rd_dir):
        if not os.path.isdir(d):
            print(f"[eval] 없음: {d}", flush=True)
            sys.exit(2)

    names = sorted(set(os.listdir(gt_dir)) & set(os.listdir(rd_dir)))
    names = [n for n in names if n.lower().endswith(".png")]

    l1s, psnrs, per_image = [], [], []
    for n in names:
        gt = np.asarray(Image.open(os.path.join(gt_dir, n)).convert("RGB"),
                        dtype=np.float32) / 255.0
        rd = np.asarray(Image.open(os.path.join(rd_dir, n)).convert("RGB"),
                        dtype=np.float32) / 255.0
        if gt.shape != rd.shape:
            print(f"[eval] 크기 불일치 {n}: {gt.shape} vs {rd.shape} — 건너뜀", flush=True)
            continue
        diff = np.abs(gt - rd)
        l1 = float(diff.mean())
        mse = float((diff.astype(np.float64) ** 2).mean())
        psnr = float("inf") if mse == 0 else 20.0 * np.log10(1.0 / np.sqrt(mse))
        l1s.append(l1)
        psnrs.append(float(psnr))
        per_image.append({"name": n, "l1": l1, "psnr": float(psnr)})
        print(f"[eval] {n}: L1={l1:.6f} PSNR={psnr:.4f}", flush=True)

    n_ok = len(l1s)
    result = {
        "n": n_ok,
        "n_expected": args.expect,
        "complete": (args.expect == 0 or n_ok == args.expect),
        "mean_l1": float(np.mean(l1s)) if n_ok else None,
        "mean_psnr": float(np.mean(psnrs)) if n_ok else None,
        "per_image": per_image,
    }

    out = args.out or os.path.join(args.render_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[eval] n={n_ok}/{args.expect} mean_l1={result['mean_l1']} "
          f"mean_psnr={result['mean_psnr']} -> {out}", flush=True)

    if not result["complete"]:
        print(f"[eval] 불완전: {n_ok}/{args.expect} 장만 채점됨 — 비교에 쓸 수 없음", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
