#!/usr/bin/env python3
"""완주한 adaptive-tile 런을 MLflow 에 사후 기록한다 (AI Center 표준 스키마).

설계 의도 — **학습 경로를 건드리지 않는다.**
  래퍼가 이미 adaptive_state.json 에 모든 사실(타일별 완주/OOM/분류/시간)을 남기므로,
  기록은 학습이 끝난 뒤 그 파일을 읽어 올리는 별도 단계로 둔다. 학습 중 네트워크 호출이
  0회이므로 MLflow 서버 장애가 학습을 죽일 수 없다 (팀 2026-08-31 사고의 구조적 회피).
  같은 이유로 몇 번이든 다시 돌릴 수 있고, 과거 런 백필도 동일 코드로 된다.

스키마 (MLflow 네이밍 & 파라미터 정책):
  experiment   lab2_aer-samsung_ada-3dgs   ({team}_{project}_{task})
  job_type     reconstruction              (3DGS = per-scene 최적화, 학습→추론 구조 없음)
  dataset_version ← 데이터셋이 아니라 **scene 식별자** (정책 §3-2 가 정한 재사용)
  config_path  생략 가능 (reconstruction 은 config 필수 면제)

Usage:
  python scripts/mlflow_report.py --run-dir output/ours_2gpu --scene samsung-dong-aerial-30 \
      [--metrics output/eval_ours/metrics.json] [--full-metrics ortho/full_metrics_all.json --arm eval_ours]
      [--log-file run_ours.log] [--run-name ours_2gpu_71.5M] [--gpus 2] [--dry-run]
"""
import argparse, json, os, re, subprocess, sys
from datetime import datetime
from pathlib import Path

DEFAULT_EXPERIMENT = "lab2_aer-samsung_ada-3dgs"
DEFAULT_TRACKING = "http://10.5.0.52:5000"


def parse_state(run_dir: Path):
    """adaptive_state.json → 트리 요약 + 타일별 상세. 없으면 통짜(oracle) 런."""
    sp = run_dir / "adaptive_state.json"
    if not sp.exists():
        return None
    st = json.loads(sp.read_text())["tiles"]
    leaves = [t for t, v in st.items() if v.get("status") == "completed"]
    splits = [t for t, v in st.items() if v.get("status") == "split"]
    cats = {}
    for t in splits:
        c = st[t].get("oom_category")
        key = f"cat{c}" if c else "preemptive"
        cats[key] = cats.get(key, 0) + 1

    # 부모-자식 관계(bbox 이분할)로 깊이 산출 — 타일 ID 역산은 선제분할에서 틀린다
    def bbox(t):
        v = [float(x) for x in st[t]["bbox"].split(",")]
        return v[0], v[1], v[3], v[4]
    EPS = 1e-3
    parent = {}
    for pid in splits:
        px0, py0, px1, py1 = bbox(pid)
        for a in st:
            if a == pid:
                continue
            ax0, ay0, ax1, ay1 = bbox(a)
            hit = None
            if abs(ax0-px0) < EPS and abs(ay0-py0) < EPS and abs(ay1-py1) < EPS and ax1 < px1-EPS:
                hit = ("x", ax1)
            elif abs(ay0-py0) < EPS and abs(ax0-px0) < EPS and abs(ax1-px1) < EPS and ay1 < py1-EPS:
                hit = ("y", ay1)
            if not hit:
                continue
            for b in st:
                bx0, by0, bx1, by1 = bbox(b)
                ok = ((hit[0] == "x" and abs(bx0-hit[1]) < EPS and abs(bx1-px1) < EPS
                       and abs(by0-py0) < EPS and abs(by1-py1) < EPS) or
                      (hit[0] == "y" and abs(by0-hit[1]) < EPS and abs(by1-py1) < EPS
                       and abs(bx0-px0) < EPS and abs(bx1-px1) < EPS))
                if ok:
                    parent[a] = pid
                    parent[b] = pid
    def depth(t):
        d = 0
        while t in parent:
            t = parent[t]
            d += 1
        return d
    max_depth = max((depth(t) for t in leaves), default=0)

    losses = [(st[t].get("quality") or {}).get("final_epoch_loss") for t in leaves]
    losses = sorted(x for x in losses if x is not None)
    converged = sum(1 for t in leaves
                    if (st[t].get("quality") or {}).get("done_reason") == "converged")
    x0, y0, x1, y1 = bbox("tile_00000000") if "tile_00000000" in st else (0, 0, 0, 0)

    return {
        "tiles_total": len(st),
        "tiles_leaf": len(leaves),
        "tiles_split": len(splits),
        "tiles_converged": converged,
        "tiles_skipped": sum(1 for v in st.values() if v.get("status") == "skipped"),
        "max_depth": max_depth,
        "split_by_cat": cats,
        "tile_loss_median": losses[len(losses)//2] if losses else None,
        "tile_loss_best": losses[0] if losses else None,
        "tile_loss_worst": losses[-1] if losses else None,
        "scene_area_km2": abs((x1-x0) * (y1-y0)) / 1e6,
        "retries_total": sum(v.get("retry_count", 0) for v in st.values()),
    }


def parse_gaussians(run_dir: Path):
    """병합 PLY 헤더에서 최종 가우시안 수 (GB 를 읽지 않고 헤더만)."""
    ply = run_dir / "merged" / "scene_point_cloud.ply"
    if not ply.exists():
        # oracle 통짜 런: point_cloud/iteration_NNNNN/point_cloud.ply (최대 iter)
        cands = sorted((run_dir / "point_cloud").glob("iteration_*/point_cloud.ply"),
                       key=lambda q: int(q.parent.name.split("_")[1])) if (run_dir / "point_cloud").exists() else []
        if not cands:
            return None
        ply = cands[-1]
    with open(ply, "rb") as f:
        head = f.read(4096).decode("latin-1")
    m = re.search(r"element vertex (\d+)", head)
    return int(m.group(1)) if m else None


def parse_walltime(log_path: Path):
    """런 로그의 시작/종료 타임스탬프 → 벽시계 시간(h). 실패하면 None."""
    if not log_path or not log_path.exists():
        return None, None, None
    pat = re.compile(r"(?:시작|Started at|Finished at|완주|종료)[^\n]*?"
                     r"([A-Z][a-z]{2} +[A-Z][a-z]{2} +\d+ +\d{2}:\d{2}:\d{2} +[A-Z]{3} +\d{4})")
    stamps = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    stamps.append(datetime.strptime(m.group(1), "%a %b %d %H:%M:%S %Z %Y"))
                except ValueError:
                    pass
    if len(stamps) < 2:
        return None, None, None
    t0, t1 = min(stamps), max(stamps)
    return (t1 - t0).total_seconds() / 3600.0, t0, t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--scene", required=True, help="scene 식별자 (dataset_version 슬롯)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT))
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING))
    ap.add_argument("--metrics", default=None, help="eval metrics.json (PSNR/L1)")
    ap.add_argument("--full-metrics", default=None, help="full_metrics_all.json (3지표)")
    ap.add_argument("--arm", default=None, help="full-metrics 안에서 읽을 키 (예: eval_ours)")
    ap.add_argument("--log-file", default=None, help="벽시계 산출용 런 로그")
    ap.add_argument("--gpus", type=int, default=None, help="사용 GPU 수 (자원 집계용)")
    ap.add_argument("--wallclock-hours", type=float, default=None,
                    help="벽시계 시간(h) 직접 주입 — 로그 파서 실패 시 사용")
    ap.add_argument("--dataset-md5", default=None,
                    help="dvc add 가 산출한 데이터 md5 (sandbox 형상관리)")
    ap.add_argument("--docker-tag", default=os.environ.get("DOCKER_TAG", "ogs:v1-936f3a4"))
    ap.add_argument("--note", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 유동 태그 방지: 로컬 도커에서 이미지 ID(다이제스트)를 해석해 태그에 덧붙인다.
    # 태그는 시간이 지나면 다른 이미지를 가리킬 수 있지만 sha256 ID 는 내용물 그 자체다.
    docker_tag = args.docker_tag
    try:
        img_id = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", docker_tag],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if img_id.startswith("sha256:"):
            docker_tag = f"{docker_tag}@{img_id[:19]}"
    except Exception:
        pass  # 도커 없는 환경에서도 리포트는 가능해야 한다


    run_dir = Path(args.run_dir)
    summary = parse_state(run_dir)
    gaussians = parse_gaussians(run_dir)
    hours, t0, t1 = parse_walltime(Path(args.log_file) if args.log_file else None)

    params = {
        "scene": args.scene,
        "gpus_used": args.gpus if args.gpus else "",
        "run_dir": str(run_dir),
    }
    metrics = {}
    if summary is not None:
        params.update({
            "tiles_total": summary["tiles_total"],
            "tiles_leaf": summary["tiles_leaf"],
            "tiles_split": summary["tiles_split"],
            "max_depth": summary["max_depth"],
        })
        for k, v in summary["split_by_cat"].items():
            params[f"split_{k}"] = v
        metrics.update({
            "tiles_leaf": summary["tiles_leaf"],
            "tiles_converged": summary["tiles_converged"],
            "tiles_skipped": summary["tiles_skipped"],
            "oom_splits": sum(v for k, v in summary["split_by_cat"].items() if k.startswith("cat")),
            "retries_total": summary["retries_total"],
            "scene_area_km2": round(summary["scene_area_km2"], 3),
        })
        for k in ("tile_loss_median", "tile_loss_best", "tile_loss_worst"):
            if summary[k] is not None:
                metrics[k] = summary[k]
    else:
        params["tiles_total"] = 1  # 통짜(oracle) 런 — 분할 없음
    if gaussians:
        metrics["gaussians_final"] = gaussians
    if args.wallclock_hours is not None:
        hours = args.wallclock_hours
    if hours:
        metrics["wallclock_hours"] = round(hours, 3)
        if args.gpus:
            metrics["gpu_hours"] = round(hours * args.gpus, 2)

    # 화질 지표
    if args.metrics and Path(args.metrics).exists():
        d = json.loads(Path(args.metrics).read_text())
        if d.get("mean_psnr") is not None:
            metrics["psnr"] = d["mean_psnr"]
            metrics["l1"] = d["mean_l1"]
            metrics["scored_views"] = d.get("n", 0)
    if args.full_metrics and args.arm and Path(args.full_metrics).exists():
        d = json.loads(Path(args.full_metrics).read_text()).get(args.arm)
        if d:
            for k in ("psnr", "ssim", "lpips"):
                if k in d:
                    metrics[k] = d[k]
            metrics["scored_views"] = d.get("n", metrics.get("scored_views", 0))

    print("=== 기록할 내용 ===")
    print("experiment :", args.experiment)
    print("run_name   :", args.run_name or run_dir.name)
    print("params     :", json.dumps(params, ensure_ascii=False))
    if args.dataset_md5:
        params["dataset_md5"] = args.dataset_md5
    print("metrics    :", json.dumps(metrics, ensure_ascii=False))
    print("docker_tag :", docker_tag)
    if args.dry_run:
        print("(dry-run — 서버에 올리지 않음)")
        return 0

    import mlflow
    from dabeeo_mlflow import log_run_meta, safe_log_artifact, safe_run

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    rid_file = run_dir / ".mlflow_run_id"
    resume_id = rid_file.read_text().strip() if rid_file.exists() else None
    if resume_id:
        print(f"(기존 RUNNING 런 {resume_id} 에 이어서 기록)")
    with safe_run(run_id=resume_id) if resume_id else safe_run(run_name=args.run_name or run_dir.name) as run:
        log_run_meta(
            dataset_version=(f"sandbox-{args.dataset_md5[:8]}" if args.dataset_md5
                             else args.scene),  # 정책 Q1: sandbox 는 dvc md5 로 고정
            job_type="reconstruction",
            task="ada-3dgs",
            team="lab2",
            docker_tag=docker_tag,
            note=args.note or "",
        )
        mlflow.log_params(params)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        # 트리 구조·병합 명세는 아티팩트로 (MLflow 에 트리 개념이 없으므로)
        for f in ("adaptive_state.json", "merged/merge_manifest.json"):
            p = run_dir / f
            if p.exists():
                safe_log_artifact(str(p))
        for f in ("merged/scene_merged_topdown.png",
                  "merged/scene_quality_heatmap.png",
                  "visualizations/tile_split_tree/tile_split_tree.png"):
            p = run_dir / f
            if p.exists():
                safe_log_artifact(str(p))
        rid = run.info.run_id if run else "?"
    print(f"\n기록 완료 — run_id={rid}")
    print(f"확인: {args.tracking_uri}/#/experiments  ({args.experiment})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
