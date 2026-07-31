"""Cat 3 OOM 시각화 유틸리티.

두 가지 시각화를 제공:
1. save_per_rank_cat3_viz: save_ply_callback 안에서 호출. 해당 rank 가
   들고 있는 가우시안의 공간 분포 + 부모/자식 a,b bbox 오버레이.
2. save_merged_cat3_viz: wrapper 의 merge 직후 호출. rank 별 PLY 를
   읽어 색상으로 구분, 자식 bbox 와 함께 표시.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


_MAX_POINTS_PER_RANK = 80000


def _bbox_tuple(bbox) -> tuple:
    return (bbox.x_min, bbox.y_min, bbox.z_min,
            bbox.x_max, bbox.y_max, bbox.z_max)


def _draw_bbox_xy(ax, bbox, *, edgecolor, facecolor, alpha, label, linestyle="-"):
    rect = patches.Rectangle(
        (bbox.x_min, bbox.y_min),
        bbox.x_max - bbox.x_min,
        bbox.y_max - bbox.y_min,
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        linestyle=linestyle,
        label=label,
    )
    ax.add_patch(rect)


def _draw_bbox_xz(ax, bbox, *, edgecolor, facecolor, alpha, label, linestyle="-"):
    rect = patches.Rectangle(
        (bbox.x_min, bbox.z_min),
        bbox.x_max - bbox.x_min,
        bbox.z_max - bbox.z_min,
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        linestyle=linestyle,
        label=label,
    )
    ax.add_patch(rect)


def _classify_xy(xyz: np.ndarray, tile_a, tile_b) -> np.ndarray:
    """Return integer label per point: 0=in A, 1=in B, 2=outside both."""
    labels = np.full(xyz.shape[0], 2, dtype=np.int8)
    in_a = (
        (xyz[:, 0] >= tile_a.x_min) & (xyz[:, 0] <= tile_a.x_max)
        & (xyz[:, 1] >= tile_a.y_min) & (xyz[:, 1] <= tile_a.y_max)
        & (xyz[:, 2] >= tile_a.z_min) & (xyz[:, 2] <= tile_a.z_max)
    )
    in_b = (
        (xyz[:, 0] >= tile_b.x_min) & (xyz[:, 0] <= tile_b.x_max)
        & (xyz[:, 1] >= tile_b.y_min) & (xyz[:, 1] <= tile_b.y_max)
        & (xyz[:, 2] >= tile_b.z_min) & (xyz[:, 2] <= tile_b.z_max)
    )
    labels[in_a] = 0
    labels[in_b & ~in_a] = 1
    return labels


def _subsample(xyz: np.ndarray, max_points: int) -> np.ndarray:
    n = xyz.shape[0]
    if n <= max_points:
        return xyz
    idx = np.random.default_rng(42).choice(n, size=max_points, replace=False)
    return xyz[idx]


def save_per_rank_cat3_viz(
    xyz_local: np.ndarray,
    *,
    rank: int,
    iteration: int,
    tile_id: str,
    parent_bbox,
    tile_a,
    tile_b,
    out_dir: str | os.PathLike,
    count_a: Optional[int] = None,
    count_b: Optional[int] = None,
) -> Optional[str]:
    """Per-rank Cat3 OOM 시각화 PNG 저장. 실패 시 None 반환 (예외 안 던짐)."""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if xyz_local.size == 0:
            return None
        xyz = _subsample(xyz_local.astype(np.float32, copy=False), _MAX_POINTS_PER_RANK)
        labels = _classify_xy(xyz, tile_a, tile_b)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        colors = np.array([
            [0.20, 0.55, 0.85],  # in A
            [0.95, 0.55, 0.15],  # in B
            [0.55, 0.55, 0.55],  # outside
        ])
        # XY (top-down)
        ax_xy = axes[0]
        _draw_bbox_xy(ax_xy, parent_bbox, edgecolor="black",
                      facecolor="none", alpha=1.0, label="parent", linestyle="--")
        _draw_bbox_xy(ax_xy, tile_a, edgecolor="#1c6ec4",
                      facecolor="#1c6ec4", alpha=0.10, label="child A")
        _draw_bbox_xy(ax_xy, tile_b, edgecolor="#d97706",
                      facecolor="#d97706", alpha=0.10, label="child B")
        ax_xy.scatter(xyz[:, 0], xyz[:, 1], s=1.0, c=colors[labels],
                      alpha=0.6, linewidths=0)
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.set_title(f"Top-down (XY)  n={xyz.shape[0]:,} (sampled)")
        ax_xy.legend(loc="upper right", fontsize=8)

        # XZ (side)
        ax_xz = axes[1]
        _draw_bbox_xz(ax_xz, parent_bbox, edgecolor="black",
                      facecolor="none", alpha=1.0, label="parent", linestyle="--")
        _draw_bbox_xz(ax_xz, tile_a, edgecolor="#1c6ec4",
                      facecolor="#1c6ec4", alpha=0.10, label="child A")
        _draw_bbox_xz(ax_xz, tile_b, edgecolor="#d97706",
                      facecolor="#d97706", alpha=0.10, label="child B")
        ax_xz.scatter(xyz[:, 0], xyz[:, 2], s=1.0, c=colors[labels],
                      alpha=0.6, linewidths=0)
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_aspect("equal", adjustable="datalim")
        ax_xz.set_title("Side (XZ)")
        ax_xz.legend(loc="upper right", fontsize=8)

        ca = "?" if count_a is None else f"{count_a:,}"
        cb = "?" if count_b is None else f"{count_b:,}"
        fig.suptitle(
            f"Cat3 OOM per-rank | tile={tile_id}  iter={iteration}  rank={rank}  "
            f"local={xyz_local.shape[0]:,}  saved A={ca} B={cb}",
            fontsize=11,
        )
        fig.tight_layout()

        path = out_dir / f"cat3_iter{iteration}_{tile_id}_rank{rank}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)
    except Exception as e:  # 시각화 실패는 학습/저장 흐름을 막지 않음
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[oom_viz] per-rank viz failed: {e}", flush=True)
        return None


def save_count_timeline_viz(
    rank_jsonl_paths: Sequence[str],
    *,
    tile_id: str,
    out_path: str | os.PathLike,
) -> Optional[str]:
    """rank 별 가우시안 카운트 타임라인 PNG. 실패 시 None 반환."""
    try:
        import json as _json

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        per_rank: dict[int, list] = {}
        for p in rank_jsonl_paths:
            try:
                rank_str = Path(p).stem.split("_rank")[-1]
                rank = int(rank_str)
            except Exception:
                rank = -1
            recs = []
            try:
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        recs.append(_json.loads(line))
            except Exception as e:
                print(f"[oom_viz] failed to read {p}: {e}", flush=True)
                continue
            per_rank.setdefault(rank, []).extend(recs)

        if not per_rank:
            return None

        from matplotlib.ticker import MaxNLocator, FuncFormatter

        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        oom_iters = []
        all_iters = []
        all_counts = []
        for rank, recs in sorted(per_rank.items()):
            recs.sort(key=lambda r: r["iter"])
            iters = [r["iter"] for r in recs]
            counts = [r["count"] for r in recs]
            all_iters.extend(iters)
            all_counts.extend(counts)
            color = cmap(rank % 10)
            # marker="o" 로 단일 점도 보이게
            ax.plot(iters, counts, "-o", color=color, linewidth=1.2, markersize=4,
                    alpha=0.85, label=f"rank {rank} (n={len(iters)})")

            densify_pts = [(r["iter"], r["count"]) for r in recs if r["event"].startswith("densify")]
            if densify_pts:
                xs, ys = zip(*densify_pts)
                ax.scatter(xs, ys, s=28, c=[color], marker="^", edgecolors="black",
                           linewidths=0.4, zorder=4)

            for r in recs:
                if r["event"].startswith("oom"):
                    oom_iters.append(r["iter"])

        for it in sorted(set(oom_iters)):
            ax.axvline(it, color="#cc0000", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(it, ax.get_ylim()[1], f"OOM iter={it}",
                    rotation=90, va="top", ha="right", fontsize=8, color="#cc0000")

        ax.set_xlabel("iteration")
        ax.set_ylabel("local gaussian count")
        ax.set_title(f"Gaussian count timeline | tile={tile_id}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # x, y 모두 정수 카운트 → 분수 tick / scientific offset 금지.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v):,}"))
        ax.ticklabel_format(useOffset=False, style="plain", axis="x")
        if all_iters:
            i_min, i_max = min(all_iters), max(all_iters)
            if i_min == i_max:
                ax.set_xlim(i_min - 1, i_min + 1)
        if all_counts:
            c_min, c_max = min(all_counts), max(all_counts)
            if c_max - c_min < 10:
                pad = max(1, (c_max - c_min) * 5 + 5)
                ax.set_ylim(c_min - pad, c_max + pad)

        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return str(out_path)
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[oom_viz] count timeline viz failed: {e}", flush=True)
        return None


def save_all_tiles_timeline_viz(
    jsonl_dir: str | os.PathLike,
    *,
    out_path: str | os.PathLike,
    target_count: Optional[int] = None,
) -> Optional[str]:
    """모든 타일의 가우시안 카운트 타임라인을 한 PNG 에 겹쳐 그림.

    - jsonl_dir 안의 ``*_rank*.jsonl`` 모두 스캔
    - tile_id 별로 색상 부여, rank 0 solid / rank 1 dashed
    - densify 이벤트 = 삼각형, OOM 이벤트 = 빨간 X
    - 각 타일의 leaf node 에는 최종 status 라벨
    """
    try:
        import json as _json
        from matplotlib.ticker import MaxNLocator, FuncFormatter

        jsonl_dir = Path(jsonl_dir)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # tile_id, rank, [records] 수집
        files = sorted(jsonl_dir.glob("*_rank*.jsonl"))
        if not files:
            return None
        per_tile: dict[str, dict[int, list]] = {}
        for f in files:
            stem = f.stem
            try:
                tile_id, rank_part = stem.rsplit("_rank", 1)
                rank = int(rank_part)
            except Exception:
                continue
            recs = []
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        recs.append(_json.loads(line))
            except Exception:
                continue
            recs.sort(key=lambda r: r["iter"])
            per_tile.setdefault(tile_id, {})[rank] = recs

        if not per_tile:
            return None

        cmap = plt.get_cmap("tab20")
        sorted_tiles = sorted(per_tile.keys())
        tile_color = {tid: cmap(i % 20) for i, tid in enumerate(sorted_tiles)}

        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

        all_iters: list[int] = []
        all_counts: list[int] = []
        oom_xs: list[int] = []
        oom_ys: list[int] = []

        for tid in sorted_tiles:
            color = tile_color[tid]
            ranks_dict = per_tile[tid]
            for rank, recs in sorted(ranks_dict.items()):
                if not recs:
                    continue
                iters = [r["iter"] for r in recs]
                counts = [r["count"] for r in recs]
                all_iters.extend(iters)
                all_counts.extend(counts)
                ls = "-" if rank == 0 else "--"
                ax.plot(iters, counts, ls + "o", color=color, linewidth=1.2,
                        markersize=4, alpha=0.85,
                        label=f"{tid} rk{rank} (n={len(iters)})")
                # densify events
                d_pts = [(r["iter"], r["count"]) for r in recs if r["event"].startswith("densify")]
                if d_pts:
                    xs, ys = zip(*d_pts)
                    ax.scatter(xs, ys, s=30, c=[color], marker="^",
                               edgecolors="black", linewidths=0.4, zorder=4)
                # oom events
                for r in recs:
                    if r["event"].startswith("oom"):
                        oom_xs.append(r["iter"]); oom_ys.append(r["count"])

        if oom_xs:
            ax.scatter(oom_xs, oom_ys, s=80, c="red", marker="x",
                       linewidths=2.0, label="OOM", zorder=5)

        ax.set_xlabel("iteration (per tile, local)")
        ax.set_ylabel("local gaussian count (per rank)")
        title = f"All tiles timeline | tiles={len(sorted_tiles)}"
        if target_count is not None:
            title += f"  (verify target={target_count})"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v):,}"))
        ax.ticklabel_format(useOffset=False, style="plain", axis="x")
        if all_iters:
            i_min, i_max = min(all_iters), max(all_iters)
            if i_min == i_max:
                ax.set_xlim(i_min - 1, i_min + 1)
        if all_counts:
            c_min, c_max = min(all_counts), max(all_counts)
            if c_max - c_min < 10:
                pad = max(1, (c_max - c_min) * 5 + 5)
                ax.set_ylim(c_min - pad, c_max + pad)
        # legend 가 너무 길면 figure 바깥
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  ncol=1, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[oom_viz] all-tiles timeline viz failed: {e}", flush=True)
        return None


def save_oom_progression_viz(
    jsonl_dir: str | os.PathLike,
    *,
    out_path: str | os.PathLike,
) -> Optional[str]:
    """타일/OOM 진행 시퀀스 뷰.

    x = 타일 처리 순서 (OOM 발생 횟수 +1), y = 그 타일이 도달한 가우시안 카운트.
    한 점 = 한 (타일, rank). rank 0/1 색 분리.
    각 x 에 tile_id 라벨, 마지막 이벤트(OOM 또는 normal end) 마커.

    학습 step 기반 PNG 와 달리, 이 뷰는 wrapper-level 흐름 ("OOM 이 거듭되면서
    타일이 어떻게 쪼개지고 카운트가 어떻게 변하는지") 을 한 눈에 보여 줌.
    """
    try:
        import json as _json
        from matplotlib.ticker import MaxNLocator, FuncFormatter

        jsonl_dir = Path(jsonl_dir)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        files = sorted(jsonl_dir.glob("*_rank*.jsonl"))
        if not files:
            return None

        # tile_id -> rank -> (recs, mtime)
        per_tile: dict[str, dict[int, tuple[list, float]]] = {}
        for f in files:
            stem = f.stem
            try:
                tile_id, rank_part = stem.rsplit("_rank", 1)
                rank = int(rank_part)
            except Exception:
                continue
            try:
                recs = []
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        recs.append(_json.loads(line))
                recs.sort(key=lambda r: r["iter"])
            except Exception:
                recs = []
            per_tile.setdefault(tile_id, {})[rank] = (recs, f.stat().st_mtime)

        if not per_tile:
            return None

        # 타일 처리 순서: rank 0 jsonl 의 mtime 기준 (없으면 가장 빠른 rank)
        def tile_first_mtime(tid: str) -> float:
            ranks = per_tile[tid]
            return min(m for _r, (_, m) in ranks.items())

        ordered_tiles = sorted(per_tile.keys(), key=tile_first_mtime)

        # 각 타일에서 rank 별 max count, last event
        rank_xy: dict[int, list[tuple[int, int]]] = {}  # rank -> [(seq, count)]
        oom_marks: list[tuple[int, int]] = []           # (seq, count) — OOM 발생 좌표
        seq_to_tile: list[str] = []
        for seq, tid in enumerate(ordered_tiles):
            seq_to_tile.append(tid)
            for rank, (recs, _mt) in per_tile[tid].items():
                if not recs:
                    continue
                max_count = max(r["count"] for r in recs)
                rank_xy.setdefault(rank, []).append((seq, max_count))
                last = recs[-1]
                if last["event"].startswith("oom"):
                    oom_marks.append((seq, last["count"]))

        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(1, 1, figsize=(max(10, len(ordered_tiles) * 1.0), 6))

        for rank, pairs in sorted(rank_xy.items()):
            pairs.sort(key=lambda p: p[0])
            xs, ys = zip(*pairs)
            color = cmap(rank % 10)
            ax.plot(xs, ys, "-o", color=color, linewidth=1.5, markersize=6,
                    label=f"rank {rank} max-count")

        if oom_marks:
            xs, ys = zip(*oom_marks)
            ax.scatter(xs, ys, s=110, c="red", marker="x", linewidths=2.2,
                       label="OOM (last event)", zorder=5)

        ax.set_xticks(range(len(ordered_tiles)))
        ax.set_xticklabels(ordered_tiles, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("tile processing sequence (OOM count)")
        ax.set_ylabel("max gaussian count reached (per rank)")
        ax.set_title(f"OOM / tile progression  |  {len(ordered_tiles)} tiles processed")
        ax.grid(True, alpha=0.3, axis="y")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v):,}"))
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[oom_viz] oom-progression viz failed: {e}", flush=True)
        return None


def save_resume_viz(
    xyz_local: np.ndarray,
    *,
    rank: int,
    tile_id: str,
    tile_bbox,
    camera_positions: np.ndarray,
    out_dir: str | os.PathLike,
    pretrained_ply_path: str = "",
    iteration: Optional[int] = None,
) -> Optional[str]:
    """Cat3 resume 검증 시각화. 자식 타일 학습 시작 시 호출.

    - 자식 bbox
    - 부모에서 로드된 pre-trained 가우시안 (이 rank 에 분배된 것)
    - 자식 visible cameras 위치
    """
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if xyz_local.size == 0:
            xyz = np.zeros((0, 3), dtype=np.float32)
        else:
            xyz = _subsample(xyz_local.astype(np.float32, copy=False), _MAX_POINTS_PER_RANK)

        if camera_positions is None:
            cam_xyz = np.zeros((0, 3), dtype=np.float32)
        else:
            cam_xyz = np.asarray(camera_positions, dtype=np.float32).reshape(-1, 3)

        # In/out classification w.r.t. child bbox
        if xyz.shape[0] > 0:
            inside = (
                (xyz[:, 0] >= tile_bbox.x_min) & (xyz[:, 0] <= tile_bbox.x_max)
                & (xyz[:, 1] >= tile_bbox.y_min) & (xyz[:, 1] <= tile_bbox.y_max)
                & (xyz[:, 2] >= tile_bbox.z_min) & (xyz[:, 2] <= tile_bbox.z_max)
            )
        else:
            inside = np.zeros((0,), dtype=bool)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # XY (top-down)
        ax_xy = axes[0]
        _draw_bbox_xy(ax_xy, tile_bbox, edgecolor="#0a8a3a",
                      facecolor="#0a8a3a", alpha=0.10, label="child bbox")
        if xyz.shape[0] > 0:
            ax_xy.scatter(xyz[~inside, 0], xyz[~inside, 1], s=0.8, c="#cc4444",
                          alpha=0.5, linewidths=0, label=f"loaded outside ({(~inside).sum():,})")
            ax_xy.scatter(xyz[inside, 0], xyz[inside, 1], s=0.8, c="#1c6ec4",
                          alpha=0.5, linewidths=0, label=f"loaded inside ({inside.sum():,})")
        if cam_xyz.shape[0] > 0:
            ax_xy.scatter(cam_xyz[:, 0], cam_xyz[:, 1], s=40, c="#000000",
                          marker="^", label=f"cameras ({cam_xyz.shape[0]})", zorder=5)
        ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y")
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.set_title("Top-down (XY)")
        ax_xy.legend(loc="upper right", fontsize=8, markerscale=3)

        # XZ (side)
        ax_xz = axes[1]
        _draw_bbox_xz(ax_xz, tile_bbox, edgecolor="#0a8a3a",
                      facecolor="#0a8a3a", alpha=0.10, label="child bbox")
        if xyz.shape[0] > 0:
            ax_xz.scatter(xyz[~inside, 0], xyz[~inside, 2], s=0.8, c="#cc4444",
                          alpha=0.5, linewidths=0)
            ax_xz.scatter(xyz[inside, 0], xyz[inside, 2], s=0.8, c="#1c6ec4",
                          alpha=0.5, linewidths=0)
        if cam_xyz.shape[0] > 0:
            ax_xz.scatter(cam_xyz[:, 0], cam_xyz[:, 2], s=40, c="#000000",
                          marker="^", zorder=5)
        ax_xz.set_xlabel("X"); ax_xz.set_ylabel("Z")
        ax_xz.set_aspect("equal", adjustable="datalim")
        ax_xz.set_title("Side (XZ)")

        ply_short = Path(pretrained_ply_path).name if pretrained_ply_path else "?"
        n_in = int(inside.sum()) if xyz.shape[0] > 0 else 0
        n_out = int((~inside).sum()) if xyz.shape[0] > 0 else 0
        fig.suptitle(
            f"Resume verify | tile={tile_id} rank={rank}  "
            f"loaded={xyz_local.shape[0]:,} (in={n_in:,} out={n_out:,})  "
            f"cams={cam_xyz.shape[0]}  ply={ply_short}",
            fontsize=10,
        )
        fig.tight_layout()

        suffix = f"_iter{iteration}" if iteration is not None else ""
        path = out_dir / f"resume_{tile_id}{suffix}_rank{rank}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[oom_viz] resume viz failed: {e}", flush=True)
        return None


def save_merged_cat3_viz(
    rank_ply_paths: Sequence[str],
    *,
    iteration: int,
    tile_id: str,
    parent_bbox,
    tile_a,
    tile_b,
    out_path: str | os.PathLike,
) -> Optional[str]:
    """Merge 후 rank 별 색상 시각화. 실패 시 None 반환."""
    try:
        from plyfile import PlyData  # type: ignore
    except Exception as e:
        print(f"[oom_viz] plyfile import failed: {e}", flush=True)
        return None

    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        per_rank_xyz = []
        for ply_path in rank_ply_paths:
            try:
                pdata = PlyData.read(str(ply_path))
                v = pdata["vertex"]
                xyz = np.column_stack([
                    np.asarray(v["x"], dtype=np.float32),
                    np.asarray(v["y"], dtype=np.float32),
                    np.asarray(v["z"], dtype=np.float32),
                ])
                per_rank_xyz.append(_subsample(xyz, _MAX_POINTS_PER_RANK))
            except Exception as e:
                print(f"[oom_viz] failed to read {ply_path}: {e}", flush=True)
                per_rank_xyz.append(np.zeros((0, 3), dtype=np.float32))

        cmap = plt.get_cmap("tab10")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax_xy = axes[0]
        _draw_bbox_xy(ax_xy, parent_bbox, edgecolor="black",
                      facecolor="none", alpha=1.0, label="parent", linestyle="--")
        _draw_bbox_xy(ax_xy, tile_a, edgecolor="#1c6ec4",
                      facecolor="#1c6ec4", alpha=0.08, label="child A")
        _draw_bbox_xy(ax_xy, tile_b, edgecolor="#d97706",
                      facecolor="#d97706", alpha=0.08, label="child B")

        ax_xz = axes[1]
        _draw_bbox_xz(ax_xz, parent_bbox, edgecolor="black",
                      facecolor="none", alpha=1.0, label="parent", linestyle="--")
        _draw_bbox_xz(ax_xz, tile_a, edgecolor="#1c6ec4",
                      facecolor="#1c6ec4", alpha=0.08, label="child A")
        _draw_bbox_xz(ax_xz, tile_b, edgecolor="#d97706",
                      facecolor="#d97706", alpha=0.08, label="child B")

        total = 0
        for r, xyz in enumerate(per_rank_xyz):
            if xyz.shape[0] == 0:
                continue
            color = [cmap(r % 10)]
            ax_xy.scatter(xyz[:, 0], xyz[:, 1], s=1.0, c=color,
                          alpha=0.55, linewidths=0, label=f"rank {r} ({xyz.shape[0]:,})")
            ax_xz.scatter(xyz[:, 0], xyz[:, 2], s=1.0, c=color,
                          alpha=0.55, linewidths=0)
            total += xyz.shape[0]

        ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y")
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.set_title(f"Merged top-down (XY)  shown={total:,} (sampled)")
        ax_xy.legend(loc="upper right", fontsize=8, markerscale=4)

        ax_xz.set_xlabel("X"); ax_xz.set_ylabel("Z")
        ax_xz.set_aspect("equal", adjustable="datalim")
        ax_xz.set_title("Merged side (XZ)")

        fig.suptitle(
            f"Cat3 OOM merged | tile={tile_id}  iter={iteration}  ranks={len(rank_ply_paths)}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return str(out_path)
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"[oom_viz] merged viz failed: {e}", flush=True)
        return None
