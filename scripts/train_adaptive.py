#!/usr/bin/env python3
"""
Adaptive Tile Training Wrapper Script

This script manages OOM-aware tile-based training:
1. Loads COLMAP data and computes scene extent
2. Manages tile queue (split on OOM)
3. Computes visible cameras for each tile
4. Calls torchrun with computed parameters

Usage:
    python scripts/train_adaptive.py \
        --source_path /path/to/colmap \
        --output_path /path/to/output \
        --num_gpus 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PNG saving
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

# Add paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Grendel-GS"))

# OOM signal file (must match train_internal.py)
OOM_SIGNAL_FILENAME = "oom_signal.json"


def read_expected_counts_from_done_files(ply_dir: Path, num_ranks: int) -> dict:
    """Read expected gaussian counts from done files.

    Returns dict with:
        - 'total': sum of all ranks' total saved
        - 'count_a': sum of all ranks' Child A counts
        - 'count_b': sum of all ranks' Child B counts
        - 'per_rank': dict of {rank: {total, count_a, count_b}}
    """
    result = {
        'total': 0,
        'count_a': 0,
        'count_b': 0,
        'per_rank': {},
        'ranks_found': 0,
    }

    for rank in range(num_ranks):
        done_file = ply_dir / f"oom_done_rank{rank}.json"
        if done_file.exists():
            try:
                with open(done_file) as f:
                    data = json.load(f)
                rank_total = data.get('gaussians_saved', 0)
                rank_a = data.get('count_a', 0)
                rank_b = data.get('count_b', 0)
                result['total'] += rank_total
                result['count_a'] += rank_a
                result['count_b'] += rank_b
                result['per_rank'][rank] = {
                    'total': rank_total,
                    'count_a': rank_a,
                    'count_b': rank_b,
                }
                result['ranks_found'] += 1
            except (json.JSONDecodeError, IOError) as e:
                print(f"[validation] Warning: Failed to read done file for rank {rank}: {e}")

    return result


class MergeError(Exception):
    """Exception raised when PLY merge fails due to missing files."""
    pass


def merge_ply_files(ply_prefix: str, num_ranks: int, output_path: str, strict: bool = True) -> int:
    """
    Merge multiple PLY files from distributed training into a single file.

    When OOM occurs during distributed training, each rank saves its local gaussians
    to separate files: {ply_prefix}_rank0.ply, {ply_prefix}_rank1.ply, etc.
    This function merges them into a single PLY file.

    Args:
        ply_prefix: Path prefix for rank files (without _rank{N}.ply suffix)
        num_ranks: Number of ranks that saved files
        output_path: Path for the merged output file
        strict: If True, raise MergeError if any rank file is missing

    Returns:
        Total number of gaussians in merged file

    Raises:
        MergeError: If strict=True and any rank file is missing
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("[merge_ply] ERROR: plyfile not installed, cannot merge PLY files", flush=True)
        if strict:
            raise MergeError("plyfile not installed")
        return 0

    # First pass: check all files exist (strict mode)
    missing_ranks = []
    for rank in range(num_ranks):
        rank_file = f"{ply_prefix}_rank{rank}.ply"
        if not Path(rank_file).exists():
            missing_ranks.append(rank)

    if missing_ranks:
        print(f"[merge_ply] FATAL: Missing rank files: {missing_ranks}", flush=True)
        print(f"[merge_ply]   Expected {num_ranks} files with prefix: {ply_prefix}", flush=True)
        for rank in missing_ranks:
            print(f"[merge_ply]   Missing: {ply_prefix}_rank{rank}.ply", flush=True)
        if strict:
            raise MergeError(f"Missing {len(missing_ranks)} rank files: ranks {missing_ranks}")
        return 0

    all_vertices = []
    total_count = 0

    for rank in range(num_ranks):
        rank_file = f"{ply_prefix}_rank{rank}.ply"
        try:
            plydata = PlyData.read(rank_file)
            vertices = plydata['vertex']
            count = len(vertices.data)
            all_vertices.append(vertices.data)
            total_count += count
            print(f"[merge_ply] Read rank {rank}: {count:,} gaussians from {rank_file}", flush=True)
        except Exception as e:
            print(f"[merge_ply] ERROR reading {rank_file}: {e}", flush=True)
            if strict:
                raise MergeError(f"Failed to read {rank_file}: {e}")
            continue

    if not all_vertices:
        print("[merge_ply] ERROR: No valid rank files found", flush=True)
        if strict:
            raise MergeError("No valid rank files found")
        return 0

    # Merge all vertices
    merged_data = np.concatenate(all_vertices)
    merged_element = PlyElement.describe(merged_data, 'vertex')

    # Write merged file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([merged_element]).write(output_path)
    print(f"[merge_ply] Merged {total_count:,} gaussians -> {output_path}", flush=True)

    # Verify merged file was created
    if not Path(output_path).exists():
        print(f"[merge_ply] ERROR: Merged file was not created: {output_path}", flush=True)
        if strict:
            raise MergeError(f"Merged file was not created: {output_path}")
        return 0

    return total_count


def effective_tile_iterations(base_iterations: int, num_cameras: int,
                              epoch_cap: int, floor_iters: int = 1500) -> int:
    """타일별 유효 iteration = min(base, max(epoch_cap × 카메라수, floor)).

    표준 3DGS 의 30k iter 는 카메라 수백 장(≈100~200에폭) 기준.
    카메라 6장짜리 타일에 30k(=5000에폭)를 돌리면 훈련뷰 암기(과적합)로
    novel view 가 무너짐 (2026-07-20 tile_0031 실측). 에폭 수를 상한해
    과적합 구간 진입을 차단한다. epoch_cap=0 은 비활성.
    floor 는 SH 램프/densify 초기 구간이 최소한 돌 수 있게 하는 하한.
    """
    if epoch_cap <= 0 or num_cameras <= 0:
        return base_iterations
    cap = max(epoch_cap * num_cameras, floor_iters)
    return min(base_iterations, cap)


def find_final_tile_ply(output_path, tile_id: str):
    """완료 타일의 최종 point_cloud PLY 경로 (없으면 None).

    merge_final_scene 과 wrapper resume 검증이 공유.
    """
    model_dir = Path(output_path) / "models" / tile_id / "point_cloud"
    if not model_dir.exists():
        return None
    iter_folders = sorted(model_dir.glob("iteration_*"),
                          key=lambda x: int(x.name.split("_")[1]))
    if not iter_folders:
        return None
    folder = iter_folders[-1]
    exact = folder / "point_cloud.ply"
    if exact.exists():
        return exact
    candidates = sorted(folder.glob("point_cloud*.ply"))
    return candidates[0] if len(candidates) == 1 else None


def merge_final_scene(output_path, filter_mode: str = "bbox"):
    """완료 타일들의 최종 point_cloud.ply 를 하나의 scene PLY 로 머지.

    adaptive_state.json 의 타일 bbox/status 를 기준으로 함 (학습 재실행 불필요).
    filter_mode="bbox": 각 타일에서 자기 bbox 내부(half-open)의 가우시안만 유지해
      타일 경계의 중복을 제거. 단, 어떤 완료 타일 bbox 에도 속하지 않는 가우시안
      (타일 밖으로 드리프트한 배경 등)은 유지.
    filter_mode="none": 필터 없이 단순 concat.

    Returns: 머지된 PLY 경로 (Path) 또는 실패 시 None
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("[merge_scene] ERROR: plyfile not installed, cannot merge", flush=True)
        return None

    output_path = Path(output_path)
    state_file = output_path / "adaptive_state.json"
    if not state_file.exists():
        print(f"[merge_scene] state file not found: {state_file}", flush=True)
        return None
    with open(state_file) as f:
        state = json.load(f)

    completed = {
        tid: info for tid, info in state.get("tiles", {}).items()
        if info.get("status") == "completed"
    }
    if not completed:
        print("[merge_scene] no completed tiles — nothing to merge", flush=True)
        return None

    def find_final_ply(tile_id: str):
        return find_final_tile_ply(output_path, tile_id)

    bboxes = {tid: BBox.from_string(info["bbox"]) for tid, info in completed.items()}

    def in_bbox(xyz, b):
        return ((xyz[:, 0] >= b.x_min) & (xyz[:, 0] < b.x_max)
                & (xyz[:, 1] >= b.y_min) & (xyz[:, 1] < b.y_max)
                & (xyz[:, 2] >= b.z_min) & (xyz[:, 2] < b.z_max))

    merged_chunks = []
    merged_tids = []
    manifest = {}
    skipped = []
    for tid in sorted(completed.keys()):
        src = find_final_ply(tid)
        if src is None:
            print(f"[merge_scene] WARNING: no final PLY for {tid} — skipped", flush=True)
            skipped.append(tid)
            manifest[tid] = {"source": None}
            continue
        vertices = PlyData.read(str(src))["vertex"].data
        total = len(vertices)
        if filter_mode == "bbox" and total > 0:
            xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
            own = in_bbox(xyz, bboxes[tid])
            inside_any = np.zeros(total, dtype=bool)
            for b in bboxes.values():
                inside_any |= in_bbox(xyz, b)
            keep = own | ~inside_any
        else:
            keep = np.ones(total, dtype=bool)
        kept = int(keep.sum())
        merged_chunks.append(vertices[keep])
        merged_tids.append(tid)
        quality = None
        done_info_path = output_path / "models" / tid / "done_info.json"
        if done_info_path.exists():
            try:
                quality = json.loads(done_info_path.read_text())
            except Exception:
                pass
        manifest[tid] = {"source": str(src), "total": total,
                         "kept": kept, "dropped": total - kept,
                         "quality": quality}
        print(f"[merge_scene] {tid}: kept {kept:,}/{total:,} from {src.name}", flush=True)

    if not merged_chunks:
        print("[merge_scene] ERROR: no tile PLYs found to merge", flush=True)
        return None

    merged = np.concatenate(merged_chunks)
    out_dir = output_path / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "scene_point_cloud.ply"
    PlyData([PlyElement.describe(merged, "vertex")]).write(str(out_ply))
    with open(out_dir / "merge_manifest.json", "w") as f:
        json.dump({
            "filter_mode": filter_mode,
            "num_tiles_merged": len(merged_chunks),
            "num_tiles_skipped": len(skipped),
            "skipped_tiles": skipped,
            "total_gaussians": int(len(merged)),
            "tiles": manifest,
        }, f, indent=2)
    print(f"[merge_scene] merged {len(merged):,} gaussians from "
          f"{len(merged_chunks)} tiles -> {out_ply}", flush=True)
    if skipped:
        print(f"[merge_scene] WARNING: {len(skipped)} completed tiles had no PLY: {skipped}", flush=True)

    # Top-down 병합 분포 시각화 (타일별 색 + 미완료 리프 = 구멍 표시). 실패해도 머지는 유효.
    try:
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=(14, 14))
        cmap = plt.get_cmap("tab20")
        rng = np.random.default_rng(0)
        for i, (tid, verts) in enumerate(zip(merged_tids, merged_chunks)):
            n = len(verts)
            if n == 0:
                continue
            idx = rng.choice(n, size=min(n, 40000), replace=False)
            color = cmap(i % 20)
            ax.scatter(np.asarray(verts["x"])[idx], np.asarray(verts["y"])[idx],
                       s=0.3, color=color, alpha=0.5, linewidths=0, rasterized=True)
            b = bboxes[tid]
            ax.add_patch(Rectangle((b.x_min, b.y_min), b.x_max - b.x_min, b.y_max - b.y_min,
                                   fill=False, edgecolor=color, linewidth=1.2))
            ax.text((b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2,
                    f"{tid}\n{manifest[tid]['kept']:,}",
                    ha="center", va="center", fontsize=7)
        for tid, info in state.get("tiles", {}).items():
            if info.get("status") in ("skipped", "failed"):
                b = BBox.from_string(info["bbox"])
                ax.add_patch(Rectangle((b.x_min, b.y_min), b.x_max - b.x_min, b.y_max - b.y_min,
                                       fill=False, edgecolor="red", linestyle="--", linewidth=1.5))
                ax.text((b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2,
                        f"{tid}\n({info['status']})",
                        ha="center", va="center", fontsize=7, color="red")
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Merged scene top-down | tiles={len(merged_chunks)} "
                     f"gaussians={len(merged):,} filter={filter_mode} "
                     f"(red dashed = skipped/failed hole)")
        png_path = out_dir / "scene_merged_topdown.png"
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[merge_scene] top-down viz -> {png_path}", flush=True)
    except Exception as viz_e:
        print(f"[merge_scene] WARNING: top-down viz failed: {viz_e}", flush=True)

    # 타일 품질 히트맵 — 타일 간 편차 가시화 (final_epoch_loss: 초록=좋음, 빨강=나쁨)
    try:
        from matplotlib.patches import Rectangle
        import matplotlib.colors as mcolors
        losses = {tid: m["quality"]["final_epoch_loss"]
                  for tid, m in manifest.items()
                  if m.get("quality") and m["quality"].get("final_epoch_loss") is not None}
        fig, ax = plt.subplots(figsize=(14, 14))
        cmap = plt.get_cmap("RdYlGn_r")
        norm = None
        if losses:
            vmin, vmax = min(losses.values()), max(losses.values())
            if vmax > vmin:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        all_b = []
        for tid in merged_tids:
            b = bboxes[tid]
            all_b.append(b)
            loss = losses.get(tid)
            if loss is None:
                color = (0.85, 0.85, 0.85, 1.0)  # 지표 없음(구버전 타일) = 회색
            elif norm is None:
                color = cmap(0.5)
            else:
                color = cmap(norm(loss))
            ax.add_patch(Rectangle((b.x_min, b.y_min), b.x_max - b.x_min,
                                   b.y_max - b.y_min, facecolor=color,
                                   edgecolor="black", linewidth=1.0, alpha=0.9))
            q = manifest[tid].get("quality") or {}
            label = f"{tid}\n" + (f"loss={loss:.4f}" if loss is not None else "(n/a)")
            if q:
                label += f"\ncams={q.get('num_cameras', '?')} {q.get('done_reason', '')}"
            ax.text((b.x_min + b.x_max) / 2, (b.y_min + b.y_max) / 2, label,
                    ha="center", va="center", fontsize=7)
        for tid, info in state.get("tiles", {}).items():
            if info.get("status") in ("skipped", "failed"):
                b = BBox.from_string(info["bbox"])
                all_b.append(b)
                ax.add_patch(Rectangle((b.x_min, b.y_min), b.x_max - b.x_min,
                                       b.y_max - b.y_min, fill=False,
                                       edgecolor="red", linestyle="--", linewidth=1.5))
        if all_b:
            ax.set_xlim(min(b.x_min for b in all_b), max(b.x_max for b in all_b))
            ax.set_ylim(min(b.y_min for b in all_b), max(b.y_max for b in all_b))
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if losses:
            vals = sorted(losses.values())
            spread = (vals[-1] - vals[0]) / max(vals[0], 1e-9) * 100
            ax.set_title(f"Tile quality heatmap (final epoch loss) | "
                         f"best={vals[0]:.4f} worst={vals[-1]:.4f} 편차={spread:.0f}%")
        else:
            ax.set_title("Tile quality heatmap (지표 없음 — 구버전 타일)")
        qpng = out_dir / "scene_quality_heatmap.png"
        fig.savefig(qpng, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[merge_scene] quality heatmap -> {qpng}", flush=True)
    except Exception as viz_e:
        print(f"[merge_scene] WARNING: quality heatmap failed: {viz_e}", flush=True)

    return out_ply


@dataclass
class BBox:
    """3D bounding box."""
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def size(self) -> Tuple[float, float, float]:
        return (self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min)

    def to_string(self) -> str:
        return f"{self.x_min},{self.y_min},{self.z_min},{self.x_max},{self.y_max},{self.z_max}"

    @classmethod
    def from_string(cls, s: str) -> "BBox":
        parts = [float(x) for x in s.split(",")]
        return cls(*parts)

    def split(self) -> Tuple["BBox", "BBox"]:
        """Split along longest axis (X or Y only, never Z)."""
        dx, dy, dz = self.size
        # Only split along X or Y axis, not Z
        if dx >= dy:
            mid = (self.x_min + self.x_max) / 2
            return (
                BBox(self.x_min, self.y_min, self.z_min, mid, self.y_max, self.z_max),
                BBox(mid, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max),
            )
        else:
            mid = (self.y_min + self.y_max) / 2
            return (
                BBox(self.x_min, self.y_min, self.z_min, self.x_max, mid, self.z_max),
                BBox(self.x_min, mid, self.z_min, self.x_max, self.y_max, self.z_max),
            )


@dataclass
class TileInfo:
    """Information about a tile."""
    tile_id: str
    bbox: BBox
    status: str  # pending, in_progress, completed, split, failed, skipped
    fail_iter: Optional[int] = None  # Iteration at which tile failed/split
    num_cameras: Optional[int] = None  # Number of visible cameras
    ply_path: Optional[str] = None  # Path to pre-trained gaussians PLY (for Category 3 OOM resume)
    oom_type: Optional[str] = None  # "gpu" (exit 42) or "ram" (exit -9)
    oom_category: Optional[int] = None  # OOM category: 1=early, 2=mid, 3=densification
    retry_count: int = 0  # wrapper 재시작(resume) 시 failed → pending 강등 횟수
    quality: Optional[dict] = None  # done 시점 품질 지표 (final_epoch_loss 등, done_info.json)


class AdaptiveTileTrainer:
    """
    Manages adaptive tile-based training.

    Computes tile info and visible cameras in Python,
    then calls torchrun with the computed parameters.
    """

    EXIT_CODE_SUCCESS = 0
    EXIT_CODE_OOM = 42

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source_path = Path(args.source_path)
        self.output_path = Path(args.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.grendel_dir = ROOT / "Grendel-GS"
        self.state_file = self.output_path / "adaptive_state.json"
        self.ply_dir = self.output_path / "ply"
        self.ply_dir.mkdir(parents=True, exist_ok=True)

        # Resume 판단: state 파일이 있고 --fresh_start 가 아니면 이어서 진행
        # (무인 장기 런에서 wrapper/호스트가 죽어도 재실행만으로 이어지게 하는 것이 목적)
        loaded_state = None
        if self.state_file.exists() and not getattr(args, "fresh_start", False):
            loaded_state = self._try_load_state()
        self.resume_mode = loaded_state is not None

        # fresh 시작일 때만 visualizations 폴더 전체 삭제 후 재생성
        # (count_timeline / cat3_oom / resume / tile_map / tile_split_tree / projection_debug /
        #  gt_compare / render_debug_json / render_debug_png 모두 이 아래에 만들어짐)
        import shutil
        self.viz_root = self.output_path / "visualizations"
        if not self.resume_mode and self.viz_root.exists():
            shutil.rmtree(self.viz_root)
            print(f"[Init] Cleared folder: {self.viz_root}")
        self.viz_root.mkdir(parents=True, exist_ok=True)

        # 후속 코드 호환: 기존 attribute 명 유지 (이제 viz_root 하위)
        self.debug_images_dir = self.viz_root / "gt_compare"
        self.debug_images_dir.mkdir(parents=True, exist_ok=True)
        self.projection_debug_dir = self.viz_root / "projection_debug"
        # projection_debug 는 필요할 때 생성

        # Load scene info
        self.scene_bbox, self.num_points = self._load_scene_info()
        self.cam_infos = self._load_camera_infos()

        # Load or initialize tile queue
        self.tiles: Dict[str, TileInfo] = {}
        self.tile_counter = 0
        self.vis_counter = 0  # Counter for visualization filenames
        self.min_successful_level = None  # Track minimum level (smallest = largest tile) that completed successfully
        self.initial_area = None  # Area of the initial tile (level 0)
        if self.resume_mode:
            self._apply_resume_state(loaded_state)
            self._vis_cleared = True  # run() 의 lazy clear 방지
        else:
            # Fresh start: clear all output folders
            self._clear_output_folders()
            self._init_tiles()

    def _load_scene_info(self) -> Tuple[BBox, int]:
        """Load point cloud and compute scene bounding box."""
        from plyfile import PlyData

        ply_path = self.source_path / "sparse" / "0" / "points3D.ply"
        if not ply_path.exists():
            # Try to convert from bin
            bin_path = self.source_path / "sparse" / "0" / "points3D.bin"
            if bin_path.exists():
                from scene.colmap_loader import read_points3D_binary
                xyz, rgb, _ = read_points3D_binary(str(bin_path))
            else:
                raise FileNotFoundError(f"No point cloud found in {self.source_path}")
        else:
            ply = PlyData.read(str(ply_path))
            vertex = ply['vertex']
            xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

        # 축별로 stable percentile 을 자동 탐색해 SfM outlier 를 제거.
        # 알고리즘: 후보 percentile 마다 "percentile 1% 늘어날 때 size 가 줄어드는 rate"
        # 를 계산. outlier 제거 구간에서는 rate 가 매우 큼 → 정상 데이터 trim 구간으로
        # 들어서면 rate 가 평탄해진다. median rate × 1.5 미만이 되는 첫 percentile 이
        # 그 축의 stable 지점.
        candidates = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

        def stable_axis_bounds(vals_1d):
            """Return (chosen_pct, lo, hi)."""
            sizes, los, his = [], [], []
            for p in candidates:
                lo = float(np.percentile(vals_1d, p))
                hi = float(np.percentile(vals_1d, 100 - p))
                los.append(lo); his.append(hi)
                sizes.append(hi - lo)
            # rate[i] = (size[i]-size[i+1]) / (candidates[i+1]-candidates[i])
            rates = []
            for i in range(len(candidates) - 1):
                dp = candidates[i + 1] - candidates[i]
                ds = sizes[i] - sizes[i + 1]
                rates.append(ds / dp if dp > 0 else 0.0)
            if not rates:
                return candidates[0], los[0], his[0]
            sorted_rates = sorted(rates)
            median_rate = sorted_rates[len(sorted_rates) // 2]
            threshold = 1.5 * max(median_rate, 1e-9)
            for i, rate in enumerate(rates):
                if rate <= threshold:
                    return candidates[i], los[i], his[i]
            # 모든 rate 가 threshold 초과 (이론상 거의 없음) → 가장 큰 candidate
            return candidates[-1], los[-1], his[-1]

        xyz_min_full = xyz.min(axis=0)
        xyz_max_full = xyz.max(axis=0)
        chosen_pct = [0.0, 0.0, 0.0]
        clean_lo = [0.0, 0.0, 0.0]
        clean_hi = [0.0, 0.0, 0.0]
        for ax in range(3):
            p, lo, hi = stable_axis_bounds(xyz[:, ax])
            chosen_pct[ax] = p
            clean_lo[ax] = lo
            clean_hi[ax] = hi

        bbox = BBox(
            clean_lo[0], clean_lo[1], clean_lo[2],
            clean_hi[0], clean_hi[1], clean_hi[2],
        )

        axis_names = ["X", "Y", "Z"]
        print(f"[Scene] Loaded {len(xyz)} points")
        print(f"[Scene] Extent (per-axis auto-percentile, outliers excluded):")
        for ax in range(3):
            print(f"  {axis_names[ax]}: {clean_lo[ax]:.3f} ~ {clean_hi[ax]:.3f} "
                  f"(size: {clean_hi[ax]-clean_lo[ax]:.3f}, pct: {chosen_pct[ax]:.2f}%-{100-chosen_pct[ax]:.2f}%)")
        print(f"  Full extent (with outliers):")
        for ax in range(3):
            full_size = float(xyz_max_full[ax] - xyz_min_full[ax])
            print(f"  {axis_names[ax]}: {float(xyz_min_full[ax]):.3f} ~ {float(xyz_max_full[ax]):.3f} (size: {full_size:.3f})")

        return bbox, len(xyz)

    def _load_camera_infos(self) -> List[Dict[str, Any]]:
        """Load camera information from COLMAP."""
        from scene.colmap_loader import (
            read_extrinsics_binary, read_extrinsics_text,
            read_intrinsics_binary,
            qvec2rotmat, Camera
        )
        from utils.graphics_utils import focal2fov

        sparse_path = self.source_path / "sparse" / "0"

        # Load extrinsics (images.bin or images.txt)
        images_bin = sparse_path / "images.bin"
        images_txt = sparse_path / "images.txt"
        if images_bin.exists():
            cameras_extrinsic = read_extrinsics_binary(str(images_bin))
        elif images_txt.exists():
            cameras_extrinsic = read_extrinsics_text(str(images_txt))
        else:
            raise FileNotFoundError(f"No images.bin or images.txt in {sparse_path}")

        # Load intrinsics (cameras.bin preferred - no PINHOLE restriction)
        cameras_bin = sparse_path / "cameras.bin"
        cameras_txt = sparse_path / "cameras.txt"
        if cameras_bin.exists():
            cameras_intrinsic = read_intrinsics_binary(str(cameras_bin))
        elif cameras_txt.exists():
            # Use flexible text reader (no PINHOLE assertion)
            cameras_intrinsic = {}
            with open(cameras_txt, "r") as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0 and line[0] != "#":
                        elems = line.split()
                        camera_id = int(elems[0])
                        model = elems[1]
                        width = int(elems[2])
                        height = int(elems[3])
                        params = np.array([float(x) for x in elems[4:]])
                        cameras_intrinsic[camera_id] = Camera(
                            id=camera_id, model=model, width=width, height=height, params=params
                        )
        else:
            raise FileNotFoundError(f"No cameras.bin or cameras.txt in {sparse_path}")

        cam_infos = []
        for idx, key in enumerate(sorted(cameras_extrinsic.keys())):
            extr = cameras_extrinsic[key]
            intr = cameras_intrinsic[extr.camera_id]

            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                focal_x = intr.params[0]
                focal_y = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model in ["PINHOLE", "OPENCV"]:
                focal_x = intr.params[0]
                focal_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
            else:
                focal_x = intr.params[0]
                focal_y = intr.params[0]
                cx = intr.width / 2.0
                cy = intr.height / 2.0

            fov_x = focal2fov(focal_x, intr.width)
            fov_y = focal2fov(focal_y, intr.height)

            # Strip extension to match Scene's camera naming convention
            image_name = os.path.splitext(extr.name)[0]
            cam_infos.append({
                "idx": idx,
                "image_name": image_name,
                "R": R,
                "T": T,
                "fov_x": fov_x,
                "fov_y": fov_y,
                "focal_x": focal_x,
                "focal_y": focal_y,
                "cx": cx,
                "cy": cy,
                "width": intr.width,
                "height": intr.height,
            })

        print(f"[Scene] Loaded {len(cam_infos)} cameras")
        return cam_infos

    def _generate_visual_debug_for_tile(self, tile_bbox: BBox, tile_id: str):
        """Generate visual debug images for a newly created tile."""
        if not self.args.visual_debug or self.args.visual_debug_only:
            # Skip if visual debug is disabled or in visual-debug-only mode
            return
            
        print(f"  [Visual Debug] Generating debug images for {tile_id}...")
        visible_cameras = self._compute_visible_cameras(
            tile_bbox, margin=self.args.tile_crop_margin,
            tile_id=tile_id, visual_debug=True
        )
        if visible_cameras:
            debug_dir = self.output_path / "visualizations" / "projection_debug" / tile_id
            if debug_dir.exists():
                print(f"  [Visual Debug] Generated debug images in {debug_dir}")
        
    def _compute_visible_cameras(self, tile_bbox: BBox, margin: int = 100, 
                                tile_id: str = None, visual_debug: bool = False) -> List[str]:
        """
        Compute which cameras can see the tile.

        Returns list of visible camera image names.
        """
        import math
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix

        visible = []
        
        # Setup visual debug if requested
        if visual_debug and tile_id:
            visual_debug_dir = self.output_path / "visualizations" / "projection_debug" / tile_id
            # Remove existing debug directory and recreate it
            import shutil
            if visual_debug_dir.exists():
                shutil.rmtree(visual_debug_dir)
            visual_debug_dir.mkdir(parents=True, exist_ok=True)
        
        corners = np.array([
            [tile_bbox.x_min, tile_bbox.y_min, tile_bbox.z_min],
            [tile_bbox.x_min, tile_bbox.y_min, tile_bbox.z_max],
            [tile_bbox.x_min, tile_bbox.y_max, tile_bbox.z_min],
            [tile_bbox.x_min, tile_bbox.y_max, tile_bbox.z_max],
            [tile_bbox.x_max, tile_bbox.y_min, tile_bbox.z_min],
            [tile_bbox.x_max, tile_bbox.y_min, tile_bbox.z_max],
            [tile_bbox.x_max, tile_bbox.y_max, tile_bbox.z_min],
            [tile_bbox.x_max, tile_bbox.y_max, tile_bbox.z_max],
        ], dtype=np.float32)

        for cam in self.cam_infos:
            # Build full_proj_transform
            Rt = getWorld2View2(cam["R"], cam["T"])
            world_view = Rt.T  # transpose
            proj = getProjectionMatrix(
                znear=0.01, zfar=100.0,
                fovX=cam["fov_x"], fovY=cam["fov_y"]
            ).cpu().numpy().T  # transpose
            full_proj = world_view @ proj

            # Project corners
            corners_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
            clip = corners_h @ full_proj

            w = clip[:, 3]
            valid = w > 0.001
            if not np.any(valid):
                continue

            # NDC coordinates
            ndc = np.zeros((8, 2))
            ndc[valid] = clip[valid, :2] / w[valid, np.newaxis]

            # Pixel coordinates
            px = (ndc[:, 0] + 1.0) * 0.5 * cam["width"]
            py = (ndc[:, 1] + 1.0) * 0.5 * cam["height"]

            # Check if any valid corner projects into image (with margin)
            valid_px = px[valid]
            valid_py = py[valid]

            x_min = valid_px.min() - margin
            x_max = valid_px.max() + margin
            y_min = valid_py.min() - margin
            y_max = valid_py.max() + margin

            # Check overlap with image
            if x_max > 0 and x_min < cam["width"] and y_max > 0 and y_min < cam["height"]:
                visible.append(cam["image_name"])
                
                # Save visual debug if requested
                if visual_debug and tile_id:
                    self._save_visual_debug(cam, tile_bbox, tile_id, x_min, x_max, y_min, y_max, visual_debug_dir)

        return visible
    
    def _save_visual_debug(self, cam, tile_bbox, tile_id, x_min, x_max, y_min, y_max, visual_debug_dir):
        """Save visual debug using the new side-by-side visualization."""
        try:
            # Import the new visualization function
            import os
            import numpy as np
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Grendel-GS', 'scene'))
            from adaptive_tile_utils import save_projection_debug_visualization, TileBBox, CropRegion
            from utils.graphics_utils import getWorld2View2, getProjectionMatrix
            
            # Convert to proper format
            from src.tile_storage import BBox as StorageBBox
            
            # Current tile bbox
            tile_storage_bbox = StorageBBox(
                min_xyz=np.array([tile_bbox.x_min, tile_bbox.y_min, tile_bbox.z_min]),
                max_xyz=np.array([tile_bbox.x_max, tile_bbox.y_max, tile_bbox.z_max])
            )
            tile_bbox_3d = TileBBox(
                tile_storage_bbox.min[0], tile_storage_bbox.min[1], tile_storage_bbox.min[2],
                tile_storage_bbox.max[0], tile_storage_bbox.max[1], tile_storage_bbox.max[2]
            )
            
            # Full scene bbox for fixed 3D view
            full_scene_storage_bbox = StorageBBox(
                min_xyz=np.array([self.scene_bbox.x_min, self.scene_bbox.y_min, self.scene_bbox.z_min]),
                max_xyz=np.array([self.scene_bbox.x_max, self.scene_bbox.y_max, self.scene_bbox.z_max])
            )
            full_scene_bbox_3d = TileBBox(
                full_scene_storage_bbox.min[0], full_scene_storage_bbox.min[1], full_scene_storage_bbox.min[2],
                full_scene_storage_bbox.max[0], full_scene_storage_bbox.max[1], full_scene_storage_bbox.max[2]
            )
            
            # Create crop region
            crop = CropRegion(
                x_min=max(0, int(x_min)),
                y_min=max(0, int(y_min)),
                x_max=min(cam["width"], int(x_max)),
                y_max=min(cam["height"], int(y_max))
            )
            
            # Build projection matrix
            Rt = getWorld2View2(cam["R"], cam["T"])
            world_view = Rt.T
            proj = getProjectionMatrix(
                znear=0.01, zfar=100.0,
                fovX=cam["fov_x"], fovY=cam["fov_y"]
            ).cpu().numpy().T
            full_proj = world_view @ proj
            
            # Create camera info object
            class CamInfo:
                def __init__(self, cam_dict):
                    self.R = cam_dict["R"]
                    self.T = cam_dict["T"] 
                    self.width = cam_dict["width"]
                    self.height = cam_dict["height"]
                    self.image_name = cam_dict["image_name"]
                    self.cx = cam_dict.get("cx", None)
                    self.cy = cam_dict.get("cy", None)
            
            cam_info = CamInfo(cam)
            
            # Calculate fixed image bounds to include all corner projections across all cameras
            # Use FULL SCENE corners (not current tile) for consistent bounds
            corners = np.array([
                [self.scene_bbox.x_min, self.scene_bbox.y_min, self.scene_bbox.z_min],
                [self.scene_bbox.x_min, self.scene_bbox.y_min, self.scene_bbox.z_max],
                [self.scene_bbox.x_min, self.scene_bbox.y_max, self.scene_bbox.z_min],
                [self.scene_bbox.x_min, self.scene_bbox.y_max, self.scene_bbox.z_max],
                [self.scene_bbox.x_max, self.scene_bbox.y_min, self.scene_bbox.z_min],
                [self.scene_bbox.x_max, self.scene_bbox.y_min, self.scene_bbox.z_max],
                [self.scene_bbox.x_max, self.scene_bbox.y_max, self.scene_bbox.z_min],
                [self.scene_bbox.x_max, self.scene_bbox.y_max, self.scene_bbox.z_max],
            ], dtype=np.float32)
            
            # Track min/max projected coordinates across all cameras
            all_px_min, all_py_min = float('inf'), float('inf')
            all_px_max, all_py_max = float('-inf'), float('-inf')
            
            # Also track max image dimensions
            max_width = 0
            max_height = 0
            
            for cam_dict in self.cam_infos:
                max_width = max(max_width, cam_dict["width"])
                max_height = max(max_height, cam_dict["height"])
                
                # Build projection matrix for this camera
                Rt_cam = getWorld2View2(cam_dict["R"], cam_dict["T"])
                world_view_cam = Rt_cam.T
                proj_cam = getProjectionMatrix(
                    znear=0.01, zfar=100.0,
                    fovX=cam_dict["fov_x"], fovY=cam_dict["fov_y"]
                ).cpu().numpy().T
                full_proj_cam = world_view_cam @ proj_cam
                
                # Project corners
                corners_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
                clip = corners_h @ full_proj_cam
                
                w = clip[:, 3]
                valid = w > 0.001
                if np.any(valid):
                    # NDC to pixel coordinates
                    ndc = np.zeros((8, 2))
                    ndc[valid] = clip[valid, :2] / w[valid, np.newaxis]
                    
                    px = (ndc[:, 0] + 1.0) * 0.5 * cam_dict["width"]
                    py = (ndc[:, 1] + 1.0) * 0.5 * cam_dict["height"]
                    
                    # Update global min/max (including invalid/behind-camera points)
                    all_px_min = min(all_px_min, px.min())
                    all_px_max = max(all_px_max, px.max())
                    all_py_min = min(all_py_min, py.min())
                    all_py_max = max(all_py_max, py.max())
            
            # Add margin to ensure all corners are visible
            margin_ratio = 0.2  # 20% additional margin
            x_range = all_px_max - all_px_min
            y_range = all_py_max - all_py_min
            
            # Use the larger of: projection range or image size
            x_span = max(x_range, max_width)
            y_span = max(y_range, max_height)
            
            margin_x = x_span * margin_ratio
            margin_y = y_span * margin_ratio
            
            # Set bounds to include all projected corners with margin
            fixed_image_bounds = (
                min(all_px_min - margin_x, -margin_x),
                min(all_py_min - margin_y, -margin_y),
                max(all_px_max + margin_x, max_width + margin_x),
                max(all_py_max + margin_y, max_height + margin_y)
            )
            
            # Use new visualization with all cameras
            save_projection_debug_visualization(
                tile_bbox_3d, cam_info, crop, full_proj,
                str(visual_debug_dir), tile_id, 0, 
                full_scene_bbox=full_scene_bbox_3d,
                all_cameras=self.cam_infos,
                fixed_image_bounds=fixed_image_bounds
            )
            
        except Exception as e:
            print(f"[Visual Debug] Failed to save visualization: {e}")
            import traceback
            traceback.print_exc()

    def _clear_visualizations(self):
        """Clear visualizations folder."""
        import shutil
        vis_dir = self.output_path / "visualizations"
        if vis_dir.exists():
            shutil.rmtree(vis_dir)
            print(f"[Init] Cleared visualizations folder: {vis_dir}")
        vis_dir.mkdir(parents=True, exist_ok=True)
        self.vis_counter = 0  # Reset counter

    def _clear_output_folders(self):
        """Clear all output folders when not resuming (fresh start).

        visualizations 는 __init__ 에서 이미 청소하고 gt_compare 등 하위
        폴더를 만들어 둔 상태이므로 여기서는 건드리지 않는다.
        """
        import shutil
        folders_to_clear = ["models", "logs", "ply"]
        for folder_name in folders_to_clear:
            folder_path = self.output_path / folder_name
            if folder_path.exists():
                shutil.rmtree(folder_path)
                print(f"[Init] Cleared folder: {folder_path}")
        # Also remove state file if exists
        if self.state_file.exists():
            self.state_file.unlink()
            print(f"[Init] Removed state file: {self.state_file}")
        self._vis_cleared = True

    def _init_tiles(self):
        """Initialize with single tile covering entire scene."""
        margin = self.args.scene_margin
        dx, dy, dz = self.scene_bbox.size
        initial_bbox = BBox(
            self.scene_bbox.x_min - dx * margin,
            self.scene_bbox.y_min - dy * margin,
            self.scene_bbox.z_min - dz * margin,
            self.scene_bbox.x_max + dx * margin,
            self.scene_bbox.y_max + dy * margin,
            self.scene_bbox.z_max + dz * margin,
        )

        # Store initial area for level calculation
        self.initial_area = initial_bbox.size[0] * initial_bbox.size[1]

        tile_id = self._next_tile_id()
        self.tiles[tile_id] = TileInfo(
            tile_id=tile_id,
            bbox=initial_bbox,
            status="pending"
        )
        self._save_state()

    def _next_tile_id(self) -> str:
        tile_id = f"tile_{self.tile_counter:04d}"
        self.tile_counter += 1
        return tile_id

    def _get_tile_level(self, tile_area: float) -> int:
        """
        Calculate tile level based on area.

        Level 0 = initial tile (full scene)
        Level 1 = half area (2 tiles)
        Level 2 = quarter area (4 tiles)
        Level n = 1/(2^n) area

        Returns integer level (rounded).
        """
        import math
        if self.initial_area is None or self.initial_area <= 0:
            return 0
        if tile_area <= 0:
            return 0
        ratio = self.initial_area / tile_area
        if ratio <= 1:
            return 0
        return int(round(math.log2(ratio)))

    def _save_state(self):
        """Save current state to file."""
        state = {
            "source_path": str(self.source_path),
            "tile_counter": self.tile_counter,
            "vis_counter": self.vis_counter,
            "initial_area": self.initial_area,
            "min_successful_level": self.min_successful_level,
            "tiles": {
                tid: {
                    "tile_id": t.tile_id,
                    "bbox": t.bbox.to_string(),
                    "status": t.status,
                    "fail_iter": t.fail_iter,
                    "num_cameras": t.num_cameras,
                    "ply_path": t.ply_path,
                    "oom_type": t.oom_type,
                    "oom_category": t.oom_category,
                    "retry_count": t.retry_count,
                    "quality": t.quality,
                }
                for tid, t in self.tiles.items()
            }
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    MAX_TILE_RETRIES = 2  # resume 시 failed 타일을 pending 으로 되돌리는 최대 횟수

    def _try_load_state(self):
        """기존 adaptive_state.json 을 읽어 resume 가능하면 dict 반환.

        파싱 실패·source_path 불일치·빈 상태면 파일을 .bak 으로 밀어두고
        None 을 반환해 fresh start 로 진행 (무인 운영에서 절대 멈추지 않기 위함).
        """
        import time
        import shutil
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            saved_source = state.get("source_path")
            if saved_source and saved_source != str(self.source_path):
                print(f"[Resume] source_path 불일치: state={saved_source}, "
                      f"현재={self.source_path} — fresh start 로 전환")
                raise ValueError("source_path mismatch")
            if not state.get("tiles"):
                raise ValueError("state has no tiles")
            return state
        except Exception as e:
            backup = self.state_file.parent / f"adaptive_state.json.bak_{int(time.time())}"
            try:
                shutil.move(str(self.state_file), str(backup))
                print(f"[Resume] state 사용 불가 ({e}) — {backup.name} 으로 백업 후 fresh start")
            except Exception:
                pass
            return None

    def _apply_resume_state(self, state: dict):
        """로드된 state 로 타일 큐 복원 + 검증/강등.

        - completed 인데 최종 PLY 가 없으면 pending 으로 강등 (재학습)
        - in_progress (크래시 당시 학습 중) 는 pending 으로 강등
        - failed 는 retry_count < MAX_TILE_RETRIES 일 때만 pending 으로 강등
        - pending 인데 pre-trained PLY 파일이 사라졌으면 스크래치로 전환 (진행 보장)
        """
        self.tile_counter = state.get("tile_counter", 0)
        self.vis_counter = state.get("vis_counter", 0)
        self.initial_area = state.get("initial_area")
        self.min_successful_level = state.get("min_successful_level")

        kept_completed = 0
        demoted = []
        missing_final = []
        lost_pretrained = []
        for tid, t in state["tiles"].items():
            info = TileInfo(
                tile_id=t["tile_id"],
                bbox=BBox.from_string(t["bbox"]),
                status=t["status"],
                fail_iter=t.get("fail_iter"),
                num_cameras=t.get("num_cameras"),
                ply_path=t.get("ply_path"),
                oom_type=t.get("oom_type"),
                oom_category=t.get("oom_category"),
                retry_count=t.get("retry_count", 0),
                quality=t.get("quality"),
            )
            if info.status == "completed":
                if find_final_tile_ply(self.output_path, tid) is None:
                    info.status = "pending"
                    missing_final.append(tid)
                else:
                    kept_completed += 1
            elif info.status == "in_progress":
                info.status = "pending"
                demoted.append(tid)
            elif info.status == "failed" and info.retry_count < self.MAX_TILE_RETRIES:
                info.retry_count += 1
                info.status = "pending"
                demoted.append(f"{tid}(retry {info.retry_count}/{self.MAX_TILE_RETRIES})")
            if (info.status == "pending" and info.ply_path
                    and not Path(info.ply_path).exists()):
                lost_pretrained.append(tid)
                info.ply_path = None
            self.tiles[tid] = info

        from collections import Counter
        counts = Counter(t.status for t in self.tiles.values())
        print("\n" + "=" * 60)
        print("[Resume] 기존 state 에서 이어서 진행")
        print(f"  타일 {len(self.tiles)}개: " +
              ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(f"  완료 유지(최종 PLY 확인됨): {kept_completed}")
        if missing_final:
            print(f"  완료였으나 PLY 소실 → 재학습: {missing_final}")
        if demoted:
            print(f"  pending 으로 강등: {demoted}")
        if lost_pretrained:
            print(f"  pre-trained PLY 소실 → 스크래치 전환: {lost_pretrained}")
        print("=" * 60 + "\n")
        self._save_state()

    def _compute_camera_footprints(self) -> List[Optional[np.ndarray]]:
        """
        Compute the XY-plane footprint of each camera's frustum.

        For each camera, unprojects the 4 image corners into world-space rays,
        then intersects them with Z = z_mid (median Z of scene bbox).

        Returns:
            List of Nx2 arrays (the XY polygon vertices), or None for cameras
            whose rays do not intersect the Z plane.
        """
        z_mid = (self.scene_bbox.z_min + self.scene_bbox.z_max) / 2.0
        footprints = []

        for cam in self.cam_infos:
            R = cam["R"]       # R_c2w (transposed from COLMAP's R_w2c)
            T = cam["T"]       # COLMAP tvec
            fx = cam["focal_x"]
            fy = cam["focal_y"]
            cx = cam["cx"]
            cy = cam["cy"]
            W = cam["width"]
            H = cam["height"]

            # Camera position in world space: -R_c2w @ T
            cam_pos = -R @ T

            # 4 image corners
            corners_px = np.array([
                [0, 0],
                [W, 0],
                [W, H],
                [0, H],
            ], dtype=np.float64)

            # Unproject to camera-space ray directions
            ray_dirs_cam = np.stack([
                (corners_px[:, 0] - cx) / fx,
                (corners_px[:, 1] - cy) / fy,
                np.ones(4),
            ], axis=1)  # (4, 3)

            # Transform to world-space ray directions
            ray_dirs_world = (R @ ray_dirs_cam.T).T  # (4, 3)

            # Intersect each ray with Z = z_mid plane
            # cam_pos[2] + t * ray_dir_z = z_mid  =>  t = (z_mid - cam_pos[2]) / ray_dir_z
            dz = ray_dirs_world[:, 2]
            valid = np.abs(dz) > 1e-8
            t_vals = np.where(valid, (z_mid - cam_pos[2]) / dz, -1.0)

            # All 4 rays must hit the plane with t > 0 (camera looking toward the plane)
            if not np.all(t_vals > 0):
                footprints.append(None)
                continue

            # Intersection points
            points = cam_pos[np.newaxis, :] + t_vals[:, np.newaxis] * ray_dirs_world  # (4, 3)
            footprints.append(points[:, :2])  # Only XY

        return footprints

    def _visualize_tile_map(self, current_tile_id: str = None):
        """
        Visualize tile map as a 2D X-Y plot and save to PNG.

        Args:
            current_tile_id: The tile currently being processed (highlighted)
        """
        # Status colors (for status indicator circles)
        # Note: "split" is further divided based on fail_iter and oom_type
        status_colors = {
            "completed": "#4CAF50",           # Green
            "in_progress": "#2196F3",         # Blue
            "pending": "#9E9E9E",             # Gray
            "split_cat1": "#FFEB3B",          # Yellow - Cat 1: early OOM (SSIM on large image)
            "split_cat2": "#FF9800",          # Orange - Cat 2: mid-training OOM
            "split_cat3": "#F44336",          # Red - Cat 3: densification OOM (gaussians saved)
            "split_cat4": "#FF5722",          # Deep Orange - Cat 4: pre-trained too large (reuse parent PLY)
            "split_ram_oom": "#E91E63",       # Pink - RAM OOM (exit -9)
            "split_preemptive": "#9C27B0",    # Purple - Preemptive split (skipped without trying)
            "split_gpu_oom": "#FF9800",       # Orange - fallback for unknown category
            "split": "#FF9800",               # Orange - fallback for old data
            "failed": "#F44336",              # Red
            "skipped": "#795548",             # Brown
        }

        # Generate tile colors using golden ratio for better distribution
        # This ensures adjacent tile numbers get visually distinct colors
        def generate_tile_color(tile_num: int) -> str:
            """Generate a distinct color for a tile using golden ratio hue distribution."""
            golden_ratio = 0.618033988749895
            hue = (tile_num * golden_ratio) % 1.0
            # Convert HSV to RGB (saturation=0.7, value=0.9 for vibrant but not harsh colors)
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        fig, ax = plt.subplots(1, 1, figsize=(14, 12))

        # Draw scene bbox as background
        scene_rect = patches.Rectangle(
            (self.scene_bbox.x_min, self.scene_bbox.y_min),
            self.scene_bbox.size[0], self.scene_bbox.size[1],
            linewidth=2, edgecolor='black', facecolor='#f5f5f5',
            linestyle='--', label='Scene extent'
        )
        ax.add_patch(scene_rect)

        # Draw camera frustum footprints on the XY plane (behind tiles)
        from matplotlib.patches import Polygon as MplPolygon
        footprints = self._compute_camera_footprints()
        num_valid_footprints = sum(1 for fp in footprints if fp is not None)
        cam_cmap = plt.cm.tab20
        num_cams = len(self.cam_infos)
        for i, (cam, fp) in enumerate(zip(self.cam_infos, footprints)):
            if fp is None:
                continue
            color = cam_cmap(i / max(num_cams - 1, 1))
            poly = MplPolygon(
                fp, closed=True,
                facecolor=(*color[:3], 0.07),
                edgecolor=(*color[:3], 0.3),
                linewidth=0.5,
                zorder=2,
            )
            ax.add_patch(poly)
            # Small marker at camera position
            R = cam["R"]
            T = cam["T"]
            cam_pos = -R @ T
            ax.plot(cam_pos[0], cam_pos[1], '.', color=color, markersize=2, zorder=3)

        # Calculate max tile size for linewidth scaling
        max_tile_area = max(t.bbox.size[0] * t.bbox.size[1] for t in self.tiles.values())
        import math

        # Sort tiles by size (largest first, so smaller tiles draw on top)
        sorted_tiles = sorted(self.tiles.items(),
                              key=lambda x: x[1].bbox.size[0] * x[1].bbox.size[1],
                              reverse=True)

        # Collect for legend (with counts)
        display_status_counts = {}
        text_labels = []  # Store text labels to draw last

        # Draw all tiles (largest first)
        for idx, (tile_id, tile) in enumerate(sorted_tiles):
            bbox = tile.bbox
            # Use tile number for consistent color (not affected by other tiles being added)
            tile_num = int(tile_id.split("_")[-1]) if "_" in tile_id else idx
            tile_color = generate_tile_color(tile_num)

            # Determine status color (split by OOM category)
            if tile.status == "split":
                if tile.fail_iter == 0:
                    display_status = "split_preemptive"
                elif tile.oom_type == "ram":
                    display_status = "split_ram_oom"
                elif tile.oom_category == 1:
                    display_status = "split_cat1"
                elif tile.oom_category == 2:
                    display_status = "split_cat2"
                elif tile.oom_category == 3:
                    display_status = "split_cat3"
                elif tile.oom_category == 4:
                    display_status = "split_cat4"
                else:
                    display_status = "split_gpu_oom"  # fallback for unknown category
            else:
                display_status = tile.status
            status_color = status_colors.get(display_status, "#9E9E9E")

            # Linewidth based on split level: 30 for original, -6 for each halving
            tile_area = bbox.size[0] * bbox.size[1]
            area_ratio = max_tile_area / tile_area
            if area_ratio > 1:
                split_level = math.log2(area_ratio)
            else:
                split_level = 0
            linewidth = max(2, 30 - 6 * split_level)

            # Highlight current tile
            if tile_id == current_tile_id:
                linewidth = max(linewidth, 5.0)

            # zorder based on size (smaller = higher zorder = drawn on top)
            zorder = 10 + (max_tile_area - tile_area) / max_tile_area * 10

            rect = patches.Rectangle(
                (bbox.x_min, bbox.y_min),
                bbox.size[0], bbox.size[1],
                linewidth=linewidth,
                edgecolor=tile_color,
                facecolor='none',  # No fill
                zorder=zorder,
            )
            ax.add_patch(rect)

            # Prepare tile label (draw later)
            center_x = (bbox.x_min + bbox.x_max) / 2
            center_y = (bbox.y_min + bbox.y_max) / 2

            # Font size based on tile size (1.5x larger)
            font_size = min(12, max(7.5, min(bbox.size[0], bbox.size[1]) / 50 * 1.5))

            label_text = tile_id.replace("tile_", "")
            if tile_id == current_tile_id:
                label_text = f"► {label_text} ◄"
                font_size = 15

            # Add info for tiles that have been processed
            if tile.fail_iter is not None or tile.num_cameras is not None:
                info_parts = []
                if tile.fail_iter is not None:
                    if tile.fail_iter == 0:
                        info_parts.append("preemptive")
                    else:
                        # Show OOM type and category
                        oom_prefix = "RAM" if tile.oom_type == "ram" else "GPU"
                        cat_str = f"C{tile.oom_category}" if tile.oom_category else ""
                        info_parts.append(f"{oom_prefix}{cat_str}@{tile.fail_iter}")
                if tile.num_cameras is not None:
                    info_parts.append(f"C{tile.num_cameras}")
                label_text += f"\n({', '.join(info_parts)})"

            text_labels.append((center_x, center_y, label_text, font_size,
                               tile_id == current_tile_id, tile_color, status_color, display_status))

            # Track status counts for legend (use display_status to distinguish split types)
            display_status_counts[display_status] = display_status_counts.get(display_status, 0) + 1

        # Draw all text labels last (on top)
        for center_x, center_y, label_text, font_size, is_current, tile_color, status_color, status in text_labels:
            # Split label into tile ID and info
            lines = label_text.split('\n')
            tile_id_text = lines[0]
            info_text = '\n'.join(lines[1:]) if len(lines) > 1 else None

            # Draw tile ID with background box colored by status
            bbox_props = dict(
                boxstyle='round,pad=0.3',
                facecolor=status_color,
                edgecolor='none',
                alpha=0.9
            )
            # Offset for tile ID (move up if there's info text)
            y_offset = font_size * 2.0 if info_text else 0
            ax.text(center_x, center_y + y_offset, tile_id_text,
                   ha='center', va='center', fontsize=font_size,
                   fontweight='bold' if is_current else 'normal',
                   color=tile_color,
                   bbox=bbox_props,
                   zorder=102)

            # Draw info text without background box
            if info_text:
                ax.text(center_x, center_y - font_size * 1.5, info_text,
                       ha='center', va='center', fontsize=font_size,
                       color=tile_color,
                       zorder=102)

        # Set axis limits with margin
        all_x = [self.scene_bbox.x_min, self.scene_bbox.x_max]
        all_y = [self.scene_bbox.y_min, self.scene_bbox.y_max]
        for tile in self.tiles.values():
            all_x.extend([tile.bbox.x_min, tile.bbox.x_max])
            all_y.extend([tile.bbox.y_min, tile.bbox.y_max])

        margin_x = (max(all_x) - min(all_x)) * 0.05
        margin_y = (max(all_y) - min(all_y)) * 0.05
        ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)

        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Title
        current_info = ""
        if current_tile_id and current_tile_id in self.tiles:
            tile = self.tiles[current_tile_id]
            current_info = f"\nCurrent: {current_tile_id} (Size: {tile.bbox.size[0]:.1f} x {tile.bbox.size[1]:.1f})"

        # Count tiles by status
        status_counts = {}
        for tile in self.tiles.values():
            status_counts[tile.status] = status_counts.get(tile.status, 0) + 1

        status_summary = " | ".join([f"{s}: {c}" for s, c in sorted(status_counts.items())])

        ax.set_title(f'Tile Map - {status_summary}{current_info}', fontsize=14, fontweight='bold')

        # Add legend for status colors with counts (draw last, on top)
        legend_handles = []
        for status, color in status_colors.items():
            if status in display_status_counts:
                count = display_status_counts[status]
                handle = patches.Patch(facecolor=color, edgecolor='white',
                                       label=f"{status.capitalize()} ({count})")
                legend_handles.append(handle)
        legend = ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
        legend.set_zorder(200)

        plt.tight_layout()

        # Save to PNG
        vis_dir = self.output_path / "visualizations" / "tile_map"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Save with iteration counter and tile ID in filename
        if current_tile_id:
            filename = f"tile_map_{self.vis_counter:03d}_{current_tile_id}.png"
            self.vis_counter += 1
        else:
            filename = f"tile_map_{self.vis_counter:03d}_final.png"

        save_path = vis_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  [Visualization] Saved: {save_path} (camera footprints: {num_valid_footprints}/{num_cams})", flush=True)

    def _visualize_split_tree(self):
        """타일 분할 트리 (parent → children) PNG 저장.

        타일 ID 규칙 ``tile_{N:04d}`` 기준으로 부모(``(N-1)//2``)를 추론.
        노드 색상은 status / oom_category, 라벨에 fail_iter, num_cameras 표시.
        """
        if not self.tiles:
            return

        def parse_num(tid: str) -> Optional[int]:
            try:
                return int(tid.split("_")[-1])
            except Exception:
                return None

        nums = {tid: parse_num(tid) for tid in self.tiles}
        # parent_num -> [child_num, ...]
        children: Dict[int, List[int]] = {}
        roots: List[int] = []
        for tid, n in nums.items():
            if n is None:
                continue
            if n == 0:
                roots.append(0)
                continue
            parent_n = (n - 1) // 2
            if any(nums[t] == parent_n for t in nums):
                children.setdefault(parent_n, []).append(n)
            else:
                roots.append(n)
        if not roots:
            roots = [n for n in nums.values() if n is not None and n == 0] or [min(n for n in nums.values() if n is not None)]

        # BFS 깊이 계산
        depth: Dict[int, int] = {}
        order: List[int] = []
        from collections import deque
        q = deque([(r, 0) for r in roots])
        while q:
            n, d = q.popleft()
            if n in depth:
                continue
            depth[n] = d
            order.append(n)
            for c in sorted(children.get(n, [])):
                q.append((c, d + 1))

        # 같은 깊이끼리 모은 다음 x 좌표를 균등 분포
        by_depth: Dict[int, List[int]] = {}
        for n, d in depth.items():
            by_depth.setdefault(d, []).append(n)
        for d in by_depth:
            by_depth[d].sort()

        max_d = max(by_depth) if by_depth else 0
        max_w = max(len(v) for v in by_depth.values()) if by_depth else 1
        positions: Dict[int, Tuple[float, float]] = {}
        for d, ns in by_depth.items():
            count = len(ns)
            for i, n in enumerate(ns):
                x = (i + 1) / (count + 1) * max(max_w, 2)
                y = -d
                positions[n] = (x, y)

        # tid 매핑 (num -> tile_id 문자열)
        num_to_tid: Dict[int, str] = {n: tid for tid, n in nums.items() if n is not None}

        status_colors = {
            "completed": "#43A047",
            "split": "#F4A261",
            "failed": "#E53935",
            "skipped": "#9E9E9E",
            "in_progress": "#1E88E5",
            "pending": "#BDBDBD",
        }

        fig_w = max(10, max_w * 1.4)
        fig_h = max(5, (max_d + 1) * 1.4)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

        # 엣지
        for parent_n, child_list in children.items():
            if parent_n not in positions:
                continue
            px, py = positions[parent_n]
            for c in child_list:
                if c not in positions:
                    continue
                cx, cy = positions[c]
                ax.plot([px, cx], [py, cy], "-", color="#888888", linewidth=1.0, zorder=1)

        # 노드
        for n, (x, y) in positions.items():
            tid = num_to_tid.get(n)
            if tid is None or tid not in self.tiles:
                continue
            t = self.tiles[tid]
            color = status_colors.get(t.status, "#777777")
            edge = "black"
            if t.status == "split" and t.oom_category is not None:
                edge_map = {1: "#FFB300", 2: "#FB8C00", 3: "#D32F2F", 4: "#6A1B9A"}
                edge = edge_map.get(t.oom_category, "black")
            ax.scatter([x], [y], s=900, c=[color], edgecolors=edge,
                       linewidths=2.0, zorder=2)
            label = tid.replace("tile_", "")
            details = []
            if t.oom_category is not None:
                details.append(f"cat{t.oom_category}")
            if t.fail_iter is not None:
                details.append(f"iter{t.fail_iter}")
            if t.num_cameras is not None:
                details.append(f"cams={t.num_cameras}")
            label_full = label
            if details:
                label_full += "\n" + " ".join(details)
            ax.text(x, y - 0.18, label_full, ha="center", va="top",
                    fontsize=7, zorder=3)

        # 범례
        legend_handles = []
        for status, color in status_colors.items():
            legend_handles.append(patches.Patch(facecolor=color, edgecolor="black",
                                                label=status))
        legend_handles.append(patches.Patch(facecolor="white", edgecolor="#FFB300",
                                            linewidth=2, label="split: cat 1"))
        legend_handles.append(patches.Patch(facecolor="white", edgecolor="#FB8C00",
                                            linewidth=2, label="split: cat 2"))
        legend_handles.append(patches.Patch(facecolor="white", edgecolor="#D32F2F",
                                            linewidth=2, label="split: cat 3"))
        legend_handles.append(patches.Patch(facecolor="white", edgecolor="#6A1B9A",
                                            linewidth=2, label="split: cat 4"))
        ax.legend(handles=legend_handles, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)

        ax.set_xlim(0, max(max_w + 1, 3))
        ax.set_ylim(-(max_d + 0.7), 0.7)
        ax.set_xticks([])
        ax.set_yticks(list(range(0, -(max_d + 1), -1)))
        ax.set_yticklabels([f"L{abs(d)}" for d in range(0, -(max_d + 1), -1)])
        ax.set_title(f"Tile split tree | total={len(self.tiles)}  depth={max_d}")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        out_dir = self.output_path / "visualizations" / "tile_split_tree"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "tile_split_tree.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualization] Saved tile split tree -> {out_path}", flush=True)

    def _save_completed_ply(self, tile: TileInfo, tile_level: int):
        """Copy completed tile PLY to ply folder with informative name."""
        import shutil
        import glob

        # Find the source PLY (in models folder)
        model_dir = self.output_path / "models" / tile.tile_id / "point_cloud"
        if not model_dir.exists():
            print(f"  [Warning] Model dir not found: {model_dir}")
            return

        # Find the latest iteration folder
        iter_folders = sorted(model_dir.glob("iteration_*"), key=lambda x: int(x.name.split("_")[1]))
        if not iter_folders:
            print(f"  [Warning] No iteration folders found in {model_dir}")
            return

        latest_iter_folder = iter_folders[-1]
        iteration = int(latest_iter_folder.name.split("_")[1])
        source_ply = latest_iter_folder / "point_cloud.ply"

        if not source_ply.exists():
            # 비분산/distributed_save 모드는 point_cloud_rk*_ws*.ply 로 저장됨
            candidates = sorted(latest_iter_folder.glob("point_cloud*.ply"))
            if len(candidates) == 1:
                source_ply = candidates[0]
            elif len(candidates) > 1:
                print(f"  [Warning] Multiple PLY shards in {latest_iter_folder} "
                      f"({[c.name for c in candidates]}) — merge required, skipping copy")
                return
            else:
                print(f"  [Warning] Source PLY not found: {source_ply}")
                return

        # Count gaussians (by reading first line of PLY or checking file size)
        try:
            with open(source_ply, 'rb') as f:
                # Read header to find vertex count
                for line in f:
                    line = line.decode('ascii', errors='ignore').strip()
                    if line.startswith('element vertex'):
                        gaussian_count = int(line.split()[-1])
                        break
                else:
                    gaussian_count = 0
        except:
            gaussian_count = 0

        # Create informative filename: {tile_id}_L{level}_completed_iter{iteration}_{count}gs.ply
        ply_name = f"{tile.tile_id}_L{tile_level}_completed_iter{iteration}_{gaussian_count}gs.ply"
        dest_ply = self.ply_dir / ply_name

        # Copy PLY to ply folder
        shutil.copy2(source_ply, dest_ply)
        print(f"  [PLY] Saved: {dest_ply}")

    def _get_next_tile(self) -> Optional[TileInfo]:
        """Get next pending tile."""
        for tile in self.tiles.values():
            if tile.status == "pending":
                return tile
        return None

    def _run_torchrun(self, tile: TileInfo, visible_cameras: List[str]) -> int:
        """Run torchrun for a single tile."""
        tile_model_path = self.output_path / "models" / tile.tile_id
        tile_log_path = self.output_path / "logs" / tile.tile_id
        tile_model_path.mkdir(parents=True, exist_ok=True)
        tile_log_path.mkdir(parents=True, exist_ok=True)

        # 에폭 상한: 카메라 적은 타일의 과적합 방지 (iterations ∝ 뷰 수)
        eff_iterations = effective_tile_iterations(
            self.args.iterations, len(visible_cameras),
            getattr(self.args, "epoch_cap", 0))
        if eff_iterations != self.args.iterations:
            print(f"  [epoch-cap] cameras={len(visible_cameras)} → iterations "
                  f"{self.args.iterations} → {eff_iterations} "
                  f"(cap {self.args.epoch_cap} epochs)", flush=True)

        # Clear any leftover OOM signal, ack, and done files from previous runs
        signal_file = self.ply_dir / OOM_SIGNAL_FILENAME
        if signal_file.exists():
            signal_file.unlink()
            print(f"  [Cleared leftover OOM signal file]")
        # Clear ack files (used for dynamic waiting during OOM coordination)
        for ack_file in self.ply_dir.glob("oom_ack_rank*.json"):
            ack_file.unlink()
            print(f"  [Cleared leftover ack file: {ack_file.name}]")
        # Clear done files
        for done_file in self.ply_dir.glob("oom_done_rank*.json"):
            done_file.unlink()
            print(f"  [Cleared leftover done file: {done_file.name}]")

        # Calculate tile level
        tile_area = tile.bbox.size[0] * tile.bbox.size[1]
        tile_level = self._get_tile_level(tile_area)

        # Ensure densify_from_iter is at least 2x the number of visible cameras
        num_visible = len(visible_cameras) if visible_cameras else 1
        effective_densify_from = max(self.args.densify_from_iter, 2 * num_visible)
        if effective_densify_from != self.args.densify_from_iter:
            print(f"  [Adjusted densify_from_iter: {self.args.densify_from_iter} -> {effective_densify_from} (2 x {num_visible} cameras)]")

        cmd = [
            "torchrun",
            f"--nproc_per_node={self.args.num_gpus}",
            str(self.grendel_dir / "train.py"),
            "--source_path", str(self.source_path),
            "--model_path", str(tile_model_path),
            "--log_folder", str(tile_log_path),
            "--iterations", str(eff_iterations),
            "--backend", self.args.backend,
            "--bsz", str(self.args.bsz),
            "--adaptive_tile_enabled",
            f"--tile_bbox={tile.bbox.to_string()}",
            "--tile_id", tile.tile_id,
            "--tile_level", str(tile_level),
            "--tile_output_dir", str(self.ply_dir),
            "--tile_crop_margin", str(self.args.tile_crop_margin),
            "--ndc_limit", str(self.args.ndc_limit),
            "--densify_from_iter", str(effective_densify_from),
            "--densification_interval", str(self.args.densification_interval),
            "--densify_grad_threshold", str(
                self.args.child_densify_grad_threshold
                if (tile.ply_path and self.args.child_densify_grad_threshold is not None)
                else self.args.densify_grad_threshold
            ),
            "--test_iterations", "999999999",  # Disable testing during adaptive training
        ]

        # Add visible cameras
        if visible_cameras:
            cmd.extend(["--visible_cameras", ",".join(visible_cameras)])

        # Add pre-trained gaussians PLY path for Category 3 OOM resume
        if tile.ply_path:
            cmd.extend(["--pretrained_ply", tile.ply_path])

        print(f"\n[Tile {tile.tile_id}] Running torchrun...")
        print(f"  BBox: {tile.bbox.to_string()}")
        print(f"  Visible cameras: {len(visible_cameras)}")
        if tile.ply_path:
            print(f"  *** RESUME MODE: Using pre-trained gaussians ***")
            print(f"  Pre-trained PLY: {tile.ply_path}")
            if self.args.child_densify_grad_threshold is not None:
                print(f"  Child densify threshold: {self.args.child_densify_grad_threshold} "
                      f"(parent: {self.args.densify_grad_threshold})")
        else:
            print(f"  Starting from scratch (SfM points)")

        # Run torchrun and capture output to train_adaptive.log via tee'd stdout
        # Use Popen to stream output in real-time (so it goes through TeeOutput)
        sys.stdout.flush()
        sys.stderr.flush()

        # Set environment variables for better NCCL error handling
        # TORCH_NCCL_ASYNC_ERROR_HANDLING=1: Store errors and throw as Python exceptions
        # instead of calling std::terminate() which kills the process immediately
        env = os.environ.copy()
        env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # Also set NCCL debug level for better diagnostics
        env["NCCL_DEBUG"] = "WARN"

        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            env=env,
        )
        # Stream output line by line (this goes through TeeOutput -> train_adaptive.log)
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
        result_code = process.returncode
        sys.stdout.flush()

        print(f"  [torchrun returned exit_code={result_code}]", flush=True)
        # If render debug abort flag exists, stop the whole wrapper
        if os.environ.get("RENDER_DEBUG_ABORT", "0") == "1":
            try:
                flag = self.ply_dir.parent / "visualizations" / "render_debug_png" / "render_debug_abort.flag"
                if flag.exists():
                    print(f"  [Render debug] Abort flag found: {flag}. Stopping wrapper.", flush=True)
                    flag.unlink(missing_ok=True)
                    return result_code, {"abort": "render_debug"}
            except Exception:
                pass

        # Check for OOM via state file (torchrun returns 1 even when worker exits with 42)
        state_file = self.ply_dir / "adaptive_tile_state.json"
        signal_file = self.ply_dir / OOM_SIGNAL_FILENAME
        oom_info = None

        # First, check adaptive_tile_state.json (written by Category 3 OOM with full info)
        if state_file.exists():
            try:
                with open(state_file) as f:
                    oom_state = json.load(f)
                if oom_state.get("oom_occurred"):
                    oom_info = {
                        "iteration": oom_state.get("iteration"),
                        "oom_category": oom_state.get("oom_category"),
                        "oom_cause": oom_state.get("oom_cause"),
                        "tile_a": oom_state.get("tile_a"),
                        "tile_b": oom_state.get("tile_b"),
                        "num_ranks": oom_state.get("num_ranks", 1),  # For Category 3 PLY merge
                    }
                    print(f"  [OOM detected via state file at iteration {oom_info['iteration']}]")
                    print(f"  [OOM category: {oom_info['oom_category']}, cause: {oom_info['oom_cause']}]")
                    # Remove the state file after reading
                    state_file.unlink()
                    # Also clear OOM signal file
                    if signal_file.exists():
                        signal_file.unlink()
                        print(f"  [Cleared OOM signal file]")
                    # Clear ack files (used for dynamic waiting during OOM coordination)
                    for ack_file in self.ply_dir.glob("oom_ack_rank*.json"):
                        ack_file.unlink()
                        print(f"  [Cleared ack file: {ack_file.name}]")

                    # Read expected counts from done files BEFORE clearing them
                    # This is used for validation during merge
                    num_ranks = oom_info.get("num_ranks", 1)
                    expected_counts = read_expected_counts_from_done_files(self.ply_dir, num_ranks)
                    if expected_counts['ranks_found'] > 0:
                        print(f"  [Validation] Expected counts from {expected_counts['ranks_found']} rank(s):")
                        print(f"    Child A: {expected_counts['count_a']:,} gaussians")
                        print(f"    Child B: {expected_counts['count_b']:,} gaussians")
                        print(f"    Total: {expected_counts['total']:,} gaussians")
                        oom_info['expected_counts'] = expected_counts

                    # Clear done files
                    for done_file in self.ply_dir.glob("oom_done_rank*.json"):
                        done_file.unlink()
                        print(f"  [Cleared done file: {done_file.name}]")
                    return self.EXIT_CODE_OOM, oom_info
            except (json.JSONDecodeError, IOError) as e:
                print(f"  [Warning] Failed to read OOM state file: {e}")

        # Second, check oom_signal.json (written by Category 1/2 OOM, no gaussians saved)
        # This happens when OOM occurs early (before densification) - no state file is written
        if signal_file.exists() and oom_info is None:
            try:
                with open(signal_file) as f:
                    signal_data = json.load(f)
                category = signal_data.get("category", 1)
                iteration = signal_data.get("iteration", 0)
                print(f"  [OOM detected via signal file: category={category}, iter={iteration}]", flush=True)
                oom_info = {
                    "iteration": iteration,
                    "oom_category": category,
                    "oom_cause": "gpu",
                    "tile_a": None,
                    "tile_b": None,
                    "num_ranks": 1,
                }
                signal_file.unlink()
                print(f"  [Cleared OOM signal file]")
                return self.EXIT_CODE_OOM, oom_info
            except (json.JSONDecodeError, IOError) as e:
                print(f"  [Warning] Failed to read OOM signal file: {e}")

        return result_code, oom_info

    def run(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Adaptive Tile Training")
        print("=" * 60)
        print(f"Scene: {self.source_path}")
        print(f"Output: {self.output_path}")
        print(f"GPUs: {self.args.num_gpus}")
        print(f"Tiles: {len(self.tiles)}")
        print(f"Initial area (level 0): {self.initial_area:.1f}")
        print(f"Min successful level: {self.min_successful_level}")
        print("=" * 60)

        max_tiles = 1000  # Safety limit
        completed = 0

        while completed < max_tiles:
            tile = self._get_next_tile()
            if tile is None:
                print("\n[Done] No more pending tiles.")
                break

            print("\n" + "=" * 60, flush=True)
            print(f">>>  STARTING TILE: {tile.tile_id}  <<<", flush=True)
            print("=" * 60, flush=True)
            print(f"  BBox: {tile.bbox.to_string()}", flush=True)
            print(f"  Size: ({tile.bbox.size[0]:.1f}, {tile.bbox.size[1]:.1f}, {tile.bbox.size[2]:.1f})", flush=True)

            # Print level info
            tile_area = tile.bbox.size[0] * tile.bbox.size[1]
            current_level = self._get_tile_level(tile_area)
            print(f"  Area: {tile_area:.1f}, Level: {current_level}", flush=True)
            print(f"  [Level tracking] min_successful_level: {self.min_successful_level}", flush=True)

            # Clear visualizations folder only once at the very beginning
            if not hasattr(self, '_vis_cleared'):
                self._clear_visualizations()
                self._vis_cleared = True

            # Visualize tile map before starting this tile
            try:
                self._visualize_tile_map(current_tile_id=tile.tile_id)
            except Exception as e:
                print(f"  [Visualization ERROR] {e}", flush=True)
                import traceback
                traceback.print_exc()

            # Compute visible cameras
            visible_cameras = self._compute_visible_cameras(
                tile.bbox, self.args.tile_crop_margin,
                tile_id=tile.tile_id, visual_debug=self.args.visual_debug
            )
            tile.num_cameras = len(visible_cameras)
            print(f"  Visible cameras: {len(visible_cameras)} / {len(self.cam_infos)}")
            if visible_cameras:
                print(f"    Names: {visible_cameras[:5]}{'...' if len(visible_cameras) > 5 else ''}")

            
            # If visual_debug_only, generate images for all levels up to target
            if self.args.visual_debug_only:
                print(f"  [Visual Debug] Generated images for level {current_level}")
                print(f"  Debug images saved to: {self.output_path}/visualizations/projection_debug/{tile.tile_id}/")
                
                if current_level == self.args.visual_debug_level:
                    print("\n" + "=" * 60)
                    print(f"Visual Debug Complete! Generated images for levels 0-{current_level}")
                    print("Exiting (--visual-debug-only mode)")
                    print("=" * 60)
                    sys.exit(0)
                elif current_level < self.args.visual_debug_level:
                    print(f"  [Visual Debug] Continuing to level {current_level + 1} (target: level {self.args.visual_debug_level})")
                    # Simulate OOM to force split for reaching next level
                    print(f"  [Visual Debug] Simulating OOM to split tile")
                    
                    # Split tile logic (same as preemptive split)
                    bbox_a, bbox_b = tile.bbox.split()
                    tile_a_id = self._next_tile_id()
                    tile_b_id = self._next_tile_id()
                    
                    tile.status = "split"
                    tile.fail_iter = 0  # Mark as visual debug split
                    tile.oom_type = "visual_debug"
                    
                    # Create child tiles
                    child_a = TileInfo(tile_a_id, bbox_a, "pending")
                    child_b = TileInfo(tile_b_id, bbox_b, "pending")
                    
                    # Add to tiles dict and save state
                    self.tiles[tile_a_id] = child_a
                    self.tiles[tile_b_id] = child_b
                    self._save_state()
                    
                    print(f"  Split into {tile_a_id} and {tile_b_id} for visual debug")
                    continue

            if len(visible_cameras) == 0:
                print(f"  [Warning] No cameras see this tile, skipping...")
                tile.status = "completed"
                self._save_state()
                completed += 1
                continue

            # Preemptive split based on level:
            # - If we know level N succeeds, try level N-1 (one step larger)
            # - But don't try level N-2 or larger (2+ steps), split immediately
            tile_area = tile.bbox.size[0] * tile.bbox.size[1]
            current_level = self._get_tile_level(tile_area)

            should_preemptive_split = False
            if self.min_successful_level is not None:
                # min_successful_level - 1 = one step larger (allowed to try)
                # min_successful_level - 2 or less = two+ steps larger (preemptive split)
                allowed_level = self.min_successful_level - 1
                level_diff = self.min_successful_level - current_level
                print(f"  [Level check] current={current_level}, min_success={self.min_successful_level}, allowed={allowed_level}, diff={level_diff}", flush=True)
                if current_level < self.min_successful_level - 1:
                    should_preemptive_split = True
                    print(f"  [Level check] WILL PREEMPTIVE SPLIT (level {current_level} < allowed {allowed_level})", flush=True)
                else:
                    print(f"  [Level check] OK to try (level {current_level} >= allowed {allowed_level})", flush=True)
            else:
                print(f"  [Level check] No successful level yet, will try this tile", flush=True)

            if should_preemptive_split:
                print("\n" + "#" * 60, flush=True)
                print("###  PREEMPTIVE SPLIT (2+ levels larger than known successful)  ###", flush=True)
                print("#" * 60, flush=True)
                print(f"  Current tile level: {current_level}")
                print(f"  Min successful level: {self.min_successful_level}")
                print(f"  (Level {self.min_successful_level - 1} would be tried, but {current_level} is too large)")

                # Check minimum tile size before splitting
                MIN_TILE_SIZE = 50.0
                tile_size = tile.bbox.size
                min_dim = min(tile_size[0], tile_size[1])

                if min_dim < MIN_TILE_SIZE:
                    print(f"  Cannot split further (min dimension {min_dim:.1f} < {MIN_TILE_SIZE})")
                    print(f"  Will attempt training anyway...")
                else:
                    # Split tile without attempting training
                    bbox_a, bbox_b = tile.bbox.split()
                    tile_a_id = self._next_tile_id()
                    tile_b_id = self._next_tile_id()

                    tile.status = "split"
                    tile.fail_iter = 0  # Mark as preemptive split (iter 0)

                    # Calculate levels for new tiles
                    size_a = bbox_a.size
                    size_b = bbox_b.size
                    area_a = size_a[0] * size_a[1]
                    area_b = size_b[0] * size_b[1]
                    level_a = self._get_tile_level(area_a)
                    level_b = self._get_tile_level(area_b)

                    print(f"  Split into:", flush=True)
                    print(f"    {tile_a_id}: size=({size_a[0]:.1f}, {size_a[1]:.1f}, {size_a[2]:.1f}), level={level_a}", flush=True)
                    print(f"    {tile_b_id}: size=({size_b[0]:.1f}, {size_b[1]:.1f}, {size_b[2]:.1f}), level={level_b}", flush=True)
                    print("#" * 60 + "\n", flush=True)

                    new_tiles = {}
                    new_tiles[tile_a_id] = TileInfo(tile_a_id, bbox_a, "pending")
                    new_tiles[tile_b_id] = TileInfo(tile_b_id, bbox_b, "pending")
                    for tid, t in self.tiles.items():
                        new_tiles[tid] = t
                    self.tiles = new_tiles

                    self._save_state()
                    
                    # Generate visual debug for new tiles
                    self._generate_visual_debug_for_tile(bbox_a, tile_a_id)
                    self._generate_visual_debug_for_tile(bbox_b, tile_b_id)
                    
                    continue

            # Mark as in_progress
            tile.status = "in_progress"
            self._save_state()

            # Run training
            exit_code, oom_info = self._run_torchrun(tile, visible_cameras)

            # Count timeline 시각화: 매 타일 종료 후 두 aggregate PNG 갱신 (best-effort)
            #  - all_tiles_timeline.png: x=학습 iter, 각 타일 색 분리
            #  - oom_progression.png:    x=타일 처리 순서, wrapper 시퀀스 뷰
            try:
                count_dir = Path(self.args.output_path) / "visualizations" / "count_timeline"
                if count_dir.exists() and any(count_dir.glob("*_rank*.jsonl")):
                    sys.path.insert(0, str(ROOT / "Grendel-GS"))
                    from utils.oom_viz import save_all_tiles_timeline_viz, save_oom_progression_viz

                    agg_png = count_dir / "all_tiles_timeline.png"
                    save_all_tiles_timeline_viz(count_dir, out_path=agg_png, target_count=None)
                    prog_png = count_dir / "oom_progression.png"
                    save_oom_progression_viz(count_dir, out_path=prog_png)
                    print(f"  [viz] timeline updated -> {agg_png.name}, {prog_png.name}", flush=True)
            except Exception as _ct_viz_e:
                print(f"  [viz] count timeline skipped: {_ct_viz_e}", flush=True)

            # Tile split tree 매 타일 처리 후 갱신 (best-effort)
            try:
                self._visualize_split_tree()
            except Exception as _tree_e:
                print(f"  [viz] tile split tree skipped: {_tree_e}", flush=True)

            # Log exit code for debugging
            print(f"\n[Tile {tile.tile_id}] torchrun exit_code = {exit_code}", flush=True)
            print(f"  oom_info = {oom_info}", flush=True)
            print(f"  EXIT_CODE_OOM = {self.EXIT_CODE_OOM}", flush=True)
            print(f"  Recognized OOM codes: {self.EXIT_CODE_OOM}, -9, 143, -15, 137", flush=True)
            if oom_info and oom_info.get("abort") == "render_debug":
                print("  [Render debug] Aborting adaptive run due to render debug abort.", flush=True)
                sys.exit(1)

            if exit_code == self.EXIT_CODE_SUCCESS:
                tile.status = "completed"
                # 타일 품질 지표 로드 (train_internal 이 저장 시점에 기록)
                done_info_path = self.output_path / "models" / tile.tile_id / "done_info.json"
                if done_info_path.exists():
                    try:
                        with open(done_info_path) as f:
                            tile.quality = json.load(f)
                        print(f"  [quality] final_epoch_loss="
                              f"{tile.quality.get('final_epoch_loss')} "
                              f"({tile.quality.get('done_reason')})", flush=True)
                    except Exception as e:
                        print(f"  [quality] done_info 읽기 실패: {e}", flush=True)
                # Track successful tile level (smaller level = larger tile)
                tile_area = tile.bbox.size[0] * tile.bbox.size[1]
                tile_level = self._get_tile_level(tile_area)
                if self.min_successful_level is None or tile_level < self.min_successful_level:
                    self.min_successful_level = tile_level
                    print(f"  [Success] Updated min successful level: {tile_level} (area: {tile_area:.1f})")
                self._save_state()
                completed += 1
                print(f"[Tile {tile.tile_id}] Completed successfully (level {tile_level}).")

                # Copy completed PLY to ply folder with informative name
                self._save_completed_ply(tile, tile_level)

            elif exit_code == self.EXIT_CODE_OOM or exit_code == -9 or exit_code in (143, -15, 137):
                # exit_code meanings:
                #   42: EXIT_CODE_OOM (explicit OOM handling)
                #   -9: SIGKILL (OS killed due to RAM OOM)
                #   143: SIGTERM (128+15, torchrun shutdown - likely OOM from another rank)
                #   -15: SIGTERM (negative signal)
                #   137: SIGKILL (128+9, torchrun shutdown)
                # Determine OOM type
                is_ram_oom = (exit_code == -9)
                is_sigterm = (exit_code in (143, -15, 137))
                if is_sigterm:
                    print(f"[Tile {tile.tile_id}] SIGTERM/SIGKILL exit ({exit_code}) - treating as OOM", flush=True)
                oom_type_str = "RAM" if is_ram_oom else "GPU"

                # RAM OOM: wait for OS to reclaim memory from dead processes
                if is_ram_oom:
                    import gc
                    import psutil

                    gc.collect()  # Force Python garbage collection

                    # Wait until RAM usage drops below 70%
                    max_wait = 300  # Maximum wait time: 5 minutes
                    waited = 0
                    mem = psutil.virtual_memory()
                    print(f"\n[RAM OOM] Current RAM usage: {mem.percent:.1f}%", flush=True)

                    while mem.percent > 70 and waited < max_wait:
                        print(f"[RAM OOM] RAM usage {mem.percent:.1f}% > 70%, waiting... ({waited}s/{max_wait}s)", flush=True)
                        time.sleep(10)
                        waited += 10
                        gc.collect()
                        mem = psutil.virtual_memory()

                    if mem.percent > 70:
                        print(f"[RAM OOM] WARNING: RAM still at {mem.percent:.1f}% after {max_wait}s, proceeding anyway...", flush=True)
                    else:
                        print(f"[RAM OOM] RAM usage dropped to {mem.percent:.1f}%, resuming.", flush=True)

                oom_iteration = oom_info.get("iteration") if oom_info else None
                oom_category = oom_info.get("oom_category") if oom_info else None

                print("\n" + "#" * 60, flush=True)
                print(f"###  {oom_type_str} OOM DETECTED - SPLITTING TILE  ###", flush=True)
                print("#" * 60, flush=True)
                tile_area = tile.bbox.size[0] * tile.bbox.size[1]
                tile_level = self._get_tile_level(tile_area)
                print(f"[Tile {tile.tile_id}] {oom_type_str} OOM at iteration {oom_iteration}, level={tile_level}, splitting...", flush=True)
                tile.fail_iter = oom_iteration
                tile.oom_type = "ram" if is_ram_oom else "gpu"
                tile.oom_category = oom_category

                # Category 4: pre-trained gaussians too large before densification.
                # Split and reuse parent PLY — no new PLY saving needed.
                if oom_category == 4:
                    print(f"  [Category 4] Pre-trained gaussians too large (iter {oom_iteration}, before densification).", flush=True)
                    print(f"  Splitting tile. Children will reuse parent PLY filtered by bbox.", flush=True)

                # Category 2: memory fragmentation, not a size problem.
                # Retry the same tile — the torchrun process already exited,
                # so GPU memory is freed and fragmentation is gone.
                MAX_CAT2_RETRIES = 3
                if oom_category == 2:
                    cat2_retries = getattr(tile, '_cat2_retries', 0)
                    if cat2_retries < MAX_CAT2_RETRIES:
                        tile._cat2_retries = cat2_retries + 1
                        tile.status = "pending"
                        tile.fail_iter = None
                        tile.oom_type = None
                        tile.oom_category = None
                        print(f"  [Category 2 RETRY] Memory fragmentation OOM at iter {oom_iteration}.", flush=True)
                        print(f"  Retrying same tile ({tile._cat2_retries}/{MAX_CAT2_RETRIES})...", flush=True)
                        print("#" * 60 + "\n", flush=True)
                        self._save_state()
                        continue
                    else:
                        print(f"  [Category 2] Exhausted {MAX_CAT2_RETRIES} retries, falling through to split.", flush=True)
                # Check minimum tile size before splitting
                MIN_TILE_SIZE = 50.0  # Minimum size in world units
                tile_size = tile.bbox.size
                min_dim = min(tile_size)

                if min_dim < MIN_TILE_SIZE:
                    print(f"  WARNING: Tile is already very small: ({tile_size[0]:.1f}, {tile_size[1]:.1f}, {tile_size[2]:.1f})", flush=True)
                    print(f"  Cannot split further (min dimension {min_dim:.1f} < {MIN_TILE_SIZE})", flush=True)
                    print(f"  Marking tile as SKIPPED (too small to process)", flush=True)
                    print("#" * 60 + "\n", flush=True)
                    tile.status = "skipped"
                    self._save_state()
                    completed += 1
                    continue

                # Split tile
                bbox_a, bbox_b = tile.bbox.split()

                # Use tile IDs from state file if available (ensures PLY filename matches tile ID)
                # Note: use "or {}" because get() returns None if key exists but value is None
                tile_a_info = (oom_info.get("tile_a") or {}) if oom_info else {}
                tile_b_info = (oom_info.get("tile_b") or {}) if oom_info else {}

                if tile_a_info.get("tile_id") and tile_b_info.get("tile_id"):
                    # Use IDs from train_internal.py to match PLY filenames
                    tile_a_id = tile_a_info["tile_id"]
                    tile_b_id = tile_b_info["tile_id"]
                    # Update counter to avoid future collisions
                    a_num = int(tile_a_id.split("_")[-1]) if "_" in tile_a_id else 0
                    b_num = int(tile_b_id.split("_")[-1]) if "_" in tile_b_id else 0
                    self.tile_counter = max(self.tile_counter, a_num + 1, b_num + 1)
                    print(f"  [Using tile IDs from state: {tile_a_id}, {tile_b_id}]", flush=True)
                else:
                    # Fallback to sequential counter
                    tile_a_id = self._next_tile_id()
                    tile_b_id = self._next_tile_id()

                # Calculate sizes for display
                size_a = bbox_a.size
                size_b = bbox_b.size

                tile.status = "split"

                # Calculate levels for new tiles
                area_a = size_a[0] * size_a[1]
                area_b = size_b[0] * size_b[1]
                level_a = self._get_tile_level(area_a)
                level_b = self._get_tile_level(area_b)

                print(f"  Original tile: {tile.tile_id} (level {tile_level})", flush=True)
                print(f"  Split into:", flush=True)
                print(f"    {tile_a_id}: size=({size_a[0]:.1f}, {size_a[1]:.1f}, {size_a[2]:.1f}), level={level_a}", flush=True)
                print(f"    {tile_b_id}: size=({size_b[0]:.1f}, {size_b[1]:.1f}, {size_b[2]:.1f}), level={level_b}", flush=True)

                # For Category 3 OOM (increased gaussians), use saved PLY for resume
                # For Category 4 OOM (pre-trained too large), reuse parent's PLY
                ply_path_a = None
                ply_path_b = None
                if oom_category == 4 and tile.ply_path:
                    # Reuse parent's PLY — children will filter by their bbox on load
                    ply_path_a = tile.ply_path
                    ply_path_b = tile.ply_path
                    print(f"\n  >>> CATEGORY 4 OOM: Reusing parent PLY for children <<<", flush=True)
                    print(f"    Parent PLY: {tile.ply_path}", flush=True)

                elif oom_category == 3 and oom_info:
                    print(f"\n  >>> CATEGORY 3 OOM: Pre-trained gaussians will be used <<<", flush=True)
                    tile_a_info = oom_info.get("tile_a") or {}
                    tile_b_info = oom_info.get("tile_b") or {}
                    num_ranks = oom_info.get("num_ranks", 1)

                    # Check if PLY files need to be merged (distributed save with local_only=True)
                    ply_prefix_a = tile_a_info.get("ply_path")
                    ply_prefix_b = tile_b_info.get("ply_path")
                    is_prefix_a = tile_a_info.get("ply_is_prefix", False)
                    is_prefix_b = tile_b_info.get("ply_is_prefix", False)

                    # Merge rank files if needed
                    # Try BOTH tiles before failing - so we at least save what we can
                    merge_errors = []
                    count_a = 0
                    count_b = 0

                    # Wait for PLY files to be saved (configurable timeout)
                    if num_ranks > 1:
                        wait_timeout = int(os.environ.get('PLY_WAIT_TIMEOUT', 3600))  # Default 1 hour
                        print(f"  [Waiting] Checking for PLY files from {num_ranks} ranks (timeout: {wait_timeout}s)...", flush=True)
                        wait_start = time.time()
                        check_interval = 5  # Check every 5 seconds
                        
                        while time.time() - wait_start < wait_timeout:
                            # Check if all rank files exist
                            all_exist = True
                            for prefix in [ply_prefix_a, ply_prefix_b]:
                                if prefix and prefix != "None":
                                    for rank in range(num_ranks):
                                        rank_file = f"{prefix}_rank{rank}.ply"
                                        if not Path(rank_file).exists():
                                            all_exist = False
                                            break
                                if not all_exist:
                                    break
                            
                            if all_exist:
                                print(f"    All PLY files found after {time.time() - wait_start:.1f}s", flush=True)
                                break
                            
                            time.sleep(check_interval)
                        else:
                            print(f"    WARNING: Timeout after {wait_timeout}s waiting for PLY files", flush=True)

                    # Try Tile A merge
                    if ply_prefix_a and is_prefix_a and num_ranks > 1:
                        print(f"  [Merging] Tile A: {num_ranks} rank files...", flush=True)
                        merged_path_a = f"{ply_prefix_a}_merged.ply"
                        try:
                            count_a = merge_ply_files(ply_prefix_a, num_ranks, merged_path_a, strict=True)
                            ply_path_a = merged_path_a
                            print(f"    Merged {count_a:,} gaussians -> {merged_path_a}", flush=True)
                        except MergeError as e:
                            merge_errors.append(f"Tile A: {e}")
                            print(f"    FAILED: {e}", flush=True)
                    elif ply_prefix_a and not is_prefix_a:
                        # Single file path (legacy or single-GPU)
                        if Path(ply_prefix_a).exists():
                            ply_path_a = ply_prefix_a
                        else:
                            merge_errors.append(f"Tile A: PLY file not found: {ply_prefix_a}")

                    # Try Tile B merge (even if Tile A failed)
                    if ply_prefix_b and is_prefix_b and num_ranks > 1:
                        print(f"  [Merging] Tile B: {num_ranks} rank files...", flush=True)
                        merged_path_b = f"{ply_prefix_b}_merged.ply"
                        try:
                            count_b = merge_ply_files(ply_prefix_b, num_ranks, merged_path_b, strict=True)
                            ply_path_b = merged_path_b
                            print(f"    Merged {count_b:,} gaussians -> {merged_path_b}", flush=True)
                        except MergeError as e:
                            merge_errors.append(f"Tile B: {e}")
                            print(f"    FAILED: {e}", flush=True)
                    elif ply_prefix_b and not is_prefix_b:
                        # Single file path (legacy or single-GPU)
                        if Path(ply_prefix_b).exists():
                            ply_path_b = ply_prefix_b
                        else:
                            merge_errors.append(f"Tile B: PLY file not found: {ply_prefix_b}")

                    # Cat3 merged 시각화 (best-effort, multi-rank 인 경우만)
                    try:
                        if num_ranks > 1 and is_prefix_a and is_prefix_b and tile_a_info.get("bbox") and tile_b_info.get("bbox"):
                            sys.path.insert(0, str(ROOT / "Grendel-GS"))
                            from utils.oom_viz import save_merged_cat3_viz
                            from scene.adaptive_tile_utils import TileBBox as _TBB
                            parent_bbox_obj = tile.bbox
                            tile_a_bbox_obj = _TBB.from_string(tile_a_info["bbox"])
                            tile_b_bbox_obj = _TBB.from_string(tile_b_info["bbox"])
                            iter_for_viz = oom_info.get("iteration", 0) if isinstance(oom_info, dict) else 0
                            tile_id_for_viz = getattr(tile, "tile_id", "tile")
                            viz_dir = Path(self.args.output_path) / "visualizations" / "cat3_oom"
                            viz_dir.mkdir(parents=True, exist_ok=True)

                            for child_label, child_prefix, child_bbox in [
                                ("A", ply_prefix_a, tile_a_bbox_obj),
                                ("B", ply_prefix_b, tile_b_bbox_obj),
                            ]:
                                if not child_prefix:
                                    continue
                                rank_files = [f"{child_prefix}_rank{r}.ply" for r in range(num_ranks)]
                                rank_files = [p for p in rank_files if Path(p).exists()]
                                if not rank_files:
                                    continue
                                out_png = viz_dir / f"cat3_iter{iter_for_viz}_tile{tile_id_for_viz}_child{child_label}_merged.png"
                                save_merged_cat3_viz(
                                    rank_files,
                                    iteration=iter_for_viz,
                                    tile_id=f"{tile_id_for_viz}_child{child_label}",
                                    parent_bbox=parent_bbox_obj,
                                    tile_a=tile_a_bbox_obj,
                                    tile_b=tile_b_bbox_obj,
                                    out_path=out_png,
                                )
                                print(f"  [viz] saved merged Cat3 viz -> {out_png}", flush=True)
                    except Exception as _viz_e:
                        print(f"  [viz] merged Cat3 viz skipped: {_viz_e}", flush=True)

                    # If any merge failed, try to use partial results
                    if merge_errors:
                        print(f"\n{'!'*60}", flush=True)
                        print(f"  WARNING: Category 3 OOM PLY merge partially failed!", flush=True)
                        for err in merge_errors:
                            print(f"  - {err}", flush=True)
                        print(f"  ", flush=True)
                        print(f"  This indicates that some GPU ranks did not complete saving.", flush=True)
                        print(f"  Possible causes:", flush=True)
                        print(f"    1. Ranks were killed before completing PLY save", flush=True)
                        print(f"    2. Disk space or I/O error during save", flush=True)
                        print(f"    3. SIGTERM timeout too short", flush=True)
                        print(f"  ", flush=True)
                        # Show which merges succeeded (if any)
                        if ply_path_a:
                            print(f"  [Partial success] Tile A merged: {ply_path_a}", flush=True)
                        if ply_path_b:
                            print(f"  [Partial success] Tile B merged: {ply_path_b}", flush=True)
                        print(f"  ", flush=True)
                        
                        # Allow partial results if at least one tile has data
                        if ply_path_a or ply_path_b:
                            print(f"  [PARTIAL SUCCESS] Continuing with available PLY files:", flush=True)
                            if ply_path_a:
                                print(f"    - Tile A: {ply_path_a}", flush=True)
                            if ply_path_b:
                                print(f"    - Tile B: {ply_path_b}", flush=True)
                            print(f"  Note: Missing rank data will result in incomplete gaussians", flush=True)
                            print(f"{'!'*60}\n", flush=True)
                        else:
                            print(f"  FATAL: No PLY files available at all. Cannot continue.", flush=True)
                            print(f"{'!'*60}\n", flush=True)
                            sys.exit(1)

                    if ply_path_a or ply_path_b:
                        print(f"  [Category 3 Resume] Child tiles will load pre-trained gaussians:", flush=True)
                        if ply_path_a:
                            print(f"    {tile_a_id} -> {ply_path_a}", flush=True)
                        if ply_path_b:
                            print(f"    {tile_b_id} -> {ply_path_b}", flush=True)

                        # Validate merged counts against expected counts from done files
                        expected_counts = oom_info.get('expected_counts')
                        if expected_counts and expected_counts.get('ranks_found', 0) == num_ranks:
                            print(f"\n  [Validation] Checking merged gaussian counts...")
                            expected_a = expected_counts.get('count_a', 0)
                            expected_b = expected_counts.get('count_b', 0)
                            actual_a = count_a if (ply_prefix_a and is_prefix_a) else 0
                            actual_b = count_b if (ply_prefix_b and is_prefix_b) else 0

                            valid = True
                            if expected_a > 0:
                                if actual_a == expected_a:
                                    print(f"    ✓ Child A: {actual_a:,} == expected {expected_a:,}")
                                else:
                                    print(f"    ✗ Child A: {actual_a:,} != expected {expected_a:,} (diff: {actual_a - expected_a:+,})")
                                    valid = False
                            if expected_b > 0:
                                if actual_b == expected_b:
                                    print(f"    ✓ Child B: {actual_b:,} == expected {expected_b:,}")
                                else:
                                    print(f"    ✗ Child B: {actual_b:,} != expected {expected_b:,} (diff: {actual_b - expected_b:+,})")
                                    valid = False

                            if valid:
                                print(f"    [Validation PASSED] All gaussian counts match!")
                                # Write validation info for load-time check
                                if ply_path_a:
                                    val_file_a = Path(ply_path_a).with_suffix('.validation.json')
                                    with open(val_file_a, 'w') as f:
                                        json.dump({'expected_count': actual_a, 'num_ranks': num_ranks}, f)
                                if ply_path_b:
                                    val_file_b = Path(ply_path_b).with_suffix('.validation.json')
                                    with open(val_file_b, 'w') as f:
                                        json.dump({'expected_count': actual_b, 'num_ranks': num_ranks}, f)
                            else:
                                print(f"    [Validation FAILED] Gaussian count mismatch detected!")
                                print(f"    This may indicate incomplete saves or merge issues.")
                        elif expected_counts:
                            print(f"\n  [Validation] Skipped: only {expected_counts.get('ranks_found', 0)}/{num_ranks} done files found")
                    else:
                        # This should not happen - Category 3 OOM should always have PLY paths
                        print(f"\n{'!'*60}", flush=True)
                        print(f"  FATAL ERROR: Category 3 OOM but no PLY paths in oom_info!", flush=True)
                        print(f"  tile_a_info: {tile_a_info}", flush=True)
                        print(f"  tile_b_info: {tile_b_info}", flush=True)
                        print(f"{'!'*60}\n", flush=True)
                        sys.exit(1)
                else:
                    print(f"  Category {oom_category} OOM: Child tiles will start from scratch", flush=True)

                print("#" * 60 + "\n", flush=True)

                # Insert split tiles at the beginning so they are processed immediately
                # Rebuild tiles dict with split tiles first
                new_tiles = {}
                new_tiles[tile_a_id] = TileInfo(tile_a_id, bbox_a, "pending", ply_path=ply_path_a)
                new_tiles[tile_b_id] = TileInfo(tile_b_id, bbox_b, "pending", ply_path=ply_path_b)
                for tid, t in self.tiles.items():
                    new_tiles[tid] = t
                self.tiles = new_tiles

                self._save_state()
                
                # Generate visual debug for new tiles after OOM split
                self._generate_visual_debug_for_tile(bbox_a, tile_a_id)
                self._generate_visual_debug_for_tile(bbox_b, tile_b_id)

            else:
                # Other failures - mark as failed and continue
                print(f"[Tile {tile.tile_id}] Failed with exit code {exit_code}", flush=True)
                print(f"  Marking tile as FAILED and continuing...", flush=True)
                tile.status = "failed"
                self._save_state()
                completed += 1

        # Final visualization showing all completed tiles
        try:
            self._visualize_tile_map(current_tile_id=None)
            print("[Final] Saved final tile map visualization")
        except Exception as e:
            print(f"[Final visualization ERROR] {e}")

        # Tile split tree (시간/인과 관계)
        try:
            self._visualize_split_tree()
            print("[Final] Saved tile split tree visualization")
        except Exception as e:
            print(f"[Final split tree ERROR] {e}")

        # 완료 타일들의 최종 PLY 를 하나의 scene 으로 머지
        try:
            merge_final_scene(self.output_path)
        except Exception as e:
            print(f"[merge_scene ERROR] {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Count successful vs failed/split tiles
        # split 타일은 자식으로 대체된 내부 노드이므로 완료 판정은 리프 기준
        success_count = sum(1 for tile in self.tiles.values() if tile.status == "completed")
        failed_count = sum(1 for tile in self.tiles.values() if tile.status == "failed")
        split_count = sum(1 for tile in self.tiles.values() if tile.status == "split")
        leaf_count = len(self.tiles) - split_count

        print("\n" + "=" * 60)
        if success_count == leaf_count and failed_count == 0:
            print("🎉 Adaptive Tile Training - ALL TILES COMPLETED! 🎉")
        else:
            print("⚠️  Adaptive Tile Training - PARTIAL COMPLETION ⚠️")
        print(f"  Successfully completed tiles: {success_count} / {leaf_count} (leaf)")
        print(f"  Failed tiles: {failed_count}")
        print(f"  Split tiles (OOM, replaced by children): {split_count}")
        print(f"  Total tiles: {len(self.tiles)}")
        if success_count < leaf_count or failed_count > 0:
            print("  ⚠️  Some tiles did not complete successfully - check logs for errors")
        print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Tile Training Wrapper")

    # Required
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to COLMAP source data")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory")

    # Training
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs")
    parser.add_argument("--iterations", type=int, default=30000,
                        help="Training iterations per tile")
    parser.add_argument("--backend", type=str, default="default",
                        help="Rendering backend (gsplat or default)")
    parser.add_argument("--bsz", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--tile_crop_margin", type=int, default=100,
                        help="Pixel margin for camera crops")
    parser.add_argument("--scene_margin", type=float, default=0.0,
                        help="Scene bbox margin ratio (e.g., 0.1 = 10%%)")
    parser.add_argument("--ndc_limit", type=float, default=1.0,
                        help="NDC limit for projection filtering (default: 1.0)")
    parser.add_argument("--densify_from_iter", type=int, default=500,
                        help="Start densification from this iteration (default: 500)")
    parser.add_argument("--densification_interval", type=int, default=100,
                        help="Densification interval (default: 100)")
    parser.add_argument("--epoch_cap", type=int, default=300,
                        help="타일별 iteration 상한 = epoch_cap × 카메라 수 (하한 1500). "
                             "카메라 적은 타일의 과적합 방지. 0=비활성 "
                             "(표준 3DGS 30k ≈ 100~200 에폭 기준)")
    parser.add_argument("--fresh_start", action="store_true",
                        help="기존 adaptive_state.json 이 있어도 무시하고 처음부터 시작 "
                             "(기본: state 가 있으면 이어서 진행)")
    parser.add_argument("--child_densify_grad_threshold", type=float, default=None,
                        help="Resume 자식 타일 전용 densify threshold. "
                             "explosive 부모에서 물려받은 자식이 재폭증해 완주 못 하는 것을 방지 "
                             "(default: None = 부모와 동일)")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002,
                        help="Gradient threshold for densification (default: 0.0002, lower=faster growth)")
    parser.add_argument("--visual_debug", action="store_true",
                        help="Enable visual debugging for projection and crop calculations")
    parser.add_argument("--visual_debug_only", action="store_true",
                        help="Run only visual debugging and exit (no training)")
    parser.add_argument("--visual_debug_level", type=int, default=0,
                        help="Debug at specific tile level (default: 0, use higher for split tiles)")

    return parser.parse_args()


class TeeOutput:
    """Write to both stdout and a file."""
    def __init__(self, file_path, stdout):
        self.file = open(file_path, 'a', buffering=1)  # Line buffered
        self.stdout = stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    args = parse_args()

    # Set up logging to file unless disabled by environment variable
    # If disabled, rely on caller (e.g., run_adaptive.sh) to capture stdout/stderr.
    disable_log = os.environ.get("TRAIN_ADAPTIVE_NO_LOG", "0") == "1"

    import datetime
    # Always capture originals for safe cleanup in exception handler
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = None
    tee_stderr = None
    if not disable_log:
        log_file = Path(args.output_path) / "train_adaptive.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Tee stdout and stderr to both console and file
        tee_stdout = TeeOutput(log_file, original_stdout)
        tee_stderr = TeeOutput(log_file, original_stderr)

        sys.stdout = tee_stdout
        sys.stderr = tee_stderr

        print(f"\n{'='*60}")
        print(f"[{datetime.datetime.now().isoformat()}] train_adaptive.py started")
        print(f"Log file: {log_file}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"[{datetime.datetime.now().isoformat()}] train_adaptive.py started")
        print("Log file: (disabled; using caller stdout/stderr)")
        print(f"{'='*60}\n")

    try:
        trainer = AdaptiveTileTrainer(args)
        trainer.run()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Exception occurred:", flush=True)
        traceback.print_exc()
        raise
    finally:
        print(f"\n[{datetime.datetime.now().isoformat()}] train_adaptive.py finished")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if tee_stdout is not None:
            tee_stdout.close()
        if tee_stderr is not None:
            tee_stderr.close()


if __name__ == "__main__":
    main()
