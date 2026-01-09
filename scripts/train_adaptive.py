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
    status: str  # pending, in_progress, completed, split
    fail_iter: Optional[int] = None  # Iteration at which tile failed/split
    num_cameras: Optional[int] = None  # Number of visible cameras


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
        self.tiles_dir = self.output_path / "tiles"
        self.tiles_dir.mkdir(parents=True, exist_ok=True)

        # Load scene info
        self.scene_bbox, self.num_points = self._load_scene_info()
        self.cam_infos = self._load_camera_infos()

        # Load or initialize tile queue
        self.tiles: Dict[str, TileInfo] = {}
        self.tile_counter = 0
        self.vis_counter = 0  # Counter for visualization filenames
        self.min_successful_level = None  # Track minimum level (smallest = largest tile) that completed successfully
        self.initial_area = None  # Area of the initial tile (level 0)
        if self.state_file.exists() and args.resume:
            self._load_state()
        else:
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

        # Use percentiles to exclude outliers (0.1% ~ 99.9%)
        pct_low, pct_high = 0.1, 99.9
        x_min, y_min, z_min = np.percentile(xyz, pct_low, axis=0)
        x_max, y_max, z_max = np.percentile(xyz, pct_high, axis=0)

        bbox = BBox(
            float(x_min), float(y_min), float(z_min),
            float(x_max), float(y_max), float(z_max)
        )

        print(f"[Scene] Loaded {len(xyz)} points")
        print(f"[Scene] Extent (percentile {pct_low}%-{pct_high}%, excluding outliers):")
        print(f"  X: {bbox.x_min:.3f} ~ {bbox.x_max:.3f} (size: {bbox.size[0]:.3f})")
        print(f"  Y: {bbox.y_min:.3f} ~ {bbox.y_max:.3f} (size: {bbox.size[1]:.3f})")
        print(f"  Z: {bbox.z_min:.3f} ~ {bbox.z_max:.3f} (size: {bbox.size[2]:.3f})")

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
            elif intr.model in ["PINHOLE", "OPENCV"]:
                focal_x = intr.params[0]
                focal_y = intr.params[1]
            else:
                focal_x = intr.params[0]
                focal_y = intr.params[0]

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
                "width": intr.width,
                "height": intr.height,
            })

        print(f"[Scene] Loaded {len(cam_infos)} cameras")
        return cam_infos

    def _compute_visible_cameras(self, tile_bbox: BBox, margin: int = 100) -> List[str]:
        """
        Compute which cameras can see the tile.

        Returns list of visible camera image names.
        """
        import math
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix

        visible = []
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

        return visible

    def _clear_visualizations(self):
        """Clear visualizations folder."""
        import shutil
        vis_dir = self.output_path / "visualizations"
        if vis_dir.exists():
            shutil.rmtree(vis_dir)
            print(f"[Init] Cleared visualizations folder: {vis_dir}")
        vis_dir.mkdir(parents=True, exist_ok=True)
        self.vis_counter = 0  # Reset counter

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
                    "num_cameras": t.num_cameras
                }
                for tid, t in self.tiles.items()
            }
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from file."""
        with open(self.state_file, "r") as f:
            state = json.load(f)

        self.tile_counter = state["tile_counter"]
        self.vis_counter = state.get("vis_counter", 0)  # Default to 0 for old state files
        self.initial_area = state.get("initial_area")
        self.min_successful_level = state.get("min_successful_level")
        self.tiles = {}
        for tid, t in state["tiles"].items():
            self.tiles[tid] = TileInfo(
                tile_id=t["tile_id"],
                bbox=BBox.from_string(t["bbox"]),
                status=t["status"],
                fail_iter=t.get("fail_iter"),
                num_cameras=t.get("num_cameras")
            )
        print(f"[Resume] Loaded {len(self.tiles)} tiles from state")
        # If initial_area is not in state (old state file), compute from scene_bbox
        if self.initial_area is None:
            margin = self.args.scene_margin
            dx, dy, dz = self.scene_bbox.size
            self.initial_area = (dx * (1 + 2 * margin)) * (dy * (1 + 2 * margin))
            print(f"[Resume] Computed initial_area from scene_bbox: {self.initial_area:.1f}")
        if self.min_successful_level is not None:
            print(f"[Resume] Min successful level: {self.min_successful_level}")

    def _visualize_tile_map(self, current_tile_id: str = None):
        """
        Visualize tile map as a 2D X-Y plot and save to PNG.

        Args:
            current_tile_id: The tile currently being processed (highlighted)
        """
        # Status colors (for status indicator circles)
        # Note: "split" is further divided into "split_oom" and "split_preemptive" based on fail_iter
        status_colors = {
            "completed": "#4CAF50",       # Green
            "in_progress": "#2196F3",     # Blue
            "pending": "#9E9E9E",         # Gray
            "split_oom": "#FF9800",       # Orange - OOM split (tried and failed)
            "split_preemptive": "#9C27B0", # Purple - Preemptive split (skipped without trying)
            "split": "#FF9800",           # Orange - fallback for old data
            "failed": "#F44336",          # Red
            "skipped": "#795548",         # Brown
        }

        # Tile border/label colors (high contrast, distinguishable)
        tile_colors = [
            "#FF0000",  # Red
            "#0000FF",  # Blue
            "#00AA00",  # Green
            "#FF00FF",  # Magenta
            "#00CCCC",  # Cyan
            "#FF8800",  # Orange
            "#8800FF",  # Purple
            "#888800",  # Olive
            "#FF0088",  # Pink
            "#0088FF",  # Sky blue
            "#00FF88",  # Spring green
            "#880000",  # Dark red
            "#000088",  # Dark blue
            "#008800",  # Dark green
            "#880088",  # Dark magenta
            "#008888",  # Dark cyan
            "#884400",  # Brown
            "#440088",  # Indigo
            "#448800",  # Dark olive
            "#004488",  # Navy
        ]

        fig, ax = plt.subplots(1, 1, figsize=(14, 12))

        # Draw scene bbox as background
        scene_rect = patches.Rectangle(
            (self.scene_bbox.x_min, self.scene_bbox.y_min),
            self.scene_bbox.size[0], self.scene_bbox.size[1],
            linewidth=2, edgecolor='black', facecolor='#f5f5f5',
            linestyle='--', label='Scene extent'
        )
        ax.add_patch(scene_rect)

        # Calculate max tile size for linewidth scaling
        max_tile_area = max(t.bbox.size[0] * t.bbox.size[1] for t in self.tiles.values())
        import math

        # Sort tiles by size (largest first, so smaller tiles draw on top)
        sorted_tiles = sorted(self.tiles.items(),
                              key=lambda x: x[1].bbox.size[0] * x[1].bbox.size[1],
                              reverse=True)

        # Collect for legend
        status_drawn = set()
        text_labels = []  # Store text labels to draw last

        # Draw all tiles (largest first)
        for idx, (tile_id, tile) in enumerate(sorted_tiles):
            bbox = tile.bbox
            tile_color = tile_colors[idx % len(tile_colors)]

            # Determine status color (split into oom vs preemptive)
            if tile.status == "split":
                if tile.fail_iter == 0:
                    display_status = "split_preemptive"
                else:
                    display_status = "split_oom"
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
                        info_parts.append(f"OOM@{tile.fail_iter}")
                if tile.num_cameras is not None:
                    info_parts.append(f"C{tile.num_cameras}")
                label_text += f"\n({', '.join(info_parts)})"

            text_labels.append((center_x, center_y, label_text, font_size,
                               tile_id == current_tile_id, tile_color, status_color, display_status))

            # Track status for legend (use display_status to distinguish split types)
            status_drawn.add(display_status)

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

        # Add legend for status colors (draw last, on top)
        legend_handles = []
        for status, color in status_colors.items():
            if status in status_drawn:
                handle = patches.Patch(facecolor=color, edgecolor='white', label=status.capitalize())
                legend_handles.append(handle)
        legend = ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
        legend.set_zorder(200)

        plt.tight_layout()

        # Save to PNG
        vis_dir = self.output_path / "visualizations"
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

        print(f"  [Visualization] Saved: {save_path}", flush=True)

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

        cmd = [
            "torchrun",
            f"--nproc_per_node={self.args.num_gpus}",
            str(self.grendel_dir / "train.py"),
            "--source_path", str(self.source_path),
            "--model_path", str(tile_model_path),
            "--log_folder", str(tile_log_path),
            "--iterations", str(self.args.iterations),
            "--backend", self.args.backend,
            "--bsz", str(self.args.bsz),
            "--adaptive_tile_enabled",
            f"--tile_bbox={tile.bbox.to_string()}",
            "--tile_id", tile.tile_id,
            "--tile_output_dir", str(self.tiles_dir),
            "--tile_crop_margin", str(self.args.tile_crop_margin),
            "--ndc_limit", str(self.args.ndc_limit),
        ]

        # Add visible cameras
        if visible_cameras:
            cmd.extend(["--visible_cameras", ",".join(visible_cameras)])

        print(f"\n[Tile {tile.tile_id}] Running torchrun...")
        print(f"  BBox: {tile.bbox.to_string()}")
        print(f"  Visible cameras: {len(visible_cameras)}")

        result = subprocess.run(cmd, cwd=str(ROOT))

        # Check for OOM via state file (torchrun returns 1 even when worker exits with 42)
        state_file = self.tiles_dir / "adaptive_tile_state.json"
        oom_iteration = None
        if state_file.exists():
            try:
                with open(state_file) as f:
                    oom_state = json.load(f)
                if oom_state.get("oom_occurred"):
                    oom_iteration = oom_state.get("iteration")
                    print(f"  [OOM detected via state file at iteration {oom_iteration}]")
                    # Remove the state file after reading
                    state_file.unlink()
                    return self.EXIT_CODE_OOM, oom_iteration
            except (json.JSONDecodeError, IOError) as e:
                print(f"  [Warning] Failed to read OOM state file: {e}")

        return result.returncode, oom_iteration

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
            visible_cameras = self._compute_visible_cameras(tile.bbox, self.args.tile_crop_margin)
            tile.num_cameras = len(visible_cameras)
            print(f"  Visible cameras: {len(visible_cameras)} / {len(self.cam_infos)}")
            if visible_cameras:
                print(f"    Names: {visible_cameras[:5]}{'...' if len(visible_cameras) > 5 else ''}")

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
                    continue

            # Mark as in_progress
            tile.status = "in_progress"
            self._save_state()

            # Run training
            exit_code, oom_iteration = self._run_torchrun(tile, visible_cameras)

            if exit_code == self.EXIT_CODE_SUCCESS:
                tile.status = "completed"
                # Track successful tile level (smaller level = larger tile)
                tile_area = tile.bbox.size[0] * tile.bbox.size[1]
                tile_level = self._get_tile_level(tile_area)
                if self.min_successful_level is None or tile_level < self.min_successful_level:
                    self.min_successful_level = tile_level
                    print(f"  [Success] Updated min successful level: {tile_level} (area: {tile_area:.1f})")
                self._save_state()
                completed += 1
                print(f"[Tile {tile.tile_id}] Completed successfully (level {tile_level}).")

            elif exit_code == self.EXIT_CODE_OOM:
                print("\n" + "#" * 60, flush=True)
                print("###  OOM DETECTED - SPLITTING TILE  ###", flush=True)
                print("#" * 60, flush=True)
                tile_level = self._get_tile_level(tile_area)
                print(f"[Tile {tile.tile_id}] OOM at iteration {oom_iteration}, level={tile_level}, splitting...", flush=True)
                tile.fail_iter = oom_iteration

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
                print("#" * 60 + "\n", flush=True)

                # Insert split tiles at the beginning so they are processed immediately
                # Rebuild tiles dict with split tiles first
                new_tiles = {}
                new_tiles[tile_a_id] = TileInfo(tile_a_id, bbox_a, "pending")
                new_tiles[tile_b_id] = TileInfo(tile_b_id, bbox_b, "pending")
                for tid, t in self.tiles.items():
                    new_tiles[tid] = t
                self.tiles = new_tiles

                self._save_state()

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

        print("\n" + "=" * 60)
        print("Adaptive Tile Training Complete!")
        print(f"  Completed tiles: {completed}")
        print(f"  Total tiles: {len(self.tiles)}")
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

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing state")

    return parser.parse_args()


def main():
    args = parse_args()
    trainer = AdaptiveTileTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
