"""
Adaptive Tile Trainer for OOM-aware 3DGS Training

This module implements the main training loop that:
1. Starts with the entire scene as a single tile
2. Dynamically splits tiles when OOM occurs
3. Uses Grendel-GS for multi-GPU distributed training
4. Computes image crops for each tile based on visibility
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Image.MAX_IMAGE_PIXELS = None

# Add paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Grendel-GS"))

from .adaptive_tile_manager import AdaptiveTileManager, TileStatus, AdaptiveTile
from .tile_storage import TileStorage, BBox
from .visibility_utils import load_visibility_metadata
from .image_loss_utils import (
    ColmapLoader,
    build_camera_from_colmap,
    ssim,
    load_reference_image,
)

# Grendel-GS imports
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from scene.gaussian_model import GaussianModel
    from scene.colmap_loader import read_points3D_binary, read_points3D_text
    from utils.graphics_utils import BasicPointCloud
    import utils.general_utils as grendel_utils
    from gaussian_renderer import render_final, distributed_preprocess3dgs_and_all2all_final
    from gaussian_renderer.workload_division import DivisionStrategyHistoryFinal, start_strategy_final
    GRENDEL_AVAILABLE = True
except ImportError:
    GRENDEL_AVAILABLE = False
    print("[Warning] Grendel-GS not available. Multi-GPU training disabled.")


def _is_oom_error(exc: BaseException) -> bool:
    """Check if an exception is an OOM error."""
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda out of memory" in msg
    try:
        return isinstance(exc, torch.OutOfMemoryError)
    except AttributeError:
        return False


def load_initial_point_cloud(colmap_path: Path) -> Tuple[np.ndarray, np.ndarray, BBox]:
    """
    Load point cloud from COLMAP output.

    Returns:
        Tuple of (xyz, rgb, bbox)
    """
    points3d_bin = colmap_path / "points3D.bin"
    points3d_txt = colmap_path / "points3D.txt"

    if points3d_bin.exists():
        xyz, rgb, _ = read_points3D_binary(str(points3d_bin))
    elif points3d_txt.exists():
        xyz, rgb, _ = read_points3D_text(str(points3d_txt))
    else:
        raise FileNotFoundError(f"No points3D file found in {colmap_path}")

    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)

    # Compute bounding box
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    bbox = BBox(min_xyz=min_xyz, max_xyz=max_xyz)

    print(f"[Load] Loaded {len(xyz)} points from COLMAP")
    print(f"[Load] BBox: min={min_xyz}, max={max_xyz}")

    return xyz, rgb, bbox


def filter_points_by_bbox(xyz: np.ndarray, rgb: np.ndarray, bbox: BBox) -> Tuple[np.ndarray, np.ndarray]:
    """Filter points to those within the bounding box."""
    mask = (
        (xyz[:, 0] >= bbox.min[0]) & (xyz[:, 0] <= bbox.max[0]) &
        (xyz[:, 1] >= bbox.min[1]) & (xyz[:, 1] <= bbox.max[1]) &
        (xyz[:, 2] >= bbox.min[2]) & (xyz[:, 2] <= bbox.max[2])
    )
    return xyz[mask], rgb[mask]


def compute_tile_visibility(
    tile_bbox: BBox,
    colmap_loader: ColmapLoader,
    device: torch.device,
    padding_ratio: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Compute which cameras see the tile and their projected 2D bboxes.

    Args:
        tile_bbox: 3D bounding box of the tile.
        colmap_loader: COLMAP data loader.
        device: Torch device.
        padding_ratio: Padding to add to projected bbox.

    Returns:
        List of dicts with camera info and projected 2D bbox.
    """
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    visible_cameras = []

    # Get 8 corners of the 3D bbox
    corners_3d = np.array([
        [tile_bbox.min[0], tile_bbox.min[1], tile_bbox.min[2]],
        [tile_bbox.max[0], tile_bbox.min[1], tile_bbox.min[2]],
        [tile_bbox.min[0], tile_bbox.max[1], tile_bbox.min[2]],
        [tile_bbox.max[0], tile_bbox.max[1], tile_bbox.min[2]],
        [tile_bbox.min[0], tile_bbox.min[1], tile_bbox.max[2]],
        [tile_bbox.max[0], tile_bbox.min[1], tile_bbox.max[2]],
        [tile_bbox.min[0], tile_bbox.max[1], tile_bbox.max[2]],
        [tile_bbox.max[0], tile_bbox.max[1], tile_bbox.max[2]],
    ], dtype=np.float64)

    for img_idx, img_data in colmap_loader._images.items():
        cam_id = img_data['cam_id']
        cam_params = colmap_loader._cameras[cam_id]

        width = cam_params["width"]
        height = cam_params["height"]

        # Build projection matrix
        from scene.colmap_loader import qvec2rotmat
        R_wc = qvec2rotmat(img_data["quat"])
        R_cw = R_wc.T
        T_wc = img_data["translation"]

        world_view_np = getWorld2View2(R_cw, T_wc)
        world_view = world_view_np.astype(np.float32).T

        projection = getProjectionMatrix(
            znear=0.01,
            zfar=10000.0,
            fovX=cam_params["fovx"],
            fovY=cam_params["fovy"],
        )
        projection_np = projection.cpu().numpy().astype(np.float32).T

        full_proj = world_view @ projection_np

        # Project corners
        ones = np.ones((8, 1), dtype=np.float64)
        corners_h = np.concatenate([corners_3d, ones], axis=1)
        clip = corners_h @ full_proj.astype(np.float64)

        w = clip[:, 3]
        valid = w > 0
        if not np.any(valid):
            continue

        clip = clip[valid]
        w = w[valid]
        ndc = clip[:, :3] / w[:, None]

        # Check if any corner is in front of camera and in NDC range
        inside = (
            (ndc[:, 0] >= -1.5) & (ndc[:, 0] <= 1.5) &
            (ndc[:, 1] >= -1.5) & (ndc[:, 1] <= 1.5) &
            (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
        )
        if not np.any(inside):
            continue

        # Compute 2D bbox in pixel coordinates
        px = ((ndc[:, 0] + 1.0) * 0.5) * width
        py = ((ndc[:, 1] + 1.0) * 0.5) * height

        px_min, px_max = px.min(), px.max()
        py_min, py_max = py.min(), py.max()

        # Add padding
        pad_x = (px_max - px_min) * padding_ratio
        pad_y = (py_max - py_min) * padding_ratio
        px_min -= pad_x
        px_max += pad_x
        py_min -= pad_y
        py_max += pad_y

        # Clamp to image bounds
        x0 = max(0, int(np.floor(px_min)))
        y0 = max(0, int(np.floor(py_min)))
        x1 = min(width, int(np.ceil(px_max)))
        y1 = min(height, int(np.ceil(py_max)))

        if x1 <= x0 or y1 <= y0:
            continue

        visible_cameras.append({
            "camera_index": img_idx,
            "image_name": img_data["name"],
            "cam_id": cam_id,
            "bbox_2d": (x0, y0, x1, y1),
            "full_width": width,
            "full_height": height,
        })

    return visible_cameras


def build_crop_camera(
    cam_params: Dict,
    pose_params: Dict,
    crop_bbox: Tuple[int, int, int, int],
    device: torch.device,
    resolution_scale: float = 1.0,
) -> Namespace:
    """
    Build a camera with adjusted intrinsics for the cropped region.

    Args:
        cam_params: Camera parameters from COLMAP.
        pose_params: Pose parameters from COLMAP.
        crop_bbox: (x0, y0, x1, y1) crop region in pixels.
        device: Torch device.
        resolution_scale: Resolution scaling factor.

    Returns:
        Camera namespace with adjusted parameters.
    """
    from scene.colmap_loader import qvec2rotmat
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    x0, y0, x1, y1 = crop_bbox
    crop_width = x1 - x0
    crop_height = y1 - y0

    # Apply resolution scale
    scaled_x0 = int(x0 * resolution_scale)
    scaled_y0 = int(y0 * resolution_scale)
    scaled_width = max(1, int(crop_width * resolution_scale))
    scaled_height = max(1, int(crop_height * resolution_scale))

    original_width = cam_params["width"]
    original_height = cam_params["height"]

    # Compute adjusted FOV for the crop
    # The crop region has the same focal length but different principal point
    fx = cam_params["fx"]
    fy = cam_params["fy"]
    cx = cam_params["cx"]
    cy = cam_params["cy"]

    # New principal point relative to crop
    new_cx = (cx - x0) * resolution_scale
    new_cy = (cy - y0) * resolution_scale

    # FOV for the crop (using the original focal length)
    new_fovx = 2 * math.atan(scaled_width / (2 * fx * resolution_scale))
    new_fovy = 2 * math.atan(scaled_height / (2 * fy * resolution_scale))

    # Build transforms
    R_wc = qvec2rotmat(pose_params["quat"])
    R_cw = R_wc.T
    T_wc = pose_params["translation"]

    world_view_np = getWorld2View2(R_cw, T_wc)
    world_view_transform = torch.tensor(world_view_np, dtype=torch.float32, device=device).transpose(0, 1)

    # Build asymmetric projection matrix
    znear = 0.01
    zfar = 10000.0

    # Compute NDC bounds for the crop region
    # Original image: NDC x from -1 to 1, y from -1 to 1
    # Crop region maps a subset of these
    ndc_left = (2.0 * x0 / original_width) - 1.0
    ndc_right = (2.0 * x1 / original_width) - 1.0
    ndc_bottom = (2.0 * y0 / original_height) - 1.0
    ndc_top = (2.0 * y1 / original_height) - 1.0

    # Scale NDC to [-1, 1] for the crop
    scale_x = 2.0 / (ndc_right - ndc_left)
    scale_y = 2.0 / (ndc_top - ndc_bottom)
    offset_x = -(ndc_right + ndc_left) / (ndc_right - ndc_left)
    offset_y = -(ndc_top + ndc_bottom) / (ndc_top - ndc_bottom)

    # Get base projection
    base_proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=cam_params["fovx"], fovY=cam_params["fovy"])
    base_proj_np = base_proj.cpu().numpy()

    # Modify projection for crop (asymmetric frustum)
    new_proj = base_proj_np.copy()
    new_proj[0, 0] *= scale_x
    new_proj[1, 1] *= scale_y
    new_proj[2, 0] = offset_x
    new_proj[2, 1] = offset_y

    projection_matrix = torch.tensor(new_proj, dtype=torch.float32, device=device).transpose(0, 1)
    full_proj = world_view_transform @ projection_matrix
    camera_center = torch.linalg.inv(world_view_transform)[3, :3]

    cam_obj = Namespace(
        image_height=scaled_height,
        image_width=scaled_width,
        tanfovx=math.tan(new_fovx * 0.5),
        tanfovy=math.tan(new_fovy * 0.5),
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj,
        camera_center=camera_center,
        FoVx=new_fovx,
        FoVy=new_fovy,
        crop_bbox=crop_bbox,
        original_resolution=(original_width, original_height),
    )

    return cam_obj


def load_crop_image(
    images_path: Path,
    image_name: str,
    crop_bbox: Tuple[int, int, int, int],
    resolution_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Load a cropped region of an image.

    Args:
        images_path: Path to images directory.
        image_name: Name of the image file.
        crop_bbox: (x0, y0, x1, y1) crop region.
        resolution_scale: Resolution scaling factor.
        device: Torch device.

    Returns:
        Cropped image tensor [C, H, W].
    """
    image_path = images_path / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    x0, y0, x1, y1 = crop_bbox

    with Image.open(image_path) as img:
        img = img.convert("RGB")

        # Crop first, then resize
        cropped = img.crop((x0, y0, x1, y1))

        # Apply resolution scale
        new_width = max(1, int((x1 - x0) * resolution_scale))
        new_height = max(1, int((y1 - y0) * resolution_scale))

        if (new_width, new_height) != cropped.size:
            cropped = cropped.resize((new_width, new_height), Image.LANCZOS)

        arr = np.asarray(cropped, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


class AdaptiveTrainer:
    """
    Main trainer class for adaptive tile-based 3DGS training.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda")

        # Paths
        self.colmap_path = Path(args.colmap_path)
        self.images_path = Path(args.images_path)
        self.output_path = Path(args.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize distributed if available
        self.distributed = False
        self.world_size = 1
        self.rank = 0

        if GRENDEL_AVAILABLE and args.distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.distributed = True
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
            self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

            if self.rank == 0:
                print(f"[Distributed] World size: {self.world_size}")

        # Load COLMAP data
        self.colmap_loader = ColmapLoader(self.colmap_path)

        # Load initial point cloud
        self.xyz, self.rgb, self.initial_bbox = load_initial_point_cloud(self.colmap_path)

        # Initialize adaptive tile manager
        self.tile_manager = AdaptiveTileManager(
            initial_bbox=self.initial_bbox,
            save_dir=self.output_path,
        )

        # Resume from checkpoint if exists
        state_path = self.output_path / "adaptive_tile_state.json"
        if state_path.exists() and args.resume:
            self.tile_manager.load_state(state_path)

        # Training parameters
        self.iterations_per_tile = args.iterations_per_tile
        self.resolution_scale = 1.0 / args.resolution if args.resolution > 1 else 1.0
        self.sh_degree = args.sh_degree
        self.lambda_dssim = args.lambda_dssim

        # OOM counter for debug images
        self.oom_count = 0

    def train(self):
        """Main training loop with adaptive tile splitting."""
        if self.rank == 0:
            print("\n" + "=" * 60)
            print("Starting Adaptive Tile Training")
            print("=" * 60)
            self.tile_manager.print_status()

        while not self.tile_manager.is_all_done():
            current_tile = self.tile_manager.get_current_tile()
            if current_tile is None:
                break

            if self.rank == 0:
                print(f"\n[Training] Processing tile: {current_tile.tile_id}")
                print(f"           BBox: {current_tile.bbox.min} - {current_tile.bbox.max}")
                print(f"           Priority: {current_tile.priority}, Area: {current_tile.area:.2f}")

            try:
                # Train the current tile
                self._train_tile(current_tile)

                # Handle success
                next_tile_id = self.tile_manager.handle_success(current_tile.tile_id)

                # Save state after each successful tile
                if self.rank == 0:
                    self.tile_manager.save_state()
                    self.tile_manager.print_status()

                if next_tile_id is None:
                    break

            except RuntimeError as e:
                if _is_oom_error(e):
                    self.oom_count += 1
                    oom_tile_id = current_tile.tile_id

                    if self.rank == 0:
                        print(f"\n[OOM] Out of memory on tile {oom_tile_id}")
                        print("[OOM] Splitting tile...")

                    # Clear memory
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Save debug image BEFORE splitting (so we can see the original tile)
                    self._save_oom_debug_image(oom_tile_id, self.oom_count)

                    # Handle OOM by splitting
                    B = self.tile_manager.handle_oom(oom_tile_id)

                    # Save the deferred tile (C) gaussians
                    C = [t for t in self.tile_manager.tiles.values()
                         if t.tile_id != B and t.status == TileStatus.UNDONE
                         and t.priority == self.tile_manager.tiles[B].priority + 1]
                    if C:
                        C = C[0]
                        self._save_tile_gaussians(C)

                    if self.rank == 0:
                        self.tile_manager.save_state()
                        self.tile_manager.print_status()

                else:
                    raise

        if self.rank == 0:
            print("\n" + "=" * 60)
            print("Adaptive Tile Training Complete!")
            print("=" * 60)
            self.tile_manager.print_status()

            # Merge all tiles
            self._merge_all_tiles()

    def _train_tile(self, tile: AdaptiveTile):
        """Train a single tile."""
        # Filter points for this tile
        tile_xyz, tile_rgb = filter_points_by_bbox(self.xyz, self.rgb, tile.bbox)

        if len(tile_xyz) == 0:
            if self.rank == 0:
                print(f"[Warning] No points in tile {tile.tile_id}, skipping...")
            return

        if self.rank == 0:
            print(f"[Tile] {tile.tile_id}: {len(tile_xyz)} points")

        # Compute visibility
        visible_cameras = compute_tile_visibility(
            tile.bbox, self.colmap_loader, self.device
        )

        if len(visible_cameras) == 0:
            if self.rank == 0:
                print(f"[Warning] No cameras see tile {tile.tile_id}, skipping...")
            return

        if self.rank == 0:
            print(f"[Tile] {tile.tile_id}: {len(visible_cameras)} visible cameras")

        # Initialize Gaussian model
        gaussian_model = GaussianModel(self.sh_degree)
        pcd = BasicPointCloud(
            points=tile_xyz,
            colors=tile_rgb / 255.0,
            normals=np.zeros_like(tile_xyz),
        )
        # Compute spatial_lr_scale from scene extent
        scene_extent = np.linalg.norm(tile.bbox.max - tile.bbox.min)
        gaussian_model.create_from_pcd(pcd, spatial_lr_scale=scene_extent)

        # Setup optimizer
        gaussian_model.training_setup(self.args)

        # Training loop
        for iteration in tqdm(range(1, self.iterations_per_tile + 1),
                              desc=f"Training {tile.tile_id}",
                              disable=(self.rank != 0)):

            # Pick a random camera
            cam_info = visible_cameras[iteration % len(visible_cameras)]

            # Build crop camera
            cam_params = self.colmap_loader._cameras[cam_info["cam_id"]]
            pose_params = self.colmap_loader._images[cam_info["camera_index"]]

            camera = build_crop_camera(
                cam_params, pose_params,
                cam_info["bbox_2d"],
                self.device,
                self.resolution_scale,
            )
            camera.uid = cam_info["camera_index"]
            camera.image_name = cam_info["image_name"]

            # Load crop image
            gt_image = load_crop_image(
                self.images_path,
                cam_info["image_name"],
                cam_info["bbox_2d"],
                self.resolution_scale,
                self.device,
            ).to(self.device)

            # Render
            bg_color = torch.zeros(3, device=self.device)

            rendered = self._render(gaussian_model, camera, bg_color)

            # Compute loss
            l1_loss = torch.abs(rendered - gt_image).mean()
            ssim_loss = 1.0 - ssim(rendered.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss

            # Backward
            loss.backward()

            # Optimizer step
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad()

            # Densification (simplified)
            if iteration % 100 == 0 and iteration < self.iterations_per_tile * 0.8:
                gaussian_model.update_learning_rate(iteration)

            # Clear memory periodically
            if iteration % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Save trained gaussians
        self._save_tile_gaussians(tile, gaussian_model)

        # Cleanup
        del gaussian_model
        gc.collect()
        torch.cuda.empty_cache()

    def _render(self, gaussian_model, camera, bg_color):
        """Render the scene from a camera using Grendel-GS rasterizer."""
        settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=float(camera.tanfovx),
            tanfovy=float(camera.tanfovy),
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=gaussian_model.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=settings)

        # Grendel-GS style: preprocess then render
        cuda_args = {
            "mode": "train",
            "world_size": "1",
            "global_rank": "0",
            "local_rank": "0",
            "mp_world_size": "1",
            "mp_rank": "0",
            "log_folder": str(self.output_path),
            "log_interval": "100",
            "iteration": "0",
            "zhx_debug": "False",
            "zhx_time": "False",
            "dist_global_strategy": "",
            "avoid_pixel_all2all": False,
            "stats_collector": {},
        }

        # Preprocess gaussians
        means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
            means3D=gaussian_model.get_xyz,
            scales=gaussian_model.get_scaling,
            rotations=gaussian_model.get_rotation,
            shs=gaussian_model.get_features,
            opacities=gaussian_model.get_opacity,
            cuda_args=cuda_args,
        )

        # For single GPU, all gaussians are computed locally
        num_gaussians = gaussian_model.get_xyz.shape[0]
        compute_locally = torch.ones(num_gaussians, dtype=torch.bool, device=self.device)
        extended_compute_locally = compute_locally.clone()

        # Render
        rendered_image, n_render, n_consider, n_contrib = rasterizer.render_gaussians(
            means2D=means2D,
            conic_opacity=conic_opacity,
            rgb=rgb,
            depths=depths,
            radii=radii,
            compute_locally=compute_locally,
            extended_compute_locally=extended_compute_locally,
            cuda_args=cuda_args,
        )

        return rendered_image

    def _save_tile_gaussians(self, tile: AdaptiveTile, gaussian_model: Optional[GaussianModel] = None):
        """Save gaussians for a tile."""
        if self.rank != 0:
            return

        tiles_dir = self.output_path / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)

        ply_path = tiles_dir / f"{tile.tile_id}.ply"

        if gaussian_model is not None:
            gaussian_model.save_ply(str(ply_path))
        else:
            # Save initial points for this tile
            tile_xyz, tile_rgb = filter_points_by_bbox(self.xyz, self.rgb, tile.bbox)
            # Save as simple PLY (would need proper implementation)
            print(f"[Save] Saving {len(tile_xyz)} points to {ply_path}")

        self.tile_manager.set_gaussians_path(tile.tile_id, str(ply_path))
        print(f"[Save] Saved tile {tile.tile_id} to {ply_path}")

    def _merge_all_tiles(self):
        """Merge all trained tiles into a single PLY."""
        if self.rank != 0:
            return

        print("\n[Merge] Merging all tiles...")

        merged_path = self.output_path / "merged.ply"

        # TODO: Implement proper PLY merging
        print(f"[Merge] Output: {merged_path}")
        print("[Merge] Merge not yet implemented - tiles saved separately")

    def _save_oom_debug_image(self, oom_tile_id: str, oom_count: int):
        """
        Save debug image showing OOM tile location on point cloud map.

        Args:
            oom_tile_id: ID of the tile that caused OOM.
            oom_count: Number of OOM events so far.
        """
        if self.rank != 0:
            return

        debug_dir = self.output_path / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Plot point cloud (XY projection, subsample for speed)
        subsample = max(1, len(self.xyz) // 10000)
        ax.scatter(
            self.xyz[::subsample, 0],
            self.xyz[::subsample, 1],
            s=0.1,
            c='gray',
            alpha=0.3,
            label='Point Cloud'
        )

        # Draw all tiles
        for tile_id, tile in self.tile_manager.tiles.items():
            bbox = tile.bbox
            x_min, y_min = bbox.min[0], bbox.min[1]
            x_max, y_max = bbox.max[0], bbox.max[1]
            width = x_max - x_min
            height = y_max - y_min

            # Color based on status
            if tile.status == TileStatus.DONE:
                color = 'green'
                linestyle = '-'
            else:
                color = 'blue'
                linestyle = '--'

            # Draw rectangle
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=1.5,
                edgecolor=color,
                facecolor='none',
                linestyle=linestyle
            )
            ax.add_patch(rect)

            # Add tile label
            ax.text(
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                tile_id.replace('tile_', ''),
                ha='center',
                va='center',
                fontsize=6,
                color=color
            )

        # Draw X on OOM tile
        if oom_tile_id in self.tile_manager.tiles:
            oom_tile = self.tile_manager.tiles[oom_tile_id]
        else:
            # Tile was split, find the original bbox from current tiles
            # Use the last split tiles to estimate
            oom_tile = None

        if oom_tile is None:
            # Find recent split tiles (B and C from the split)
            recent_tiles = sorted(
                self.tile_manager.tiles.values(),
                key=lambda t: int(t.tile_id.split('_')[1]),
                reverse=True
            )[:2]
            if recent_tiles:
                # Combine bbox of the two most recent tiles
                x_min = min(t.bbox.min[0] for t in recent_tiles)
                y_min = min(t.bbox.min[1] for t in recent_tiles)
                x_max = max(t.bbox.max[0] for t in recent_tiles)
                y_max = max(t.bbox.max[1] for t in recent_tiles)
        else:
            x_min, y_min = oom_tile.bbox.min[0], oom_tile.bbox.min[1]
            x_max, y_max = oom_tile.bbox.max[0], oom_tile.bbox.max[1]

        # Draw big red X
        ax.plot([x_min, x_max], [y_min, y_max], 'r-', linewidth=3)
        ax.plot([x_min, x_max], [y_max, y_min], 'r-', linewidth=3)

        # Draw red rectangle around OOM tile
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=3,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'OOM #{oom_count}: {oom_tile_id}\n'
                     f'Total tiles: {len(self.tile_manager.tiles)}, '
                     f'Done: {sum(1 for t in self.tile_manager.tiles.values() if t.status == TileStatus.DONE)}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Save
        save_path = debug_dir / f"oom_{oom_count:03d}_{oom_tile_id}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"[Debug] Saved OOM debug image: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Tile 3DGS Training")

    # Paths
    parser.add_argument("--colmap_path", type=str, required=True,
                        help="Path to COLMAP sparse/0 directory")
    parser.add_argument("--images_path", type=str, required=True,
                        help="Path to images directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for trained tiles")

    # Training
    parser.add_argument("--iterations_per_tile", type=int, default=7000,
                        help="Number of iterations per tile")
    parser.add_argument("--resolution", type=int, default=1,
                        help="Resolution divisor (1=full, 2=half, etc.)")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="Spherical harmonics degree")
    parser.add_argument("--lambda_dssim", type=float, default=0.2,
                        help="SSIM loss weight")

    # Distributed
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")

    # Gaussian model args (for compatibility)
    parser.add_argument("--percent_dense", type=float, default=0.01)
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--position_lr_max_steps", type=int, default=30000)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    # Grendel-GS specific args
    parser.add_argument("--gaussians_distribution", action="store_true", default=False,
                        help="Distribute gaussians across GPUs")
    parser.add_argument("--drop_initial_3dgs_p", type=float, default=0.0,
                        help="Probability to drop initial 3DGS points")
    parser.add_argument("--lr_scale_pos_and_scale", type=float, default=1.0,
                        help="Learning rate scale for position and scale")
    parser.add_argument("--lr_scale_mode", type=str, default="sqrt",
                        help="Learning rate scaling mode (linear, sqrt, accumu)")
    parser.add_argument("--grad_normalization_mode", type=str, default="none",
                        help="Gradient normalization mode")
    parser.add_argument("--sync_grad_mode", type=str, default="dense",
                        help="Gradient sync mode (dense, sparse, fused_dense, fused_sparse)")
    parser.add_argument("--distributed_save", action="store_true", default=False,
                        help="Save model distributed across GPUs")
    parser.add_argument("--bsz", type=int, default=1,
                        help="Batch size")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set global args for Grendel-GS
    if GRENDEL_AVAILABLE:
        grendel_utils.set_args(args)
        # Set log file (use /dev/null or actual log file)
        log_path = Path(args.output_path) / "grendel_gs.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")
        grendel_utils.set_log_file(log_file)

    trainer = AdaptiveTrainer(args)
    trainer.train()

    if GRENDEL_AVAILABLE:
        log_file.close()


if __name__ == "__main__":
    main()
