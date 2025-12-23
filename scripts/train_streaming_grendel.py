#!/usr/bin/env python3
"""
Dynamic Single-Image Grendel Trainer.
Combines Streaming (OOC) with Grendel's Distributed Rendering.
"""

import argparse
import json
import sys
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import math
import os
import torch.distributed as dist
import queue
import threading
from dataclasses import dataclass
from tqdm import tqdm


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 0:
        return f"{num_bytes}B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(num_bytes)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TiB"


def _numel_from_shape(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return int(n)


def _tensor_bytes_from_shape(shape, dtype: torch.dtype) -> int:
    return _numel_from_shape(shape) * torch.tensor([], dtype=dtype).element_size()


def estimate_gaussian_param_vram_bytes(num_gaussians: int, sh_degree: int, *, dtype: torch.dtype = torch.float32) -> Dict[str, int]:
    """Estimate VRAM bytes for trainable Gaussian parameters + grads + densification stats.

    This is an upper-bound style estimate assuming:
    - Parameters live on GPU in fp32.
    - Autograd keeps a gradient tensor for each parameter (same shape).
    - Densification stats stored in GaussianModel: max_radii2D [N], xyz_gradient_accum [N,1], denom [N,1], send_to_gpui_cnt [N,1 int32].
    """

    # Shapes match how TileBatchTensors.populate_gaussian_model creates Parameters.
    bands_rest = (sh_degree + 1) ** 2 - 1

    param_shapes = {
        "xyz": (num_gaussians, 3),
        "scaling": (num_gaussians, 3),
        "rotation": (num_gaussians, 4),
        "opacity": (num_gaussians, 1),
        "features_dc": (num_gaussians, 1, 3),
        "features_rest": (num_gaussians, bands_rest, 3),
    }

    bytes_params = 0
    bytes_grads = 0
    for _name, shape in param_shapes.items():
        b = _tensor_bytes_from_shape(shape, dtype)
        bytes_params += b
        bytes_grads += b

    # GaussianModel extra buffers used by densification.
    bytes_densify = 0
    bytes_densify += _tensor_bytes_from_shape((num_gaussians,), dtype)  # max_radii2D
    bytes_densify += _tensor_bytes_from_shape((num_gaussians, 1), dtype)  # xyz_gradient_accum
    bytes_densify += _tensor_bytes_from_shape((num_gaussians, 1), dtype)  # denom
    bytes_densify += _tensor_bytes_from_shape((num_gaussians, 1), torch.int32)  # send_to_gpui_cnt

    return {
        "params": int(bytes_params),
        "grads": int(bytes_grads),
        "densify_buffers": int(bytes_densify),
        "total": int(bytes_params + bytes_grads + bytes_densify),
    }


def estimate_adam_state_vram_bytes(num_gaussians: int, sh_degree: int, *, dtype: torch.dtype = torch.float32) -> Dict[str, int]:
    """Estimate VRAM bytes for Adam optimizer states.

    The codebase caches per-parameter momentum buffers (m, v) for:
    xyz, features_dc, features_rest, opacity, scaling, rotation.
    That is 2x tensors per parameter.
    """
    bands_rest = (sh_degree + 1) ** 2 - 1
    shapes = {
        "xyz": (num_gaussians, 3),
        "features_dc": (num_gaussians, 1, 3),
        "features_rest": (num_gaussians, bands_rest, 3),
        "opacity": (num_gaussians, 1),
        "scaling": (num_gaussians, 3),
        "rotation": (num_gaussians, 4),
    }

    bytes_m = 0
    bytes_v = 0
    for _name, shape in shapes.items():
        b = _tensor_bytes_from_shape(shape, dtype)
        bytes_m += b
        bytes_v += b
    return {
        "adam_m": int(bytes_m),
        "adam_v": int(bytes_v),
        "total": int(bytes_m + bytes_v),
    }


def estimate_ssim_working_vram_bytes(chunk_h: int, chunk_w: int, *, channels: int = 3, dtype: torch.dtype = torch.float32, window_size: int = 11) -> Dict[str, int]:
    """Estimate peak-ish extra VRAM used by SSIM() for a single chunk.

    ssim() calls several conv2d ops and keeps intermediates:
      mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12, ssim_map
    Each is roughly [N=1, C, H, W] in fp32.
    This ignores temporary CUDA workspace and autograd saved tensors; treat as a conservative-but-not-perfect indicator.
    """
    # Assume batch=1
    base = _tensor_bytes_from_shape((1, channels, chunk_h, chunk_w), dtype)
    # 9 large feature maps + (small) window tensor
    large_maps = 9
    window = _tensor_bytes_from_shape((channels, 1, window_size, window_size), dtype)
    return {
        "feature_maps": int(large_maps * base),
        "window": int(window),
        "total": int(large_maps * base + window),
    }

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "Grendel-GS"))
    # Add compiled extension path
    sys.path.insert(0, str(ROOT / "Grendel-GS" / "submodules" / "diff-gaussian-rasterization" / "build" / "lib.linux-x86_64-cpython-38"))

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
import utils.general_utils as utils
from src.tile_storage import TileStorage, BBox
from src.tiled_scene import LRUCache
from src.tile_training_utils import TileBatchTensors
from src.visibility_utils import load_visibility_metadata
from src.image_patch_cache import ImagePatchCache, SharedMemoryImageCache
from src.image_loss_utils import prepare_image_loss, ColmapLoader, build_camera_from_colmap, ssim, load_reference_image, split_camera_vertical
from src.spatial_index import UniformGrid
from scene.colmap_loader import qvec2rotmat

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from argparse import Namespace

# Grendel-GS Distributed Imports
from gaussian_renderer import render_final, distributed_preprocess3dgs_and_all2all_final
from gaussian_renderer.workload_division import DivisionStrategyHistoryFinal, start_strategy_final, finish_strategy_final
import torch.distributed.nn.functional as dist_nn

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def isatty(self):
        return False

class PersistentLRUCache(LRUCache):
    def __init__(self, capacity, storage, output_storage=None, on_evict=None):
        super().__init__(capacity)
        self.storage = storage
        self.output_storage = output_storage if output_storage else storage
        self.on_evict = on_evict

    def put(self, tile_coord, tile_data):
        if len(self.cache) >= self.capacity and tile_coord not in self.cache:
            oldest_coord = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            data_to_save, _ = self.cache[oldest_coord]
            self.output_storage.save_tile(oldest_coord, data_to_save)
            if self.on_evict:
                self.on_evict(oldest_coord)
        super().put(tile_coord, tile_data)

class StreamingScene:
    def __init__(self, tiles_root: Path, device: torch.device, output_dir: Optional[Path] = None, cache_size: int = 200, sh_degree: int = 3, resume_from: Optional[Path] = None):
        self.tiles_root = tiles_root
        self.output_dir = output_dir if output_dir else tiles_root
        self.device = device
        self.sh_degree = sh_degree
        self.resume_from = resume_from
        
        # If resuming, load from checkpoint instead of initial tiles
        if resume_from and resume_from.exists():
            if utils.GLOBAL_RANK == 0:
                print(f"[Resume] Loading checkpoint from {resume_from}")
            self.storage = TileStorage(resume_from)
        else:
            self.storage = TileStorage(tiles_root)
            
        if self.output_dir != self.tiles_root:
            self.output_storage = TileStorage(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy global metadata from input storage to output storage
            # This ensures grid_size, scene_bbox, etc. are preserved
            modified = False
            for k, v in self.storage.metadata.items():
                if k != 'tiles' and k not in self.output_storage.metadata:
                    self.output_storage.metadata[k] = v
                    modified = True
            
            if modified:
                self.output_storage._save_metadata()
        else:
            self.output_storage = self.storage
            
        self.cache = PersistentLRUCache(
            capacity=cache_size, 
            storage=self.storage, 
            output_storage=self.output_storage,
            on_evict=self._on_tile_evicted
        )
        self.metadata = self.storage.metadata
        
        if 'scene_bbox' not in self.metadata:
            meta_path = tiles_root / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    extra_meta = json.load(f)
                    self.metadata.update(extra_meta)
        
        self.grid_size = tuple(self.metadata.get('grid_size', [32, 32, 1]))
        self.overlap = self.metadata.get('overlap_meters', 10.0)
        
        self.active_tile_ids: Set[str] = set()
        self.tile_batch: Optional[TileBatchTensors] = None
        self.gaussian_model: Optional[GaussianModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.opt_state_cache: Dict[str, Dict] = {}
        
        # Load optimizer states from checkpoint if resuming
        if resume_from and resume_from.exists():
            self._load_optimizer_states_from_checkpoint(resume_from)
        
        scene_bbox = self.metadata.get('scene_bbox', {})
        if scene_bbox:
            min_xyz = np.array(scene_bbox['min'])
            self.spatial_index = UniformGrid(
                points=np.array([min_xyz, np.array(scene_bbox['max'])]), 
                grid_size=self.grid_size,
                overlap_meters=self.overlap
            )
            self.spatial_index.scene_min = np.array(scene_bbox['min'])
            self.spatial_index.scene_max = np.array(scene_bbox['max'])
            self.spatial_index.scene_extent = self.spatial_index.scene_max - self.spatial_index.scene_min
            self.spatial_index.tile_size = self.spatial_index.scene_extent / self.spatial_index.grid_size
        else:
            self.spatial_index = None
            
        # Start with SH degree 0 (DC only), will be upgraded to 1 at iter 1200
        # This matches the original 3DGS training strategy
        self.active_sh_degree = 0

    def _load_tile_data(self, tile_id: str) -> Dict:
        coord = self.storage._tile_id_to_coord(tile_id)
        cached = self.cache.get(coord)
        if cached is not None:
            return cached
        
        data = None
        if self.output_storage != self.storage:
            data = self.output_storage.load_tile(coord)
        if data is None:
            data = self.storage.load_tile(coord)
            
        if data is None:
            raise FileNotFoundError(f"Tile {tile_id} not found")

        self.cache.put(coord, data)
        return data

    def _save_tile_data(self, tile_id: str, data: Dict):
        coord = self.storage._tile_id_to_coord(tile_id)
        self.cache.put(coord, data)

    def update_active_tiles(self, needed_tile_ids: List[str], current_iter: int, reset_interval: int, xyz_lr: float = 0.00016):
        needed_set = set(needed_tile_ids)
        
        if needed_set == self.active_tile_ids and self.gaussian_model is not None:
            if self.optimizer is not None:
                for param_group in self.optimizer.param_groups:
                    if "name" in param_group and param_group["name"] == "xyz":
                        param_group["lr"] = xyz_lr
            return

        if self.gaussian_model is not None:
            self._extract_and_cache_current_state()
            
            # Save tiles that are being deactivated to prevent data loss and race conditions
            dropping_tiles = self.active_tile_ids - needed_set
            for tile_id in dropping_tiles:
                coord = self.storage._tile_id_to_coord(tile_id)
                if coord in self.cache.cache:
                    data, _ = self.cache.cache[coord]
                    self.output_storage.save_tile(coord, data)
                
                if tile_id in self.opt_state_cache:
                    state = self.opt_state_cache[tile_id]
                    opt_state_dir = self.output_dir / "opt_state"
                    opt_state_dir.mkdir(exist_ok=True, parents=True)
                    np.savez_compressed(opt_state_dir / f"opt_state_{tile_id}.npz", **state)
                    # Aggressively free memory for deactivated tiles
                    del self.opt_state_cache[tile_id]
        
        new_active_ids = list(needed_set)
        self.tile_batch = self._load_batch(new_active_ids)
        self.active_tile_ids = set(new_active_ids)

        # --- Deduplication & Re-binning ---
        # Disabled to prevent data loss/duplication issues during streaming
        if False and self.tile_batch.xyz.shape[0] > 0:
            xyz_np = self.tile_batch.xyz.cpu().numpy()
            xyz_rounded = np.round(xyz_np, 5)
            _, unique_indices = np.unique(xyz_rounded, axis=0, return_index=True)
            
            if len(unique_indices) < len(xyz_np):
                if utils.GLOBAL_RANK == 0:
                    print(f"[Load] Deduplication: {len(xyz_np)} -> {len(unique_indices)} points")
                
                unique_indices_tensor = torch.from_numpy(unique_indices).to(self.device)
                
                # 1. Filter Batch Tensors
                self.tile_batch.xyz = self.tile_batch.xyz[unique_indices_tensor]
                self.tile_batch.scales = self.tile_batch.scales[unique_indices_tensor]
                self.tile_batch.quats = self.tile_batch.quats[unique_indices_tensor]
                self.tile_batch.opacities = self.tile_batch.opacities[unique_indices_tensor]
                self.tile_batch.features_dc = self.tile_batch.features_dc[unique_indices_tensor]
                self.tile_batch.features_rest = self.tile_batch.features_rest[unique_indices_tensor]
                if self.tile_batch.accum_grad is not None:
                    self.tile_batch.accum_grad = self.tile_batch.accum_grad[unique_indices_tensor]
                if self.tile_batch.accum_count is not None:
                    self.tile_batch.accum_count = self.tile_batch.accum_count[unique_indices_tensor]
                if self.tile_batch.max_radii2D is not None:
                    self.tile_batch.max_radii2D = self.tile_batch.max_radii2D[unique_indices_tensor]

                # 2. Handle Opt State (Collect -> Filter)
                param_names = ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']
                collected_state = {}
                has_state = any(tid in self.opt_state_cache for tid in new_active_ids)
                
                if has_state:
                    for name in param_names:
                        for suffix in ['m', 'v']:
                            key = f"{name}_{suffix}"
                            tensors = []
                            for tile_id in new_active_ids:
                                sl = self.tile_batch.tile_slices[tile_id]
                                size = sl.stop - sl.start
                                if tile_id in self.opt_state_cache and key in self.opt_state_cache[tile_id]:
                                    tensors.append(torch.from_numpy(self.opt_state_cache[tile_id][key]).to(self.device))
                                else:
                                    # Create zeros matching shape
                                    if name == 'xyz': shape = (size, 3)
                                    elif name == 'features_dc': shape = (size, 1, 3)
                                    elif name == 'features_rest': shape = (size, self.tile_batch.features_rest.shape[1], 3)
                                    elif name == 'opacity': shape = (size, 1)
                                    elif name == 'scaling': shape = (size, 3)
                                    elif name == 'rotation': shape = (size, 4)
                                    tensors.append(torch.zeros(shape, device=self.device, dtype=torch.float32))
                            collected_state[key] = torch.cat(tensors)[unique_indices_tensor]

                # 3. Re-binning (Assign to Tiles)
                xyz_np = self.tile_batch.xyz.cpu().numpy()
                # scales_np = torch.exp(self.tile_batch.scales).cpu().numpy()
                assignments = self.spatial_index.assign_points_to_tiles_strict(xyz_np)
                
                new_xyz, new_scales, new_quats, new_opacities, new_sh0, new_shN = [], [], [], [], [], []
                new_accum_grad, new_accum_count, new_max_radii2D = [], [], []
                new_slices, new_coords = {}, {}
                
                offset = 0
                for coord, indices in assignments.items():
                    tile_id = self.storage._tile_coord_to_id(coord)
                    if tile_id in self.active_tile_ids:
                        indices = np.array(indices)
                        count = len(indices)
                        new_slices[tile_id] = slice(offset, offset + count)
                        new_coords[tile_id] = coord
                        offset += count
                        
                        idx_tensor = torch.from_numpy(indices).to(self.device)
                        
                        new_xyz.append(self.tile_batch.xyz[idx_tensor])
                        new_scales.append(self.tile_batch.scales[idx_tensor])
                        new_quats.append(self.tile_batch.quats[idx_tensor])
                        new_opacities.append(self.tile_batch.opacities[idx_tensor])
                        new_sh0.append(self.tile_batch.features_dc[idx_tensor])
                        new_shN.append(self.tile_batch.features_rest[idx_tensor])
                        
                        if self.tile_batch.accum_grad is not None:
                            new_accum_grad.append(self.tile_batch.accum_grad[idx_tensor])
                        if self.tile_batch.accum_count is not None:
                            new_accum_count.append(self.tile_batch.accum_count[idx_tensor])
                        if self.tile_batch.max_radii2D is not None:
                            new_max_radii2D.append(self.tile_batch.max_radii2D[idx_tensor])
                        
                        if has_state:
                            tile_state = {}
                            old_step = self.opt_state_cache.get(tile_id, {}).get('step', 0)
                            tile_state['step'] = old_step
                            for k in collected_state:
                                tile_state[k] = collected_state[k][idx_tensor].cpu().numpy()
                            self.opt_state_cache[tile_id] = tile_state

                # 4. Update TileBatch
                if new_xyz:
                    self.tile_batch.xyz = torch.cat(new_xyz)
                    self.tile_batch.scales = torch.cat(new_scales)
                    self.tile_batch.quats = torch.cat(new_quats)
                    self.tile_batch.opacities = torch.cat(new_opacities)
                    self.tile_batch.features_dc = torch.cat(new_sh0)
                    self.tile_batch.features_rest = torch.cat(new_shN)
                    self.tile_batch.tile_slices = new_slices
                    self.tile_batch.coords = new_coords
                    
                    if new_accum_grad: self.tile_batch.accum_grad = torch.cat(new_accum_grad)
                    if new_accum_count: self.tile_batch.accum_count = torch.cat(new_accum_count)
                    if new_max_radii2D: self.tile_batch.max_radii2D = torch.cat(new_max_radii2D)

        self.gaussian_model = GaussianModel(sh_degree=self.sh_degree)
        self.tile_batch.populate_gaussian_model(self.gaussian_model)
        self.gaussian_model.active_sh_degree = self.active_sh_degree
        
        num_points = self.gaussian_model.get_xyz.shape[0]
        if num_points == 0:
            self.optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=0.0)
            self.gaussian_model.optimizer = self.optimizer
            return

        l = [
            {'params': [self.gaussian_model._xyz], 'lr': xyz_lr, "name": "xyz"},
            {'params': [self.gaussian_model._features_dc], 'lr': 0.0025, "name": "f_dc"},
            {'params': [self.gaussian_model._features_rest], 'lr': 0.0025/20.0, "name": "f_rest"},
            {'params': [self.gaussian_model._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self.gaussian_model._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self.gaussian_model._rotation], 'lr': 0.001, "name": "rotation"}
        ]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.gaussian_model.optimizer = self.optimizer
        self._restore_optimizer_state()

    def _load_batch(self, tile_ids: List[str]) -> TileBatchTensors:
        xyz_list = []
        scales_list = []
        quats_list = []
        opacities_list = []
        sh0_list = []
        sh_rest_list = []
        accum_grad_list = []
        accum_count_list = []
        max_radii2D_list = []
        
        tile_slices: Dict[str, slice] = {}
        coords: Dict[str, Tuple[int, int, int]] = {}

        offset = 0
        for tile_id in tile_ids:
            try:
                data = self._load_tile_data(tile_id)
            except FileNotFoundError:
                continue

            coord = self.storage._tile_id_to_coord(tile_id)
            count = len(data["means"])
            tile_slices[tile_id] = slice(offset, offset + count)
            coords[tile_id] = coord
            offset += count

            xyz_list.append(torch.from_numpy(data["means"]))
            scales_list.append(torch.from_numpy(data["scales"]))
            quats_list.append(torch.from_numpy(data["quats"]))
            opacities = data["opacities"].reshape(-1, 1)
            opacities_list.append(torch.from_numpy(opacities))
            sh0_list.append(torch.from_numpy(data["sh0"]))
            
            # Slice shN based on requested sh_degree to save memory
            shN_data = data["shN"]
            target_sh_dim = (self.sh_degree + 1) ** 2 - 1
            if shN_data.shape[1] > target_sh_dim:
                shN_data = shN_data[:, :target_sh_dim]
            
            sh_rest_list.append(torch.from_numpy(shN_data))
            
            if 'accum_grad' in data:
                accum_grad_list.append(torch.from_numpy(data['accum_grad']))
            else:
                accum_grad_list.append(torch.zeros((count, 1), dtype=torch.float32))
            if 'accum_count' in data:
                accum_count_list.append(torch.from_numpy(data['accum_count']))
            else:
                accum_count_list.append(torch.zeros((count, 1), dtype=torch.float32))
            if 'max_radii2D' in data:
                max_radii2D_list.append(torch.from_numpy(data['max_radii2D']))
            else:
                max_radii2D_list.append(torch.zeros((count,), dtype=torch.float32))

        def cat(l, shape_suffix): 
            if not l: return torch.empty((0,) + shape_suffix, device=self.device, dtype=torch.float32)
            return torch.cat(l, dim=0).to(self.device, dtype=torch.float32).contiguous()
        
        # Infer SH degree from actual data shape
        if sh_rest_list:
            actual_sh_dim = sh_rest_list[0].shape[1]
        else:
            actual_sh_dim = (self.sh_degree + 1) ** 2 - 1
        
        return TileBatchTensors(
            xyz=cat(xyz_list, (3,)),
            scales=cat(scales_list, (3,)),
            quats=cat(quats_list, (4,)),
            opacities=cat(opacities_list, (1,)),
            features_dc=cat(sh0_list, (1, 3)),
            features_rest=cat(sh_rest_list, (actual_sh_dim, 3)),
            tile_slices=tile_slices,
            coords=coords,
            accum_grad=cat(accum_grad_list, (1,)),
            accum_count=cat(accum_count_list, (1,)),
            max_radii2D=cat(max_radii2D_list, ())
        )

    def _extract_and_cache_current_state(self):
        if self.gaussian_model is None or self.tile_batch is None:
            return

        self.tile_batch.copy_from_model(self.gaussian_model)
        
        for tile_id in self.active_tile_ids:
            if tile_id not in self.tile_batch.tile_slices: continue
            slice_idx = self.tile_batch.tile_slices[tile_id]
            
            data = {
                'means': self.tile_batch.xyz[slice_idx].detach().cpu().numpy(),
                'scales': self.tile_batch.scales[slice_idx].detach().cpu().numpy(),
                'quats': self.tile_batch.quats[slice_idx].detach().cpu().numpy(),
                'opacities': self.tile_batch.opacities[slice_idx].detach().cpu().numpy().flatten(),
                'sh0': self.tile_batch.features_dc[slice_idx].detach().cpu().numpy(),
                'shN': self.tile_batch.features_rest[slice_idx].detach().cpu().numpy(),
            }
            
            if self.tile_batch.accum_grad is not None:
                data['accum_grad'] = self.tile_batch.accum_grad[slice_idx].detach().cpu().numpy()
            if self.tile_batch.accum_count is not None:
                data['accum_count'] = self.tile_batch.accum_count[slice_idx].detach().cpu().numpy()
            if self.tile_batch.max_radii2D is not None:
                data['max_radii2D'] = self.tile_batch.max_radii2D[slice_idx].detach().cpu().numpy()
            
            if len(data['means']) > 0:
                min_xyz = data['means'].min(axis=0)
                max_xyz = data['means'].max(axis=0)
                data['bbox'] = BBox(min_xyz, max_xyz)
            else:
                data['bbox'] = BBox([0,0,0], [0,0,0])

            self._save_tile_data(tile_id, data)
            
            opt_state = self._extract_opt_state_slice(slice_idx)
            self.opt_state_cache[tile_id] = opt_state

    def _extract_opt_state_slice(self, span: slice) -> Optional[Dict]:
        param_names = ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']
        saved_state = {}
        
        for i, name in enumerate(param_names):
            group = self.optimizer.param_groups[i]
            p = group["params"][0]
            s = self.optimizer.state.get(p)
            
            if s:
                m = s.get("exp_avg")
                v = s.get("exp_avg_sq")
                step = s.get("step")
                
                if m is not None:
                    saved_state[f"{name}_m"] = m[span].detach().cpu().numpy()
                if v is not None:
                    saved_state[f"{name}_v"] = v[span].detach().cpu().numpy()
                if step is not None:
                    saved_state["step"] = int(step.item()) if isinstance(step, torch.Tensor) else step
                
        return saved_state

    def _restore_optimizer_state(self):
        param_names = ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']
        
        for tile_id in self.active_tile_ids:
            if tile_id not in self.opt_state_cache:
                continue
                
            state_dict = self.opt_state_cache[tile_id]
            span = self.tile_batch.tile_slices[tile_id]
            
            current_size = span.stop - span.start
            if 'xyz_m' in state_dict:
                cached_size = state_dict['xyz_m'].shape[0]
                if cached_size != current_size:
                    del self.opt_state_cache[tile_id]
                    continue
            
            for i, name in enumerate(param_names):
                group = self.optimizer.param_groups[i]
                p = group["params"][0]
                m_key = f"{name}_m"
                v_key = f"{name}_v"
                
                if m_key in state_dict and v_key in state_dict:
                    if p not in self.optimizer.state:
                        self.optimizer.state[p] = {}
                        self.optimizer.state[p]['step'] = torch.tensor(state_dict.get("step", 0), device=self.device)
                        self.optimizer.state[p]['exp_avg'] = torch.zeros_like(p)
                        self.optimizer.state[p]['exp_avg_sq'] = torch.zeros_like(p)
                    
                    m_data = torch.from_numpy(state_dict[m_key]).to(self.device)
                    v_data = torch.from_numpy(state_dict[v_key]).to(self.device)
                    
                    self.optimizer.state[p]['exp_avg'][span] = m_data
                    self.optimizer.state[p]['exp_avg_sq'][span] = v_data

    def save_all_active_tiles(self, backup_iteration: Optional[int] = None):
        self._extract_and_cache_current_state()
        
        # 1. Save current active tiles to main output
        for coord, (data, _) in self.cache.cache.items():
            self.output_storage.save_tile(coord, data)
        
        # 2. Save current active opt_state to main output
        opt_state_dir = self.output_dir / "opt_state"
        opt_state_dir.mkdir(exist_ok=True, parents=True)
        for tile_id, state in self.opt_state_cache.items():
            np.savez_compressed(opt_state_dir / f"opt_state_{tile_id}.npz", **state)
            
        if self.output_storage != self.storage:
            self.output_storage._save_metadata()

        # 3. Backup to iteration folder if requested
        if backup_iteration is not None:
            backup_dir = self.output_dir / f"iteration_{backup_iteration}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize backup storage
            backup_storage = TileStorage(backup_dir)
            
            # Optimization: Copy ALL metadata first (including 'tiles')
            # This avoids re-generating metadata for inactive tiles
            import copy
            backup_storage.metadata = copy.deepcopy(self.output_storage.metadata)
            
            import shutil
            
            # A. Backup Tiles: Cache (Active) + Disk (Inactive)
            # First, save what's in cache (most recent) - this updates metadata/files for active tiles
            for coord, (data, _) in self.cache.cache.items():
                backup_storage.save_tile(coord, data)
            
            # Then, copy remaining tiles from disk that are NOT in cache
            # Optimization: Use direct file copy instead of load+save (Decompress+Compress)
            for tile_coord_str in self.output_storage.metadata.get('tiles', {}):
                coord = self.storage._tile_id_to_coord(tile_coord_str)
                if coord not in self.cache.cache:
                    tile_id = self.storage._tile_coord_to_id(coord)
                    src_path = self.output_storage.tiles_dir / f"tile_{tile_id}.npz"
                    dst_path = backup_storage.tiles_dir / f"tile_{tile_id}.npz"
                    
                    if src_path.exists():
                        shutil.copy(src_path, dst_path)
            
            backup_storage._save_metadata()
            
            # B. Backup Opt State: Cache (Active) + Disk (Inactive)
            backup_opt_dir = backup_dir / "opt_state"
            backup_opt_dir.mkdir(parents=True, exist_ok=True)
            
            # First, save what's in cache (most recent)
            for tile_id, state in self.opt_state_cache.items():
                np.savez_compressed(backup_opt_dir / f"opt_state_{tile_id}.npz", **state)
            
            # Then, copy remaining opt_states from disk that are NOT in cache
            if opt_state_dir.exists():
                for opt_file in opt_state_dir.glob("opt_state_*.npz"):
                    tile_id = opt_file.stem.replace("opt_state_", "")
                    if tile_id not in self.opt_state_cache:
                        shutil.copy(opt_file, backup_opt_dir / opt_file.name)
                
            if utils.GLOBAL_RANK == 0:
                print(f"[Save] Backup created at {backup_dir}")

    def _extract_full_opt_state(self) -> Dict[str, torch.Tensor]:
        param_names = ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']
        full_state = {}
        
        for i, name in enumerate(param_names):
            group = self.optimizer.param_groups[i]
            p = group["params"][0]
            s = self.optimizer.state.get(p)
            
            if s:
                m = s.get("exp_avg")
                v = s.get("exp_avg_sq")
                if m is not None:
                    full_state[f"{name}_m"] = m
                if v is not None:
                    full_state[f"{name}_v"] = v
        return full_state

    def densify_and_prune(self, viewspace_point_tensor, visibility_filter, radii, threshold=0.0002, xyz_lr=0.00016, skip_add_stats=False):
        self.gaussian_model.max_radii2D[visibility_filter] = torch.max(
            self.gaussian_model.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
        if not skip_add_stats:
            self.gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)
        
        scene_extent = np.max(self.spatial_index.scene_extent) if self.spatial_index else 100.0
        
        grads = self.gaussian_model.xyz_gradient_accum / self.gaussian_model.denom
        grads[grads.isnan()] = 0.0

        before_count = self.gaussian_model.get_xyz.shape[0]
        
        if utils.GLOBAL_RANK == 0:
            max_grad = grads.max().item()
            print(f"[Densify] Max Gradient Norm: {max_grad:.8f} (Threshold: {threshold})")

        self.gaussian_model.densify_and_clone(grads, threshold, scene_extent)
        after_count = self.gaussian_model.get_xyz.shape[0]
        
        if after_count > before_count:
            with torch.no_grad():
                noise = (torch.rand(after_count - before_count, 3, device=self.device) - 0.5) * 0.0001
                self.gaussian_model._xyz.data[before_count:] += noise

        self.gaussian_model.densify_and_split(grads, threshold, scene_extent)
        
        min_opacity = 0.005
        max_screen_size = 20.0
        
        prune_mask = (self.gaussian_model.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.gaussian_model.max_radii2D > max_screen_size
            big_points_ws = self.gaussian_model.get_scaling.max(dim=1).values > 0.001 * scene_extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.gaussian_model.prune_points(prune_mask)
        
        torch.cuda.empty_cache()
        
        xyz = self.gaussian_model._xyz.detach()
        scales = self.gaussian_model._scaling.detach()
        rots = self.gaussian_model._rotation.detach()
        opacities = self.gaussian_model._opacity.detach()
        sh0 = self.gaussian_model._features_dc.detach()
        shN = self.gaussian_model._features_rest.detach()
        
        opt_state = self._extract_full_opt_state()
        
        xyz_np = xyz.cpu().numpy()
        
        # Deduplication removed from here to avoid cracking. 
        # It should be handled during tile loading if necessary.
        # xyz_rounded = np.round(xyz_np, 5)
        # _, unique_indices = np.unique(xyz_rounded, axis=0, return_index=True)
        
        # if utils.GLOBAL_RANK == 0:
        #     print(f"[Densify] Deduplication: {len(xyz)} -> {len(unique_indices)} points")
        
        # unique_indices = torch.from_numpy(unique_indices).to(self.device)
        
        # xyz = xyz[unique_indices]
        # scales = scales[unique_indices]
        # rots = rots[unique_indices]
        # opacities = opacities[unique_indices]
        # sh0 = sh0[unique_indices]
        # shN = shN[unique_indices]
        
        # for key in opt_state:
        #     opt_state[key] = opt_state[key][unique_indices]
        
        xyz_np = xyz.cpu().numpy()
        # scales_np = torch.exp(scales).cpu().numpy();
        
        assignments = self.spatial_index.assign_points_to_tiles_strict(xyz_np)
        
        new_xyz = []
        new_scales = []
        new_quats = []
        new_opacities = []
        new_sh0 = []
        new_shN = []
        
        new_slices = {}
        new_coords = {};
        
        param_names = ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']
        new_opt_m = {name: [] for name in param_names}
        new_opt_v = {name: [] for name in param_names}
        
        offset = 0;
        
        for coord, indices in assignments.items():
            tile_id = self.storage._tile_coord_to_id(coord);
            
            if tile_id in self.active_tile_ids:
                indices = np.array(indices);
                count = len(indices);
                new_slices[tile_id] = slice(offset, offset + count);
                new_coords[tile_id] = coord;
                offset += count;
                
                idx_tensor = torch.from_numpy(indices).to(self.device);
                
                new_xyz.append(xyz[idx_tensor]);
                new_scales.append(scales[idx_tensor]);
                new_quats.append(rots[idx_tensor]);
                new_opacities.append(opacities[idx_tensor]);
                new_sh0.append(sh0[idx_tensor]);
                new_shN.append(shN[idx_tensor]);
                
                for name in param_names:
                    if f"{name}_m" in opt_state:
                        new_opt_m[name].append(opt_state[f"{name}_m"][idx_tensor]);
                    if f"{name}_v" in opt_state:
                        new_opt_v[name].append(opt_state[f"{name}_v"][idx_tensor]);
            else:
                # Log dropped points for debugging
                # if utils.GLOBAL_RANK == 0:
                #     print(f"[Densify] Warning: Dropping {len(indices)} points moving to inactive tile {tile_id}")
                pass

        if not new_xyz:
            print("Warning: Densification resulted in empty active tiles!")
            return

        self.tile_batch = TileBatchTensors(
            xyz=torch.cat(new_xyz),
            scales=torch.cat(new_scales),
            quats=torch.cat(new_quats),
            opacities=torch.cat(new_opacities),
            features_dc=torch.cat(new_sh0),
            features_rest=torch.cat(new_shN),
            tile_slices=new_slices,
            coords=new_coords
        )
        
        self.gaussian_model = GaussianModel(sh_degree=self.sh_degree)
        self.tile_batch.populate_gaussian_model(self.gaussian_model)
        self.gaussian_model.active_sh_degree = self.active_sh_degree
        
        l = [
            {'params': [self.gaussian_model._xyz], 'lr': xyz_lr, "name": "xyz"},
            {'params': [self.gaussian_model._features_dc], 'lr': 0.0025, "name": "f_dc"},
            {'params': [self.gaussian_model._features_rest], 'lr': 0.0025/20.0, "name": "f_rest"},
            {'params': [self.gaussian_model._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self.gaussian_model._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self.gaussian_model._rotation], 'lr': 0.001, "name": "rotation"}
        ]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.gaussian_model.optimizer = self.optimizer
        
        for i, name in enumerate(param_names):
            group = self.optimizer.param_groups[i]
            p = group["params"][0]
            
            if new_opt_m[name]:
                m_data = torch.cat(new_opt_m[name])
                v_data = torch.cat(new_opt_v[name])
                
                self.optimizer.state[p] = {}
                self.optimizer.state[p]['step'] = torch.tensor(0, device=self.device) # Reset step? Or keep max?
                self.optimizer.state[p]['exp_avg'] = m_data
                self.optimizer.state[p]['exp_avg_sq'] = v_data

    def _on_tile_evicted(self, tile_coord):
        tile_id = self.storage._tile_coord_to_id(tile_coord)
        if tile_id in self.opt_state_cache:
            del self.opt_state_cache[tile_id]

    def _load_optimizer_states_from_checkpoint(self, checkpoint_dir: Path):
        """Load all optimizer states from checkpoint directory."""
        opt_state_dir = checkpoint_dir / "opt_state"
        if not opt_state_dir.exists():
            if utils.GLOBAL_RANK == 0:
                print(f"[Resume] Warning: No opt_state directory found in {checkpoint_dir}")
            return
        
        loaded_count = 0
        for opt_file in opt_state_dir.glob("opt_state_tile_*.npz"):
            tile_id = opt_file.stem.replace("opt_state_", "")
            try:
                data = np.load(opt_file)
                state_dict = {k: data[k] for k in data.files}
                self.opt_state_cache[tile_id] = state_dict
                loaded_count += 1
            except Exception as e:
                if utils.GLOBAL_RANK == 0:
                    print(f"[Resume] Warning: Failed to load {opt_file}: {e}")
        
        if utils.GLOBAL_RANK == 0:
            print(f"[Resume] Loaded {loaded_count} optimizer states from checkpoint")

class ReduceScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rendered_image, world_size, chunk_h):
        ctx.world_size = world_size
        ctx.chunk_h = chunk_h
        ctx.input_shape = rendered_image.shape
        
        # Memory-efficient implementation using loop of reduce
        # This avoids allocating a full copy of the image (which torch.stack or contiguous() would do)
        # Instead, we process one chunk at a time, allocating only 1/N extra memory.
        
        rank = dist.get_rank()
        result = None
        
        C, H, W = rendered_image.shape
        
        for r in range(world_size):
            # Calculate slice for rank r
            y_start = r * chunk_h
            y_end = min((r + 1) * chunk_h, H)
            
            # Extract slice (view)
            # Note: rendered_image is [C, H, W], slicing dim 1 is non-contiguous
            slice_r = rendered_image[:, y_start:y_end, :]
            
            # Make contiguous (allocates 1/N of image memory)
            slice_r_cont = slice_r.contiguous()
            
            # Reduce to destination rank
            dist.reduce(slice_r_cont, dst=r, op=dist.ReduceOp.SUM)
            
            if r == rank:
                result = slice_r_cont
            else:
                del slice_r_cont
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [C, chunk_h, W] - gradient for this rank's strip
        
        # Gather all gradients from all ranks
        grad_input_list = [torch.zeros_like(grad_output) for _ in range(ctx.world_size)]
        dist.all_gather(grad_input_list, grad_output)
        
        # Memory-efficient reconstruction: directly concatenate along height dimension
        # Each grad in list: [C, chunk_h, W]
        # Concatenate along dim=1 (height) to get [C, H, W] where H = WorldSize * chunk_h
        grad_input = torch.cat(grad_input_list, dim=1)
        
        return grad_input, None, None

def reduce_scatter_autograd(rendered_image, world_size, chunk_h):
    return ReduceScatterFunction.apply(rendered_image, world_size, chunk_h)

def _is_oom_error(exc: BaseException) -> bool:
    # torch.OutOfMemoryError exists on newer PyTorch. Some stacks raise RuntimeError with 'out of memory'.
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda out of memory" in msg
    try:
        return isinstance(exc, torch.OutOfMemoryError)  # type: ignore[attr-defined]
    except Exception:
        return False


def chunked_loss(rendered, gt, lambda_dssim, chunk_size=512, *, allow_disable_ssim_on_oom: bool = True):
    """
    Computes loss in chunks to avoid OOM on large images.
    """
    # Ensure inputs are 4D for consistency if needed, but ssim handles 3D/4D
    # rendered: [3, H, W] on GPU
    # gt: [3, H, W] on CPU (or GPU)
    
    H, W = rendered.shape[1], rendered.shape[2]
    total_loss = 0.0
    total_pixels = 0.0
    
    # If SSIM OOMs (common under fragmentation), progressively reduce chunk size.
    # As a last resort, disable SSIM term for this call to avoid aborting training.
    cur_chunk = int(chunk_size)
    disable_ssim = False

    y = 0
    while y < H:
        x = 0
        while x < W:
            y_end = min(y + cur_chunk, H)
            x_end = min(x + cur_chunk, W)

            r_chunk = rendered[:, y:y_end, x:x_end].unsqueeze(0)  # [1, 3, h, w]
            g_chunk = gt[:, y:y_end, x:x_end]

            # Convert uint8 to float on CPU first to avoid a larger transient on GPU.
            if g_chunk.dtype == torch.uint8:
                g_chunk = g_chunk.float().mul_(1.0 / 255.0)

            # Move to GPU in fp16 to reduce transient memory pressure.
            # (rendered is fp32; mixing dtypes here is OK since this is a loss term)
            try:
                g_chunk = g_chunk.to(device=rendered.device, dtype=torch.float16, non_blocking=True).unsqueeze(0)
            except Exception as e:
                if not _is_oom_error(e):
                    raise
                torch.cuda.empty_cache()
                if cur_chunk > 32:
                    new_chunk = max(32, cur_chunk // 2)
                    if utils.GLOBAL_RANK == 0:
                        print(f"[Loss] OOM while moving GT chunk (chunk={cur_chunk}). Reducing to {new_chunk}.")
                    cur_chunk = new_chunk
                    continue
                raise

            # Compute L1. Use fp32 for stability, relying on adaptive chunking to handle OOM.
            try:
                l1 = (r_chunk.float() - g_chunk.float()).abs().mean()
            except Exception as e:
                if not _is_oom_error(e):
                    raise
                torch.cuda.empty_cache()
                if cur_chunk > 32:
                    new_chunk = max(32, cur_chunk // 2)
                    if utils.GLOBAL_RANK == 0:
                        print(f"[Loss] OOM during L1 (chunk={cur_chunk}). Reducing to {new_chunk}.")
                    cur_chunk = new_chunk
                    del r_chunk, g_chunk
                    continue
                raise
            
            if disable_ssim or lambda_dssim <= 0.0:
                # Skip SSIM term.
                s = None
                chunk_loss = l1
            else:
                # SSIM requires more memory due to gaussian filtering.
                # Adaptive retry: empty cache -> reduce chunk size -> optionally drop SSIM.
                try:
                    # Run SSIM in fp32 for stability.
                    s = ssim(r_chunk.float(), g_chunk.float())
                    chunk_loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - s)
                except Exception as e:
                    if not _is_oom_error(e):
                        raise
                    torch.cuda.empty_cache()
                    # 1) retry once
                    try:
                        s = ssim(r_chunk.float(), g_chunk.float())
                        chunk_loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - s)
                    except Exception as e2:
                        if not _is_oom_error(e2):
                            raise
                        # 2) reduce chunk size for future tiles
                        if cur_chunk > 32:
                            new_chunk = max(32, cur_chunk // 2)
                            if utils.GLOBAL_RANK == 0:
                                print(f"[Loss] SSIM OOM at chunk={cur_chunk}. Reducing to {new_chunk} and continuing.")
                            cur_chunk = new_chunk
                            torch.cuda.empty_cache()
                            # Fall back to L1 for this chunk; next chunks use smaller tiles.
                            disable_ssim = False
                            s = None
                            chunk_loss = l1
                        else:
                            # 3) last resort: disable SSIM for the remainder of this loss call
                            if allow_disable_ssim_on_oom:
                                if utils.GLOBAL_RANK == 0:
                                    print("[Loss] SSIM keeps OOMing even at small chunks. Disabling SSIM term for this iteration.")
                                disable_ssim = True
                                s = None
                                chunk_loss = l1
                            else:
                                raise
            
            pixels = (y_end - y) * (x_end - x)
            total_loss += chunk_loss * pixels
            total_pixels += pixels
            
            del r_chunk, g_chunk, l1, s, chunk_loss

            x = x_end
        y = y_end

    return total_loss / total_pixels

class DummyDataset:
    def __init__(self, cameras):
        self.cameras = cameras
    def __len__(self):
        return len(self.cameras)
    def __getitem__(self, idx):
        return self.cameras[idx]

class GrendelStreamingTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        
        print(f"Initializing Distributed Group... (Rank {utils.LOCAL_RANK})")
        # Initialize Distributed
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        # Update utils with correct rank info
        utils.GLOBAL_RANK = dist.get_rank()
        utils.WORLD_SIZE = dist.get_world_size()
        utils.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        
        self.tiles_root = Path(args.tiles_path)
        self.output_dir = Path(args.output_path)
        self.images_path = Path(args.images_path)

        # Setup logging
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_dir / f"rank{utils.GLOBAL_RANK}.log", 'a', buffering=1)  # Line buffering
        
        # Redirect stderr at OS level to capture CUDA/NCCL errors
        # This must be done before Python-level redirection
        self.stderr_fd = os.dup(2)  # Save original stderr file descriptor
        os.dup2(self.log_file.fileno(), 2)  # Redirect stderr to log file
        
        self.tee = Tee(sys.stdout, self.log_file)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.tee
        sys.stderr = self.tee  # Python-level stderr도 로그 파일에 기록
        
        # Install exception hook to ensure errors are logged before exit
        self._install_exception_hook()

        # Set log file for Grendel utils
        utils.set_log_file(self.tee)
        
        # Initialize Grendel distributed groups
        utils.DEFAULT_GROUP = dist.group.WORLD
        
        print(f"Distributed Group Initialized. Global Rank: {utils.GLOBAL_RANK}, Local Rank: {utils.LOCAL_RANK}")
        torch.cuda.set_device(utils.LOCAL_RANK)
        
        # Prepare resume path if specified
        resume_path = Path(args.resume_from) if args.resume_from else None
        
        self.scene = StreamingScene(
            self.tiles_root, 
            self.device, 
            self.output_dir, 
            cache_size=args.cache_size, 
            sh_degree=args.sh_degree,
            resume_from=resume_path
        )
        
        # Load Cameras
        colmap_loader = ColmapLoader(Path(args.colmap_path))
        self.cameras = {}
        
        for img_idx, img_data in colmap_loader._images.items():
            cam_id = img_data['cam_id']
            cam_params = colmap_loader._cameras[cam_id]
            
            cam_dict = build_camera_from_colmap(
                cam_params, 
                img_data, 
                self.device, 
                resolution_scale=1.0
            )
            
            # Convert dict to object
            cam_obj = Namespace(**cam_dict)
            cam_obj.uid = img_idx
            cam_obj.image_name = img_data['name']
            
            # Add FoVx and FoVy required by Grendel
            cam_obj.FoVx = cam_params["fovx"]
            cam_obj.FoVy = cam_params["fovy"]
            
            self.cameras[img_idx] = cam_obj
            
        self.train_cam_ids = sorted(list(self.cameras.keys()))
        
        # Initialize Grendel Utils
        if len(self.train_cam_ids) > 0:
            first_cam = self.cameras[self.train_cam_ids[0]]
            utils.set_img_size(first_cam.image_height, first_cam.image_width)
        
        # Visibility
        vis_path = self.tiles_root / "tile_visibility.json"
        if not vis_path.exists():
            raise FileNotFoundError(f"{vis_path} not found")
        self.visibility = load_visibility_metadata(vis_path)
        
        # Grendel Strategy
        # We need a dataset-like object for the strategy
        self.dummy_dataset = DummyDataset([self.cameras[cid] for cid in self.train_cam_ids])
        self.strategy = DivisionStrategyHistoryFinal(self.dummy_dataset, utils.WORLD_SIZE, utils.GLOBAL_RANK)
        
        self.iteration = 0
        
        # Image Cache
        if utils.GLOBAL_RANK == 0:
            print(f"Initializing Shared Memory Image Cache with capacity: {args.max_cached_images} images")
        self.image_cache = SharedMemoryImageCache(
            Path(args.images_path), 
            self.visibility, 
            max_scaled_images=args.max_cached_images
        )
        
        self.stop_densification_globally = False
        
        # Preload images to Shared Memory if requested
        if args.preload_images:
            self._preload_images_to_shm(args)

    def _preload_images_to_shm(self, args):
        """Preload all images to Shared Memory before training starts."""
        if utils.GLOBAL_RANK == 0:
            print(f"[Preload] Starting preload of {len(self.train_cam_ids)} images to Shared Memory...")
            
            # Determine scale
            if args.resolution in [-1, 1]:
                res_scale = 1.0
            else:
                res_scale = 1.0 / args.resolution
                
            count = 0
            for cam_id in tqdm(self.train_cam_ids, desc="Preloading Images", unit="img"):
                cam = self.cameras[cam_id]
                try:
                    # Load to SHM
                    self.image_cache.put_image_to_shm(cam.image_name, resolution_scale=res_scale)
                    count += 1
                except Exception as e:
                    print(f"[Preload] Warning: Failed to load {cam.image_name}: {e}")
            
            print(f"[Preload] Completed. {count} images loaded to Shared Memory.")
        
        # Wait for Rank 0 to finish loading
        dist.barrier()
        print(f"[Preload] Rank {utils.GLOBAL_RANK} ready.")

    def _install_exception_hook(self):
        """Install exception hook to ensure errors are logged before exit."""
        import traceback
        original_excepthook = sys.excepthook
        
        def exception_hook(exc_type, exc_value, exc_traceback):
            # Print full traceback to stderr (which is now redirected to log file)
            print("\n" + "="*80, file=sys.stderr)
            print("UNCAUGHT EXCEPTION:", file=sys.stderr)
            print("="*80, file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
            print("="*80 + "\n", file=sys.stderr)
            
            # Force flush all buffers
            if hasattr(self, 'log_file') and self.log_file:
                try:
                    self.log_file.flush()
                except Exception:
                    pass
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Call original hook
            original_excepthook(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = exception_hook

    def cleanup(self):
        # Force flush before cleanup
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.flush()
        
        if hasattr(self, 'image_cache'):
            self.image_cache.cleanup()
        
        # Restore OS-level stderr first
        if hasattr(self, 'stderr_fd'):
            try:
                os.dup2(self.stderr_fd, 2)  # Restore original stderr
                os.close(self.stderr_fd)
            except Exception:
                pass
        
        # Restore stdout/stderr and close log file
        if hasattr(self, 'original_stdout') and self.original_stdout:
            sys.stdout = self.original_stdout
        if hasattr(self, 'original_stderr') and self.original_stderr:
            sys.stderr = self.original_stderr
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.flush()  # Final flush before close
            self.log_file.close()

    def train_loop(self):
        start_time = time.time()
        
        current_cam_id = None
        gt_image = None
        
        # Determine starting iteration
        start_iter = self.args.start_iteration
        if self.args.resume_from and start_iter == 1:
            # Auto-detect iteration from checkpoint path (e.g., "iteration_819")
            resume_path = Path(self.args.resume_from)
            if resume_path.name.startswith("iteration_"):
                try:
                    start_iter = int(resume_path.name.split("_")[1]) + 1
                    if utils.GLOBAL_RANK == 0:
                        print(f"[Resume] Auto-detected starting iteration: {start_iter}")
                except ValueError:
                    pass
        
        # Restore active_sh_degree based on start_iter
        cycle_length = len(self.train_cam_ids) * self.args.view_iter
        if start_iter > 0:
            self.scene.active_sh_degree = min(start_iter // cycle_length, self.args.sh_degree)
            if self.scene.gaussian_model:
                self.scene.gaussian_model.active_sh_degree = self.scene.active_sh_degree
            if utils.GLOBAL_RANK == 0:
                print(f"[Resume] Restored active SH degree to {self.scene.active_sh_degree} (Iter {start_iter})")

        def print_mem(step_name):
            # Print memory stats for Rank 3 (where OOM happened) or Rank 0
            if utils.GLOBAL_RANK in [0, 3]:
                mem = torch.cuda.memory_allocated() / 1024**3
                max_mem = torch.cuda.max_memory_allocated() / 1024**3
                # print(f"[Mem Rank{utils.GLOBAL_RANK}] {step_name}: Cur {mem:.2f}GB / Max {max_mem:.2f}GB")

        # Initialize peak memory tracker
        self.last_peak_memory = 0
        self.current_split_factor = self.args.initial_split_factor # Start with user-defined split factor
        if utils.GLOBAL_RANK == 0 and self.current_split_factor > 1:
            print(f"[Config] Starting with initial split factor: {self.current_split_factor}")

        # Next-iteration split factor (decided by prediction) to avoid mid-iteration graph churn.
        self.next_split_factor = self.current_split_factor

        for iteration in range(start_iter, self.args.iterations + 1):
            # Capture peak memory of the PREVIOUS iteration before resetting
            # Note: max_memory_reserved() returns the peak reserved memory since the last reset
            current_peak = torch.cuda.max_memory_reserved(self.device)
            if iteration > start_iter:
                self.last_peak_memory = current_peak

            # Reset peak memory stats at the start of each iteration to capture the true peak of THIS iteration
            torch.cuda.reset_peak_memory_stats()
            
            print_mem(f"Start Iter {iteration}")
            # Explicitly clear variables from previous iteration to free up memory
            # This is crucial for large image training to avoid OOM
            if 'rendered_image' in locals(): del rendered_image
            if 'output_tensor' in locals(): del output_tensor
            if 'loss' in locals(): del loss
            # gt_image should NOT be deleted here as it persists across iterations when view_iter > 1

            
            # Periodic GC to handle fragmentation
            if iteration % 100 == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            self.iteration = iteration
            utils.set_cur_iter(iteration)

            # Apply any pending split-factor change at the iteration boundary.
            if hasattr(self, "next_split_factor") and self.next_split_factor != self.current_split_factor:
                old = self.current_split_factor
                self.current_split_factor = self.next_split_factor
                if utils.GLOBAL_RANK == 0:
                    print(f"[Iter {iteration}] Applying split factor change: {old} -> {self.current_split_factor}")

            # Upgrade SH degree periodically (faster than original 3DGS for faster convergence)
            # Original 3DGS: every 1000 iters, Ours: every 600 iters (1 full cycle)
            cycle_length = len(self.train_cam_ids) * self.args.view_iter
            if iteration % cycle_length == 0 and iteration > 0:
                old_degree = self.scene.active_sh_degree
                self.scene.active_sh_degree = min(self.scene.active_sh_degree + 1, self.args.sh_degree)
                if utils.GLOBAL_RANK == 0:
                    print(f"[Iter {iteration}] SH degree upgrade: {old_degree} -> {self.scene.active_sh_degree} (max: {self.args.sh_degree})")
                # Will be applied in gaussian_model during next update_active_tiles or immediately if model exists
                if self.scene.gaussian_model:
                    self.scene.gaussian_model.active_sh_degree = self.scene.active_sh_degree

            # Log current wall-clock time for this iteration (only from rank 0)
            # if utils.GLOBAL_RANK == 0:
            #     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #     print(f"[{now}] Starting iteration {iteration}, current_cam={current_cam_id}")
            
            # --- Step 1: Camera Selection (Rank 0) ---
            assignments = None
            
            # Determine if we need to switch camera
            switch_camera = (iteration == 1) or ((iteration - 1) % self.args.view_iter == 0)
            
            if utils.GLOBAL_RANK == 0:
                if switch_camera:
                    # Select Camera
                    if self.args.random_camera:
                        cam_idx = np.random.randint(0, len(self.train_cam_ids))
                    else:
                        # Round Robin
                        # We want to cycle through cameras, but stay on each for view_iter
                        # Total switches so far = (iteration - 1) // view_iter
                        cam_idx = ((iteration - 1) // self.args.view_iter) % len(self.train_cam_ids)
                    
                    current_cam_id = self.train_cam_ids[cam_idx]
                    
                    # Identify Visible Tiles
                    if current_cam_id in self.visibility.cameras:
                        cam_vis = self.visibility.cameras[current_cam_id]
                        visible_tiles = list(set([t.tile_id for t in cam_vis.tiles]))
                    else:
                        visible_tiles = []
                    
                    # Assign Tiles to Ranks
                    assignments = {i: [] for i in range(utils.WORLD_SIZE)}
                    for tile_id in visible_tiles:
                        # Simple hash-based assignment for now
                        # Ideally, we should balance based on point count, but we don't know it yet
                        rank = hash(tile_id) % utils.WORLD_SIZE
                        assignments[rank].append(tile_id)
            
            # Broadcast Plan
            if switch_camera:
                broadcast_list = [current_cam_id, assignments]
                dist.broadcast_object_list(broadcast_list, src=0)
                current_cam_id, assignments = broadcast_list
                
                # --- Step 2: Local Setup ---
                my_tiles = assignments[utils.GLOBAL_RANK]
                self.scene.update_active_tiles(my_tiles, iteration, self.args.opacity_reset_interval)
                print_mem("After Tile Update")
                
                current_cam = self.cameras[current_cam_id]
                
                # Load GT image (Only when camera switches)
                # Use SharedMemoryImageCache to load full image efficiently
                
                # Free previous image memory
                del gt_image
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Determine scale based on args.resolution
                if self.args.resolution in [-1, 1]:
                    res_scale = 1.0
                else:
                    res_scale = 1.0 / self.args.resolution

                # 1. Rank 0 loads image to Shared Memory
                shm_info = None
                if utils.GLOBAL_RANK == 0:
                    print(f"[Iter {iteration}] Loading/Preparing image for {current_cam.image_name} (Shared Memory)...")
                    try:
                        shm_info = self.image_cache.put_image_to_shm(
                            current_cam.image_name,
                            resolution_scale=res_scale
                        )
                    except Exception as e:
                        print(f"[Error] Failed to load image to SHM: {e}")
                        shm_info = None

                # 2. Broadcast metadata to all ranks
                broadcast_list = [shm_info]
                dist.broadcast_object_list(broadcast_list, src=0)
                shm_info = broadcast_list[0]

                if shm_info is None:
                    raise RuntimeError(f"Failed to load image {current_cam.image_name} on Rank 0")

                # 2.5. Wait for Rank 0 to finish writing to Shared Memory
                dist.barrier()

                # 3. All ranks read from Shared Memory (Zero-copy view)
                # The tensor is uint8 [C, H, W]
                gt_image_uint8 = self.image_cache.get_image_from_shm(shm_info)
                
                # 4. Convert to float for training (on CPU, will be moved to GPU in chunks)
                # We keep it as uint8 on CPU to save memory (4x savings).
                # Conversion to float happens inside chunked_loss or before usage.
                gt_image = gt_image_uint8

                # --- Predict VRAM usage after tile load (per-rank), and decide split factor for next iteration ---
                # We do this here because tile loading changes Gaussian count and optimizer state footprint.
                try:
                    local_ng = int(self.scene.tile_batch.xyz.shape[0]) if self.scene.tile_batch is not None else 0
                except Exception:
                    local_ng = 0

                # Determine the SSIM chunk size used in chunked_loss.
                # chunked_loss is called with chunk_size=128 currently.
                ssim_chunk = 128
                # Worst-case chunk dimensions are clamped by strip size and image size.
                # Use current image resolution (full, not split) for conservative estimate.
                full_H = int(current_cam.image_height)
                full_W = int(current_cam.image_width)
                est_chunk_h = min(ssim_chunk, full_H)
                est_chunk_w = min(ssim_chunk, full_W)

                # Parameters + grads + densification buffers depend on current active SH degree.
                sh_deg = int(self.scene.active_sh_degree)
                est_gauss = estimate_gaussian_param_vram_bytes(local_ng, sh_deg, dtype=torch.float32)
                est_adam = estimate_adam_state_vram_bytes(local_ng, sh_deg, dtype=torch.float32)

                # SSIM runs on the local strip of pixels; splitting reduces strip height per split.
                # Our chunked_loss processes the strip in 128x128 blocks, so SSIM peak largely
                # depends on chunk size not full image. Still, we scale by split factor for strip height
                # only to reflect the fact that smaller splits tend to reduce other renderer buffers.
                est_ssim = estimate_ssim_working_vram_bytes(est_chunk_h, est_chunk_w, dtype=torch.float32)

                # Total predicted working set we control directly.
                local_pred_bytes = int(est_gauss["total"] + est_adam["total"] + est_ssim["total"])

                total_mem = torch.cuda.get_device_properties(self.device).total_memory
                local_ratio = float(local_pred_bytes) / float(total_mem)
                ratio_t = torch.tensor([local_ratio], device=self.device, dtype=torch.float32)
                dist.all_reduce(ratio_t, op=dist.ReduceOp.MAX)
                global_pred_ratio = float(ratio_t.item())

                # Also consider actual reserved memory ratio (fragmentation / renderer buffers).
                # This catches cases where prediction is small but the renderer already reserved most VRAM.
                local_reserved_ratio = float(torch.cuda.memory_reserved(self.device)) / float(total_mem)
                reserved_t = torch.tensor([local_reserved_ratio], device=self.device, dtype=torch.float32)
                dist.all_reduce(reserved_t, op=dist.ReduceOp.MAX)
                global_reserved_ratio = float(reserved_t.item())

                effective_ratio = max(global_pred_ratio, global_reserved_ratio)

                # Log once on rank 0 (max across ranks).
                if utils.GLOBAL_RANK == 0:
                    print(
                        f"[Iter {iteration}] PredVRAM(max-rank): "
                        f"gauss={_format_bytes(est_gauss['total'])} (params={_format_bytes(est_gauss['params'])}, grads={_format_bytes(est_gauss['grads'])}, densify={_format_bytes(est_gauss['densify_buffers'])}), "
                        f"adam={_format_bytes(est_adam['total'])}, "
                        f"ssim≈{_format_bytes(est_ssim['total'])} (chunk={est_chunk_h}x{est_chunk_w}), "
                        f"sum≈{_format_bytes(local_pred_bytes)} => pred={global_pred_ratio*100:.1f}%, reserved={global_reserved_ratio*100:.1f}%, effective={effective_ratio*100:.1f}%"
                    )

                # Split scheduling threshold.
                # NOTE: You can tune this value (e.g., 0.55, 0.75). We use an *effective* ratio that
                # is max(predicted_working_set, actual_reserved) to react to fragmentation/renderer buffers.
                split_threshold = 0.55

                # Requirement: if split=1 and threshold exceeded, move to split=2 (next iter).
                # Then if exceeded again (e.g., after densification), move to split=4, etc.
                if effective_ratio >= split_threshold and self.current_split_factor < self.args.max_split_factor:
                    proposed = min(self.current_split_factor * 2, self.args.max_split_factor)
                    if proposed != self.current_split_factor:
                        self.next_split_factor = proposed
                        if utils.GLOBAL_RANK == 0:
                            print(
                                f"[Iter {iteration}] Memory pressure >= {split_threshold*100:.0f}% (effective={effective_ratio*100:.1f}%, pred={global_pred_ratio*100:.1f}%, reserved={global_reserved_ratio*100:.1f}%). "
                                f"Scheduling split factor increase {self.current_split_factor} -> {self.next_split_factor} starting next iteration."
                            )
            
            # Ensure current_cam is set even if we didn't switch (it persists in self.cameras)
            current_cam = self.cameras[current_cam_id]
            
            # --- Step 4: Distributed Preprocess & All2All ---
            # Note: distributed_preprocess3dgs_and_all2all_final expects a GaussianModel
            # But our GaussianModel only contains LOCAL tiles.
            # This is exactly what we want! Grendel handles the gathering.
            
            if self.args.random_background:
                # Avoid extreme white/black backgrounds to prevent opacity artifacts
                # especially when SH degree is low and dataset is small.
                while True:
                    background = torch.rand((3), dtype=torch.float32, device=self.device)
                    mean_val = background.mean().item()
                    # Retry if too dark (<0.1) or too bright (>0.9)
                    if 0.1 <= mean_val <= 0.9:
                        break
                        
                if utils.GLOBAL_RANK == 0 and iteration % 100 == 0:
                    print(f"[Debug] Iter {iteration}: Random BG active. Sample: {background.cpu().numpy()}")
            elif self.args.white_background:
                background = torch.ones((3), dtype=torch.float32, device=self.device)
                if utils.GLOBAL_RANK == 0 and iteration % 100 == 0:
                    print(f"[Debug] Iter {iteration}: White BG active.")
            else:
                background = torch.zeros((3), dtype=torch.float32, device=self.device)
                if utils.GLOBAL_RANK == 0 and iteration % 100 == 0:
                    print(f"[Debug] Iter {iteration}: Black BG active.")
            
            pipe_args = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

            # Keep the old peak-memory heuristic as a fallback safety net,
            # but prefer the prediction-based scheduling above.
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            if iteration == start_iter:
                check_mem = torch.cuda.memory_reserved(self.device)
            else:
                check_mem = self.last_peak_memory

            mem_ratio = float(check_mem) / float(total_mem)
            mem_ratio_tensor = torch.tensor([mem_ratio], device=self.device, dtype=torch.float32)
            dist.all_reduce(mem_ratio_tensor, op=dist.ReduceOp.MAX)
            global_mem_ratio = float(mem_ratio_tensor.item())

            # If we're already close to OOM by measured peak, also schedule a split increase.
            if global_mem_ratio > 0.90 and self.current_split_factor < self.args.max_split_factor:
                proposed = min(self.current_split_factor * 2, self.args.max_split_factor)
                if proposed != self.current_split_factor:
                    self.next_split_factor = proposed
                    if utils.GLOBAL_RANK == 0:
                        print(
                            f"[Iter {iteration}] PeakVRAM warning ({global_mem_ratio*100:.1f}%). "
                            f"Scheduling split factor increase {self.current_split_factor} -> {self.next_split_factor} starting next iteration."
                        )
            
            use_split = (self.current_split_factor > 1)
            
            if use_split:
                if utils.GLOBAL_RANK == 0:
                    print(f"[Iter {iteration}] High Memory Mode (Last Peak: {global_mem_ratio*100:.1f}%), splitting image into {self.current_split_factor} parts...")
                
                sub_cameras = []
                # Add overlap to prevent culling artifacts at split boundaries
                # Large Gaussians near the boundary might be culled if the guard band (in pixels) is too small
                overlap_pixels = 0
                if self.current_split_factor > 1:
                    # Use 5% overlap or at least 32 pixels
                    overlap_pixels = max(32, int(current_cam.image_height * 0.05))

                for i in range(self.current_split_factor):
                    sub_cam_dict = split_camera_vertical(current_cam, i, self.current_split_factor, overlap_pixels=overlap_pixels)
                    sub_cam = Namespace(**sub_cam_dict)
                    sub_cam.uid = current_cam.uid
                    sub_cam.image_name = current_cam.image_name
                    sub_cam.FoVx = current_cam.FoVx
                    sub_cam.FoVy = 2.0 * math.atan(sub_cam.tanfovy)
                    sub_cameras.append(sub_cam)
            else:
                sub_cameras = [current_cam]

            batch_statistic_collector = []
            
            for sub_cam_idx, sub_cam in enumerate(sub_cameras):
                batched_viewpoint_cameras = [sub_cam]
                batched_strategies, gpuid2tasks = start_strategy_final(
                    batched_viewpoint_cameras, self.strategy
                )

                batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
                    batched_viewpoint_cameras,
                    self.scene.gaussian_model,
                    pipe_args,
                    background,
                    batched_strategies=batched_strategies,
                    mode="train"
                )
                print_mem(f"After Preprocess (Split {sub_cam_idx})")
                
                batched_rendered_images, batched_compute_locally = render_final(
                    batched_screenspace_pkg,
                    batched_strategies
                )
                print_mem(f"After Render (Split {sub_cam_idx})")
                
                current_stats = [
                    cuda_args["stats_collector"]
                    for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
                ]
                batch_statistic_collector.extend(current_stats)
                
                rendered_image = batched_rendered_images[0]
                del batched_rendered_images
                
                if rendered_image is None:
                    rendered_image = torch.zeros((3, sub_cam.image_height, sub_cam.image_width), device=self.device)
                elif rendered_image.ndim == 0:
                    dummy = rendered_image
                    rendered_image = torch.zeros((3, sub_cam.image_height, sub_cam.image_width), device=self.device)
                    rendered_image = rendered_image + dummy * 0
                
                C, H, W = rendered_image.shape
                pad_h = (utils.WORLD_SIZE - (H % utils.WORLD_SIZE)) % utils.WORLD_SIZE
                if pad_h > 0:
                    rendered_image = torch.nn.functional.pad(rendered_image, (0, 0, 0, pad_h))
                
                H_pad = H + pad_h
                chunk_h = H_pad // utils.WORLD_SIZE
                
                output_tensor = reduce_scatter_autograd(rendered_image, utils.WORLD_SIZE, chunk_h)
                
                rank = utils.GLOBAL_RANK
                y_start = rank * chunk_h
                y_end = min((rank + 1) * chunk_h, H)
                
                valid_h = y_end - y_start
                output_tensor = output_tensor[:, :valid_h, :]
                
                if use_split:
                    split_info = sub_cam.split_info
                    split_y_offset = split_info['y_offset']
                else:
                    split_y_offset = 0
                    
                global_y_start = split_y_offset + y_start
                global_y_end = split_y_offset + y_end
                
                # Keep GT on CPU to avoid allocating a full strip on GPU (can OOM under fragmentation).
                # chunked_loss will move only small chunks to GPU.
                gt_strip = gt_image[:, global_y_start:global_y_end, :]
                
                t_start = time.time()

                # Loss computation can still OOM under extreme fragmentation. In that case, we must
                # make sure ALL ranks take the same path (skip backward/step) to avoid desync.
                local_loss_oom = 0
                loss = None
                try:
                    loss = chunked_loss(output_tensor, gt_strip, self.args.lambda_dssim, chunk_size=128)
                except Exception as e:
                    if _is_oom_error(e):
                        local_loss_oom = 1
                    else:
                        raise

                oom_t = torch.tensor([local_loss_oom], device=self.device, dtype=torch.int32)
                dist.all_reduce(oom_t, op=dist.ReduceOp.MAX)
                global_loss_oom = int(oom_t.item())
                if global_loss_oom:
                    # Free as much as possible before continuing.
                    if loss is not None:
                        del loss
                    del output_tensor, gt_strip
                    torch.cuda.empty_cache()
                    # Schedule more aggressive splitting immediately.
                    if self.current_split_factor < self.args.max_split_factor:
                        proposed = min(self.current_split_factor * 2, self.args.max_split_factor)
                        if proposed != self.current_split_factor:
                            self.next_split_factor = proposed
                            if utils.GLOBAL_RANK == 0:
                                print(
                                    f"[Iter {iteration}] OOM during loss/backward on some rank. "
                                    f"Scheduling split factor increase {self.current_split_factor} -> {self.next_split_factor} (next iteration)."
                                )
                    # Skip backward/step for this split (and effectively this iteration, since view_iter=1).
                    continue
                
                full_H = current_cam.image_height
                full_W = current_cam.image_width
                total_pixels = full_H * full_W
                strip_pixels = valid_h * W
                
                loss = loss * (strip_pixels / total_pixels)
                
                del output_tensor, gt_strip
                
                print_mem(f"Before Backward (Split {sub_cam_idx})")
                local_bwd_oom = 0
                try:
                    loss.backward()
                except Exception as e:
                    if _is_oom_error(e):
                        local_bwd_oom = 1
                    else:
                        raise

                bwd_t = torch.tensor([local_bwd_oom], device=self.device, dtype=torch.int32)
                dist.all_reduce(bwd_t, op=dist.ReduceOp.MAX)
                global_bwd_oom = int(bwd_t.item())
                if global_bwd_oom:
                    torch.cuda.empty_cache()
                    if self.current_split_factor < self.args.max_split_factor:
                        proposed = min(self.current_split_factor * 2, self.args.max_split_factor)
                        if proposed != self.current_split_factor:
                            self.next_split_factor = proposed
                            if utils.GLOBAL_RANK == 0:
                                print(
                                    f"[Iter {iteration}] OOM during backward on some rank. "
                                    f"Scheduling split factor increase {self.current_split_factor} -> {self.next_split_factor} (next iteration)."
                                )
                    continue
                print_mem(f"After Backward (Split {sub_cam_idx})")
                
                # Accumulate densification stats IMMEDIATELY after backward
                if not self.stop_densification_globally and "batched_locally_preprocessed_visibility_filter" in batched_screenspace_pkg:
                    local_vis_filter = batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"][0]
                    local_viewspace_points = batched_screenspace_pkg["batched_locally_preprocessed_mean2D"][0]
                    local_radii = batched_screenspace_pkg["batched_locally_preprocessed_radii"][0]
                    
                    self.scene.gaussian_model.add_densification_stats(local_viewspace_points, local_vis_filter)
                    
                    if sub_cam_idx == 0:
                        self.accum_visibility_filter = local_vis_filter
                        self.accum_radii = local_radii
                    else:
                        self.accum_visibility_filter = torch.logical_or(self.accum_visibility_filter, local_vis_filter)
                        self.accum_radii = torch.max(self.accum_radii, local_radii)

                torch.cuda.synchronize()
                t_end = time.time()
                
                for stats in current_stats:
                    stats["forward_loss_time"] = (t_end - t_start) * 1000.0
                
                del batched_screenspace_pkg
                torch.cuda.empty_cache()
                
                finish_strategy_final(batched_viewpoint_cameras, self.strategy, batched_strategies, current_stats)

            # --- Step 6: Densification ---
            if self.iteration % self.args.densification_interval == 0 and self.iteration > self.args.densify_from_iter and self.iteration <= self.args.densify_until_iter:
                if self.stop_densification_globally:
                    if utils.GLOBAL_RANK == 0:
                        print(f"[Densify] Skipping densification at iter {self.iteration} (Globally disabled due to memory limit)")
                else:
                    total_mem = torch.cuda.get_device_properties(self.device).total_memory
                    peak_mem = torch.cuda.max_memory_allocated(self.device)
                    usage_ratio = peak_mem / total_mem
                    
                    local_stop = usage_ratio > self.args.densify_memory_limit_percentage
                    
                    stop_tensor = torch.tensor([1 if local_stop else 0], device=self.device)
                    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                    
                    if stop_tensor.item() > 0:
                        self.stop_densification_globally = True
                        if utils.GLOBAL_RANK == 0:
                            print(f"[Warning] VRAM usage exceeded {self.args.densify_memory_limit_percentage*100:.1f}% on at least one rank. Disabling densification permanently.")

                    if not self.stop_densification_globally and hasattr(self, 'accum_visibility_filter'):
                        self.scene.densify_and_prune(
                            None,
                            self.accum_visibility_filter,
                            self.accum_radii,
                            threshold=self.args.densify_grad_threshold,
                            skip_add_stats=True
                        )
                        print_mem("After Densification")
                    
                    if hasattr(self, 'accum_visibility_filter'):
                        del self.accum_visibility_filter
                        del self.accum_radii

            # --- Step 5: Optimization ---
            if self.scene.optimizer:
                self.scene.optimizer.step()
                self.scene.optimizer.zero_grad(set_to_none=True)
            print_mem("After Optimizer")
            
            # --- Step 7: Save/Evict ---
            if self.iteration % self.args.save_interval == 0 or self.iteration in self.args.save_iterations:
                self.scene.save_all_active_tiles(backup_iteration=self.iteration)
            
            # Synchronize at cycle boundaries (every N_images * view_iter iterations)
            cycle_length = len(self.train_cam_ids) * self.args.view_iter
            if self.iteration % cycle_length == 0:
                if utils.GLOBAL_RANK == 0:
                    print(f"[Iter {self.iteration}] Cycle boundary reached. Synchronizing all tiles to disk...")
                
                # Ensure all GPU computations are complete
                torch.cuda.synchronize()
                
                # Force save all active tiles (blocking I/O)
                self.scene.save_all_active_tiles()
                
                # Wait for all ranks to finish saving
                dist.barrier()
                
                if utils.GLOBAL_RANK == 0:
                    print(f"[Iter {self.iteration}] Cycle synchronization complete. Continuing training...")
                
            if utils.GLOBAL_RANK == 0 and self.iteration % 10 == 0:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cache_stats = self.image_cache.stats
                print(f"[{now}] Iteration {self.iteration}: Cam {current_cam_id} Done. Cache: {cache_stats}")

        # Save at the end of training
        self.scene.save_all_active_tiles()
        if utils.GLOBAL_RANK == 0:
            print("Training finished. Final tiles saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--colmap_path", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--cache_size", type=int, default=200)
    parser.add_argument("--max_cached_images", type=int, default=16, help="Max number of full images to cache in RAM")
    parser.add_argument("--view_iter", type=int, default=100, help="Number of iterations to train on the same view before switching")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval to save trained tiles")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[], help="Specific iterations to save trained tiles")
    parser.add_argument("--densification_interval", type=int, default=100, help="Interval for densification")
    parser.add_argument("--opacity_reset_interval", type=int, default=3000, help="Interval to reset opacity")
    parser.add_argument("--densify_from_iter", type=int, default=15000, help="Iteration to start densification")
    parser.add_argument("--densify_until_iter", type=int, default=15000, help="Iteration to stop densification")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002, help="Gradient threshold for densification")
    parser.add_argument("--random_background", action="store_true", help="Enable random background color (Black is default)")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree (default: 3)")
    parser.add_argument("--resolution", type=int, default=-1, help="Resolution scale (1=full, 2=1/2, 4=1/4). Default -1 (full)")
    parser.add_argument("--random_camera", action="store_true", help="Randomly select camera instead of round-robin")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory (e.g., output/run/iteration_819)")
    parser.add_argument("--start_iteration", type=int, default=1, help="Starting iteration (auto-detected from --resume_from if provided)")
    
    # Grendel args (needed for strategy)
    parser.add_argument("--refine_interval", type=int, default=100)
    parser.add_argument("--refine_start_iter", type=int, default=500)
    parser.add_argument("--refine_stop_iter", type=int, default=15000)
    parser.add_argument("--preload_images", action="store_true", help="Preload all images to Shared Memory before training")
    parser.add_argument("--densify_memory_limit_percentage", type=float, default=0.65, help="VRAM memory limit for densification (0.0-1.0)")
    
    # New arguments
    parser.add_argument("--max_split_factor", type=int, default=4, help="Max split factor for image splitting (1, 2, 4, ...)")
    parser.add_argument("--initial_split_factor", type=int, default=1, help="Initial split factor (1, 2, 4, ...). Default 1.")
    parser.add_argument("--lambda_dssim", type=float, default=0.2, help="SSIM loss weight")
    
    args = parser.parse_args()

    # Set random_background based on disable flag
    if args.white_background:
        args.random_background = False
    # else:
    #     args.random_background is already set by argparse (default False)
        
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        if args.random_background:
            print("[Config] Random background enabled.")
        elif args.white_background:
            print("[Config] White background enabled.")
        else:
            print("[Config] Black background enabled.")
    
    # Add Grendel args
    args.local_sampling = False
    args.border_divpos_coeff = 0.0
    args.heuristic_decay = 0.0
    args.adjust_strategy_warmp_iterations = 100
    args.no_heuristics_update = False
    args.bsz = 1
    
    # Distribution params
    args.image_distribution = True
    args.image_distribution_mode = "final"
    args.gaussians_distribution = True
    args.redistribute_gaussians_mode = "random_redistribute"
    args.redistribute_gaussians_frequency = 10
    args.redistribute_gaussians_threshold = 1.1
    args.sync_grad_mode = "dense"
    args.grad_normalization_mode = "none"
    args.distributed_dataset_storage = True
    args.distributed_save = False
    args.preload_dataset_to_gpu = False
    args.preload_dataset_to_gpu_threshold = 10
    args.multiprocesses_image_loading = True
    args.num_train_cameras = -1
    args.num_test_cameras = -1
    
    # Optimization params
    # args.densify_grad_threshold = 0.0002 # Moved to argparse
    # args.densify_memory_limit_percentage = 0.35  # Lowered from 0.7 to prevent OOM during rendering
    args.disable_auto_densification = False
    args.opacity_reset_until_iter = -1
    # args.random_background = False # Use command line arg
    args.min_opacity = 0.005
    args.lr_scale_mode = "sqrt"
    args.lambda_dssim = 0.2
    args.percent_dense = 0.01
    # args.densification_interval = 100 # Moved to argparse
    # args.opacity_reset_interval = 3000 # Moved to argparse
    # args.densify_from_iter = 500 # Moved to argparse
    # args.densify_until_iter = 5000 # Moved to argparse
    
    # Benchmark params
    args.enable_timer = False
    args.end2end_time = True
    args.zhx_time = False
    args.check_gpu_memory = False
    args.check_cpu_memory = False
    args.log_memory_summary = False
    
    # Debug params
    args.zhx_debug = False
    args.stop_update_param = False
    args.time_image_loading = False
    args.nsys_profile = False
    args.drop_initial_3dgs_p = 0.0
    args.drop_duplicate_gaussians_coeff = 1.0
    
    # Model params
    # args.sh_degree = 3 # Moved to argparse
    args.source_path = ""
    args.model_path = args.output_path
    args.images = "images"
    # args.resolution = -1 # Moved to argparse
    # args.white_background = False # Use command line arg
    args.data_device = "cuda"
    args.eval = False
    
    # Pipeline params
    args.convert_SHs_python = False
    args.compute_cov3D_python = False
    args.debug = False
    
    # Auxiliary params
    args.log_interval = 100
    args.log_folder = args.output_path
    
    utils.set_args(args)
    
    trainer = GrendelStreamingTrainer(args)
    try:
        trainer.train_loop()
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()
