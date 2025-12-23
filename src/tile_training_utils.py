"""Utilities for building tile-based training batches on a single GPU.

This module provides helpers for loading tile parameter shards from disk, assembling
batched tensors on the target device, and writing updated parameters back to the
original tile artifacts.  The routines are intentionally lightweight so they can be
reused both by the high level training script and future unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple
import math

import numpy as np
import torch
from torch import nn

from src.tile_storage import TileStorage


@dataclass
class TileBatchTensors:
    """Flat tensor views over a collection of tiles.

    Attributes:
        xyz: Concatenated Gaussian means on the target device ``[N, 3]``.
        scales: Log-space scales ``[N, 3]``.
        quats: Quaternion rotations ``[N, 4]``.
        opacities: Logit-space opacities ``[N, 1]``.
        features_dc: DC SH coefficients ``[N, 1, 3]``.
        features_rest: Higher order SH coefficients ``[N, M, 3]`` where ``M`` is
            ``(sh_degree + 1) ** 2 - 1``.
        tile_slices: Mapping from tile id to the slice range inside the flat tensors.
        coords: Mapping from tile id to the original integer grid coordinate.
    """

    xyz: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    tile_slices: Dict[str, slice]
    coords: Dict[str, Tuple[int, int, int]]
    
    # Optional densification stats
    accum_grad: Optional[torch.Tensor] = None
    accum_count: Optional[torch.Tensor] = None
    max_radii2D: Optional[torch.Tensor] = None

    def populate_gaussian_model(self, model: "GaussianModel") -> None:
        """Copy the flat tensors into an existing :class:`GaussianModel` instance."""

        total = self.xyz.shape[0]
        device = self.xyz.device

        model._xyz = nn.Parameter(self.xyz.clone().requires_grad_(True))
        model._scaling = nn.Parameter(self.scales.clone().requires_grad_(True))
        model._rotation = nn.Parameter(self.quats.clone().requires_grad_(True))
        model._opacity = nn.Parameter(self.opacities.clone().requires_grad_(True))
        model._features_dc = nn.Parameter(self.features_dc.clone().requires_grad_(True))
        model._features_rest = nn.Parameter(
            self.features_rest.clone().requires_grad_(True)
        )
        
        # Restore densification stats if available
        if self.max_radii2D is not None:
            model.max_radii2D = self.max_radii2D.clone().to(device)
        else:
            model.max_radii2D = torch.zeros((total,), device=device)
            
        if self.accum_grad is not None:
            model.xyz_gradient_accum = self.accum_grad.clone().to(device)
        else:
            model.xyz_gradient_accum = torch.zeros((total, 1), device=device)
            
        if self.accum_count is not None:
            model.denom = self.accum_count.clone().to(device)
        else:
            model.denom = torch.zeros((total, 1), device=device)
            
        model.send_to_gpui_cnt = torch.zeros((total, 1), dtype=torch.int, device=device)
        model.percent_dense = 0.01

    def copy_from_model(self, model: "GaussianModel") -> None:
        """Update the batch tensors with the latest parameters from ``model``."""

        with torch.no_grad():
            self.xyz.copy_(model._xyz.detach())
            self.scales.copy_(model._scaling.detach())
            self.quats.copy_(model._rotation.detach())
            self.opacities.copy_(model._opacity.detach())
            self.features_dc.copy_(model._features_dc.detach())
            self.features_rest.copy_(model._features_rest.detach())
            
            # Copy densification stats
            if hasattr(model, 'xyz_gradient_accum'):
                if self.accum_grad is None:
                    self.accum_grad = torch.zeros_like(model.xyz_gradient_accum)
                self.accum_grad.copy_(model.xyz_gradient_accum.detach())
                
            if hasattr(model, 'denom'):
                if self.accum_count is None:
                    self.accum_count = torch.zeros_like(model.denom)
                self.accum_count.copy_(model.denom.detach())
                
            if hasattr(model, 'max_radii2D'):
                if self.max_radii2D is None:
                    self.max_radii2D = torch.zeros_like(model.max_radii2D)
                self.max_radii2D.copy_(model.max_radii2D.detach())

    def infer_sh_degree(self) -> int:
        """Infer the spherical-harmonics degree represented by this batch."""

        bands = self.features_dc.shape[1] + self.features_rest.shape[1]
        degree = int(round(math.sqrt(bands) - 1))
        if (degree + 1) ** 2 != bands:
            raise ValueError(
                f"Unable to infer SH degree from features (total bands={bands})"
            )
        return degree

    def parameter_slices(self, tile_id: str) -> slice:
        """Return the span inside the flat tensors that corresponds to ``tile_id``."""

        return self.tile_slices[tile_id]


def _load_single_tile(
    storage: TileStorage, tile_id: str
) -> Tuple[Mapping[str, np.ndarray], Tuple[int, int, int]]:
    meta = storage.metadata["tiles"].get(tile_id)
    if meta is None:
        raise KeyError(f"Tile {tile_id} not found in metadata")
    coord = tuple(meta["coord"])
    data = storage.load_tile(coord)
    if data is None:
        raise FileNotFoundError(f"Tile file missing for id {tile_id}")
    return data, coord  # type: ignore[return-value]


def load_tiles_as_tensors(
    storage: TileStorage,
    tile_ids: Iterable[str],
    device: torch.device,
) -> TileBatchTensors:
    """Load the requested tiles and concatenate their parameters on ``device``."""

    xyz_list = []
    scales_list = []
    quats_list = []
    opacities_list = []
    sh0_list = []
    sh_rest_list = []
    tile_slices: Dict[str, slice] = {}
    coords: Dict[str, Tuple[int, int, int]] = {}

    offset = 0
    for tile_id in tile_ids:
        data, coord = _load_single_tile(storage, tile_id)
        count = data["means"].shape[0]
        tile_slices[tile_id] = slice(offset, offset + count)
        coords[tile_id] = coord
        offset += count

        xyz_list.append(torch.from_numpy(data["means"]))
        scales_list.append(torch.from_numpy(data["scales"]))
        quats_list.append(torch.from_numpy(data["quats"]))
        opacities = data["opacities"].reshape(-1, 1)
        opacities_list.append(torch.from_numpy(opacities))
        sh0_list.append(torch.from_numpy(data["sh0"]))
        sh_rest_list.append(torch.from_numpy(data["shN"]))

    if offset == 0:
        raise ValueError("No Gaussians loaded for the provided tile ids")

    xyz = torch.cat(xyz_list, dim=0).to(device=device, dtype=torch.float32).contiguous()
    scales = (
        torch.cat(scales_list, dim=0)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    quats = (
        torch.cat(quats_list, dim=0)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    opacities = (
        torch.cat(opacities_list, dim=0)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    features_dc = (
        torch.cat(sh0_list, dim=0)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    features_rest = (
        torch.cat(sh_rest_list, dim=0)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )

    return TileBatchTensors(
        xyz=xyz,
        scales=scales,
        quats=quats,
        opacities=opacities,
        features_dc=features_dc,
        features_rest=features_rest,
        tile_slices=tile_slices,
        coords=coords,
    )


def write_back_tiles(
    storage: TileStorage,
    tensors: TileBatchTensors,
    *,
    output_root: Optional[Path] = None,
) -> None:
    """Persist the current tensor values back to individual NPZ tile files."""

    xyz = tensors.xyz.detach().cpu().numpy()
    scales = tensors.scales.detach().cpu().numpy()
    quats = tensors.quats.detach().cpu().numpy()
    opacities = tensors.opacities.detach().cpu().numpy()
    sh0 = tensors.features_dc.detach().cpu().numpy()
    shN = tensors.features_rest.detach().cpu().numpy()

    for tile_id, span in tensors.tile_slices.items():
        meta = storage.metadata["tiles"][tile_id]
        rel_path = Path(meta["file_path"])
        target = storage.root_dir / rel_path
        if output_root is not None:
            target = Path(output_root) / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)

        idx = slice(span.start, span.stop)
        np.savez_compressed(
            target,
            means=xyz[idx],
            scales=scales[idx],
            quats=quats[idx],
            opacities=opacities[idx, 0],
            sh0=sh0[idx],
            shN=shN[idx],
        )
