"""Utilities to persist optimizer state for tile-based training.

This module provides helpers to extract optimizer buffers (Adam-style m/v)
for slices corresponding to tiles and save them to per-tile NPZ files, and to
restore them back into an optimizer. The code assumes flat parameter tensors
are used (as in TileBatchTensors.populate_gaussian_model which creates
flat nn.Parameter tensors on the model).

The functions are intentionally minimal and file-format stable: each NPZ will
contain 'opt_m', 'opt_v' arrays (float32) and an integer 'opt_step'.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def _param_flattened_view(param: torch.nn.Parameter) -> torch.Tensor:
    return param.detach().reshape(-1)


def save_optimizer_state_per_tile(
    optimizer: torch.optim.Optimizer,
    tensors_tile_slices: Dict[str, slice],
    out_dir: str,
    device: torch.device = torch.device("cpu"),
    prefix: str = "opt_state_",
) -> None:
    """Save per-tile optimizer m/v and step for Adam-like optimizers.

    Args:
        optimizer: torch Optimizer (expects single param group and Adam-like state)
        tensors_tile_slices: mapping tile_id -> slice(start, stop) into flat params
        out_dir: directory to write per-tile NPZ files
        device: target device for reading state tensors
        prefix: filename prefix for saved files
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect flat parameter state from optimizer state dict. We assume optimizer
    # has a single parameter tensor (or that parameters are ordered to match slices).
    state = optimizer.state
    # Build concatenated m/v if present
    # Find first parameter state that contains 'exp_avg' (Adam m)
    exp_avg = None
    exp_avg_sq = None
    global_step = 0

    for p in optimizer.param_groups[0]["params"]:
        s = state.get(p)
        if not s:
            continue
        if "exp_avg" in s:
            exp_avg = s["exp_avg"].detach().cpu().reshape(-1).clone()
        if "exp_avg_sq" in s:
            exp_avg_sq = s["exp_avg_sq"].detach().cpu().reshape(-1).clone()
        if "step" in s:
            global_step = int(s.get("step", global_step))
        # Break after first param found
        break

    if exp_avg is None or exp_avg_sq is None:
        # Nothing to save for this optimizer
        return

    # If optimizer params are multiple tensors, concatenate in param_groups order
    # to form a flat vector compatible with TileBatchTensors slices.
    # For simplicity, if multiple params present, concatenate all their states.
    exp_avg_list = []
    exp_avg_sq_list = []
    for p in optimizer.param_groups[0]["params"]:
        s = state.get(p)
        if not s:
            continue
        if "exp_avg" in s:
            exp_avg_list.append(s["exp_avg"].detach().cpu().reshape(-1))
        if "exp_avg_sq" in s:
            exp_avg_sq_list.append(s["exp_avg_sq"].detach().cpu().reshape(-1))

    if exp_avg_list:
        exp_avg = torch.cat(exp_avg_list).clone()
    if exp_avg_sq_list:
        exp_avg_sq = torch.cat(exp_avg_sq_list).clone()

    exp_avg_np = exp_avg.numpy().astype(np.float32)
    exp_avg_sq_np = exp_avg_sq.numpy().astype(np.float32)

    # Now write per-tile slices
    for tile_id, span in tensors_tile_slices.items():
        s = slice(span.start, span.stop)
        m_slice = exp_avg_np[s]
        v_slice = exp_avg_sq_np[s]
        target = out_path / f"{prefix}{tile_id}.npz"
        np.savez_compressed(target, opt_m=m_slice, opt_v=v_slice, opt_step=global_step)


def load_optimizer_state_for_tile(
    optimizer: torch.optim.Optimizer,
    tile_id: str,
    tensors_tile_slices: Dict[str, slice],
    state_dir: str,
) -> bool:
    """Load optimizer state for a single tile and inject into optimizer state.

    Returns True if state was loaded, False otherwise.
    """
    path = Path(state_dir) / f"opt_state_{tile_id}.npz"
    if not path.exists():
        return False

    data = np.load(path)
    opt_m = torch.from_numpy(data["opt_m"]).to(torch.float32)
    opt_v = torch.from_numpy(data["opt_v"]).to(torch.float32)
    opt_step = int(data.get("opt_step", 0))

    # Build flat buffers from optimizer params
    params = [p for p in optimizer.param_groups[0]["params"]]
    # Concatenate existing buffers shapes to know offsets
    total_len = sum(p.numel() for p in params)

    span = tensors_tile_slices[tile_id]
    start = span.start
    stop = span.stop

    # Inject into optimizer state: we distribute the slice into the first param's state
    # by offsetting. This is a simple approach and assumes params are concatenated
    # in the same order when saving.
    offset = 0
    for p in params:
        num = p.numel()
        p_state = optimizer.state.setdefault(p, {})
        # compute overlap between [start,stop) and [offset, offset+num)
        lo = max(start, offset)
        hi = min(stop, offset + num)
        if lo < hi:
            local_lo = lo - offset
            local_hi = hi - offset
            # extract corresponding slice from opt_m/opt_v
            seg_lo = lo - start
            seg_hi = hi - start
            # Ensure state buffers exist
            if "exp_avg" not in p_state or p_state.get("exp_avg").numel() != num:
                p_state["exp_avg"] = torch.zeros_like(p.data)
            if "exp_avg_sq" not in p_state or p_state.get("exp_avg_sq").numel() != num:
                p_state["exp_avg_sq"] = torch.zeros_like(p.data)
            # Copy into the correct positions
            p_state["exp_avg"].view(-1)[local_lo:local_hi].copy_(opt_m[seg_lo:seg_hi])
            p_state["exp_avg_sq"].view(-1)[local_lo:local_hi].copy_(opt_v[seg_lo:seg_hi])
            # PyTorch's optimizer expects 'step' to be a singleton tensor for some backends
            try:
                p_state["step"] = torch.tensor([int(opt_step)], dtype=torch.long, device=p.data.device)
            except Exception:
                # fallback to plain int if tensor creation fails for any reason
                p_state["step"] = int(opt_step)
        offset += num

    return True
