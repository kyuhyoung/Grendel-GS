import os
import sys
import gc
import torch
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from utils.loss_utils import l1_loss
from gaussian_renderer import (
    distributed_preprocess3dgs_and_all2all_final,
    render_final,
    gsplat_distributed_preprocess3dgs_and_all2all_final,
    gsplat_render_final,
)
from torch.cuda import nvtx
from scene import Scene, GaussianModel, SceneDataset
from gaussian_renderer.workload_division import (
    start_strategy_final,
    finish_strategy_final,
    DivisionStrategyHistoryFinal,
)
from gaussian_renderer.loss_distribution import (
    load_camera_from_cpu_to_all_gpu,
    load_camera_from_cpu_to_all_gpu_for_eval,
    batched_loss_computation,
)
from utils.general_utils import prepare_output_and_logger, globally_sync_for_timer
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
import torch.distributed as dist
from densification import densification, gsplat_densification

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.image_loss_utils import (
    prepare_image_loss,
    render_gaussian_model,
    save_abs_diff_image,
    save_image_tensor,
)
from src.tile_storage import TileStorage
from src.tile_training_utils import load_tiles_as_tensors, write_back_tiles
from scene.adaptive_tile_utils import TileBBox


# Exit codes for adaptive tile training
EXIT_CODE_SUCCESS = 0
EXIT_CODE_OOM = 42  # Special exit code to signal OOM to wrapper script


# ============================================================================
# GPU Memory Logging for OOM Diagnosis
# ============================================================================

class GPUMemoryTracker:
    """Track GPU memory usage and current operation for OOM diagnosis."""

    def __init__(self):
        self.current_operation = "initialization"
        self.memory_log = []
        self.enabled = True

    def set_operation(self, operation: str):
        """Set the current operation being performed."""
        self.current_operation = operation

    def log_memory(self, label: str = None, force: bool = False):
        """Log current GPU memory usage."""
        if not self.enabled and not force:
            return

        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

        entry = {
            "operation": self.current_operation,
            "label": label or self.current_operation,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
        }
        self.memory_log.append(entry)

        if force or (label and "OOM" in label.upper()):
            utils.print_rank_0(
                f"[GPU-MEM] {entry['label']}: "
                f"alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, max={max_allocated:.2f}GB"
            )

    def get_memory_summary(self) -> str:
        """Get a summary of memory usage for OOM diagnosis."""
        if not torch.cuda.is_available():
            return "CUDA not available"

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        # Get device properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / 1024**3

        summary = [
            f"\n{'='*60}",
            f"GPU MEMORY ANALYSIS (OOM occurred during: {self.current_operation})",
            f"{'='*60}",
            f"  Device: {props.name} (GPU {device})",
            f"  Total GPU Memory: {total_memory:.2f} GB",
            f"  Currently Allocated: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)",
            f"  Reserved by PyTorch: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)",
            f"  Peak Allocated: {max_allocated:.2f} GB ({max_allocated/total_memory*100:.1f}%)",
            f"  Free (estimated): {total_memory - reserved:.2f} GB",
            f"{'='*60}",
        ]

        # Add recent memory log
        if self.memory_log:
            summary.append("Recent Memory Usage:")
            for entry in self.memory_log[-10:]:  # Last 10 entries
                summary.append(
                    f"  [{entry['operation']}] {entry['label']}: {entry['allocated_gb']:.2f}GB"
                )
            summary.append(f"{'='*60}")

        return "\n".join(summary)

    def reset_peak(self):
        """Reset peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


# Global memory tracker instance
_memory_tracker = GPUMemoryTracker()


def get_memory_tracker() -> GPUMemoryTracker:
    """Get the global memory tracker instance."""
    return _memory_tracker


def log_gpu_memory(label: str = None, force: bool = False):
    """Convenience function to log GPU memory."""
    _memory_tracker.log_memory(label, force)


def set_current_operation(operation: str):
    """Set the current operation for OOM tracking."""
    _memory_tracker.set_operation(operation)


def _is_oom_error(exception: BaseException) -> bool:
    """Check if an exception is a CUDA out of memory error."""
    return isinstance(exception, RuntimeError) and (
        "out of memory" in str(exception).lower() or
        "CUDA" in str(exception) and "memory" in str(exception).lower()
    )


def handle_adaptive_tile_oom(
    gaussians: GaussianModel,
    args,
    scene: Scene,
    iteration: int,
    log_file,
    num_cameras: int = 0,
    total_pixels: int = 0,
    total_cameras_in_tile: int = 0,
):
    """
    Handle OOM in adaptive tile mode by splitting the tile and saving state.

    This function:
    1. Splits current tile bbox in half
    2. Filters gaussians for each half
    3. Saves one half as PLY (completed)
    4. Saves state for the other half (to be resumed)
    5. Exits with special code for wrapper to handle

    Args:
        gaussians: Current gaussian model
        args: Training arguments
        scene: Scene object with tile_bbox
        iteration: Current iteration
        log_file: Log file handle
        num_cameras: Number of cameras in current batch
        total_pixels: Total pixels across all cameras in batch
        total_cameras_in_tile: Total number of cameras for this tile (C)
    """
    if not getattr(args, "adaptive_tile_enabled", False):
        return False  # Not in adaptive tile mode

    tile_bbox = scene.tile_bbox
    if tile_bbox is None:
        return False

    tile_id = getattr(args, "tile_id", "unknown")
    tile_output_dir = getattr(args, "tile_output_dir", "") or args.model_path
    tile_output_dir = Path(tile_output_dir)
    tile_output_dir.mkdir(parents=True, exist_ok=True)

    # Get memory tracker for detailed analysis
    memory_tracker = get_memory_tracker()

    utils.print_rank_0("\n" + "!" * 60)
    utils.print_rank_0("!!!  GPU OOM DETECTED  !!!")
    utils.print_rank_0("!" * 60)

    # Print memory analysis
    utils.print_rank_0(memory_tracker.get_memory_summary())

    utils.print_rank_0(f"[adaptive-tile] Tile: {tile_id}, Iteration: {iteration}")
    log_file.write(f"[adaptive-tile] OOM detected on tile {tile_id} at iteration {iteration}\n")
    log_file.write(memory_tracker.get_memory_summary() + "\n")

    # Show current tile info
    dx = tile_bbox.x_max - tile_bbox.x_min
    dy = tile_bbox.y_max - tile_bbox.y_min
    dz = tile_bbox.z_max - tile_bbox.z_min
    utils.print_rank_0(f"[adaptive-tile] Current tile size: X={dx:.2f}, Y={dy:.2f}, Z={dz:.2f}")

    # Show OOM cause analysis
    num_gaussians = gaussians.get_xyz.shape[0] if gaussians.get_xyz is not None else 0
    utils.print_rank_0(f"\n[OOM Cause Analysis]")
    utils.print_rank_0(f"  Failed during: {memory_tracker.current_operation}")
    utils.print_rank_0(f"  Iteration: {iteration}")
    utils.print_rank_0(f"  Gaussians: {num_gaussians:,}")
    if num_cameras > 0:
        utils.print_rank_0(f"  Cameras in batch: {num_cameras}")
        utils.print_rank_0(f"  Total cameras (C): {num_cameras}")
    if total_pixels > 0:
        utils.print_rank_0(f"  Total pixels: {total_pixels:,} ({total_pixels/1e6:.1f}M)")
        # Estimate memory for rendered image (float32 RGB)
        estimated_img_mem = total_pixels * 3 * 4 / 1024**3  # GB
        utils.print_rank_0(f"  Estimated image memory: {estimated_img_mem:.2f} GB")
    # Estimate gaussian memory (rough: ~200 bytes per gaussian for all attributes)
    estimated_gauss_mem = num_gaussians * 200 / 1024**3  # GB
    utils.print_rank_0(f"  Estimated gaussian memory: {estimated_gauss_mem:.2f} GB")

    # Analyze likely OOM cause based on iteration vs camera count and densification
    densify_from = getattr(args, 'densify_from_iter', 500)
    densify_interval = getattr(args, 'densification_interval', 100)
    first_densify_iter = densify_from + densify_interval
    C = total_cameras_in_tile if total_cameras_in_tile > 0 else num_cameras

    utils.print_rank_0(f"\n[OOM Cause Diagnosis]")
    utils.print_rank_0(f"  Iteration: {iteration}")
    utils.print_rank_0(f"  Total cameras in tile (C): {C}")
    utils.print_rank_0(f"  First densification at: ~{first_densify_iter}")

    if C > 0 and iteration <= C:
        # First cycle through cameras
        likely_cause = "SSIM on large image (first pass through cameras)"
        utils.print_rank_0(f"  Iteration({iteration}) <= C({C}): Still in first camera cycle")
        utils.print_rank_0(f"  --> Likely cause: {likely_cause}")
    elif iteration <= first_densify_iter:
        # After first cycle but before densification
        likely_cause = "Memory fragmentation or leak (same images succeeded before, no densification yet)"
        utils.print_rank_0(f"  C({C}) < Iteration({iteration}) <= first_densify({first_densify_iter}): Before densification")
        utils.print_rank_0(f"  --> Likely cause: {likely_cause}")
    else:
        # After densification started
        likely_cause = "Increased gaussians from densification"
        utils.print_rank_0(f"  Iteration({iteration}) > first_densify({first_densify_iter}): Densification has occurred")
        utils.print_rank_0(f"  --> Likely cause: {likely_cause}")

    # Determine split axis
    if dx >= dy and dx >= dz:
        split_axis = "X"
        mid = (tile_bbox.x_min + tile_bbox.x_max) / 2
        utils.print_rank_0(f"[adaptive-tile] Splitting along {split_axis} axis (longest): mid={mid:.2f}")
    elif dy >= dz:
        split_axis = "Y"
        mid = (tile_bbox.y_min + tile_bbox.y_max) / 2
        utils.print_rank_0(f"[adaptive-tile] Splitting along {split_axis} axis (longest): mid={mid:.2f}")
    else:
        split_axis = "Z"
        mid = (tile_bbox.z_min + tile_bbox.z_max) / 2
        utils.print_rank_0(f"[adaptive-tile] Splitting along {split_axis} axis (longest): mid={mid:.2f}")

    # Split the tile
    tile_a, tile_b = tile_bbox.split()
    utils.print_rank_0(f"[adaptive-tile] Splitting tile into:")
    utils.print_rank_0(f"  Tile A: {tile_a.to_string()}")
    utils.print_rank_0(f"  Tile B: {tile_b.to_string()}")

    # Get current gaussians as numpy
    with torch.no_grad():
        xyz = gaussians.get_xyz.cpu().numpy()

    # Filter gaussians for each tile
    mask_a = tile_a.contains_points(xyz)
    mask_b = tile_b.contains_points(xyz)

    count_a = mask_a.sum()
    count_b = mask_b.sum()
    utils.print_rank_0(f"[adaptive-tile] Tile A: {count_a} gaussians")
    utils.print_rank_0(f"[adaptive-tile] Tile B: {count_b} gaussians")

    # Generate new tile IDs
    tile_num = int(tile_id.split("_")[-1]) if "_" in tile_id else 0
    tile_a_id = f"tile_{tile_num * 2 + 1:04d}"
    tile_b_id = f"tile_{tile_num * 2 + 2:04d}"

    # Save tile A gaussians as PLY (this half is "done" for now)
    # tile_output_dir is already the tiles directory
    tile_output_dir.mkdir(parents=True, exist_ok=True)
    ply_path_a = tile_output_dir / f"{tile_a_id}.ply"

    # Create a filtered gaussian model for tile A and save
    # Note: This is a simplified save - may need refinement for full state
    if count_a > 0:
        utils.print_rank_0(f"[adaptive-tile] Saving tile A to {ply_path_a}")
        gaussians.save_ply_masked(str(ply_path_a), mask_a)

    # Save state for wrapper script to handle
    state_file = getattr(args, "tile_state_file", "") or (tile_output_dir / "adaptive_tile_state.json")
    state_file = Path(state_file)

    oom_state = {
        "oom_occurred": True,
        "original_tile_id": tile_id,
        "original_tile_bbox": tile_bbox.to_string(),
        "iteration": iteration,
        "tile_a": {
            "tile_id": tile_a_id,
            "bbox": tile_a.to_string(),
            "ply_path": str(ply_path_a) if count_a > 0 else None,
            "gaussian_count": int(count_a),
            "status": "saved",
        },
        "tile_b": {
            "tile_id": tile_b_id,
            "bbox": tile_b.to_string(),
            "gaussian_count": int(count_b),
            "status": "pending",
        },
    }

    with open(state_file, "w") as f:
        json.dump(oom_state, f, indent=2)

    utils.print_rank_0(f"[adaptive-tile] Saved OOM state to {state_file}")
    log_file.write(f"[adaptive-tile] Saved OOM state to {state_file}\n")
    log_file.flush()

    return True  # OOM was handled


class TileImageLossScheduler:
    """Schedules periodic COLMAP-driven image supervision during training."""
    def __init__(self, args, background):
        self.enabled = bool(getattr(args, "tile_loss_enable", False))
        self._disabled_reason = None
        if not self.enabled:
            return

        self.interval = max(1, int(getattr(args, "tile_loss_interval", 1)))
        start_iter = int(getattr(args, "tile_loss_start_iteration", self.interval))
        self.next_trigger = max(0, start_iter)
        self.weight = float(getattr(args, "tile_loss_weight", 1.0))

        colmap_arg = getattr(args, "tile_loss_colmap", "")
        if not colmap_arg:
            self._disable("--tile-loss-colmap not provided")
            return
        self.colmap_dir = Path(colmap_arg)
        if not self.colmap_dir.exists():
            self._disable(f"COLMAP directory not found: {self.colmap_dir}")
            return

        images_arg = getattr(args, "tile_loss_images_dir", "")
        if images_arg:
            self.images_dir = Path(images_arg)
        else:
            self.images_dir = self.colmap_dir.parent / "images"
        if not self.images_dir.exists():
            self._disable(f"Images directory not found: {self.images_dir}")
            return

        self.image_idx = int(getattr(args, "tile_loss_image_idx", 0))
        self.resolution_scale = float(getattr(args, "tile_loss_resolution_scale", 0.05))

        output_arg = getattr(args, "tile_loss_output_dir", "")
        self.output_dir = Path(output_arg) if output_arg else None
        if self.output_dir and utils.GLOBAL_RANK == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.background = background
        self._prepared = False
        self._context = None
        self._bg_color = None
        self._device = None
        self.last_preview_path = None
        self.last_diff_path = None

    def _disable(self, message: str) -> None:
        self.enabled = False
        self._disabled_reason = message
        utils.print_rank_0(f"[tile-loss] Disabled tile image loss: {message}")

    def _ensure_prepared(self, device: torch.device) -> None:
        if not self.enabled or self._prepared:
            return
        self._device = device
        self._context = prepare_image_loss(
            device=device,
            colmap_dir=self.colmap_dir,
            images_dir=self.images_dir,
            image_idx=self.image_idx,
            resolution_scale=self.resolution_scale,
        )
        if self.background is not None:
            self._bg_color = self.background.detach().to(device=device)
        else:
            bg = self._context.get("bg_color")
            if bg is None:
                self._bg_color = torch.ones(3, device=device)
            else:
                self._bg_color = bg.detach().to(device=device)
        self._prepared = True

    def maybe_run(self, iteration: int, gaussians: GaussianModel) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        if not self.enabled:
            return None, None
        if iteration < self.next_trigger:
            return None, None

        device = gaussians.get_xyz.device
        self._ensure_prepared(device)
        if not self._prepared:
            return None, None

        render = render_gaussian_model(gaussians, self._context["camera"], self._bg_color)
        target = self._context["target"]
        diff = render - target
        raw_loss = torch.mean(diff * diff)
        scaled_loss = raw_loss * self.weight

        if self.output_dir and utils.GLOBAL_RANK == 0:
            iter_idx = utils.get_cur_iter()
            preview_path = self.output_dir / f"tile_loss_render_{iter_idx:06d}.png"
            diff_path = self.output_dir / f"tile_loss_diff_{iter_idx:06d}.png"
            save_image_tensor(render, preview_path)
            save_abs_diff_image(render, self._context["target_cpu"], diff_path)
            self.last_preview_path = preview_path
            self.last_diff_path = diff_path

        self.next_trigger = iteration + self.interval
        return scaled_loss, float(raw_loss.detach().cpu())


class TileBatchManager:
    """Manage loading and saving tile batches for out-of-core training."""

    def __init__(self, args, device: torch.device, log_file) -> None:
        self.enabled = bool(getattr(args, "tile_scene_root", ""))
        self._device = device
        self._log_file = log_file
        self._opt_args = None
        if not self.enabled:
            self.storage = None
            return

        root = Path(args.tile_scene_root)
        if not root.exists():
            raise FileNotFoundError(f"Tile scene directory not found: {root}")
        self.storage = TileStorage(root)

        self.steps_per_batch = max(1, int(getattr(args, "tile_steps_per_batch", 1)))
        self.batch_size = max(1, int(getattr(args, "tile_batch_size", 1)))
        self.shuffle = bool(getattr(args, "tile_plan_shuffle", False))
        self.cycle = bool(getattr(args, "tile_cycle_batches", True))
        write_back_root = getattr(args, "tile_write_back_root", "")
        self.write_back_root = Path(write_back_root) if write_back_root else None
        if self.write_back_root and utils.GLOBAL_RANK == 0:
            self.write_back_root.mkdir(parents=True, exist_ok=True)

        self.tile_batches = self._build_batches(args)
        if not self.tile_batches:
            raise RuntimeError("No tile batches available for training")

        self.batch_index = -1
        self.steps_remaining = 0
        self.current_batch = None
        self.current_tile_ids: List[str] = []
        self._args = args

        if not getattr(args, "disable_auto_densification", False):
            utils.print_rank_0("[tile-ooc] Disabling auto densification for tile streaming mode")
            args.disable_auto_densification = True

    @staticmethod
    def _expand_tile_tokens(tokens: Iterable[str]) -> List[str]:
        ids: List[str] = []
        for token in tokens:
            for part in str(token).split(","):
                tid = part.strip()
                if not tid:
                    continue
                if tid.startswith("tile_"):
                    tid = tid[len("tile_") :]
                ids.append(tid)
        return ids

    def _build_batches(self, args) -> List[List[str]]:
        plan_path = getattr(args, "tile_plan_json", "")
        batches: List[List[str]] = []
        if plan_path:
            plan_file = Path(plan_path)
            if not plan_file.exists():
                raise FileNotFoundError(f"Tile plan JSON not found: {plan_file}")
            with open(plan_file, "r", encoding="utf-8") as handle:
                plan = json.load(handle)
            raw_batches = plan.get("batches", [])
            for entry in raw_batches:
                tiles = entry.get("tiles")
                if not tiles:
                    continue
                batches.append(self._expand_tile_tokens(tiles))
        else:
            all_tiles = sorted(self.storage.metadata["tiles"].keys())
            if self.shuffle:
                random.shuffle(all_tiles)
            for idx in range(0, len(all_tiles), self.batch_size):
                batches.append(all_tiles[idx : idx + self.batch_size])

        if self.shuffle and plan_path:
            random.shuffle(batches)
        return batches

    def _assign_batch_to_model(self, gaussians: GaussianModel) -> None:
        assert self.current_batch is not None
        total = self.current_batch.xyz.shape[0]
        gaussians._xyz = torch.nn.Parameter(self.current_batch.xyz.clone().requires_grad_(True))
        gaussians._scaling = torch.nn.Parameter(self.current_batch.scales.clone().requires_grad_(True))
        gaussians._rotation = torch.nn.Parameter(self.current_batch.quats.clone().requires_grad_(True))
        gaussians._opacity = torch.nn.Parameter(self.current_batch.opacities.clone().requires_grad_(True))
        gaussians._features_dc = torch.nn.Parameter(self.current_batch.features_dc.clone().requires_grad_(True))
        gaussians._features_rest = torch.nn.Parameter(self.current_batch.features_rest.clone().requires_grad_(True))
        gaussians.max_radii2D = torch.zeros((total,), device=self._device)
        gaussians.xyz_gradient_accum = torch.zeros((total, 1), device=self._device)
        gaussians.denom = torch.zeros((total, 1), device=self._device)
        gaussians.active_sh_degree = min(gaussians.max_sh_degree, self.current_batch.infer_sh_degree())

    def _switch_to_next_batch(self, gaussians: GaussianModel, *, initial: bool = False) -> bool:
        if self.current_batch is not None and not initial:
            self.current_batch.copy_from_model(gaussians)
            write_back_tiles(self.storage, self.current_batch, output_root=self.write_back_root)
            self.current_batch = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.batch_index += 1
        if self.batch_index >= len(self.tile_batches):
            if not self.cycle:
                utils.print_rank_0("[tile-ooc] Tile batches exhausted; stopping reuse.")
                self.enabled = False
                return False
            self.batch_index = 0
            if self.shuffle:
                random.shuffle(self.tile_batches)

        tile_ids = self.tile_batches[self.batch_index]
        self.current_batch = load_tiles_as_tensors(self.storage, tile_ids, device=self._device)
        self.current_tile_ids = tile_ids
        self._assign_batch_to_model(gaussians)
        if self._opt_args is not None:
            gaussians.training_setup(self._opt_args)
        self.steps_remaining = self.steps_per_batch
        utils.print_rank_0(
            f"[tile-ooc] Activated batch {self.batch_index + 1}/{len(self.tile_batches)} "
            f"({len(tile_ids)} tiles, {self.current_batch.xyz.shape[0]} gaussians)"
        )
        self._log_file.write(
            f"[tile-ooc] batch_index={self.batch_index} tiles={tile_ids} count={self.current_batch.xyz.shape[0]}\n"
        )
        return True

    def bootstrap(self, gaussians: GaussianModel, opt_args) -> None:
        if not self.enabled:
            return
        self._opt_args = opt_args
        switched = self._switch_to_next_batch(gaussians, initial=True)
        if not switched:
            raise RuntimeError("Failed to activate initial tile batch")

    def prepare_for_iteration(self, iteration: int, gaussians: GaussianModel, opt_args) -> Optional[Dict[str, object]]:
        if not self.enabled:
            return None
        self._opt_args = opt_args
        if self.current_batch is None or self.steps_remaining <= 0:
            if not self._switch_to_next_batch(gaussians):
                return {"halt": True}
            just_switched = True
        else:
            just_switched = False
        self.steps_remaining -= 1
        return {
            "batch_index": self.batch_index,
            "tile_ids": list(self.current_tile_ids),
            "steps_remaining": self.steps_remaining,
            "just_switched": just_switched,
        }

    def finalize(self, gaussians: GaussianModel) -> None:
        if not self.enabled:
            return
        if self.current_batch is not None:
            self.current_batch.copy_from_model(gaussians)
            write_back_tiles(self.storage, self.current_batch, output_root=self.write_back_root)
            self.current_batch = None
        if self.storage is not None:
            self.storage.close()

def training(dataset_args, opt_args, pipe_args, args, log_file):

    # Init auxiliary tools

    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1

    # Initialize memory tracker
    memory_tracker = get_memory_tracker()
    set_current_operation("scene_initialization")
    log_gpu_memory("training_start", force=True)

    # Init parameterized scene
    gaussians = GaussianModel(dataset_args.sh_degree)

    with torch.no_grad():
        set_current_operation("scene_loading")
        scene = Scene(args, gaussians)
        log_gpu_memory("after_scene_load", force=True)

        if args.start_checkpoint != "":
            model_params, start_from_this_iteration = utils.load_checkpoint(args)
            gaussians.restore(model_params, opt_args)
            utils.print_rank_0(
                "Restored from checkpoint: {}".format(args.start_checkpoint)
            )
            log_file.write(
                "Restored from checkpoint: {}\n".format(args.start_checkpoint)
            )

    utils.check_initial_gpu_memory_usage("after init and before training loop")

    # Init dataset
    train_dataset = SceneDataset(scene.getTrainCameras())
    if args.adjust_strategy_warmp_iterations == -1:
        args.adjust_strategy_warmp_iterations = len(train_dataset.cameras)
        # use one epoch to warm up. do not use the first epoch's running time for adjustment of strategy.

    # Init distribution strategy history
    strategy_history = DivisionStrategyHistoryFinal(
        train_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )

    # Init background
    background = None
    if args.backend == "gsplat":
        bg_color = [1, 1, 1] if dataset_args.white_background else None
    else:
        bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    tile_manager = TileBatchManager(
        args,
        device=background.device if background is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        log_file=log_file,
    )

    if tile_manager.enabled and args.start_checkpoint != "":
        raise RuntimeError("Checkpoint restore is not supported in tile out-of-core mode yet")

    if not tile_manager.enabled:
        gaussians.training_setup(opt_args)
    else:
        tile_manager.bootstrap(gaussians, opt_args)

    scene.log_scene_info_to_file(log_file, "Scene Info Before Training")

    tile_loss_scheduler = TileImageLossScheduler(args, background)

    # Training Loop
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc="Training progress",
        disable=(utils.LOCAL_RANK != 0),
    )
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0

    ema_loss_for_log = 0
    adaptive_tile_mode = getattr(args, "adaptive_tile_enabled", False)
    current_iteration = start_from_this_iteration

    try:
        for iteration in range(
            start_from_this_iteration, opt_args.iterations + 1, args.bsz
        ):
            current_iteration = iteration
            # Debug logging for first few iterations
            debug_first_iters = iteration <= 3

            # Reset operation tracker for this iteration
            set_current_operation("iteration_start")

            if debug_first_iters:
                utils.print_rank_0(f"\n[DEBUG] ===== Iteration {iteration} START =====")
                # Print key stats for OOM diagnosis
                num_gaussians = gaussians.get_xyz.shape[0] if gaussians.get_xyz is not None else 0
                utils.print_rank_0(f"[DEBUG] [{iteration}] Gaussians: {num_gaussians:,}")

            # Step Initialization
            tile_loss_raw = None
            tile_info = tile_manager.prepare_for_iteration(iteration, gaussians, opt_args)
            if tile_info and tile_info.get("halt"):
                utils.print_rank_0("[tile-ooc] Exhausted tile batches; ending training loop early")
                break
            if tile_info and tile_info.get("just_switched"):
                log_file.write(
                    f"[tile-ooc] iteration={iteration} active_batch={tile_info['batch_index']} tiles={tile_info['tile_ids']}\n"
                )
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "TileBatch": str(tile_info["batch_index"]),
                })
            if iteration // args.bsz % 30 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            # NOTE: progress_bar.update() moved to end of iteration (after backward pass succeeds)
            utils.set_cur_iter(iteration)
            gaussians.update_learning_rate(iteration)
            num_trained_batches += 1
            timers.clear()
            if args.nsys_profile:
                nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
                gaussians.oneupSHdegree()

            # Prepare data: Pick random Cameras for training
            if args.local_sampling:
                assert (
                    args.bsz % utils.WORLD_SIZE == 0
                ), "Batch size should be divisible by the number of GPUs."
                batched_cameras_idx = train_dataset.get_batched_cameras_idx(
                    args.bsz // utils.WORLD_SIZE
                )
                batched_all_cameras_idx = torch.zeros(
                    (utils.WORLD_SIZE, len(batched_cameras_idx)), device="cuda", dtype=int
                )
                batched_cameras_idx = torch.tensor(
                    batched_cameras_idx, device="cuda", dtype=int
                )
                torch.distributed.all_gather_into_tensor(
                    batched_all_cameras_idx, batched_cameras_idx, group=utils.DEFAULT_GROUP
                )
                batched_all_cameras_idx = batched_all_cameras_idx.cpu().numpy().squeeze()
                batched_cameras = train_dataset.get_batched_cameras_from_idx(
                    batched_all_cameras_idx
                )
            else:
                batched_cameras = train_dataset.get_batched_cameras(args.bsz)

            # Set image size for current camera(s) - critical for adaptive tile with varying crop sizes
            # All ranks must call this with the same values to stay synchronized
            if len(batched_cameras) == 1:
                cam = batched_cameras[0]
                utils.set_img_size(cam.image_height, cam.image_width)
            elif len(batched_cameras) > 1:
                # For batch size > 1, verify all cameras have same dimensions
                first_cam = batched_cameras[0]
                for cam in batched_cameras[1:]:
                    if cam.image_width != first_cam.image_width or cam.image_height != first_cam.image_height:
                        raise RuntimeError(
                            f"Batch contains cameras with different sizes: "
                            f"{first_cam.image_name}={first_cam.image_width}x{first_cam.image_height} vs "
                            f"{cam.image_name}={cam.image_width}x{cam.image_height}. "
                            f"Use --bsz 1 for adaptive tile training with varying crop sizes."
                        )
                utils.set_img_size(first_cam.image_height, first_cam.image_width)

            # Track camera info for OOM analysis
            current_num_cameras = len(batched_cameras)
            current_total_pixels = sum(c.image_width * c.image_height for c in batched_cameras)

            with torch.no_grad():
                # Prepare Workload division strategy
                set_current_operation("prepare_strategies")
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] Preparing strategies...")
                timers.start("prepare_strategies")
                batched_strategies, gpuid2tasks = start_strategy_final(
                    batched_cameras, strategy_history
                )
                timers.stop("prepare_strategies")
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] Strategies prepared")

                # Load ground-truth images to GPU
                set_current_operation("load_images_to_gpu")
                log_gpu_memory("before_load_images")
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] Loading cameras to GPU...")
                    for i, cam in enumerate(batched_cameras):
                        utils.print_rank_0(f"[DEBUG] [{iteration}]   Camera {i}: {cam.image_name} {cam.image_width}x{cam.image_height}")
                timers.start("load_cameras")
                load_camera_from_cpu_to_all_gpu(
                    batched_cameras, batched_strategies, gpuid2tasks
                )
                timers.stop("load_cameras")
                log_gpu_memory("after_load_images")
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] Cameras loaded to GPU")

            if debug_first_iters:
                utils.print_rank_0(f"[DEBUG] [{iteration}] Starting rendering (backend={args.backend})...")

            if args.backend == "gsplat":
                set_current_operation("gsplat_preprocess")
                log_gpu_memory("before_gsplat_preprocess")
                batched_screenspace_pkg = (
                    gsplat_distributed_preprocess3dgs_and_all2all_final(
                        batched_cameras,
                        gaussians,
                        pipe_args,
                        background,
                        batched_strategies=batched_strategies,
                        mode="train",
                    )
                )
                log_gpu_memory("after_gsplat_preprocess")
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] gsplat preprocess done, rendering...")
                set_current_operation("gsplat_render")
                log_gpu_memory("before_gsplat_render")
                batched_image, batched_compute_locally = gsplat_render_final(
                    batched_screenspace_pkg, batched_strategies
                )
                log_gpu_memory("after_gsplat_render")
                batch_statistic_collector = [
                    cuda_args["stats_collector"]
                    for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
                ]
            else:
                set_current_operation("default_preprocess")
                log_gpu_memory("before_preprocess")
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] About to call distributed_preprocess3dgs_and_all2all_final...")
                batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
                    batched_cameras,
                    gaussians,
                    pipe_args,
                    background,
                    batched_strategies=batched_strategies,
                    mode="train",
                )
                log_gpu_memory("after_preprocess")
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] preprocess done, rendering...")
                set_current_operation("default_render")
                log_gpu_memory("before_render")
                batched_image, batched_compute_locally = render_final(
                    batched_screenspace_pkg, batched_strategies
                )
                log_gpu_memory("after_render")
                batch_statistic_collector = [
                    cuda_args["stats_collector"]
                    for cuda_args in batched_screenspace_pkg["batched_cuda_args"]
                ]

            if debug_first_iters:
                utils.print_rank_0(f"[DEBUG] [{iteration}] Rendering done, computing loss...")

            set_current_operation("loss_computation (L1+SSIM)")
            log_gpu_memory("before_loss")
            loss_sum, batched_losses = batched_loss_computation(
                batched_image,
                batched_cameras,
                batched_compute_locally,
                batched_strategies,
                batch_statistic_collector,
            )
            log_gpu_memory("after_loss")

            if debug_first_iters:
                utils.print_rank_0(f"[DEBUG] [{iteration}] Loss computed: {loss_sum.item():.6f}, starting backward...")

            set_current_operation("backward")
            log_gpu_memory("before_backward")
            timers.start("backward")
            loss_sum.backward()
            timers.stop("backward")
            log_gpu_memory("after_backward")
            utils.check_initial_gpu_memory_usage("after backward")

            if debug_first_iters:
                utils.print_rank_0(f"[DEBUG] [{iteration}] Backward done")

            with torch.no_grad():
                # Adjust workload division strategy.
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] Syncing for timer...")
                globally_sync_for_timer()
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] Finishing strategy...")
                timers.start("finish_strategy_final")
                finish_strategy_final(
                    batched_cameras,
                    strategy_history,
                    batched_strategies,
                    batch_statistic_collector,
                )
                timers.stop("finish_strategy_final")

                # Sync losses in the batch
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] Syncing losses...")
                timers.start("sync_loss_and_log")
                batched_losses = torch.tensor(batched_losses, device="cuda")
                if utils.DEFAULT_GROUP.size() > 1:
                    dist.all_reduce(
                        batched_losses, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                    )
                batched_loss = (1.0 - args.lambda_dssim) * batched_losses[
                    :, 0
                ] + args.lambda_dssim * (1.0 - batched_losses[:, 1])
                batched_loss_cpu = batched_loss.cpu().numpy()
                ema_loss_for_log = (
                    batched_loss_cpu.mean()
                    if ema_loss_for_log is None
                    else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
                )
                # Update Epoch Statistics
                train_dataset.update_losses(batched_loss_cpu)
                # Logging
                batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
                log_string = "iteration[{},{}) loss: {} image: {}\n".format(
                    iteration,
                    iteration + args.bsz,
                    batched_loss_cpu,
                    [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
                )
                log_file.write(log_string)
                timers.stop("sync_loss_and_log")

                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] ===== Iteration {iteration} COMPLETE =====")

                # Evaluation
                end2end_timers.stop()
                training_report(
                    iteration,
                    l1_loss,
                    args.test_iterations,
                    scene,
                    pipe_args,
                    background,
                    args.backend,
                )
                end2end_timers.start()

                # Densification
                if not tile_manager.enabled:
                    set_current_operation("densification")
                    log_gpu_memory("before_densification")
                    if args.backend == "gsplat":
                        gsplat_densification(
                            iteration, scene, gaussians, batched_screenspace_pkg
                        )
                    else:
                        densification(iteration, scene, gaussians, batched_screenspace_pkg)
                    log_gpu_memory("after_densification")

                if tile_loss_scheduler.enabled:
                    # Release heavy training buffers before running the auxiliary render.
                    del batched_screenspace_pkg
                    del batched_image
                    del batched_compute_locally
                    del batch_statistic_collector
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    with torch.enable_grad():
                        extra_loss, raw_val = tile_loss_scheduler.maybe_run(
                            iteration, gaussians
                        )
                        if extra_loss is not None:
                            timers.start("tile_loss_backward")
                            extra_loss.backward()
                            timers.stop("tile_loss_backward")
                            tile_loss_raw = raw_val
                    # Ensure Python references are cleared promptly.
                    if "extra_loss" in locals():
                        del extra_loss

                if tile_loss_raw is not None:
                    weighted_val = tile_loss_raw * tile_loss_scheduler.weight
                    utils.print_rank_0(
                        f"[ITER {iteration}] Tile image L2: {tile_loss_raw:.6f} (weighted {weighted_val:.6f})"
                    )
                    log_file.write(
                        f"[ITER {iteration}] tile_image_loss_raw={tile_loss_raw:.6f} weighted={weighted_val:.6f}\n"
                    )
                    if (
                        tile_loss_scheduler.output_dir
                        and utils.GLOBAL_RANK == 0
                        and tile_loss_scheduler.last_preview_path is not None
                    ):
                        log_file.write(
                            f"[ITER {iteration}] tile_image_render={tile_loss_scheduler.last_preview_path}\n"
                        )
                        if tile_loss_scheduler.last_diff_path is not None:
                            log_file.write(
                                f"[ITER {iteration}] tile_image_diff={tile_loss_scheduler.last_diff_path}\n"
                            )
                    if utils.LOCAL_RANK == 0:
                        progress_updates = {
                            "Loss": f"{ema_loss_for_log:.{7}f}",
                            "TileL2": f"{tile_loss_raw:.6f}",
                        }
                        progress_bar.set_postfix(progress_updates)

            # Save Gaussians
            if any(
                [
                    iteration <= save_iteration < iteration + args.bsz
                    for save_iteration in args.save_iterations
                ]
            ):
                end2end_timers.stop()
                end2end_timers.print_time(log_file, iteration + args.bsz)
                utils.print_rank_0("\n[ITER {}] Saving Gaussians".format(iteration))
                log_file.write("[ITER {}] Saving Gaussians\n".format(iteration))
                scene.save(iteration)

                if args.save_strategy_history:
                    with open(
                        args.log_folder
                        + "/strategy_history_ws="
                        + str(utils.WORLD_SIZE)
                        + "_rk="
                        + str(utils.GLOBAL_RANK)
                        + ".json",
                        "w",
                    ) as f:
                        json.dump(strategy_history.to_json(), f)
                end2end_timers.start()

            # Save Checkpoints
            if any(
                [
                    iteration <= checkpoint_iteration < iteration + args.bsz
                    for checkpoint_iteration in args.checkpoint_iterations
                ]
            ):
                end2end_timers.stop()
                utils.print_rank_0("\n[ITER {}] Saving Checkpoint".format(iteration))
                log_file.write("[ITER {}] Saving Checkpoint\n".format(iteration))
                save_folder = scene.model_path + "/checkpoints/" + str(iteration) + "/"
                if utils.DEFAULT_GROUP.rank() == 0:
                    os.makedirs(save_folder, exist_ok=True)
                    if utils.DEFAULT_GROUP.size() > 1:
                        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                elif utils.DEFAULT_GROUP.size() > 1:
                    torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                torch.save(
                    (gaussians.capture(), iteration + args.bsz),
                    save_folder
                    + "/chkpnt_ws="
                    + str(utils.WORLD_SIZE)
                    + "_rk="
                    + str(utils.GLOBAL_RANK)
                    + ".pth",
                )
                end2end_timers.start()

            # Optimizer step
            if iteration < opt_args.iterations:
                set_current_operation("optimizer_step")
                log_gpu_memory("before_optimizer_step")
                timers.start("optimizer_step")

                if (
                    args.lr_scale_mode != "accumu"
                ):  # we scale the learning rate rather than accumulate the gradients.
                    for param in gaussians.all_parameters():
                        if param.grad is not None:
                            param.grad /= args.bsz

                if not args.stop_update_param:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                timers.stop("optimizer_step")
                log_gpu_memory("after_optimizer_step")
                utils.check_initial_gpu_memory_usage("after optimizer step")

            # Finish a iteration and clean up
            torch.cuda.synchronize()
            for (
                viewpoint_cam
            ) in batched_cameras:  # Release memory of locally rendered original_image
                viewpoint_cam.original_image = None
            if args.nsys_profile:
                nvtx.range_pop()
            if utils.check_enable_python_timer():
                timers.printTimers(iteration, mode="sum")

            # Update progress bar only after successful iteration (backward pass completed)
            progress_bar.update(args.bsz)
            log_file.flush()

    except Exception as e:
        # Handle OOM in adaptive tile mode
        if adaptive_tile_mode and _is_oom_error(e):
            utils.print_rank_0(f"\n[adaptive-tile] Caught OOM error at iteration {current_iteration}")
            log_file.write(f"[adaptive-tile] OOM error: {str(e)}\n")

            # Clear CUDA cache
            gc.collect()
            torch.cuda.empty_cache()

            # Get camera info if available
            try:
                oom_num_cameras = current_num_cameras
                oom_total_pixels = current_total_pixels
            except NameError:
                oom_num_cameras = 0
                oom_total_pixels = 0

            # Get total cameras in tile from train_dataset
            try:
                oom_total_cameras_in_tile = len(train_dataset.cameras)
            except (NameError, AttributeError):
                oom_total_cameras_in_tile = 0

            # Handle the OOM by splitting tile
            handled = handle_adaptive_tile_oom(
                gaussians, args, scene, current_iteration, log_file,
                num_cameras=oom_num_cameras,
                total_pixels=oom_total_pixels,
                total_cameras_in_tile=oom_total_cameras_in_tile,
            )

            if handled:
                utils.print_rank_0(f"[adaptive-tile] Exiting with code {EXIT_CODE_OOM} for wrapper to handle")
                log_file.flush()
                tile_manager.finalize(gaussians)
                # Clear and close progress bar silently (don't print final state on OOM)
                progress_bar.clear()
                progress_bar.disable = True
                progress_bar.close()
                sys.exit(EXIT_CODE_OOM)

        # Re-raise if not OOM or not handled
        raise

    # Finish training
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)
    log_file.write(
        "Max Memory usage: {} GB.\n".format(
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        )
    )
    tile_manager.finalize(gaussians)
    progress_bar.close()


def training_report(
    iteration, l1_loss, testing_iterations, scene: Scene, pipe_args, background, backend
):
    args = utils.get_args()
    log_file = utils.get_log_file()
    # Report test and samples of training set
    while len(testing_iterations) > 0 and iteration > testing_iterations[0]:
        testing_iterations.pop(0)
    if len(testing_iterations) > 0 and utils.check_update_at_this_iter(
        iteration, utils.get_args().bsz, testing_iterations[0], 0
    ):
        testing_iterations.pop(0)
        utils.print_rank_0("\n[ITER {}] Start Testing".format(iteration))

        test_cameras = scene.getTestCameras() or []
        train_cameras = scene.getTrainCameras() or []

        validation_configs = [
            {
                "name": "train",
                "cameras": train_cameras,
                "num_cameras": max(len(train_cameras) // args.llffhold, args.bsz)
                if len(train_cameras) > 0
                else 0,
            }
        ]

        if len(test_cameras) > 0:
            validation_configs.insert(
                0,
                {
                    "name": "test",
                    "cameras": test_cameras,
                    "num_cameras": len(test_cameras),
                },
            )

        # init workload division strategy
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = torch.scalar_tensor(0.0, device="cuda")
                psnr_test = torch.scalar_tensor(0.0, device="cuda")

                # TODO: if not divisible by world size
                num_cameras = config["num_cameras"] // args.bsz * args.bsz
                eval_dataset = SceneDataset(config["cameras"])
                strategy_history = DivisionStrategyHistoryFinal(
                    eval_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
                )
                for idx in range(1, num_cameras + 1, args.bsz):
                    num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
                    if args.local_sampling:
                        # TODO: if not divisible by world size
                        batched_cameras_idx = eval_dataset.get_batched_cameras_idx(
                            args.bsz // utils.WORLD_SIZE
                        )
                        batched_all_cameras_idx = torch.zeros(
                            (utils.WORLD_SIZE, len(batched_cameras_idx)),
                            device="cuda",
                            dtype=int,
                        )
                        batched_cameras_idx = torch.tensor(
                            batched_cameras_idx, device="cuda", dtype=int
                        )
                        torch.distributed.all_gather_into_tensor(
                            batched_all_cameras_idx,
                            batched_cameras_idx,
                            group=utils.DEFAULT_GROUP,
                        )
                        batched_all_cameras_idx = (
                            batched_all_cameras_idx.cpu().numpy().squeeze()
                        )
                        batched_cameras = eval_dataset.get_batched_cameras_from_idx(
                            batched_all_cameras_idx
                        )
                    else:
                        batched_cameras = eval_dataset.get_batched_cameras(
                            num_camera_to_load
                        )
                    batched_strategies, gpuid2tasks = start_strategy_final(
                        batched_cameras, strategy_history
                    )
                    load_camera_from_cpu_to_all_gpu_for_eval(
                        batched_cameras, batched_strategies, gpuid2tasks
                    )
                    if backend == "gsplat":
                        batched_screenspace_pkg = (
                            gsplat_distributed_preprocess3dgs_and_all2all_final(
                                batched_cameras,
                                scene.gaussians,
                                pipe_args,
                                background,
                                batched_strategies=batched_strategies,
                                mode="test",
                            )
                        )
                        batched_image, _ = gsplat_render_final(
                            batched_screenspace_pkg, batched_strategies
                        )
                    else:
                        batched_screenspace_pkg = (
                            distributed_preprocess3dgs_and_all2all_final(
                                batched_cameras,
                                scene.gaussians,
                                pipe_args,
                                background,
                                batched_strategies=batched_strategies,
                                mode="test",
                            )
                        )
                        batched_image, _ = render_final(
                            batched_screenspace_pkg, batched_strategies
                        )
                    for camera_id, (image, gt_camera) in enumerate(
                        zip(batched_image, batched_cameras)
                    ):
                        if (
                            image is None or len(image.shape) == 0
                        ):  # The image is not rendered locally.
                            image = torch.zeros(
                                gt_camera.original_image.shape,
                                device="cuda",
                                dtype=torch.float32,
                            )

                        if utils.DEFAULT_GROUP.size() > 1:
                            torch.distributed.all_reduce(
                                image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP
                            )

                        image = torch.clamp(image, 0.0, 1.0)
                        gt_image = torch.clamp(
                            gt_camera.original_image / 255.0, 0.0, 1.0
                        )

                        if idx + camera_id < num_cameras + 1:
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
                        gt_camera.original_image = None
                psnr_test /= num_cameras
                l1_test /= num_cameras
                utils.print_rank_0(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                log_file.write(
                    "[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )

        torch.cuda.empty_cache()
