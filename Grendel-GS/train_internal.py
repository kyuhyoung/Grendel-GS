import os
import sys
import gc
import signal
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
# Global State for SIGTERM Handler (Distributed OOM Synchronization)
# ============================================================================
# When one rank catches OOM and exits, torchrun sends SIGTERM to other ranks.
# These global references allow the SIGTERM handler to save gaussians before exit.
_SIGTERM_GAUSSIANS = None  # Reference to GaussianModel
_SIGTERM_ARGS = None       # Reference to training args
_SIGTERM_LOG_FILE = None   # Reference to log file
_SIGTERM_ITERATION = 0     # Current iteration


def register_sigterm_state(gaussians, args, log_file, iteration: int):
    """Register state for SIGTERM handler to use when saving gaussians."""
    global _SIGTERM_GAUSSIANS, _SIGTERM_ARGS, _SIGTERM_LOG_FILE, _SIGTERM_ITERATION
    _SIGTERM_GAUSSIANS = gaussians
    _SIGTERM_ARGS = args
    _SIGTERM_LOG_FILE = log_file
    _SIGTERM_ITERATION = iteration


def _sigterm_handler(signum, frame):
    """Handle SIGTERM signal (sent by torchrun when one rank dies).

    When another rank catches OOM and exits, torchrun terminates other ranks.
    This handler checks if OOM signal file exists and saves gaussians before exit.
    """
    global _SIGTERM_GAUSSIANS, _SIGTERM_ARGS, _SIGTERM_LOG_FILE, _SIGTERM_ITERATION

    rank = utils.GLOBAL_RANK if hasattr(utils, 'GLOBAL_RANK') else 0

    # IMMEDIATE log write - this is the first thing we do
    # If this doesn't appear in log, SIGTERM handler was never called
    if _SIGTERM_LOG_FILE:
        try:
            _SIGTERM_LOG_FILE.write(f"\n{'='*60}\n")
            _SIGTERM_LOG_FILE.write(f"[SIGTERM] HANDLER CALLED! Rank {rank} at iter {_SIGTERM_ITERATION}\n")
            _SIGTERM_LOG_FILE.write(f"{'='*60}\n")
            _SIGTERM_LOG_FILE.flush()
        except:
            pass

    print(f"\n[SIGTERM] Rank {rank} received SIGTERM signal!", flush=True)

    if _SIGTERM_GAUSSIANS is None or _SIGTERM_ARGS is None:
        print(f"[SIGTERM] Rank {rank}: No gaussian state registered, exiting immediately", flush=True)
        sys.exit(EXIT_CODE_OOM)

    # Check for OOM signal file
    signal_data = check_oom_signal(_SIGTERM_ARGS)
    if signal_data is None:
        # No signal file - try to infer category from iteration
        # If we're past densification, it's likely Category 3 and worth saving
        densify_from = getattr(_SIGTERM_ARGS, 'densify_from_iter', 500)
        densify_interval = getattr(_SIGTERM_ARGS, 'densification_interval', 100)
        first_densify = densify_from + densify_interval

        if _SIGTERM_ITERATION > first_densify:
            print(f"[SIGTERM] Rank {rank}: No signal file, but iteration {_SIGTERM_ITERATION} > densify {first_densify}", flush=True)
            print(f"[SIGTERM] Rank {rank}: Inferring Category 3 OOM, will save gaussians", flush=True)
            # Create synthetic signal data for saving
            signal_data = {
                "signaling_rank": -1,  # Unknown signaling rank
                "category": 3,
                "iteration": _SIGTERM_ITERATION,
                "tile_id": getattr(_SIGTERM_ARGS, "tile_id", "unknown"),
            }
        else:
            print(f"[SIGTERM] Rank {rank}: No OOM signal file found, iteration {_SIGTERM_ITERATION} <= densify {first_densify}", flush=True)
            print(f"[SIGTERM] Rank {rank}: Likely early OOM, not worth saving", flush=True)
            sys.exit(EXIT_CODE_OOM)

    signaling_rank = signal_data.get("signaling_rank", -1)
    category = signal_data.get("category", 1)

    print(f"[SIGTERM] Rank {rank}: Found OOM signal from rank {signaling_rank}, category={category}", flush=True)

    # Only save for Category 3 OOM
    if category < 3:
        print(f"[SIGTERM] Rank {rank}: Category {category} - no need to save gaussians", flush=True)
        if _SIGTERM_LOG_FILE:
            try:
                _SIGTERM_LOG_FILE.write(f"[SIGTERM] Rank {rank}: Category {category} < 3, skipping save\n")
                _SIGTERM_LOG_FILE.flush()
            except:
                pass
        sys.exit(EXIT_CODE_OOM)

    # Note: Don't skip even if signaling_rank == rank
    # The signaling rank may not have completed saving before receiving SIGTERM
    if signaling_rank == rank:
        print(f"[SIGTERM] Rank {rank}: This rank triggered OOM, but will still save (may not have completed)", flush=True)
        if _SIGTERM_LOG_FILE:
            try:
                _SIGTERM_LOG_FILE.write(f"[SIGTERM] Rank {rank}: Signaling rank, saving anyway\n")
                _SIGTERM_LOG_FILE.flush()
            except:
                pass

    # Save gaussians
    print(f"[SIGTERM] Rank {rank}: Saving gaussians before exit...", flush=True)
    if _SIGTERM_LOG_FILE:
        try:
            _SIGTERM_LOG_FILE.write(f"[SIGTERM] Rank {rank}: Saving gaussians (cat={category})...\n")
            _SIGTERM_LOG_FILE.flush()
        except:
            pass

    try:
        handled = handle_oom_signal_from_other_rank(
            signal_data, _SIGTERM_GAUSSIANS, _SIGTERM_ARGS, _SIGTERM_ITERATION,
            _SIGTERM_LOG_FILE if _SIGTERM_LOG_FILE else sys.stdout
        )
        if handled:
            print(f"[SIGTERM] Rank {rank}: Successfully saved gaussians!", flush=True)
            if _SIGTERM_LOG_FILE:
                try:
                    _SIGTERM_LOG_FILE.write(f"[SIGTERM] Rank {rank}: Save completed successfully\n")
                    _SIGTERM_LOG_FILE.flush()
                except:
                    pass
    except Exception as e:
        print(f"[SIGTERM] Rank {rank}: Error saving gaussians: {e}", flush=True)
        if _SIGTERM_LOG_FILE:
            try:
                _SIGTERM_LOG_FILE.write(f"[SIGTERM] Rank {rank}: Error: {e}\n")
                _SIGTERM_LOG_FILE.flush()
            except:
                pass

    sys.exit(EXIT_CODE_OOM)


def install_sigterm_handler():
    """Install SIGTERM handler for graceful shutdown with gaussian saving."""
    signal.signal(signal.SIGTERM, _sigterm_handler)
    msg = f"[oom-signal] Rank {utils.GLOBAL_RANK}: Installed SIGTERM handler for distributed OOM sync"
    print(msg, flush=True)
    # Also write to log file if available
    log_file = utils.get_log_file() if hasattr(utils, 'get_log_file') else None
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


# ============================================================================
# Distributed OOM Signal Mechanism
# ============================================================================
# When one rank catches OOM, it writes a signal file. Other ranks check for
# this signal at each iteration start and enter the save-and-exit path.

OOM_SIGNAL_FILENAME = "oom_signal.json"


def get_oom_signal_path(args) -> Path:
    """Get the path to the OOM signal file."""
    tile_output_dir = getattr(args, "tile_output_dir", "")
    if tile_output_dir:
        return Path(tile_output_dir) / OOM_SIGNAL_FILENAME
    model_path = getattr(args, "model_path", "")
    if model_path:
        return Path(model_path) / OOM_SIGNAL_FILENAME
    return Path(".") / OOM_SIGNAL_FILENAME


def write_oom_signal(args, iteration: int, oom_cause: str, category: int):
    """Write OOM signal file so other ranks can detect and save.

    Args:
        args: Training arguments
        iteration: Current iteration when OOM occurred
        oom_cause: Description of OOM cause
        category: OOM category (1, 2, or 3)
    """
    signal_path = get_oom_signal_path(args)
    signal_path.parent.mkdir(parents=True, exist_ok=True)

    signal_data = {
        "oom_occurred": True,
        "iteration": iteration,
        "oom_cause": oom_cause,
        "category": category,
        "signaling_rank": utils.GLOBAL_RANK,
        "world_size": utils.WORLD_SIZE,
        "tile_id": getattr(args, "tile_id", "unknown"),
    }

    # Write atomically by writing to temp file first
    temp_path = signal_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(signal_data, f, indent=2)
    temp_path.rename(signal_path)

    # Use print (not print_rank_0) because OOM can happen on any rank
    print(f"[oom-signal] Rank {utils.GLOBAL_RANK} wrote OOM signal to {signal_path}", flush=True)
    print(f"  Signal contents: iteration={iteration}, category={category}, cause={oom_cause}", flush=True)


def check_oom_signal(args) -> Optional[dict]:
    """Check if OOM signal file exists and return its contents.

    Returns:
        Signal data dict if signal exists, None otherwise.
    """
    signal_path = get_oom_signal_path(args)
    if signal_path.exists():
        try:
            with open(signal_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def clear_oom_signal(args):
    """Remove OOM signal file (called by wrapper after handling)."""
    signal_path = get_oom_signal_path(args)
    try:
        signal_path.unlink(missing_ok=True)
    except Exception:
        pass


def get_oom_done_path(args, rank: int) -> Path:
    """Get the path to the OOM done file for a specific rank."""
    signal_path = get_oom_signal_path(args)
    return signal_path.parent / f"oom_done_rank{rank}.json"


def write_oom_done(args, rank: int, gaussians_saved: int):
    """Write OOM done file to indicate this rank has finished saving."""
    done_path = get_oom_done_path(args, rank)
    done_data = {
        "rank": rank,
        "gaussians_saved": gaussians_saved,
        "done": True,
    }
    temp_path = done_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(done_data, f)
    temp_path.rename(done_path)
    print(f"[oom-signal] Rank {rank} wrote done file: {done_path}", flush=True)


def check_all_ranks_done(args, world_size: int, timeout: float = 120.0) -> bool:
    """Wait for all ranks to write their done files.

    Args:
        args: Training arguments
        world_size: Total number of ranks
        timeout: Maximum seconds to wait

    Returns:
        True if all ranks done, False if timeout
    """
    import time
    start_time = time.time()
    check_interval = 0.5  # Check every 0.5 seconds

    while time.time() - start_time < timeout:
        all_done = True
        for rank in range(world_size):
            done_path = get_oom_done_path(args, rank)
            if not done_path.exists():
                all_done = False
                break

        if all_done:
            print(f"[oom-signal] All {world_size} ranks have completed saving!", flush=True)
            return True

        time.sleep(check_interval)
        elapsed = time.time() - start_time
        if int(elapsed) % 2 == 0 and int(elapsed) > 0:  # Print every 2 seconds
            missing = [r for r in range(world_size) if not get_oom_done_path(args, r).exists()]
            print(f"[oom-signal] Waiting for ranks {missing} to complete... ({elapsed:.0f}s)", flush=True)

    missing = [r for r in range(world_size) if not get_oom_done_path(args, r).exists()]
    print(f"[oom-signal] TIMEOUT after {timeout}s! Missing ranks: {missing}", flush=True)
    return False


def clear_oom_done_files(args, world_size: int):
    """Remove all OOM done files."""
    for rank in range(world_size):
        done_path = get_oom_done_path(args, rank)
        try:
            done_path.unlink(missing_ok=True)
        except Exception:
            pass


def handle_oom_signal_from_other_rank(
    signal_data: dict,
    gaussians,
    args,
    iteration: int,
    log_file,
) -> bool:
    """Handle OOM signal that was written by another rank.

    When another rank catches OOM and writes a signal, this rank should:
    1. Save its local gaussians
    2. Exit with OOM code

    This ensures ALL ranks save their data before torchrun kills them.

    Returns:
        True if handled (should exit), False otherwise.
    """
    signaling_rank = signal_data.get("signaling_rank", -1)
    category = signal_data.get("category", 1)
    signal_iteration = signal_data.get("iteration", iteration)
    tile_id = signal_data.get("tile_id", "unknown")

    print(f"[oom-signal] Rank {utils.GLOBAL_RANK} detected OOM signal from rank {signaling_rank}", flush=True)
    print(f"  Signal info: iteration={signal_iteration}, category={category}, tile={tile_id}", flush=True)
    log_file.write(f"[oom-signal] Rank {utils.GLOBAL_RANK} handling signal from rank {signaling_rank}\n")
    log_file.write(f"[oom-signal] Signal: iter={signal_iteration}, cat={category}, tile={tile_id}\n")
    log_file.flush()

    # Only save for Category 3 OOM (trained gaussians worth saving)
    if category < 3:
        print(f"[oom-signal] Category {category} OOM - no need to save gaussians", flush=True)
        log_file.write(f"[oom-signal] Category {category} < 3, skipping save\n")
        log_file.flush()
        return True

    log_file.write(f"[oom-signal] Category 3 - will save gaussians\n")
    log_file.flush()

    # Get tile info for saving
    tile_bbox_str = getattr(args, "tile_bbox", "")
    if not tile_bbox_str:
        print(f"[oom-signal] No tile_bbox in args, cannot save", flush=True)
        return True

    tile_bbox = TileBBox.from_string(tile_bbox_str)
    tile_output_dir = Path(getattr(args, "tile_output_dir", "output/ply"))
    tile_output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate child tile bboxes (same logic as handle_adaptive_tile_oom)
    parent_level = getattr(args, "tile_level", 0)
    child_level = parent_level + 1
    tile_a, tile_b = tile_bbox.split()
    # Use same tile ID generation as handle_adaptive_tile_oom()
    tile_num = int(tile_id.split("_")[-1]) if "_" in tile_id else 0
    tile_a_id = f"tile_{tile_num * 2 + 1:04d}"
    tile_b_id = f"tile_{tile_num * 2 + 2:04d}"

    # Use the signal iteration for file naming (matches signaling rank's files)
    use_iteration = signal_iteration

    print(f"[oom-signal] Rank {utils.GLOBAL_RANK} saving local gaussians for tile split...", flush=True)
    print(f"  Parent tile: {tile_id}, level: {parent_level} -> child level: {child_level}", flush=True)
    print(f"  Child tile A: {tile_a_id}, bbox: {tile_a.to_string()}", flush=True)
    print(f"  Child tile B: {tile_b_id}, bbox: {tile_b.to_string()}", flush=True)
    print(f"  Output dir: {tile_output_dir}", flush=True)

    # Save tile A
    temp_path_a = tile_output_dir / f"{tile_a_id}_L{child_level}_temp.ply"
    count_a = gaussians.save_ply(str(temp_path_a), filter_bbox=tile_a, local_only=True)

    # Rename to final path
    actual_path_a = tile_output_dir / f"{tile_a_id}_L{child_level}_temp_rank{utils.GLOBAL_RANK}.ply"
    final_path_a = tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{use_iteration}_rank{utils.GLOBAL_RANK}.ply"
    if actual_path_a.exists() and count_a > 0:
        actual_path_a.rename(final_path_a)
        print(f"  [SAVED] Tile A rank{utils.GLOBAL_RANK}: {count_a:,} gaussians", flush=True)
        print(f"          -> {final_path_a}", flush=True)
    elif actual_path_a.exists():
        actual_path_a.unlink(missing_ok=True)
        print(f"  [SKIP] Tile A rank{utils.GLOBAL_RANK}: 0 gaussians (file deleted)", flush=True)

    # Save tile B
    temp_path_b = tile_output_dir / f"{tile_b_id}_L{child_level}_temp.ply"
    count_b = gaussians.save_ply(str(temp_path_b), filter_bbox=tile_b, local_only=True)

    # Rename to final path
    actual_path_b = tile_output_dir / f"{tile_b_id}_L{child_level}_temp_rank{utils.GLOBAL_RANK}.ply"
    final_path_b = tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{use_iteration}_rank{utils.GLOBAL_RANK}.ply"
    if actual_path_b.exists() and count_b > 0:
        actual_path_b.rename(final_path_b)
        print(f"  [SAVED] Tile B rank{utils.GLOBAL_RANK}: {count_b:,} gaussians", flush=True)
        print(f"          -> {final_path_b}", flush=True)
    elif actual_path_b.exists():
        actual_path_b.unlink(missing_ok=True)
        print(f"  [SKIP] Tile B rank{utils.GLOBAL_RANK}: 0 gaussians (file deleted)", flush=True)

    total_saved = count_a + count_b
    print(f"[oom-signal] Rank {utils.GLOBAL_RANK} finished. Total saved: {total_saved:,} gaussians", flush=True)
    log_file.write(f"[oom-signal] Rank {utils.GLOBAL_RANK} saved {total_saved:,} gaussians\n")
    log_file.flush()

    # Write done file to signal completion to the signaling rank
    write_oom_done(args, utils.GLOBAL_RANK, total_saved)

    return True


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


# ============================================================================
# Debug Image Saving for Off-Center Projection Verification
# ============================================================================

def save_debug_images(
    batched_image,
    batched_cameras,
    batched_strategies,
    iteration: int,
    output_dir: Path,
    tile_id: str = "unknown",
):
    """
    Save GT, rendered, and diff images for debugging off-center projection.

    Each GPU saves its own portion of the image to show how the workload
    is divided. This helps verify that off-center projection is working.

    Args:
        batched_image: List of rendered images (per-camera, may be None if not local)
        batched_cameras: List of camera objects (contain GT images)
        batched_strategies: List of strategies (contain coverage info)
        iteration: Current training iteration
        output_dir: Base directory to save debug images
        tile_id: Tile identifier for filename
    """
    import torchvision

    debug_dir = output_dir / "debug_images"
    debug_dir.mkdir(parents=True, exist_ok=True)

    rank = utils.GLOBAL_RANK

    for idx, (image, camera, strategy) in enumerate(
        zip(batched_image, batched_cameras, batched_strategies)
    ):
        if image is None or len(image.shape) == 0:
            continue  # Not rendered locally

        # Get coverage info for this rank
        if rank not in strategy.gpu_ids:
            continue
        local_rank = strategy.gpu_ids.index(rank)
        tile_ids_l, tile_ids_r = (
            strategy.division_pos[local_rank],
            strategy.division_pos[local_rank + 1],
        )

        # Calculate Y coverage (row-based division)
        from gaussian_renderer.loss_distribution import get_coverage_y_min_max
        coverage_min_y, coverage_max_y = get_coverage_y_min_max(tile_ids_l, tile_ids_r)

        # Debug: print shapes and ranges
        gt_h = camera.original_image.shape[1]
        rendered_h = image.shape[1]

        # Check pixel values at different regions to understand tensor layout
        img_mean_all = image.mean().item()
        img_mean_top = image[:, :1000, :].mean().item() if image.shape[1] > 1000 else 0
        img_mean_coverage = image[:, coverage_min_y:min(coverage_max_y, image.shape[1]), :].mean().item()
        print(f"[DEBUG-IMG] GPU{rank} cam={camera.image_name}: image.shape={image.shape}, gt_h={gt_h}, coverage_y=({coverage_min_y}, {coverage_max_y})", flush=True)
        # Also check if data might be at start of tensor (local portion starting at row 0)
        local_h = coverage_max_y - coverage_min_y
        img_mean_local_start = image[:, :local_h, :].mean().item() if image.shape[1] >= local_h else 0
        print(f"[DEBUG-IMG] GPU{rank}: img_mean_all={img_mean_all:.4f}, img_mean_top1000={img_mean_top:.4f}, img_mean_coverage={img_mean_coverage:.4f}, img_mean_local_start={img_mean_local_start:.4f}", flush=True)

        # In Grendel-GS distributed rendering:
        # - Rendered image is gathered to ALL GPUs (full image)
        # - GT image: each GPU has only its coverage portion
        # So we need to slice rendered to this GPU's coverage range.

        # Get GT (this GPU's portion) - already local
        gt_image = camera.original_image / 255.0
        local_gt = gt_image.clamp(0.0, 1.0)

        # Slice rendered image to this GPU's coverage range
        local_rendered = image[:, coverage_min_y:coverage_max_y, :].detach()

        if local_gt.numel() == 0 or local_rendered.numel() == 0:
            utils.print_rank_0(f"[DEBUG-IMG] GPU{rank}: SKIP - empty tensor (local_gt={local_gt.shape}, local_rendered={local_rendered.shape})")
            continue

        # Shape check - GT and rendered should match for this GPU's portion
        if local_rendered.shape != local_gt.shape:
            utils.print_rank_0(f"[DEBUG-IMG] GPU{rank}: shape mismatch rendered={local_rendered.shape} vs gt={local_gt.shape}, adjusting...")
            min_h = min(local_rendered.shape[1], local_gt.shape[1])
            min_w = min(local_rendered.shape[2], local_gt.shape[2])
            local_rendered = local_rendered[:, :min_h, :min_w]
            local_gt = local_gt[:, :min_h, :min_w]

        print(f"[DEBUG-IMG] GPU{rank}: local_rendered.shape={local_rendered.shape}, local_gt.shape={local_gt.shape}", flush=True)

        cam_name = camera.image_name.replace("/", "_").replace("\\", "_")
        # Format: tile_xxxx_iter_yyyy_cam_zzzz_gpuW_compare.png
        prefix = f"{tile_id}_iter{iteration:04d}_cam{cam_name}_gpu{rank}"

        # Combine GT (left/top) and Rendered (right/bottom) into single image
        local_rendered_clamped = local_rendered.clamp(0.0, 1.0).cpu()
        local_gt_cpu = local_gt.cpu()

        h, w = local_rendered_clamped.shape[1], local_rendered_clamped.shape[2]
        if w > h:
            # Wide image -> stack vertically (GT on top, Rendered on bottom)
            combined = torch.cat([local_gt_cpu, local_rendered_clamped], dim=1)
            layout = "top_GT_bottom_Rendered"
        else:
            # Tall image -> stack horizontally (GT on left, Rendered on right)
            combined = torch.cat([local_gt_cpu, local_rendered_clamped], dim=2)
            layout = "left_GT_right_Rendered"

        torchvision.utils.save_image(combined, debug_dir / f"{prefix}_compare.png")

        # Save info file
        info_path = debug_dir / f"{prefix}_info.txt"
        with open(info_path, "w") as f:
            f.write(f"Camera: {camera.image_name}\n")
            f.write(f"GPU Rank: {rank}\n")
            f.write(f"GT image size: {camera.image_width}x{camera.image_height}\n")
            f.write(f"Rendered image shape: {list(image.shape)}\n")
            f.write(f"Saved comparison size: {w}x{h}\n")
            f.write(f"Layout: {layout}\n")
            f.write(f"Strategy GPU IDs: {strategy.gpu_ids}\n")
            f.write(f"Division positions: {strategy.division_pos}\n")
            # Log projection matrix info if available
            # NOTE: projection_matrix is stored TRANSPOSED, so P[2,0] and P[2,1] are the offsets
            if hasattr(camera, 'projection_matrix'):
                P = camera.projection_matrix
                f.write(f"\nProjection Matrix (transposed):\n{P}\n")
                f.write(f"P[2,0] (X offset): {P[2,0]:.6f}\n")
                f.write(f"P[2,1] (Y offset): {P[2,1]:.6f}\n")

    if rank == 0:
        utils.print_rank_0(f"[DEBUG] Saved debug images to {debug_dir}")


def should_save_debug_images(iteration: int, args) -> bool:
    """Check if debug images should be saved at this iteration."""
    # Check environment variable
    debug_iters_str = os.environ.get("DEBUG_SAVE_ITERS", "")
    if debug_iters_str:
        try:
            debug_iters = [int(x.strip()) for x in debug_iters_str.split(",")]
            return iteration in debug_iters
        except ValueError:
            pass

    # Check args
    debug_interval = getattr(args, "debug_image_interval", 0)
    if debug_interval > 0 and iteration % debug_interval == 0:
        return True

    # Default: save at iterations 1, 100, 500, 1000
    default_iters = {1, 100, 500, 1000}
    return iteration in default_iters


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

    # CRITICAL: Write OOM signal IMMEDIATELY to notify other ranks
    # This must happen BEFORE any complex operations that might fail due to memory pressure
    # Calculate OOM category early for signal
    densify_from_early = getattr(args, 'densify_from_iter', 500)
    densify_interval_early = getattr(args, 'densification_interval', 100)
    first_densify_early = densify_from_early + densify_interval_early
    is_category3_early = iteration > first_densify_early
    C_early = total_cameras_in_tile if total_cameras_in_tile > 0 else num_cameras
    oom_category_early = 3 if is_category3_early else (1 if (C_early > 0 and iteration <= C_early) else 2)
    early_cause = "densification" if is_category3_early else "early_iteration"

    # Write signal immediately - this is the most critical operation
    print(f"\n[oom-signal] Rank {utils.GLOBAL_RANK} IMMEDIATE signal write (iter={iteration}, cat={oom_category_early})", flush=True)
    write_oom_signal(args, iteration, early_cause, oom_category_early)

    # Give other ranks time to check the signal file before we continue
    # This is critical because other ranks may be in loss computation and will
    # check for OOM signal before starting backward (which would cause them to block forever)
    # Note: 20 seconds is needed because:
    #   - Other ranks may be in loss computation (can take 5-10s for large images)
    #   - They need to finish loss and reach pre-backward check point
    #   - If they enter backward (all_reduce) before seeing signal, they'll block forever
    print(f"[oom-signal] Rank {utils.GLOBAL_RANK} waiting 20s for other ranks to detect signal...", flush=True)
    import time
    time.sleep(20.0)

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
    # Note: Use print() instead of print_rank_0() because OOM can happen on any rank
    densify_from = getattr(args, 'densify_from_iter', 500)
    densify_interval = getattr(args, 'densification_interval', 100)
    first_densify_iter = densify_from + densify_interval
    C = total_cameras_in_tile if total_cameras_in_tile > 0 else num_cameras

    print(f"\n[OOM Cause Diagnosis]", flush=True)
    print(f"  Iteration: {iteration}", flush=True)
    print(f"  Total cameras in tile (C): {C}", flush=True)
    print(f"  First densification at: ~{first_densify_iter}", flush=True)

    if C > 0 and iteration <= C:
        # First cycle through cameras
        likely_cause = "SSIM on large image (first pass through cameras)"
        diagnosis_msg = f"  Iteration({iteration}) <= C({C}): Still in first camera cycle"
        print(diagnosis_msg, flush=True)
        print(f"  --> Likely cause: {likely_cause}", flush=True)
    elif iteration <= first_densify_iter:
        # After first cycle but before densification
        likely_cause = "Memory fragmentation or leak (same images succeeded before, no densification yet)"
        diagnosis_msg = f"  C({C}) < Iteration({iteration}) <= first_densify({first_densify_iter}): Before densification"
        print(diagnosis_msg, flush=True)
        print(f"  --> Likely cause: {likely_cause}", flush=True)
    else:
        # After densification started
        likely_cause = "Increased gaussians from densification"
        diagnosis_msg = f"  Iteration({iteration}) > first_densify({first_densify_iter}): Densification has occurred"
        print(diagnosis_msg, flush=True)
        print(f"  --> Likely cause: {likely_cause}", flush=True)

    # Write diagnosis to log file
    log_file.write(f"\n[OOM Cause Diagnosis]\n")
    log_file.write(f"  Iteration: {iteration}, Total cameras (C): {C}, First densify: ~{first_densify_iter}\n")
    log_file.write(f"  {diagnosis_msg}\n")
    log_file.write(f"  --> Likely cause: {likely_cause}\n")
    log_file.flush()

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

    # Note: Gaussian counts will be computed accurately inside save_ply after gathering from all GPUs
    # The filter_bbox is passed to save_ply which computes the mask on the gathered data

    # Generate new tile IDs
    tile_num = int(tile_id.split("_")[-1]) if "_" in tile_id else 0
    tile_a_id = f"tile_{tile_num * 2 + 1:04d}"
    tile_b_id = f"tile_{tile_num * 2 + 2:04d}"

    # Determine if this is Category 3 OOM (increased gaussians)
    is_category3 = iteration > first_densify_iter
    oom_category = 3 if is_category3 else (1 if (C > 0 and iteration <= C) else 2)

    print(f"\n[OOM Category Detection]", flush=True)
    print(f"  iteration={iteration}, first_densify_iter={first_densify_iter}", flush=True)
    print(f"  is_category3={is_category3}, oom_category={oom_category}", flush=True)

    # Update OOM signal with more detailed cause (signal was already written at function start)
    print(f"\n[oom-signal] Rank {utils.GLOBAL_RANK} updating signal with detailed cause: {likely_cause}", flush=True)
    write_oom_signal(args, iteration, likely_cause, oom_category)

    # Save state for wrapper script to handle
    tile_output_dir.mkdir(parents=True, exist_ok=True)

    # Get parent level and calculate child level
    parent_level = getattr(args, "tile_level", 0)
    child_level = parent_level + 1

    # Initialize counts (will be set by save_ply for Category 3 OOM)
    count_a = 0
    count_b = 0
    ply_path_a = None
    ply_path_b = None

    # For Category 3 OOM, save trained gaussians for both child tiles
    # This allows resuming training with pre-trained gaussians instead of from scratch
    # NOTE: We use local_only=True because other ranks may not be synchronized during OOM recovery
    # Each rank saves its local portion independently, wrapper will merge them
    if is_category3:
        print(f"[adaptive-tile] >>> Category 3 OOM: Saving trained gaussians for resume <<<", flush=True)
        print(f"  Using LOCAL_ONLY mode (each rank saves independently)", flush=True)

        # Each rank saves its local gaussians for tile A (with filter_bbox applied locally)
        # Files will be named: {tile_id}_L{level}_temp_rank{rank}.ply
        print(f"  Saving tile A ({tile_a_id}) - rank {utils.GLOBAL_RANK}...", flush=True)
        temp_path_a = tile_output_dir / f"{tile_a_id}_L{child_level}_temp.ply"
        count_a = gaussians.save_ply(str(temp_path_a), filter_bbox=tile_a, local_only=True)
        print(f"  Rank {utils.GLOBAL_RANK} saved {count_a:,} gaussians for tile A", flush=True)

        # Each rank saves its local gaussians for tile B
        print(f"  Saving tile B ({tile_b_id}) - rank {utils.GLOBAL_RANK}...", flush=True)
        temp_path_b = tile_output_dir / f"{tile_b_id}_L{child_level}_temp.ply"
        count_b = gaussians.save_ply(str(temp_path_b), filter_bbox=tile_b, local_only=True)
        print(f"  Rank {utils.GLOBAL_RANK} saved {count_b:,} gaussians for tile B", flush=True)

        # Record the PLY directory prefix (wrapper will find all rank files and merge them)
        # Files are: {tile_id}_L{level}_temp_rank0.ply, {tile_id}_L{level}_temp_rank1.ply, etc.
        ply_prefix_a = str(tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{iteration}")
        ply_prefix_b = str(tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{iteration}")

        # Rename temp files on this rank
        actual_path_a = tile_output_dir / f"{tile_a_id}_L{child_level}_temp_rank{utils.GLOBAL_RANK}.ply"
        final_path_a = tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{iteration}_rank{utils.GLOBAL_RANK}.ply"
        if actual_path_a.exists() and count_a > 0:
            actual_path_a.rename(final_path_a)
            print(f"  [SAVED] Tile A rank{utils.GLOBAL_RANK}: {count_a:,} gaussians -> {final_path_a.name}", flush=True)
        elif actual_path_a.exists():
            actual_path_a.unlink(missing_ok=True)

        actual_path_b = tile_output_dir / f"{tile_b_id}_L{child_level}_temp_rank{utils.GLOBAL_RANK}.ply"
        final_path_b = tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{iteration}_rank{utils.GLOBAL_RANK}.ply"
        if actual_path_b.exists() and count_b > 0:
            actual_path_b.rename(final_path_b)
            print(f"  [SAVED] Tile B rank{utils.GLOBAL_RANK}: {count_b:,} gaussians -> {final_path_b.name}", flush=True)
        elif actual_path_b.exists():
            actual_path_b.unlink(missing_ok=True)

        print(f"  Rank {utils.GLOBAL_RANK} finished saving. Total local: {count_a + count_b:,} gaussians", flush=True)

        # Set ply_path to the prefix (wrapper will look for *_rank*.ply files)
        ply_path_a = ply_prefix_a if count_a > 0 else None
        ply_path_b = ply_prefix_b if count_b > 0 else None
    else:
        print(f"[adaptive-tile] Category {oom_category} OOM: Child tiles will start from scratch", flush=True)
        print(f"  (No pre-trained gaussians saved)", flush=True)

    # All ranks save state file (race condition but same content, ensures at least one succeeds)
    state_file = getattr(args, "tile_state_file", "") or (tile_output_dir / "adaptive_tile_state.json")
    state_file = Path(state_file)

    # For Category 3, ply_path is a prefix; actual files are {prefix}_rank{0,1,2,3}.ply
    # The wrapper needs to merge these files before loading
    world_size = utils.WORLD_SIZE if is_category3 else 1

    oom_state = {
        "oom_occurred": True,
        "oom_cause": likely_cause,
        "oom_category": 3 if is_category3 else (1 if (C > 0 and iteration <= C) else 2),
        "original_tile_id": tile_id,
        "original_tile_bbox": tile_bbox.to_string(),
        "iteration": iteration,
        "num_ranks": world_size,  # Number of ranks that saved PLY files (for merging)
        "tile_a": {
            "tile_id": tile_a_id,
            "bbox": tile_a.to_string(),
            "ply_path": str(ply_path_a) if (is_category3 and ply_path_a) else None,
            "ply_is_prefix": is_category3,  # True means: look for {ply_path}_rank{0..N}.ply
            "gaussian_count_local": int(count_a),  # This rank's count (total is sum of all ranks)
            "status": "pending",
        },
        "tile_b": {
            "tile_id": tile_b_id,
            "bbox": tile_b.to_string(),
            "ply_path": str(ply_path_b) if (is_category3 and ply_path_b) else None,
            "ply_is_prefix": is_category3,  # True means: look for {ply_path}_rank{0..N}.ply
            "gaussian_count_local": int(count_b),  # This rank's count (total is sum of all ranks)
            "status": "pending",
        },
    }

    with open(state_file, "w") as f:
        json.dump(oom_state, f, indent=2)

    utils.print_rank_0(f"[adaptive-tile] Saved OOM state to {state_file}")
    log_file.write(f"[adaptive-tile] Saved OOM state to {state_file}\n")
    log_file.flush()

    # Write done file for this rank (even for Category 1/2, to unblock other ranks)
    total_saved = count_a + count_b
    write_oom_done(args, utils.GLOBAL_RANK, total_saved)

    # Wait briefly for other ranks that may be actively checking signal
    # Short timeout (10s): if other ranks are stuck in distributed collectives,
    # they won't save until they receive SIGTERM after this rank exits
    print(f"\n[oom-signal] Rank {utils.GLOBAL_RANK} waiting for other ranks (10s)...", flush=True)
    print(f"[oom-signal] Note: Stuck ranks will save via SIGTERM handler after exit.", flush=True)
    all_done = check_all_ranks_done(args, utils.WORLD_SIZE, timeout=10.0)

    if all_done:
        print(f"[oom-signal] All ranks completed! Safe to exit.", flush=True)
    else:
        print(f"[oom-signal] Some ranks not done yet - they will save via SIGTERM handler.", flush=True)

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

    # Install SIGTERM handler for distributed OOM synchronization
    # This allows other ranks to save their gaussians when one rank catches OOM
    if adaptive_tile_mode:
        install_sigterm_handler()
        # Initial state registration
        register_sigterm_state(gaussians, args, log_file, current_iteration)

    try:
        for iteration in range(
            start_from_this_iteration, opt_args.iterations + 1, args.bsz
        ):
            current_iteration = iteration

            # Update SIGTERM state with current iteration (for proper file naming on termination)
            if adaptive_tile_mode:
                register_sigterm_state(gaussians, args, log_file, iteration)

            # Debug logging for first few iterations
            debug_first_iters = iteration <= 3

            # Reset operation tracker for this iteration
            set_current_operation("iteration_start")

            # Check for OOM signal from other ranks (distributed OOM sync)
            # This allows all ranks to save their gaussians when one rank catches OOM
            if adaptive_tile_mode:
                oom_signal = check_oom_signal(args)
                if oom_signal and oom_signal.get("signaling_rank") != utils.GLOBAL_RANK:
                    # Another rank caught OOM - save our gaussians and exit
                    print(f"\n[oom-signal] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Detected OOM signal!", flush=True)
                    log_file.write(f"[oom-signal] Rank {utils.GLOBAL_RANK} detected signal at iter {iteration}\n")

                    handled = handle_oom_signal_from_other_rank(
                        oom_signal, gaussians, args, iteration, log_file
                    )
                    if handled:
                        log_file.flush()
                        tile_manager.finalize(gaussians)
                        progress_bar.clear()
                        progress_bar.disable = True
                        progress_bar.close()
                        print(f"[oom-signal] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
                        sys.exit(EXIT_CODE_OOM)

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

            # Check for OOM signal after render (more frequent check for faster response)
            if adaptive_tile_mode:
                oom_signal = check_oom_signal(args)
                if oom_signal and oom_signal.get("signaling_rank") != utils.GLOBAL_RANK:
                    print(f"\n[oom-signal] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Detected OOM signal (post-render)!", flush=True)
                    log_file.write(f"[oom-signal] Rank {utils.GLOBAL_RANK} detected signal at iter {iteration} (post-render)\n")
                    handled = handle_oom_signal_from_other_rank(
                        oom_signal, gaussians, args, iteration, log_file
                    )
                    if handled:
                        log_file.flush()
                        tile_manager.finalize(gaussians)
                        progress_bar.clear()
                        progress_bar.disable = True
                        progress_bar.close()
                        print(f"[oom-signal] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
                        sys.exit(EXIT_CODE_OOM)

            # Save debug images for off-center projection verification
            debug_iters_env = os.environ.get("DEBUG_SAVE_ITERS", "")
            should_save = should_save_debug_images(iteration, args)
            if iteration <= 3:
                utils.print_rank_0(f"[DEBUG-IMG] iter={iteration}, DEBUG_SAVE_ITERS='{debug_iters_env}', should_save={should_save}")
            if should_save:
                try:
                    # Save debug images directly under main output dir (e.g., adaptive_test/debug_images)
                    tile_output_dir = Path(getattr(args, "tile_output_dir", "") or args.model_path)
                    output_path = tile_output_dir.parent  # Go up from ply/ to adaptive_test/
                    tile_id = getattr(args, "tile_id", "unknown")
                    utils.print_rank_0(f"[DEBUG-IMG] Saving to {output_path}/debug_images/ tile={tile_id} iter={iteration}")
                    save_debug_images(
                        batched_image,
                        batched_cameras,
                        batched_strategies,
                        iteration,
                        output_path,
                        tile_id=tile_id,
                    )
                except Exception as e:
                    utils.print_rank_0(f"[DEBUG-IMG] Failed to save debug images: {e}")
                    import traceback
                    traceback.print_exc()

            # Check for OOM signal before loss computation (other rank may have OOM'd during render)
            if adaptive_tile_mode:
                oom_signal = check_oom_signal(args)
                # Log every 100 iterations to track progress without flooding
                if iteration % 100 == 0:
                    log_file.write(f"[oom-check] iter={iteration} pre-loss: signal={'YES' if oom_signal else 'no'}\n")
                    log_file.flush()
                if oom_signal and oom_signal.get("signaling_rank") != utils.GLOBAL_RANK:
                    print(f"\n[oom-signal] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Detected OOM signal (pre-loss)!", flush=True)
                    log_file.write(f"[oom-signal] Rank {utils.GLOBAL_RANK} detected signal at iter {iteration} (pre-loss)\n")
                    log_file.flush()
                    handled = handle_oom_signal_from_other_rank(
                        oom_signal, gaussians, args, iteration, log_file
                    )
                    if handled:
                        log_file.flush()
                        tile_manager.finalize(gaussians)
                        progress_bar.clear()
                        progress_bar.disable = True
                        progress_bar.close()
                        print(f"[oom-signal] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
                        sys.exit(EXIT_CODE_OOM)

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

            # Check for OOM signal before backward (critical timing - other rank may have OOM'd during loss)
            if adaptive_tile_mode:
                oom_signal = check_oom_signal(args)
                # Log every 100 iterations to track progress without flooding
                if iteration % 100 == 0:
                    log_file.write(f"[oom-check] iter={iteration} pre-backward: signal={'YES' if oom_signal else 'no'}\n")
                    log_file.flush()
                if oom_signal and oom_signal.get("signaling_rank") != utils.GLOBAL_RANK:
                    print(f"\n[oom-signal] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Detected OOM signal (pre-backward)!", flush=True)
                    log_file.write(f"[oom-signal] Rank {utils.GLOBAL_RANK} detected signal at iter {iteration} (pre-backward)\n")
                    log_file.flush()
                    handled = handle_oom_signal_from_other_rank(
                        oom_signal, gaussians, args, iteration, log_file
                    )
                    if handled:
                        log_file.flush()
                        tile_manager.finalize(gaussians)
                        progress_bar.clear()
                        progress_bar.disable = True
                        progress_bar.close()
                        print(f"[oom-signal] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
                        sys.exit(EXIT_CODE_OOM)

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
                    num_gaussians_before = gaussians.get_xyz.shape[0]
                    if args.backend == "gsplat":
                        gsplat_densification(
                            iteration, scene, gaussians, batched_screenspace_pkg
                        )
                    else:
                        densification(iteration, scene, gaussians, batched_screenspace_pkg)
                    num_gaussians_after = gaussians.get_xyz.shape[0]
                    log_gpu_memory("after_densification")
                    # Log gaussian count change (compute total across all ranks)
                    if num_gaussians_after != num_gaussians_before:
                        local_delta = num_gaussians_after - num_gaussians_before
                        sign = "+" if local_delta > 0 else ""
                        # Compute total across all GPUs (before and after)
                        counts = torch.tensor([num_gaussians_before, num_gaussians_after], device="cuda", dtype=torch.long)
                        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
                        total_before, total_after = counts[0].item(), counts[1].item()
                        total_delta = total_after - total_before
                        total_sign = "+" if total_delta > 0 else ""
                        utils.print_rank_0(f"[Densify] iter {iteration}: TOTAL {total_before:,} -> {total_after:,} ({total_sign}{total_delta:,})")

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

            # Save Gaussians (skip intermediate saves in adaptive tile mode)
            should_save = any(
                [
                    iteration <= save_iteration < iteration + args.bsz
                    for save_iteration in args.save_iterations
                ]
            )
            # In adaptive mode, only save at final iteration (handled by adaptive_trainer.py)
            if should_save and not args.adaptive_tile_enabled:
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
