import os
import sys
import gc
import signal
import time
import threading
import _thread
import torch
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Import robust OOM handler
from robust_oom_handler import (
    initialize_robust_oom_handler, 
    get_robust_oom_handler,
    detect_oom_robust,
    check_oom_signal_robust
)
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
from datetime import datetime


def _ts():
    """Get current timestamp string for logging (HH:MM:SS.mmm)"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


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
_OOM_SAVE_IN_PROGRESS = False  # Flag to prevent duplicate saves


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
    global _SIGTERM_GAUSSIANS, _SIGTERM_ARGS, _SIGTERM_LOG_FILE, _SIGTERM_ITERATION, _OOM_SAVE_IN_PROGRESS

    rank = utils.GLOBAL_RANK if hasattr(utils, 'GLOBAL_RANK') else 0

    # IMMEDIATE log write - this is the first thing we do
    # If this doesn't appear in log, SIGTERM handler was never called
    ts = _ts()
    if _SIGTERM_LOG_FILE:
        try:
            _SIGTERM_LOG_FILE.write(f"\n{'='*60}\n")
            _SIGTERM_LOG_FILE.write(f"[{ts}] [SIGTERM] HANDLER CALLED! Rank {rank} at iter {_SIGTERM_ITERATION}\n")
            _SIGTERM_LOG_FILE.write(f"{'='*60}\n")
            _SIGTERM_LOG_FILE.flush()
        except:
            pass

    print(f"\n[{ts}] [SIGTERM] Rank {rank} received SIGTERM signal!", flush=True)

    # Check if save is already in progress (via signal check in main loop)
    # If so, let that save complete - don't try to save again
    if _OOM_SAVE_IN_PROGRESS:
        print(f"[SIGTERM] Rank {rank}: Save already in progress, waiting for completion...", flush=True)
        if _SIGTERM_LOG_FILE:
            try:
                _SIGTERM_LOG_FILE.write(f"[SIGTERM] Rank {rank}: Save in progress, exiting without duplicate save\n")
                _SIGTERM_LOG_FILE.flush()
            except:
                pass
        # Wait briefly for save to complete (up to 60s)
        # This gives the main save process time to finish
        import time
        for _ in range(60):
            if not _OOM_SAVE_IN_PROGRESS:
                print(f"[SIGTERM] Rank {rank}: Save completed, exiting", flush=True)
                break
            time.sleep(1)
        sys.exit(EXIT_CODE_OOM)

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
    # Ignore further SIGTERM during save to reduce re-entrancy
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print(f"[SIGTERM] Rank {rank}: Ignoring further SIGTERM during save", flush=True)
    except Exception:
        pass

    # Write ack file (even though we received SIGTERM, write it for logging)
    try:
        write_oom_ack(_SIGTERM_ARGS, rank)
    except Exception:
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

# Global OOM signal monitor instance (set by training function) - DEPRECATED
# Now using robust OOM handler instead
_oom_signal_monitor: Optional["OOMSignalMonitor"] = None


class OOMSignalMonitor:
    """Background thread that monitors for OOM signals from other ranks.

    This solves the problem where ranks are stuck in collective operations
    (like all_reduce) and cannot check for OOM signals in the main loop.
    The monitor thread runs independently and can detect signals even when
    the main thread is blocked.

    For Category 3 OOM (where PLY saving is needed), the monitor will
    interrupt the main thread using _thread.interrupt_main() to force it
    out of blocking collective operations.
    """

    def __init__(self, args, rank: int, world_size: int):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.signal_detected = False
        self.detected_signal_data: Optional[dict] = None
        self.stop_event = threading.Event()
        self.ack_written = False
        self.main_thread_interrupted = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Start the monitor thread."""
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[oom-monitor] Rank {self.rank} started OOM signal monitor thread", flush=True)

    def stop(self):
        """Stop the monitor thread."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print(f"[oom-monitor] Rank {self.rank} stopped OOM signal monitor thread", flush=True)

    def _monitor_loop(self):
        """Main monitoring loop - runs in background thread."""
        signal_path = get_oom_signal_path(self.args)
        check_interval = 0.5  # Check every 0.5 seconds

        while not self.stop_event.is_set():
            try:
                if signal_path.exists() and not self.signal_detected:
                    with open(signal_path, "r") as f:
                        signal_data = json.load(f)

                    signaling_rank = signal_data.get("signaling_rank")

                    # Only respond to signals from OTHER ranks
                    if signaling_rank is not None and signaling_rank != self.rank:
                        with self._lock:
                            if not self.signal_detected:  # Double-check under lock
                                self.signal_detected = True
                                self.detected_signal_data = signal_data
                                category = signal_data.get("category", "?")
                                iteration = signal_data.get("iteration", "?")
                                print(f"\n[oom-monitor] Rank {self.rank} detected OOM signal from rank {signaling_rank} (cat={category}, iter={iteration})", flush=True)

                                # Write ack immediately
                                self._write_ack()

                                # For Category 3, interrupt main thread to force PLY saving
                                if category == 3 and not self.main_thread_interrupted:
                                    print(f"[oom-monitor] Rank {self.rank} Category 3 OOM - interrupting main thread for PLY save", flush=True)
                                    self.main_thread_interrupted = True
                                    # Small delay to ensure ack is written
                                    time.sleep(0.2)
                                    try:
                                        _thread.interrupt_main()
                                        print(f"[oom-monitor] Rank {self.rank} sent interrupt to main thread", flush=True)
                                    except Exception as e:
                                        print(f"[oom-monitor] Rank {self.rank} failed to interrupt main thread: {e}", flush=True)
            except (json.JSONDecodeError, IOError, KeyError):
                pass  # Ignore transient file read errors

            # Sleep in small increments to respond to stop_event quickly
            for _ in range(int(check_interval * 10)):
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)

    def _write_ack(self):
        """Write acknowledgment file."""
        if self.ack_written:
            return

        try:
            ack_path = get_oom_ack_path(self.args, self.rank)
            ack_data = {
                "rank": self.rank,
                "timestamp": time.time(),
                "from_monitor_thread": True,
            }
            temp_path = ack_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(ack_data, f)
            temp_path.rename(ack_path)
            self.ack_written = True
            print(f"[oom-monitor] Rank {self.rank} wrote ack file (from monitor thread)", flush=True)
        except Exception as e:
            print(f"[oom-monitor] Rank {self.rank} failed to write ack: {e}", flush=True)

    def check_and_get_signal(self) -> Optional[dict]:
        """Check if signal was detected (called from main thread).

        Returns:
            Signal data if detected, None otherwise.
        """
        with self._lock:
            if self.signal_detected:
                return self.detected_signal_data
            return None

    def clear_signal(self):
        """Clear the detected signal (after handling)."""
        with self._lock:
            self.signal_detected = False
            self.detected_signal_data = None
            self.ack_written = False


def get_oom_signal_monitor() -> Optional["OOMSignalMonitor"]:
    """Get the global OOM signal monitor instance."""
    return _oom_signal_monitor


def set_oom_signal_monitor(monitor: Optional["OOMSignalMonitor"]):
    """Set the global OOM signal monitor instance."""
    global _oom_signal_monitor
    _oom_signal_monitor = monitor


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

    This function first checks the monitor thread (if running) for signals
    detected while the main thread was blocked in collective operations.
    Then falls back to checking the file directly.

    Returns:
        Signal data dict if signal exists, None otherwise.
    """
    # First check the monitor thread (it can detect signals even when main thread is blocked)
    monitor = get_oom_signal_monitor()
    if monitor is not None:
        signal_data = monitor.check_and_get_signal()
        if signal_data is not None:
            return signal_data

    # Fall back to checking the file directly
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


def write_oom_done(args, rank: int, gaussians_saved: int, count_a: int = 0, count_b: int = 0):
    """Write OOM done file to indicate this rank has finished saving.

    Args:
        args: Training arguments
        rank: Rank ID
        gaussians_saved: Total gaussians saved by this rank
        count_a: Gaussians saved for Child A (for validation)
        count_b: Gaussians saved for Child B (for validation)
    """
    done_path = get_oom_done_path(args, rank)
    done_data = {
        "rank": rank,
        "gaussians_saved": gaussians_saved,
        "count_a": count_a,  # For merge validation
        "count_b": count_b,  # For merge validation
        "done": True,
    }
    temp_path = done_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(done_data, f)
    temp_path.rename(done_path)
    print(f"[oom-signal] Rank {rank} wrote done file: {done_path}", flush=True)


def check_all_ranks_done(args, world_size: int, timeout: float = 120.0, target_ranks: list = None) -> bool:
    """Wait for specific ranks to write their done files.

    Args:
        args: Training arguments
        world_size: Total number of ranks (used if target_ranks is None)
        timeout: Maximum seconds to wait
        target_ranks: List of specific ranks to wait for. If None, wait for all ranks.

    Returns:
        True if all target ranks done, False if timeout
    """
    start_time = time.time()
    check_interval = 0.5  # Check every 0.5 seconds

    # If target_ranks not specified, wait for all ranks
    if target_ranks is None:
        target_ranks = list(range(world_size))

    if not target_ranks:
        print(f"[oom-signal] No target ranks to wait for", flush=True)
        return True

    print(f"[oom-signal] Waiting for done files from ranks: {target_ranks}", flush=True)

    while time.time() - start_time < timeout:
        all_done = True
        for rank in target_ranks:
            done_path = get_oom_done_path(args, rank)
            if not done_path.exists():
                all_done = False
                break

        if all_done:
            elapsed = time.time() - start_time
            print(f"[oom-signal] All {len(target_ranks)} target ranks completed in {elapsed:.1f}s!", flush=True)
            return True

        time.sleep(check_interval)
        elapsed = time.time() - start_time
        if int(elapsed) % 2 == 0 and int(elapsed) > 0:  # Print every 2 seconds
            missing = [r for r in target_ranks if not get_oom_done_path(args, r).exists()]
            print(f"[oom-signal] Waiting for ranks {missing} to complete... ({elapsed:.0f}s/{timeout:.0f}s)", flush=True)

    missing = [r for r in target_ranks if not get_oom_done_path(args, r).exists()]
    print(f"[oom-signal] TIMEOUT after {timeout:.0f}s! Missing ranks: {missing}", flush=True)
    return False


def clear_oom_done_files(args, world_size: int):
    """Remove all OOM done files."""
    for rank in range(world_size):
        done_path = get_oom_done_path(args, rank)
        try:
            done_path.unlink(missing_ok=True)
        except Exception:
            pass


def get_oom_ack_path(args, rank: int) -> Path:
    """Get the path to the OOM acknowledgment file for a specific rank."""
    signal_path = get_oom_signal_path(args)
    return signal_path.parent / f"oom_ack_rank{rank}.json"


def write_oom_ack(args, rank: int):
    """Write OOM ack file to indicate this rank has seen the signal.

    This is called IMMEDIATELY when a rank detects the signal, BEFORE
    it starts saving gaussians. This allows the OOM rank to know that
    this rank will NOT enter backward() and get stuck in all_reduce.
    """
    ack_path = get_oom_ack_path(args, rank)
    ack_data = {
        "rank": rank,
        "acked": True,
        "timestamp": time.time(),
    }
    temp_path = ack_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(ack_data, f)
    temp_path.rename(ack_path)
    print(f"[{_ts()}] [oom-signal] Rank {rank} wrote ack file: {ack_path.name}", flush=True)


def wait_for_all_acks(args, world_size: int, signaling_rank: int, timeout: float = 60.0) -> tuple:
    """Wait for all OTHER ranks to acknowledge the OOM signal.

    This is called by the OOM rank after writing the signal file.
    It waits until all other ranks have seen the signal (and thus won't
    enter backward() and get stuck in all_reduce).

    Args:
        args: Training arguments
        world_size: Total number of ranks
        signaling_rank: The rank that wrote the signal (don't wait for itself)
        timeout: Maximum seconds to wait

    Returns:
        Tuple of (all_acked: bool, acked_ranks: list)
        - all_acked: True if all ranks acknowledged within timeout
        - acked_ranks: List of ranks that acknowledged (for done-file waiting)
    """
    start_time = time.time()
    check_interval = 0.5  # Check every 0.5 seconds

    # We wait for all ranks except the signaling rank
    expected_ranks = [r for r in range(world_size) if r != signaling_rank]

    if not expected_ranks:
        print(f"[oom-signal] No other ranks to wait for (world_size={world_size})", flush=True)
        return True, []

    print(f"[oom-signal] Rank {signaling_rank} waiting for acks from ranks {expected_ranks}...", flush=True)

    prev_acked_set = set()
    last_progress_print = 0

    while time.time() - start_time < timeout:
        acked_ranks = []
        missing_ranks = []

        for rank in expected_ranks:
            ack_path = get_oom_ack_path(args, rank)
            if ack_path.exists():
                acked_ranks.append(rank)
            else:
                missing_ranks.append(rank)

        if not missing_ranks:
            elapsed = time.time() - start_time
            print(f"[oom-signal] SUCCESS: All {len(expected_ranks)} ranks acknowledged in {elapsed:.1f}s!", flush=True)
            return True, acked_ranks

        # Print when a NEW ack is received (not every iteration)
        current_acked_set = set(acked_ranks)
        new_acks = current_acked_set - prev_acked_set
        if new_acks:
            elapsed = time.time() - start_time
            # Check if ack came from monitor thread
            for new_rank in sorted(new_acks):
                ack_path = get_oom_ack_path(args, new_rank)
                from_monitor = False
                try:
                    with open(ack_path, "r") as f:
                        ack_data = json.load(f)
                        from_monitor = ack_data.get("from_monitor_thread", False)
                except:
                    pass
                source = " (from monitor thread)" if from_monitor else ""
                print(f"[oom-signal] +++ Rank {new_rank} acked!{source} ({len(acked_ranks)}/{len(expected_ranks)}) [{elapsed:.1f}s]", flush=True)
            prev_acked_set = current_acked_set

        time.sleep(check_interval)
        elapsed = time.time() - start_time
        elapsed_int = int(elapsed)
        if elapsed_int >= last_progress_print + 5:  # Print every 5 seconds
            last_progress_print = elapsed_int
            print(f"[oom-signal] Waiting: {len(acked_ranks)}/{len(expected_ranks)} acks, missing={missing_ranks} ({elapsed_int}s/{int(timeout)}s)", flush=True)

    # Timeout - some ranks didn't ack (probably stuck in all_reduce)
    acked_ranks = [r for r in expected_ranks if get_oom_ack_path(args, r).exists()]
    missing_ranks = [r for r in expected_ranks if r not in acked_ranks]
    elapsed = time.time() - start_time
    print(f"[oom-signal] TIMEOUT after {elapsed:.0f}s! Ranks {missing_ranks} didn't ack (probably in all_reduce)", flush=True)
    print(f"[oom-signal] Acked ranks: {acked_ranks} - will wait for their done files", flush=True)
    print(f"[oom-signal] Non-acked ranks: {missing_ranks} - will save via SIGTERM handler after exit", flush=True)
    return False, acked_ranks


def clear_oom_ack_files(args, world_size: int):
    """Remove all OOM ack files."""
    for rank in range(world_size):
        ack_path = get_oom_ack_path(args, rank)
        try:
            ack_path.unlink(missing_ok=True)
        except Exception:
            pass


def get_acked_ranks(args, world_size: int, signaling_rank: int) -> list:
    """Get list of ranks that have acknowledged the OOM signal.

    This reads existing ack files to determine which ranks are NOT stuck
    in backward() and will be saving gaussians.

    Args:
        args: Training arguments
        world_size: Total number of ranks
        signaling_rank: The rank that triggered OOM (won't have ack file)

    Returns:
        List of rank IDs that have acknowledged (excluding signaling_rank)
    """
    acked = []
    for rank in range(world_size):
        if rank == signaling_rank:
            continue  # OOM rank doesn't write ack
        ack_path = get_oom_ack_path(args, rank)
        if ack_path.exists():
            acked.append(rank)
    return acked


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

    # Set flag to prevent duplicate saves from SIGTERM handler
    global _OOM_SAVE_IN_PROGRESS
    _OOM_SAVE_IN_PROGRESS = True
    print(f"[oom-signal] Rank {utils.GLOBAL_RANK}: Save started (flag set)", flush=True)

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
    # Safely handle tile split
    split_result = tile_bbox.split()
    if isinstance(split_result, tuple) and len(split_result) == 2:
        tile_a, tile_b = split_result
    else:
        print(f"[oom-signal] Rank {utils.GLOBAL_RANK} ERROR: tile_bbox.split() returned unexpected format: {split_result}", flush=True)
        print(f"[oom-signal] Expected tuple with 2 elements, got {type(split_result)} with {len(split_result) if hasattr(split_result, '__len__') else 'unknown'} elements", flush=True)
        return False
    # Use same tile ID generation as handle_adaptive_tile_oom()
    tile_num = int(tile_id.split("_")[-1]) if "_" in tile_id else 0
    tile_a_id = f"tile_{tile_num * 2 + 1:04d}"
    tile_b_id = f"tile_{tile_num * 2 + 2:04d}"

    # Use the signal iteration for file naming (matches signaling rank's files)
    use_iteration = signal_iteration

    print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} saving local gaussians for tile split...", flush=True)
    print(f"  Parent tile: {tile_id}, level: {parent_level} -> child level: {child_level}", flush=True)
    print(f"  Child tile A: {tile_a_id}, bbox: {tile_a.to_string()}", flush=True)
    print(f"  Child tile B: {tile_b_id}, bbox: {tile_b.to_string()}", flush=True)
    print(f"  Output dir: {tile_output_dir}", flush=True)

    # IMPORTANT: Save tile B FIRST, then tile A
    # Reason: Child B tends to be larger due to non-uniform gaussian distribution.
    # If SIGKILL arrives during save, we want the larger tile (B) to be saved first.
    # This way, SIGTERM-triggered ranks have a better chance of saving both tiles,
    # or at least the larger/more important one.

    # CRITICAL: Aggressively clear GPU memory before saving PLY
    import gc

    # Clear gradients on all gaussian parameters
    for param in gaussians.parameters():
        if param.grad is not None:
            param.grad = None

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"  [GPU cleanup] Cleared gradients and cache. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB", flush=True)

    # Save tile B first - directly to final path (no temp file to avoid race condition)
    # Note: save_ply with local_only=True appends _rank{N} to the path
    print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} >>> STARTING Child B save...", flush=True)
    final_base_b = tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{use_iteration}.ply"
    count_b = gaussians.save_ply(str(final_base_b), filter_bbox=tile_b, local_only=True)
    final_path_b = tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{use_iteration}_rank{utils.GLOBAL_RANK}.ply"
    if count_b > 0:
        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} <<< COMPLETED Child B: {count_b:,} gaussians -> {final_path_b.name}", flush=True)
    else:
        if final_path_b.exists():
            final_path_b.unlink(missing_ok=True)
        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} <<< SKIP Child B: 0 gaussians", flush=True)

    # Clear cache again before second save
    gc.collect()
    torch.cuda.empty_cache()

    # Save tile A second - directly to final path
    print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} >>> STARTING Child A save...", flush=True)
    final_base_a = tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{use_iteration}.ply"
    count_a = gaussians.save_ply(str(final_base_a), filter_bbox=tile_a, local_only=True)
    # Actual file created: {final_base_a}_rank{N}.ply
    final_path_a = tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{use_iteration}_rank{utils.GLOBAL_RANK}.ply"
    if count_a > 0:
        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} <<< COMPLETED Child A: {count_a:,} gaussians -> {final_path_a.name}", flush=True)
    else:
        # Remove empty file
        if final_path_a.exists():
            final_path_a.unlink(missing_ok=True)
        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} <<< SKIP Child A: 0 gaussians", flush=True)

    total_saved = count_a + count_b

    # Print split comparison
    if total_saved > 0:
        pct_a = 100.0 * count_a / total_saved
        pct_b = 100.0 * count_b / total_saved
        print(f"[oom-signal] Rank {utils.GLOBAL_RANK} split summary:", flush=True)
        print(f"  Child A ({tile_a_id}): {count_a:,} gaussians ({pct_a:.1f}%)", flush=True)
        print(f"  Child B ({tile_b_id}): {count_b:,} gaussians ({pct_b:.1f}%)", flush=True)
        print(f"  Total: {total_saved:,} gaussians", flush=True)
    else:
        print(f"[oom-signal] Rank {utils.GLOBAL_RANK} finished. Total saved: 0 gaussians", flush=True)

    log_file.write(f"[oom-signal] Rank {utils.GLOBAL_RANK} saved {total_saved:,} gaussians\n")
    log_file.flush()

    # Write adaptive_tile_state.json so the wrapper can merge/resume (Category 3)
    try:
        state_file = getattr(args, "tile_state_file", "") or (tile_output_dir / "adaptive_tile_state.json")
        state_file = Path(state_file)
        ply_path_a = str(tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{use_iteration}")
        ply_path_b = str(tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{use_iteration}")
        oom_state = {
            "oom_occurred": True,
            "oom_cause": "signal",
            "oom_category": 3,
            "original_tile_id": tile_id,
            "original_tile_bbox": tile_bbox.to_string(),
            "iteration": use_iteration,
            "num_ranks": utils.WORLD_SIZE,
            "tile_a": {
                "tile_id": tile_a_id,
                "bbox": tile_a.to_string(),
                "ply_path": ply_path_a if count_a > 0 else None,
                "ply_is_prefix": True,
                "gaussian_count_local": int(count_a),
                "status": "pending",
            },
            "tile_b": {
                "tile_id": tile_b_id,
                "bbox": tile_b.to_string(),
                "ply_path": ply_path_b if count_b > 0 else None,
                "ply_is_prefix": True,
                "gaussian_count_local": int(count_b),
                "status": "pending",
            },
        }
        with open(state_file, "w") as f:
            json.dump(oom_state, f, indent=2)
        print(f"[oom-signal] Saved OOM state to {state_file} (num_ranks={utils.WORLD_SIZE}, WORLD_SIZE={utils.WORLD_SIZE})", flush=True)
        log_file.write(f"[oom-signal] Saved OOM state to {state_file} (num_ranks={utils.WORLD_SIZE})\n")
        log_file.flush()
    except Exception as e:
        print(f"[oom-signal] WARNING: Failed to write adaptive_tile_state.json: {e}", flush=True)

    # Write done file to signal completion to the signaling rank
    # Include per-child counts for merge validation
    write_oom_done(args, utils.GLOBAL_RANK, total_saved, count_a=count_a, count_b=count_b)

    # Clear save-in-progress flag
    _OOM_SAVE_IN_PROGRESS = False
    print(f"[oom-signal] Rank {utils.GLOBAL_RANK}: Save completed (flag cleared)", flush=True)

    # CRITICAL: Wait for ALL saving ranks to finish before exiting
    # This includes:
    # - acked_ranks: non-OOM ranks that detected signal and are saving
    # - signaling_rank: the OOM rank that is also saving
    # This prevents torchrun from killing ranks that are still saving
    # when this rank exits first.
    acked_ranks = get_acked_ranks(args, utils.WORLD_SIZE, signaling_rank)

    # Include the signaling rank (OOM rank) in the wait list
    # The OOM rank also saves gaussians and writes a done file
    all_saving_ranks = list(set(acked_ranks + [signaling_rank]))
    all_saving_ranks.sort()

    if all_saving_ranks:
        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} waiting for all saving ranks {all_saving_ranks} to complete...", flush=True)
        # Wait indefinitely - no timeout. All ranks MUST complete saving.
        # If a rank crashes, torchrun will eventually kill us via SIGTERM.
        # Using a very long timeout (1 hour) as safety net, but this should never be reached.
        all_done = check_all_ranks_done(args, utils.WORLD_SIZE, timeout=3600.0, target_ranks=all_saving_ranks)
        if all_done:
            print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK}: All saving ranks completed! Now safe to exit together.", flush=True)
        else:
            # This should never happen in normal operation
            # If we reach here, something is seriously wrong
            print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK}: FATAL - TIMEOUT after 1 hour waiting for saving ranks!", flush=True)
            print(f"[oom-signal] This indicates a serious problem - some ranks may have crashed.", flush=True)
            # Don't sys.exit here - let the process end naturally so torchrun can handle it
    else:
        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK}: No other saving ranks to wait for.", flush=True)

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


def _is_nccl_timeout_error(exception: BaseException) -> bool:
    """Check if an exception is a NCCL timeout error.

    NCCL timeout errors occur when one rank fails to participate in a collective
    operation within the timeout period. This typically happens when:
    - One rank crashed (e.g., OOM)
    - One rank exited early
    - Network issues between ranks

    Common error messages include:
    - "Timed out initializing process group"
    - "NCCL communicator was aborted"
    - "Watchdog caught collective operation timeout"
    - "NCCL error: unhandled system error"
    - "Work ... timed out"
    """
    if not isinstance(exception, (RuntimeError, Exception)):
        return False

    error_msg = str(exception).lower()
    timeout_indicators = [
        "timed out",
        "timeout",
        "nccl communicator was aborted",
        "watchdog caught collective operation",
        "nccl error",
        "unhandled system error",
        "connection reset by peer",
        "software caused connection abort",
    ]
    return any(indicator in error_msg for indicator in timeout_indicators)


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
    args=None,
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

    debug_dir = output_dir / "visualizations" / "gt_compare"
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
        num_visible = getattr(args, '_num_visible_cameras', 0)
        # Format: tile_xxxx_view_NN_of_MM_cam_zzzz_gpuW_compare.png
        prefix = f"{tile_id}_view{iteration:02d}of{num_visible:02d}_cam{cam_name}_gpu{rank}"

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

        save_path = debug_dir / f"{prefix}_compare.png"
        torchvision.utils.save_image(combined, save_path)
        print(f"[DEBUG-IMG] Saved: {save_path.name}", flush=True)

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
    """Check if debug images should be saved at this iteration.

    By default, saves for the first N iterations where N = number of visible cameras.
    This ensures all views are captured (each camera appears once in first N iters).
    """
    # Check environment variable for explicit iteration list
    debug_iters_str = os.environ.get("DEBUG_SAVE_ITERS", "")
    if debug_iters_str:
        try:
            debug_iters = [int(x.strip()) for x in debug_iters_str.split(",")]
            if debug_iters != [1]:  # Skip default "1" — use auto N instead
                return iteration in debug_iters
        except ValueError:
            pass

    # Check args for interval-based saving
    debug_interval = getattr(args, "debug_image_interval", 0)
    if debug_interval > 0 and iteration % debug_interval == 0:
        return True

    # Default: save first N iterations (N = visible camera count)
    # This captures all views since each camera appears once in first N iters
    num_visible = getattr(args, '_num_visible_cameras', 30)
    return iteration <= num_visible


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

    # CRITICAL: Check if another rank already wrote OOM signal
    # If so, don't write our own signal - just ack and handle
    existing_signal = check_oom_signal(args)
    if existing_signal and existing_signal.get("signaling_rank") != utils.GLOBAL_RANK:
        signaling_rank = existing_signal.get('signaling_rank')
        signal_category = existing_signal.get('category', '?')
        print(f"\n[oom-signal] Rank {utils.GLOBAL_RANK} OOM but signal already exists from rank {signaling_rank} (cat={signal_category})", flush=True)
        print(f"[oom-signal] Rank {utils.GLOBAL_RANK} will ack and handle existing signal instead of writing new one", flush=True)
        # Handle the existing signal (this will ack, save PLY if needed, and exit)
        handled = handle_oom_signal_from_other_rank(
            existing_signal, gaussians, args, iteration, log_file
        )
        print(f"[oom-signal] Rank {utils.GLOBAL_RANK} handle_oom_signal_from_other_rank returned: {handled}", flush=True)
        if handled:
            print(f"[oom-signal] Rank {utils.GLOBAL_RANK} exiting with code 42 (handled existing signal)", flush=True)
            sys.exit(42)  # Exit with OOM code
        else:
            print(f"[oom-signal] Rank {utils.GLOBAL_RANK} handle returned False, will write own signal", flush=True)
        # If not handled, fall through to write our own signal

    # No existing signal (or it was ours), write new one
    if not existing_signal:
        print(f"\n[oom-signal] Rank {utils.GLOBAL_RANK} no existing signal found, will write new one", flush=True)

    # Calculate OOM category early for signal
    densify_from_early = getattr(args, 'densify_from_iter', 500)
    densify_interval_early = getattr(args, 'densification_interval', 100)
    first_densify_early = densify_from_early + densify_interval_early
    is_category3_early = iteration > first_densify_early
    has_pretrained_early = bool(getattr(args, "pretrained_ply", ""))
    C_early = total_cameras_in_tile if total_cameras_in_tile > 0 else num_cameras
    if is_category3_early:
        oom_category_early = 3
    elif has_pretrained_early:
        oom_category_early = 4
    elif C_early > 0 and iteration <= C_early:
        oom_category_early = 1
    else:
        oom_category_early = 2
    early_cause = "densification" if is_category3_early else "early_iteration"

    # Only use robust OOM handler for Category 3 (which needs PLY saving across ranks)
    # Category 1,2 can be handled locally without cross-rank coordination
    robust_used = False
    if oom_category_early == 3:
        print(f"\n[robust-oom] Rank {utils.GLOBAL_RANK} IMMEDIATE robust OOM signal (iter={iteration}, cat={oom_category_early})", flush=True)
        # Do NOT kill the process group here; we still need to write state files.
        # Also avoid handling SIGUSR1 on the detecting rank so it can finish state write.
        detect_oom_robust(
            iteration,
            oom_category_early,
            early_cause,
            kill_process_group=False,
            signal_self=False,
            save_self=False,
        )
        robust_used = True
    else:
        print(f"\n[simple-oom] Rank {utils.GLOBAL_RANK} IMMEDIATE category {oom_category_early} OOM - no cross-rank coordination needed", flush=True)

    # Only wait for acks if Category 3 (need PLY saving from other ranks)
    # Category 1/2: No PLY saving needed, other ranks are likely stuck in collective ops anyway
    if oom_category_early == 3 and not robust_used:
        # Wait for other ranks to acknowledge the signal (dynamic waiting)
        # This is CRITICAL: we wait until all other ranks have written ack files,
        # which means they've seen the signal and will NOT enter backward() (all_reduce).
        # If some ranks don't ack within timeout, they're probably stuck in backward()
        # already and will be saved via SIGTERM handler after we exit.
        #
        # Timeout is 60s - should be enough for even the longest loss computation.
        # We use dynamic ack-based waiting instead of fixed sleep because:
        #   - Early/large tiles have long loss computation (20s was too short)
        #   - Small tiles finish quickly (20s was wasteful)
        #   - Ack-based waiting adapts to actual completion time
        # Use environment variable for ACK timeout (default 600s for large gaussian counts 10M+)
        ack_timeout = float(os.environ.get('OOM_ACK_TIMEOUT', 600.0))
        all_acked, acked_ranks = wait_for_all_acks(args, utils.WORLD_SIZE, utils.GLOBAL_RANK, timeout=ack_timeout)

        if all_acked:
            print(f"[oom-signal] Proceeding with OOM handling - all ranks will save PLY.", flush=True)
        else:
            print(f"[oom-signal] Proceeding with OOM handling - stuck ranks will save via SIGTERM.", flush=True)
            print(f"[oom-signal] Will wait for done files only from acked ranks: {acked_ranks}", flush=True)
    elif oom_category_early == 3 and robust_used:
        # Robust handler does not use ack files; skip ack waiting to avoid timeouts.
        all_acked, acked_ranks = False, []
        print(f"[robust-oom] Skipping ack wait (robust handler in use)", flush=True)
    else:
        # Category 1/2: Skip ack waiting - no PLY saving needed
        print(f"[oom-signal] Category {oom_category_early} OOM - skipping ack wait (no PLY saving needed)", flush=True)
        all_acked, acked_ranks = False, []

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

    # Determine split axis (X or Y only, never Z)
    if dx >= dy:
        split_axis = "X"
        mid = (tile_bbox.x_min + tile_bbox.x_max) / 2
        utils.print_rank_0(f"[adaptive-tile] Splitting along {split_axis} axis (dx={dx:.1f} >= dy={dy:.1f}): mid={mid:.2f}")
    else:
        split_axis = "Y"
        mid = (tile_bbox.y_min + tile_bbox.y_max) / 2
        utils.print_rank_0(f"[adaptive-tile] Splitting along {split_axis} axis (dy={dy:.1f} > dx={dx:.1f}): mid={mid:.2f}")

    # Split the tile
    split_result = tile_bbox.split()
    if isinstance(split_result, tuple) and len(split_result) == 2:
        tile_a, tile_b = split_result
    else:
        utils.print_rank_0(f"[adaptive-tile] ERROR: tile_bbox.split() returned unexpected format: {split_result}")
        utils.print_rank_0(f"[adaptive-tile] Expected tuple with 2 elements, got {type(split_result)} with {len(split_result) if hasattr(split_result, '__len__') else 'unknown'} elements")
        return
    utils.print_rank_0(f"[adaptive-tile] Splitting tile into:")
    utils.print_rank_0(f"  Tile A: {tile_a.to_string()}")
    utils.print_rank_0(f"  Tile B: {tile_b.to_string()}")

    # Note: Gaussian counts will be computed accurately inside save_ply after gathering from all GPUs
    # The filter_bbox is passed to save_ply which computes the mask on the gathered data

    # Generate new tile IDs
    tile_num = int(tile_id.split("_")[-1]) if "_" in tile_id else 0
    tile_a_id = f"tile_{tile_num * 2 + 1:04d}"
    tile_b_id = f"tile_{tile_num * 2 + 2:04d}"

    # Determine OOM category
    has_pretrained = bool(getattr(args, "pretrained_ply", ""))
    is_category3 = iteration > first_densify_iter
    if is_category3:
        oom_category = 3  # Densification grew gaussians (save PLY + split)
    elif has_pretrained:
        oom_category = 4  # Pre-trained gaussians too large before densification (split, reuse parent PLY)
    elif C > 0 and iteration <= C:
        oom_category = 1  # First camera pass, image too large (split)
    else:
        oom_category = 2  # Memory fragmentation before densification (retry)

    print(f"\n[OOM Category Detection]", flush=True)
    print(f"  iteration={iteration}, first_densify_iter={first_densify_iter}, has_pretrained={has_pretrained}", flush=True)
    print(f"  is_category3={is_category3}, oom_category={oom_category}", flush=True)

    # Only use robust OOM handler for Category 3 (which needs PLY saving across ranks)
    # Category 1,2 can be handled locally without cross-rank coordination
    if not robust_used:
        if is_category3:
            print(f"\n[robust-oom] Rank {utils.GLOBAL_RANK} robust OOM signal with detailed cause: {likely_cause}", flush=True)
            detect_oom_robust(
                iteration,
                oom_category,
                likely_cause,
                kill_process_group=False,
                signal_self=False,
                save_self=False,
            )
            robust_used = True
        else:
            print(f"\n[simple-oom] Category {oom_category} OOM - no cross-rank coordination needed", flush=True)

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

        # CRITICAL: Aggressively clear GPU memory before saving PLY
        # When OOM happens, GPU is nearly full. We need to free as much as possible:
        # 1. Clear gradients on all gaussian parameters (frees gradient tensors)
        # 2. Move tensors to CPU temporarily to maximize GPU memory
        # 3. Force Python garbage collection multiple times
        # 4. Synchronize CUDA and clear PyTorch's memory cache
        import gc

        print(f"  [aggressive-cleanup] Starting aggressive GPU memory cleanup for PLY saving...", flush=True)
        
        # Clear gradients - this frees the .grad tensors attached to parameters
        for param in gaussians.parameters():
            if param.grad is not None:
                param.grad = None

        # First cleanup pass
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # More aggressive cleanup: remove gradient computation and force memory release
        # But avoid moving essential tensors that might break save_ply
        print(f"  [aggressive-cleanup] Performing additional memory cleanup...", flush=True)
        cpu_backup = {}
        
        # Disable gradients and clear caches before PLY saving
        # NOTE: Do NOT modify utils.WORLD_SIZE or utils.GLOBAL_RANK here.
        # save_ply(local_only=True) handles distributed ops internally.
        # Changing globals causes race conditions with SIGTERM handlers
        # that read WORLD_SIZE for state file writes.
        print(f"  [aggressive-cleanup] Disabling gradients and clearing caches for PLY saving...", flush=True)

        try:
            torch.set_grad_enabled(False)  # Temporarily disable gradients

            # Clear any remaining optimizer states if they exist
            if hasattr(gaussians, 'optimizer_state'):
                gaussians.optimizer_state = None

        except Exception as e:
            print(f"  [aggressive-cleanup] Warning: Additional cleanup failed: {e}", flush=True)

        # Multiple cleanup passes - sometimes tensors are not freed immediately
        for i in range(3):
            gc.collect()
            torch.cuda.synchronize() 
            torch.cuda.empty_cache()

        # Report memory status after aggressive cleanup
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [aggressive-cleanup] After cleanup. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB", flush=True)

        # PLY prefix for wrapper to find all rank files
        ply_prefix_a = str(tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{iteration}")
        ply_prefix_b = str(tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{iteration}")

        # IMPORTANT: Save tile B FIRST, then tile A
        # Reason: Child B tends to be larger due to non-uniform gaussian distribution.
        # If SIGKILL arrives during save, we want the larger tile (B) to be saved first.

        # Save tile B first - directly to final path (no temp file to avoid race condition)
        # save_ply with local_only=True appends _rank{N} to the path
        print(f"  Saving tile B ({tile_b_id}) - rank {utils.GLOBAL_RANK}...", flush=True)
        final_base_b = tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{iteration}.ply"
        count_b = gaussians.save_ply(str(final_base_b), filter_bbox=tile_b, local_only=True)
        final_path_b = tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{iteration}_rank{utils.GLOBAL_RANK}.ply"
        if count_b > 0:
            print(f"  [SAVED] Tile B rank{utils.GLOBAL_RANK}: {count_b:,} gaussians -> {final_path_b.name}", flush=True)
        else:
            if final_path_b.exists():
                final_path_b.unlink(missing_ok=True)
            print(f"  [SKIP] Tile B rank{utils.GLOBAL_RANK}: 0 gaussians", flush=True)

        # Clear cache again before second save
        gc.collect()
        torch.cuda.empty_cache()

        # Save tile A second - directly to final path
        print(f"  Saving tile A ({tile_a_id}) - rank {utils.GLOBAL_RANK}...", flush=True)
        final_base_a = tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{iteration}.ply"
        count_a = gaussians.save_ply(str(final_base_a), filter_bbox=tile_a, local_only=True)
        final_path_a = tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{iteration}_rank{utils.GLOBAL_RANK}.ply"
        if count_a > 0:
            print(f"  [SAVED] Tile A rank{utils.GLOBAL_RANK}: {count_a:,} gaussians -> {final_path_a.name}", flush=True)
        else:
            if final_path_a.exists():
                final_path_a.unlink(missing_ok=True)
            print(f"  [SKIP] Tile A rank{utils.GLOBAL_RANK}: 0 gaussians", flush=True)

        # Print split comparison
        total_local = count_a + count_b
        if total_local > 0:
            pct_a = 100.0 * count_a / total_local
            pct_b = 100.0 * count_b / total_local
            print(f"\n[adaptive-tile] Rank {utils.GLOBAL_RANK} split summary:", flush=True)
            print(f"  Child A ({tile_a_id}): {count_a:,} gaussians ({pct_a:.1f}%)", flush=True)
            print(f"  Child B ({tile_b_id}): {count_b:,} gaussians ({pct_b:.1f}%)", flush=True)
            print(f"  Total: {total_local:,} gaussians", flush=True)
        else:
            print(f"  Rank {utils.GLOBAL_RANK} finished saving. Total local: 0 gaussians", flush=True)

        # Re-enable gradients after PLY saving
        try:
            torch.set_grad_enabled(True)
            print(f"  [aggressive-cleanup] Re-enabled gradients after PLY saving", flush=True)
        except Exception as e:
            print(f"  [aggressive-cleanup] Warning: Failed to restore settings: {e}", flush=True)

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
        "oom_category": oom_category,
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

    utils.print_rank_0(f"[adaptive-tile] Saved OOM state to {state_file} (num_ranks={world_size}, WORLD_SIZE={utils.WORLD_SIZE})")
    log_file.write(f"[adaptive-tile] Saved OOM state to {state_file} (num_ranks={world_size})\n")
    log_file.flush()

    # Write done file for this rank (even for Category 1/2, to unblock other ranks)
    # Include per-child counts for merge validation
    total_saved = count_a + count_b
    write_oom_done(args, utils.GLOBAL_RANK, total_saved, count_a=count_a, count_b=count_b)

    # Wait ONLY for acked ranks' done files (dynamic waiting, not fixed timeout)
    #
    # Key insight:
    # - acked_ranks: saw signal before backward(), will save and write done files
    # - non-acked_ranks: stuck in backward(), can ONLY save via SIGTERM after we exit
    #
    # We wait for acked_ranks because they WILL write done files (and quickly).
    # We do NOT wait for non-acked_ranks because:
    #   1. They can't write done files until they receive SIGTERM
    #   2. SIGTERM is only sent AFTER we exit
    #   3. Waiting for them would be waiting forever
    #
    # Timeline:
    #   1. OOM rank: signal -> ack wait -> save -> done -> wait for acked ranks' done -> exit
    #   2. Acked ranks: see signal -> ack -> save -> done -> exit (before OOM rank exits)
    #   3. Stuck ranks: in backward() -> (OOM rank exits) -> SIGTERM -> save -> exit
    #
    # With this approach:
    # - Acked ranks finish quickly, OOM rank proceeds without fixed delay
    # - Stuck ranks get full SIGTERM grace period from torchrun (default ~30s)

    if robust_used and is_category3:
        wait_timeout = float(os.environ.get('PLY_WAIT_TIMEOUT', 3600))
        target_ranks = list(range(utils.WORLD_SIZE))
        print(f"\n[robust-oom] Waiting for done files from all ranks {target_ranks} (timeout {wait_timeout}s)...", flush=True)
        all_done = check_all_ranks_done(args, utils.WORLD_SIZE, timeout=wait_timeout, target_ranks=target_ranks)
        if all_done:
            print(f"[robust-oom] All ranks completed saving. Safe to exit.", flush=True)
        else:
            print(f"[robust-oom] FATAL - TIMEOUT after {wait_timeout:.0f}s waiting for all ranks!", flush=True)
            print(f"[robust-oom] Some ranks did not complete saving.", flush=True)
    elif acked_ranks:
        print(f"\n[oom-signal] Waiting for acked ranks {acked_ranks} to complete saving...", flush=True)
        # Wait indefinitely for acked ranks - they MUST complete saving.
        # Using 1 hour timeout as safety net, but should never be reached.
        all_done = check_all_ranks_done(args, utils.WORLD_SIZE, timeout=3600.0, target_ranks=acked_ranks)
        if all_done:
            print(f"[oom-signal] All acked ranks completed! Safe to exit.", flush=True)
        else:
            print(f"[oom-signal] FATAL - TIMEOUT after 1 hour waiting for acked ranks!", flush=True)
            print(f"[oom-signal] This indicates a serious problem - some ranks may have crashed.", flush=True)
    else:
        print(f"\n[oom-signal] No acked ranks to wait for (all stuck in backward())", flush=True)
        print(f"[oom-signal] Stuck ranks will save via SIGTERM handler after exit.", flush=True)

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

    # Per-rank gaussian count tracker — initialized as early as possible so any
    # OOM/exception in pre-training stages also leaves a JSONL trace. Must come
    # after Scene init (gaussians._xyz exists) but before any optional pre-flight.
    _count_tracker = None
    try:
        from utils.count_tracker import init_tracker as _init_count_tracker
        _ct_tile_id = getattr(args, "tile_id", "tile")
        _ct_out_dir = Path(getattr(args, "tile_output_dir", "") or args.model_path).parent / "visualizations" / "count_timeline"
        _count_tracker = _init_count_tracker(utils.GLOBAL_RANK, _ct_tile_id, _ct_out_dir, sample_every=20)
        _count_tracker.record_event(0, gaussians._xyz.shape[0], "start")
    except Exception as _ct_e:
        print(f"[count_tracker] init skipped: {_ct_e}", flush=True)

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
    args._num_visible_cameras = len(train_dataset.cameras)

    # Quality 기반 done: epoch_loss(카메라 한 바퀴의 평균 손실) 정체 감지로 조기 종료.
    # 단일 iter 의 loss 는 카메라마다 달라 비교 불가 → 반드시 에폭(N뷰) 단위 통계로 비교.
    # epoch_loss 는 all-reduce 된 동일 값에서 계산되므로 모든 rank 가 같은 결정을 내림.
    quality_done_enabled = (
        args.adaptive_tile_enabled
        and os.environ.get("QUALITY_DONE", "0") == "1"
    )
    quality_converged = False  # done_info 기록에서 참조하므로 비활성 시에도 정의
    if quality_done_enabled:
        # dnq ConvergenceDetector 이식 (etc/Grendel-GS dnq 브랜치, user 검증 설계):
        # best 에폭 평균 + patience — 개선(= best 보다 threshold 이상 낮은 에폭 평균)이
        # patience 에폭 연속 없으면 조기 done. densification 게이트 없음 (dnq 실전 검증).
        quality_done_min_iter = int(os.environ.get("QUALITY_DONE_MIN_ITER", "1000"))
        quality_done_threshold = float(os.environ.get("QUALITY_DONE_LOSS_THRESHOLD", "0.0001"))
        _n_cam = max(1, train_dataset.camera_size)
        _qd_pat_env = os.environ.get("QUALITY_DONE_PATIENCE", "").strip()
        if _qd_pat_env:
            quality_done_patience = int(_qd_pat_env)
        else:
            # patience = 카메라 수 power function (2뷰→60에폭, 30뷰→15에폭, clamp[5,100])
            # 기존 dnq fitting (2뷰→90, 30뷰→20):
            # _b = _math.log(90.0 / 20.0) / _math.log(2.0 / 30.0)
            # _a = 90.0 / (2.0 ** _b)
            import math as _math
            _b = _math.log(60.0 / 15.0) / _math.log(2.0 / 30.0)
            _a = 60.0 / (2.0 ** _b)
            _cc = float(min(max(_n_cam, 2), 30))
            quality_done_patience = max(5, min(int(_a * (_cc ** _b)), 100))
        # Opacity reset 직후 회복 구간은 무개선 카운트에서 제외 (grace).
        # 리셋은 loss 를 인위적으로 튀게 하므로, 회복이 patience 보다 느리면
        # 가짜 CONVERGED 가 발생 (7/21 iter 3240 실측). 회복 중 개선(기록 갱신)은 그대로 인정.
        quality_done_reset_grace = int(os.environ.get("QUALITY_DONE_RESET_GRACE", "500"))
        _qd_reset_interval = max(1, int(getattr(opt_args, "opacity_reset_interval", 3000)))
        _qd_reset_until = int(getattr(opt_args, "opacity_reset_until_iter",
                                      getattr(opt_args, "densify_until_iter", 15000)))
        quality_best_epoch_loss = float("inf")
        quality_best_epoch_num = 0
        quality_epochs_since_improvement = 0
        quality_done_last_epoch = 0
        utils.print_rank_0(
            f"[quality-done] enabled (best+patience): min_iter={quality_done_min_iter}, "
            f"threshold={quality_done_threshold}, patience={quality_done_patience} epochs "
            f"(cameras={_n_cam}, epoch={_n_cam} iters), "
            f"reset_grace={quality_done_reset_grace} iters after each opacity reset "
            f"(interval={_qd_reset_interval}, until={_qd_reset_until})"
        )
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

    # PLY save callback for robust OOM handler
    def save_ply_callback(iteration: int, rank: int) -> bool:
        """Save PLY files for robust OOM handler. Returns True if successful."""
        try:
            # Get tile split configuration
            tile_output_dir = Path(getattr(args, "tile_output_dir", "") or args.model_path)
            tile_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get current tile info from args
            current_tile_id = getattr(args, "tile_id", "tile_0000")
            parent_level = getattr(args, "tile_level", 0)
            child_level = parent_level + 1
            
            # Get tile bbox from args if available
            tile_bbox_str = getattr(args, "tile_bbox", None)
            print(f"[robust-oom] Rank {rank} DEBUG: tile_bbox_str = {tile_bbox_str}, type = {type(tile_bbox_str)}", flush=True)
            
            if tile_bbox_str is None:
                print(f"[robust-oom] Rank {rank} no tile_bbox found, saving all gaussians", flush=True)
                # Save all gaussians
                final_base = tile_output_dir / f"{current_tile_id}_L{parent_level}_oom_iter{iteration}.ply"
                count = gaussians.save_ply(str(final_base), local_only=True)
                final_path = tile_output_dir / f"{current_tile_id}_L{parent_level}_oom_iter{iteration}_rank{rank}.ply"
                count_a, count_b = 0, 0
            else:
                # Convert string to TileBBox object if needed
                from scene.adaptive_tile_utils import TileBBox
                if isinstance(tile_bbox_str, str):
                    print(f"[robust-oom] Rank {rank} converting string '{tile_bbox_str}' to TileBBox", flush=True)
                    tile_bbox = TileBBox.from_string(tile_bbox_str)
                else:
                    print(f"[robust-oom] Rank {rank} using existing TileBBox object: {tile_bbox_str}", flush=True)
                    tile_bbox = tile_bbox_str
                
                print(f"[robust-oom] Rank {rank} about to call split() on {tile_bbox}", flush=True)
                # Split and save tiles
                split_result = tile_bbox.split()
                print(f"[robust-oom] Rank {rank} split() returned: {split_result}, type: {type(split_result)}", flush=True)
                
                # Safely unpack split result
                if isinstance(split_result, tuple) and len(split_result) == 2:
                    tile_a, tile_b = split_result
                else:
                    print(f"[robust-oom] Rank {rank} ERROR: split() returned unexpected format: {split_result}", flush=True)
                    print(f"[robust-oom] Rank {rank} Expected tuple with 2 elements, got {type(split_result)} with {len(split_result) if hasattr(split_result, '__len__') else 'unknown'} elements", flush=True)
                    return False
                
                # Generate new tile IDs
                tile_num = int(current_tile_id.split("_")[-1]) if "_" in current_tile_id else 0
                base_id = "_".join(current_tile_id.split("_")[:-1]) if "_" in current_tile_id else "tile"
                tile_a_id = f"{base_id}_{(tile_num * 2 + 1):04d}"
                tile_b_id = f"{base_id}_{(tile_num * 2 + 2):04d}"
                
                print(f"[robust-oom] Rank {rank} saving split tiles: {tile_a_id}, {tile_b_id}", flush=True)
                
                # Save tile B first (larger tile)
                final_base_b = tile_output_dir / f"{tile_b_id}_L{child_level}_oom_iter{iteration}.ply"
                count_b = gaussians.save_ply(str(final_base_b), filter_bbox=tile_b, local_only=True)
                
                # Save tile A second
                final_base_a = tile_output_dir / f"{tile_a_id}_L{child_level}_oom_iter{iteration}.ply"
                count_a = gaussians.save_ply(str(final_base_a), filter_bbox=tile_a, local_only=True)
                
                count = count_a + count_b
                print(f"[robust-oom] Rank {rank} saved A:{count_a}, B:{count_b} gaussians", flush=True)

                # Per-rank Cat3 OOM 시각화 (학습/저장 흐름을 막지 않도록 best-effort)
                try:
                    from utils.oom_viz import save_per_rank_cat3_viz
                    snap = getattr(gaussians, "_oom_cpu_snapshot", None)
                    if snap is not None and snap.get("xyz") is not None:
                        xyz_local = snap["xyz"].detach().cpu().numpy()
                    else:
                        xyz_local = gaussians._xyz.detach().cpu().numpy()
                    viz_dir = tile_output_dir.parent / "visualizations" / "cat3_oom"
                    save_per_rank_cat3_viz(
                        xyz_local,
                        rank=rank,
                        iteration=iteration,
                        tile_id=current_tile_id,
                        parent_bbox=tile_bbox,
                        tile_a=tile_a,
                        tile_b=tile_b,
                        out_dir=viz_dir,
                        count_a=count_a,
                        count_b=count_b,
                    )
                except Exception as viz_e:
                    print(f"[robust-oom] Rank {rank} viz skipped: {viz_e}", flush=True)

            # Write done file for robust handler so wrapper can wait/merge
            try:
                write_oom_done(args, rank, count, count_a=count_a, count_b=count_b)
            except Exception as e:
                print(f"[robust-oom] Rank {rank} WARNING: failed to write done file: {e}", flush=True)

            # Count timeline: OOM event 기록 + flush
            try:
                from utils.count_tracker import get_tracker as _get_ct
                _ct = _get_ct()
                if _ct is not None:
                    try:
                        _cur = gaussians._xyz.shape[0]
                    except Exception:
                        _cur = -1
                    _ct.record_event(iteration, _cur, "oom_cat3")
                    _ct.flush()
            except Exception as _ct_e:
                print(f"[count_tracker] flush at OOM skipped: {_ct_e}", flush=True)

            return count > 0
            
        except Exception as e:
            print(f"[robust-oom] FATAL: Rank {rank} PLY save failed: {e}", flush=True)
            return False
    
    # Function to check and handle OOM signal from robust handler (backward compatibility)
    def check_and_handle_oom(iteration, phase="unknown"):
        """Check for OOM signal and handle if found. Returns True if OOM detected."""
        if not getattr(args, "adaptive_tile_enabled", False):
            return False
            
        # Check robust OOM handler first
        robust_handler = get_robust_oom_handler()
        if robust_handler:
            oom_signal = robust_handler.check_oom_signal()
            if oom_signal:
                print(f"\n[{_ts()}] [robust-oom] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Detected robust OOM signal! ({phase})", flush=True)
                log_file.write(f"[{_ts()}] [robust-oom] Rank {utils.GLOBAL_RANK} detected robust signal at iter {iteration} ({phase})\n")
                
                # Handle OOM through robust handler - it will call save_ply_callback and exit
                # No need to do anything else here as the signal handler will take over
                return True
        
        # Fallback to legacy OOM signal check for backward compatibility
        oom_signal = check_oom_signal(args)
        if oom_signal and oom_signal.get("signaling_rank") != utils.GLOBAL_RANK:
            print(f"\n[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Detected legacy OOM signal! ({phase})", flush=True)
            log_file.write(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} detected legacy signal at iter {iteration} ({phase})\n")
            
            # IMMEDIATELY write ack file
            write_oom_ack(args, utils.GLOBAL_RANK)
            
            handled = handle_oom_signal_from_other_rank(
                oom_signal, gaussians, args, iteration, log_file
            )
            if handled:
                log_file.flush()
                tile_manager.finalize(gaussians)
                progress_bar.clear()
                progress_bar.disable = True
                progress_bar.close()
                print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
                sys.exit(EXIT_CODE_OOM)
            return True
        return False

    # NOTE: Removed the DEBUG-IMG pre-loop that previously rendered all visible cameras
    # before training. It used the same distributed render path as a training step, so it
    # cost the same memory as one full training iteration on a giant image and routinely
    # caused CUDA OOM at the very start (before robust_oom_handler / count_tracker were
    # initialized, masking the real Cat 3 behavior we want to study). GT-vs-rendered debug
    # images are still saved during the training loop at the iterations specified by
    # DEBUG_SAVE_ITERS via should_save_debug_images / save_debug_images, which is the
    # correct (and safe) place because the training loop's OOM handler protects them.

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
    # Always enable adaptive tile mode if we're in a tile training (has tile_id)
    adaptive_tile_mode = getattr(args, "adaptive_tile_enabled", False) or hasattr(args, "tile_id")
    current_iteration = start_from_this_iteration

    # Initialize robust OOM handler for distributed OOM synchronization
    if adaptive_tile_mode:
        print(f"[robust-oom] Rank {utils.GLOBAL_RANK} initializing robust OOM handler (adaptive_tile_enabled={getattr(args, 'adaptive_tile_enabled', False)}, has_tile_id={hasattr(args, 'tile_id')}, tile_id={getattr(args, 'tile_id', 'None')})", flush=True)
        initialize_robust_oom_handler(args, utils.GLOBAL_RANK, utils.WORLD_SIZE, save_ply_callback)
        print(f"[robust-oom] Rank {utils.GLOBAL_RANK} robust OOM handler initialized successfully", flush=True)

        # Legacy SIGTERM handler as fallback (for backward compatibility)
        install_sigterm_handler()
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

            # Check for OOM signal at iteration start (CRITICAL POINT 1)
            if adaptive_tile_mode:
                check_and_handle_oom(iteration, "iter_start")

            # Periodic CPU snapshot for OOM-safe PLY saving
            if adaptive_tile_mode:
                try:
                    snap_interval = int(os.environ.get("OOM_CPU_SNAPSHOT_INTERVAL", "50"))
                except Exception:
                    snap_interval = 50
                if snap_interval > 0 and iteration % snap_interval == 0:
                    gaussians.update_oom_cpu_snapshot(iteration=iteration, reason="periodic")

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
                
                # Check for OOM signal after loading images (CRITICAL POINT 3)
                if adaptive_tile_mode:
                    check_and_handle_oom(iteration, "after_load_images")
                    
                if debug_first_iters:
                    utils.print_rank_0(f"[DEBUG] [{iteration}] Cameras loaded to GPU")

            if debug_first_iters:
                utils.print_rank_0(f"[DEBUG] [{iteration}] Starting rendering (backend={args.backend})...")

            # Check for OOM signal before forward pass (preprocessing/rendering)
            if adaptive_tile_mode:
                check_and_handle_oom(iteration, "before_forward_pass")

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
                    print(f"\n[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Detected OOM signal! (post-render)", flush=True)
                    log_file.write(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} detected signal at iter {iteration} (post-render)\n")
                    # IMMEDIATELY write ack file - this tells the OOM rank we won't enter backward()
                    write_oom_ack(args, utils.GLOBAL_RANK)
                    handled = handle_oom_signal_from_other_rank(
                        oom_signal, gaussians, args, iteration, log_file
                    )
                    if handled:
                        log_file.flush()
                        tile_manager.finalize(gaussians)
                        progress_bar.clear()
                        progress_bar.disable = True
                        progress_bar.close()
                        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
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
                    utils.print_rank_0(f"[DEBUG-IMG] Saving to {output_path}/visualizations/gt_compare/ tile={tile_id} iter={iteration}")
                    save_debug_images(
                        batched_image,
                        batched_cameras,
                        batched_strategies,
                        iteration,
                        output_path,
                        tile_id=tile_id,
                        args=args,
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
                    print(f"\n[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Detected OOM signal! (pre-loss)", flush=True)
                    log_file.write(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} detected signal at iter {iteration} (pre-loss)\n")
                    log_file.flush()
                    # IMMEDIATELY write ack file - this tells the OOM rank we won't enter backward()
                    write_oom_ack(args, utils.GLOBAL_RANK)
                    handled = handle_oom_signal_from_other_rank(
                        oom_signal, gaussians, args, iteration, log_file
                    )
                    if handled:
                        log_file.flush()
                        tile_manager.finalize(gaussians)
                        progress_bar.clear()
                        progress_bar.disable = True
                        progress_bar.close()
                        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
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
                    print(f"\n[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Detected OOM signal! (pre-backward)", flush=True)
                    log_file.write(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} detected signal at iter {iteration} (pre-backward)\n")
                    log_file.flush()
                    # IMMEDIATELY write ack file - this tells the OOM rank we won't enter backward()
                    write_oom_ack(args, utils.GLOBAL_RANK)
                    handled = handle_oom_signal_from_other_rank(
                        oom_signal, gaussians, args, iteration, log_file
                    )
                    if handled:
                        log_file.flush()
                        tile_manager.finalize(gaussians)
                        progress_bar.clear()
                        progress_bar.disable = True
                        progress_bar.close()
                        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
                        sys.exit(EXIT_CODE_OOM)

            # CRITICAL: Synchronize all ranks before backward to ensure no rank is stuck
            # This prevents deadlock when one rank hits OOM while another enters backward
            if adaptive_tile_mode and utils.DEFAULT_GROUP.size() > 1:
                try:
                    # Synchronize all ranks before backward
                    torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                except RuntimeError as e:
                    # Barrier failure might indicate rank sync issues
                    print(f"[{_ts()}] [barrier-error] Rank {utils.GLOBAL_RANK} @ iter {iteration}: Pre-backward barrier failed: {e}", flush=True)
                    # Check for OOM signal again
                    oom_signal = check_oom_signal(args)
                    if oom_signal:
                        print(f"[{_ts()}] [oom-signal] Rank {utils.GLOBAL_RANK}: OOM signal found after barrier timeout", flush=True)
                        write_oom_ack(args, utils.GLOBAL_RANK)
                        handled = handle_oom_signal_from_other_rank(
                            oom_signal, gaussians, args, iteration, log_file
                        )
                        if handled:
                            sys.exit(EXIT_CODE_OOM)
                    # If no OOM signal, re-raise the original error
                    raise e
            
            set_current_operation("backward")
            log_gpu_memory("before_backward")
            timers.start("backward")
            loss_sum.backward()
            timers.stop("backward")
            log_gpu_memory("after_backward")
            utils.check_initial_gpu_memory_usage("after backward")
            
            # Check for OOM signal after backward pass (CRITICAL POINT 2)
            if adaptive_tile_mode:
                check_and_handle_oom(iteration, "after_backward")

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

                # Quality 기반 done (dnq best+patience): 새로 완성된 에폭마다
                # best 대비 threshold 이상 개선 여부를 판정, patience 에폭 연속 무개선이면 종료
                if quality_done_enabled and len(train_dataset.epoch_loss) > quality_done_last_epoch:
                    _el = train_dataset.epoch_loss
                    _in_grace = False
                    for _e_idx in range(quality_done_last_epoch, len(_el)):
                        _e_avg = float(_el[_e_idx])
                        # 이 에폭이 opacity reset 직후 grace 구간에 걸치는지
                        _e_iter = (_e_idx + 1) * _n_cam
                        _in_grace = (
                            _e_iter >= _qd_reset_interval
                            and (_e_iter % _qd_reset_interval) <= quality_done_reset_grace
                            and _e_iter <= _qd_reset_until + quality_done_reset_grace
                        )
                        if _e_avg < quality_best_epoch_loss - quality_done_threshold:
                            quality_best_epoch_loss = _e_avg
                            quality_best_epoch_num = _e_idx + 1
                            quality_epochs_since_improvement = 0
                        elif not _in_grace:
                            quality_epochs_since_improvement += 1
                        # grace 중 무개선 에폭은 카운트하지 않음 (회복 시간 보장)
                    quality_done_last_epoch = len(_el)
                    utils.print_rank_0(
                        f"[quality-done] epoch {quality_done_last_epoch} (iter {iteration}): "
                        f"avg={float(_el[-1]):.6f} best={quality_best_epoch_loss:.6f}"
                        f"@E{quality_best_epoch_num} "
                        f"no_improve={quality_epochs_since_improvement}/{quality_done_patience}"
                        + (" (reset-grace)" if _in_grace else "")
                    )
                    if (iteration >= quality_done_min_iter
                            and quality_epochs_since_improvement >= quality_done_patience):
                        quality_converged = True
                if quality_done_enabled and quality_converged:
                    utils.print_rank_0(
                        f"\n[quality-done] CONVERGED at iteration {iteration} "
                        f"(epoch {quality_done_last_epoch}/{train_dataset.camera_size} views) "
                        f"— early done"
                    )
                    log_file.write(
                        f"[quality-done] converged at iter {iteration}, "
                        f"epoch {quality_done_last_epoch}\n"
                    )
                    progress_bar.close()
                    break
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
                    # Check for OOM signal before densification (CRITICAL POINT 4)
                    if adaptive_tile_mode:
                        check_and_handle_oom(iteration, "before_densification")
                    
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
                    
                    # Check for OOM signal after densification (CRITICAL POINT 5)
                    if adaptive_tile_mode:
                        check_and_handle_oom(iteration, "after_densification")
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
                        if _count_tracker is not None:
                            _count_tracker.record_event(
                                iteration, num_gaussians_after,
                                f"densify_{sign}{local_delta}",
                            )

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

            if _count_tracker is not None:
                try:
                    _count_tracker.maybe_record(iteration, gaussians._xyz.shape[0])
                except Exception:
                    pass

    except KeyboardInterrupt:
        # Handle KeyboardInterrupt from OOM signal monitor thread (_thread.interrupt_main())
        # This is used to force the main thread out of blocking collective operations for PLY saving
        if adaptive_tile_mode:
            monitor = get_oom_signal_monitor()
            if monitor and monitor.signal_detected and monitor.main_thread_interrupted:
                signal_data = monitor.detected_signal_data
                print(f"\n[{_ts()}] [oom-interrupt] Rank {utils.GLOBAL_RANK} received interrupt from OOM monitor", flush=True)
                log_file.write(f"[{_ts()}] [oom-interrupt] Rank {utils.GLOBAL_RANK} handling interrupt from OOM monitor\n")

                # Handle the OOM signal - this will save PLY for Category 3
                handled = handle_oom_signal_from_other_rank(
                    signal_data, gaussians, args, current_iteration, log_file
                )
                if handled:
                    log_file.flush()
                    tile_manager.finalize(gaussians)
                    progress_bar.clear()
                    progress_bar.disable = True
                    progress_bar.close()
                    print(f"[{_ts()}] [oom-interrupt] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
                    sys.exit(EXIT_CODE_OOM)

        # If not from OOM monitor, re-raise as normal KeyboardInterrupt
        print(f"\n[{_ts()}] [interrupt] Rank {utils.GLOBAL_RANK} received KeyboardInterrupt (not from OOM monitor)", flush=True)
        raise

    except Exception as e:
        # Log exception details for debugging
        print(f"\n[{_ts()}] [exception] Rank {utils.GLOBAL_RANK} caught exception: {type(e).__name__}", flush=True)
        print(f"[exception] Message: {str(e)[:300]}", flush=True)
        is_timeout = _is_nccl_timeout_error(e)
        is_oom = _is_oom_error(e)
        print(f"[exception] is_nccl_timeout={is_timeout}, is_oom={is_oom}, adaptive_tile_mode={adaptive_tile_mode}", flush=True)

        # Handle NCCL timeout in adaptive tile mode
        # This happens when another rank died (e.g., OOM) and this rank's collective operation timed out
        if adaptive_tile_mode and is_timeout:
            print(f"\n[{_ts()}] [nccl-timeout] Rank {utils.GLOBAL_RANK} caught NCCL timeout at iteration {current_iteration}", flush=True)
            print(f"[nccl-timeout] Error: {str(e)[:200]}", flush=True)
            log_file.write(f"[{_ts()}] [nccl-timeout] Rank {utils.GLOBAL_RANK} caught NCCL timeout: {str(e)[:200]}\n")

            # Check if there's an OOM signal from another rank
            oom_signal = check_oom_signal(args)
            if oom_signal and oom_signal.get("signaling_rank") != utils.GLOBAL_RANK:
                print(f"[nccl-timeout] Found OOM signal from rank {oom_signal.get('signaling_rank')}", flush=True)
                log_file.write(f"[nccl-timeout] Found OOM signal from rank {oom_signal.get('signaling_rank')}\n")

                # Write ack (in case monitor didn't)
                write_oom_ack(args, utils.GLOBAL_RANK)

                # Handle the OOM signal - this will save PLY for Category 3
                handled = handle_oom_signal_from_other_rank(
                    oom_signal, gaussians, args, current_iteration, log_file
                )
                if handled:
                    log_file.flush()
                    tile_manager.finalize(gaussians)
                    progress_bar.clear()
                    progress_bar.disable = True
                    progress_bar.close()
                    print(f"[{_ts()}] [nccl-timeout] Rank {utils.GLOBAL_RANK} exiting with code {EXIT_CODE_OOM}", flush=True)
                    sys.exit(EXIT_CODE_OOM)
            else:
                # No OOM signal - this is a real NCCL error, re-raise
                print(f"[nccl-timeout] No OOM signal found - this is a real NCCL error", flush=True)
                log_file.write(f"[nccl-timeout] No OOM signal found - re-raising\n")
                log_file.flush()
                raise

        # Handle OOM in adaptive tile mode
        if adaptive_tile_mode and is_oom:
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
                # Read the OOM category from the state file to determine if we should exit
                tile_output_dir = Path(getattr(args, "tile_output_dir", "output/adaptive_test/ply"))
                state_file = getattr(args, "tile_state_file", "") or (tile_output_dir / "adaptive_tile_state.json")
                state_file = Path(state_file)
                
                oom_category = None
                if state_file.exists():
                    try:
                        with open(state_file, 'r') as f:
                            state_data = json.load(f)
                            oom_category = state_data.get("oom_category", None)
                            utils.print_rank_0(f"[adaptive-tile] OOM category {oom_category} detected from state file")
                    except Exception as e:
                        utils.print_rank_0(f"[adaptive-tile] Warning: Failed to read OOM category from state file: {e}")
                
                # Only exit for Category 3 OOM (which requires PLY save and resume)
                # Category 1 and 2 should continue with tile splitting in the wrapper
                if oom_category == 3:
                    utils.print_rank_0(f"[adaptive-tile] Category 3 OOM: Exiting with code {EXIT_CODE_OOM} for wrapper to handle PLY resume")
                    log_file.flush()
                    tile_manager.finalize(gaussians)
                    # Clear and close progress bar silently (don't print final state on OOM)
                    progress_bar.clear()
                    progress_bar.disable = True
                    progress_bar.close()
                    sys.exit(EXIT_CODE_OOM)
                else:
                    # For Category 1 and 2, we still need to exit but the wrapper will handle tile splitting
                    utils.print_rank_0(f"[adaptive-tile] Category {oom_category if oom_category else 'unknown'} OOM: Exiting with code {EXIT_CODE_OOM} for wrapper to handle tile splitting")
                    log_file.flush()
                    tile_manager.finalize(gaussians)
                    # Clear and close progress bar silently (don't print final state on OOM)
                    progress_bar.clear()
                    progress_bar.disable = True
                    progress_bar.close()
                    sys.exit(EXIT_CODE_OOM)

        # Re-raise if not OOM or not handled
        print(f"[exception] Rank {utils.GLOBAL_RANK} re-raising unhandled exception: {type(e).__name__}", flush=True)
        raise

    # Finish training
    if args.adaptive_tile_enabled:
        # 완료 타일 최종 저장. scene.save() 는 rank 0 으로의 collective gather 를
        # 포함하므로 모든 rank 가 함께 호출해야 함 (rank guard 금지)
        # quality-done 조기 종료 시 실제 마지막 iteration 으로 저장
        final_iteration = locals().get("iteration", opt_args.iterations)
        utils.print_rank_0(
            f"[adaptive-tile] Training complete — saving final gaussians at iteration {final_iteration}"
        )
        torch.cuda.empty_cache()
        scene.save(final_iteration)
        log_file.write(
            f"[ITER {final_iteration}] Saving final Gaussians (adaptive mode)\n"
        )
        # 타일 품질 지표 기록 — 타일 간 품질 편차 측정/히트맵용.
        # epoch_loss 는 rank 공통이므로 rank 0 만 기록.
        if utils.GLOBAL_RANK == 0:
            try:
                _el = train_dataset.epoch_loss
                _tail = _el[-3:] if _el else []
                done_info = {
                    "final_iteration": int(final_iteration),
                    "num_cameras": int(train_dataset.camera_size),
                    "epochs": len(_el),
                    "final_epoch_loss": (float(sum(_tail) / len(_tail)) if _tail else None),
                    "done_reason": "converged" if quality_converged else "iter_end",
                }
                if quality_done_enabled and quality_best_epoch_loss != float("inf"):
                    done_info["best_epoch_loss"] = float(quality_best_epoch_loss)
                    done_info["best_epoch_num"] = int(quality_best_epoch_num)
                    done_info["patience_epochs"] = int(quality_done_patience)
                with open(os.path.join(args.model_path, "done_info.json"), "w") as f:
                    json.dump(done_info, f, indent=2)
                utils.print_rank_0(f"[adaptive-tile] done_info: {done_info}")
            except Exception as e:
                print(f"[adaptive-tile] WARNING: done_info 기록 실패: {e}", flush=True)
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)
    log_file.write(
        "Max Memory usage: {} GB.\n".format(
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        )
    )
    
    # Cleanup robust OOM handler
    if adaptive_tile_mode:
        robust_handler = get_robust_oom_handler()
        if robust_handler:
            print(f"[robust-oom] Rank {utils.GLOBAL_RANK} cleaning up robust OOM handler", flush=True)
            robust_handler.cleanup()
    
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

    # OOM signal monitor thread removed - no cleanup needed
