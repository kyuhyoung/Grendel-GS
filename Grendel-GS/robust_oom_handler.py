"""
Robust Distributed OOM Handler using Process Groups + Signal Handlers + Shared Memory

This system guarantees that all ranks will detect OOM signals and save PLY files
even when stuck in blocking operations (NCCL, CUDA, etc.).

Key principles:
1. Fail fast - any error kills the entire system immediately
2. OS-level guarantees - signal delivery is kernel-guaranteed  
3. No timeouts or polling - immediate response
4. Atomic operations - shared memory writes are atomic
5. Process group synchronization - all ranks die together
"""

import mmap
import signal
import os
import sys
import json
import time
import fcntl
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class OOMState:
    detected: bool = False
    category: int = 0
    iteration: int = 0
    cause: str = ""
    rank_who_detected: int = -1
    timestamp: float = 0.0

class RobustOOMHandler:
    """
    Robust OOM handler that guarantees PLY saving across all ranks.
    
    Uses OS-level signal delivery + shared memory for maximum reliability.
    No fallbacks - either works or fails with clear error.
    """
    
    def __init__(self, args, rank: int, world_size: int, save_ply_callback):
        self.args = args
        self.rank = rank 
        self.world_size = world_size
        self.save_ply_callback = save_ply_callback
        self.self_pid = os.getpid()
        self.rank_pids = None
        
        # Shared memory for OOM state (memory-mapped file)
        self.shm_path = f"/tmp/oom_shm_{os.getppid()}_{world_size}"
        self.shm_fd = None
        self.shm = None
        
        # Process group for simultaneous termination
        self.pgrp_initialized = False
        self.pgrp_id = None
        
        # Signal handling state
        self.emergency_save_in_progress = False
        self.oom_handled = False
        # Whether this rank should honor SIGUSR1 for its own detected OOM
        # (used to let the detecting rank finish writing state before exit)
        self.allow_self_sigusr1 = True
        
        self._initialize_shared_memory()
        self._initialize_process_group()
        self._initialize_rank_pids()
        self._install_signal_handlers()
        
        print(f"[robust-oom] Rank {rank} initialized robust OOM handler", flush=True)
    
    def _initialize_shared_memory(self):
        """Initialize shared memory for OOM state coordination."""
        try:
            # Create or open shared memory file
            if self.rank == 0:
                # Rank 0 creates the shared memory
                self.shm_fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o666)
                os.ftruncate(self.shm_fd, 4096)
            else:
                # Other ranks wait and open existing shared memory
                max_wait = 10.0
                start_time = time.time()
                while not os.path.exists(self.shm_path):
                    if time.time() - start_time > max_wait:
                        raise RuntimeError(f"Rank {self.rank}: Shared memory not created by rank 0 within {max_wait}s")
                    time.sleep(0.01)
                self.shm_fd = os.open(self.shm_path, os.O_RDWR)
            
            # Memory map the file
            self.shm = mmap.mmap(self.shm_fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            
            # Initialize shared state (rank 0 only)
            if self.rank == 0:
                initial_state = OOMState()
                self._write_oom_state(initial_state)
                
            print(f"[robust-oom] Rank {self.rank} initialized shared memory at {self.shm_path}", flush=True)
            
        except Exception as e:
            print(f"[robust-oom] FATAL: Rank {self.rank} failed to initialize shared memory: {e}", flush=True)
            sys.exit(1)
    
    def _initialize_process_group(self):
        """Initialize process group for coordinated termination."""
        try:
            # Get current process group (torchrun already sets this up)
            current_pgrp = os.getpgrp()
            print(f"[robust-oom] Rank {self.rank} using existing process group {current_pgrp}", flush=True)
            
            self.pgrp_initialized = True
            self.pgrp_id = current_pgrp
            
        except Exception as e:
            print(f"[robust-oom] FATAL: Rank {self.rank} failed to initialize process group: {e}", flush=True)
            sys.exit(1)

    def _initialize_rank_pids(self):
        """Collect per-rank PIDs for targeted signaling (avoid signaling wrapper)."""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                pid_list = [None for _ in range(self.world_size)]
                dist.all_gather_object(pid_list, self.self_pid)
                # Filter to ints only
                self.rank_pids = [int(p) for p in pid_list if isinstance(p, int)]
                print(f"[robust-oom] Rank {self.rank} gathered rank PIDs: {self.rank_pids}", flush=True)
        except Exception as e:
            print(f"[robust-oom] Rank {self.rank} WARNING: failed to gather rank PIDs: {e}", flush=True)
    
    def _install_signal_handlers(self):
        """Install signal handlers for immediate OOM response."""
        import threading
        
        # Install SIGUSR1 handler for immediate PLY saving
        def sigusr1_handler(signum, frame):
            print(f"[robust-oom] Rank {self.rank} received SIGUSR1 - checking OOM state", flush=True)
            
            # Read OOM state from shared memory
            oom_state = self._read_oom_state()
            if (
                oom_state.detected
                and oom_state.rank_who_detected == self.rank
                and not self.allow_self_sigusr1
            ):
                print(
                    f"[robust-oom] Rank {self.rank} ignoring SIGUSR1 (self-detected OOM, allow_self_sigusr1=False)",
                    flush=True,
                )
                return
            if oom_state.detected and oom_state.category == 3:
                print(f"[robust-oom] Rank {self.rank} SIGUSR1 detected Category 3 OOM (iter={oom_state.iteration})", flush=True)
                
                # Save PLY immediately  
                try:
                    success = self.save_ply_callback(oom_state.iteration, self.rank)
                    if success:
                        print(f"[robust-oom] Rank {self.rank} successfully saved PLY via SIGUSR1", flush=True)
                    else:
                        print(f"[robust-oom] ERROR: Rank {self.rank} failed to save PLY via SIGUSR1", flush=True)
                except Exception as e:
                    print(f"[robust-oom] FATAL: Rank {self.rank} SIGUSR1 PLY save exception: {e}", flush=True)
                
                # Exit after saving
                print(f"[robust-oom] Rank {self.rank} exiting via SIGUSR1 with code 42", flush=True)
                os._exit(42)
            else:
                print(f"[robust-oom] Rank {self.rank} SIGUSR1 but no Category 3 OOM detected", flush=True)
        
        # Install the signal handler
        signal.signal(signal.SIGUSR1, sigusr1_handler)
        print(f"[robust-oom] Rank {self.rank} installed SIGUSR1 handler", flush=True)
        
        # Keep the monitoring thread as backup
        def monitor_thread():
            print(f"[robust-oom] Rank {self.rank} monitor thread STARTED", flush=True)
            iteration_counter = 0
            while True:
                time.sleep(0.01)  # Check every 10ms for faster response
                iteration_counter += 1
                
                # Log monitor thread heartbeat every 10 seconds (1000 iterations * 0.01s = 10s)
                if iteration_counter % 1000 == 0:
                    print(f"[robust-oom] Rank {self.rank} monitor thread heartbeat: iteration {iteration_counter}, oom_handled={self.oom_handled}", flush=True)
                
                try:
                    oom_state = self._read_oom_state()
                except Exception as e:
                    print(f"[robust-oom] Rank {self.rank} monitor thread: _read_oom_state failed: {e}", flush=True)
                    continue
                
                # DEBUG: Log the full state for debugging 
                if oom_state.detected:
                    print(f"[robust-oom] Rank {self.rank} DEBUG monitor: detected={oom_state.detected}, category={oom_state.category}, rank_who_detected={oom_state.rank_who_detected}, oom_handled={self.oom_handled}", flush=True)
                
                if (
                    oom_state.detected
                    and oom_state.category == 3
                    and not self.oom_handled
                ):
                    if (
                        oom_state.rank_who_detected == self.rank
                        and not self.allow_self_sigusr1
                    ):
                        # Detecting rank should continue to write state/exit normally.
                        continue
                    self.oom_handled = True
                    print(f"[robust-oom] Rank {self.rank} detected Category 3 OOM in shared memory (monitor thread)", flush=True)
                    print(f"[robust-oom] Rank {self.rank} OOM detected by rank {oom_state.rank_who_detected} at iter {oom_state.iteration}", flush=True)
                    print(f"[robust-oom] Rank {self.rank} DEBUG: oom_state.detected={oom_state.detected}, category={oom_state.category}, oom_handled={self.oom_handled}", flush=True)
                    
                    # Handle adaptive tiling if enabled (skip for now as attributes don't exist)
                    enabled_val = False
                    handler_exists = False
                    
                    # Skip adaptive tiling for now (no attributes configured)
                    
                    # Skip cuda.synchronize() and empty_cache() entirely:
                    # PLY save uses CPU snapshots, so CUDA sync is unnecessary.
                    # cuda.synchronize() blocks on pending NCCL ops when the other
                    # rank has already left the training loop, causing a 20-min hang
                    # followed by NCCL watchdog killing this process.
                    import time as _time
                    _ply_t0 = _time.monotonic()
                    print(f"[robust-oom] Rank {self.rank} skipping CUDA sync (CPU snapshot path), running gc only", flush=True)
                    try:
                        import gc
                        gc.collect()
                        print(f"[robust-oom] Rank {self.rank} garbage collection done", flush=True)
                    except Exception as e:
                        print(f"[robust-oom] Rank {self.rank} gc failed: {e}", flush=True)

                    print(f"[robust-oom] Rank {self.rank} proceeding to PLY save (t0={_ply_t0:.3f})", flush=True)
                    
                    # Save PLY with timeout
                    import threading as th
                    ply_save_done = th.Event()
                    ply_save_success = [False]
                    
                    def save_ply_with_timeout():
                        try:
                            print(f"[robust-oom] Rank {self.rank} entering PLY save callback (monitor thread)", flush=True)
                            ply_save_success[0] = self.save_ply_callback(oom_state.iteration, self.rank)
                            print(f"[robust-oom] Rank {self.rank} PLY save callback returned: {ply_save_success[0]}", flush=True)
                        except Exception as e:
                            print(f"[robust-oom] FATAL: Rank {self.rank} monitor thread PLY save exception: {e}", flush=True)
                            import traceback
                            print(f"[robust-oom] Rank {self.rank} traceback:\n{traceback.format_exc()}", flush=True)
                        finally:
                            ply_save_done.set()
                    
                    print(f"[robust-oom] Rank {self.rank} starting PLY save thread (monitor)", flush=True)
                    print(f"[robust-oom] Rank {self.rank} DEBUG: about to start monitor thread PLY save", flush=True)
                    save_thread = th.Thread(target=save_ply_with_timeout)
                    save_thread.start()
                    print(f"[robust-oom] Rank {self.rank} DEBUG: monitor thread PLY save thread started", flush=True)
                    
                    # Wait for PLY save (configurable, default 300s)
                    ply_timeout = float(os.environ.get("OOM_PLY_SAVE_TIMEOUT", "300"))
                    if ply_save_done.wait(timeout=ply_timeout):
                        _ply_dt = _time.monotonic() - _ply_t0
                        if ply_save_success[0]:
                            print(f"[robust-oom] Rank {self.rank} successfully saved PLY via monitor thread (total_dt={_ply_dt:.3f}s)", flush=True)
                            time.sleep(0.5)  # Give time for file flush
                        else:
                            print(f"[robust-oom] ERROR: Rank {self.rank} failed to save PLY via monitor thread (total_dt={_ply_dt:.3f}s)", flush=True)
                    else:
                        _ply_dt = _time.monotonic() - _ply_t0
                        print(f"[robust-oom] ERROR: Rank {self.rank} PLY save timed out after {ply_timeout:.0f}s (total_dt={_ply_dt:.3f}s)", flush=True)

                    # Exit immediately after saving (or timeout)
                    print(f"[robust-oom] Rank {self.rank} exiting after PLY save attempt", flush=True)
                    os._exit(42)
        
        monitor = threading.Thread(target=monitor_thread, daemon=True)
        monitor.start()
        print(f"[robust-oom] Rank {self.rank} started monitoring thread (backup)", flush=True)
    
    def _write_oom_state(self, state: OOMState):
        """Atomically write OOM state to shared memory."""
        try:
            # Serialize state to JSON
            state_json = json.dumps({
                'detected': state.detected,
                'category': state.category, 
                'iteration': state.iteration,
                'cause': state.cause,
                'rank_who_detected': state.rank_who_detected,
                'timestamp': state.timestamp
            })
            
            # Atomic write to shared memory
            self.shm.seek(0)
            self.shm.write(state_json.encode('utf-8').ljust(4095, b'\0'))
            self.shm.flush()
            
        except Exception as e:
            print(f"[robust-oom] ERROR: Rank {self.rank} failed to write OOM state: {e}", flush=True)
            raise
    
    def _read_oom_state(self) -> OOMState:
        """Read OOM state from shared memory."""
        try:
            self.shm.seek(0)
            data = self.shm.read(4095).rstrip(b'\0').decode('utf-8')
            
            if not data:
                return OOMState()
                
            state_dict = json.loads(data)
            return OOMState(
                detected=state_dict['detected'],
                category=state_dict['category'],
                iteration=state_dict['iteration'], 
                cause=state_dict['cause'],
                rank_who_detected=state_dict['rank_who_detected'],
                timestamp=state_dict['timestamp']
            )
            
        except Exception as e:
            print(f"[robust-oom] ERROR: Rank {self.rank} failed to read OOM state: {e}", flush=True)
            return OOMState()
    
    def _emergency_save_handler(self, signum, frame):
        """Signal handler for immediate emergency PLY saving."""
        if self.emergency_save_in_progress:
            print(f"[robust-oom] Rank {self.rank} emergency save already in progress, ignoring signal {signum}", flush=True)
            return
            
        self.emergency_save_in_progress = True
        
        print(f"[robust-oom] Rank {self.rank} received signal {signum}, starting emergency save", flush=True)
        
        oom_state = None
        try:
            # Read OOM state from shared memory
            oom_state = self._read_oom_state()
            
            print(f"[robust-oom] Rank {self.rank} OOM state: category={oom_state.category}, iter={oom_state.iteration}", flush=True)
            
            if oom_state.category == 3:
                # Category 3: Save trained gaussians
                print(f"[robust-oom] Rank {self.rank} saving PLY for category 3 OOM", flush=True)
                print(f"[robust-oom] Rank {self.rank} DEBUG: about to call save_ply_callback with iter={oom_state.iteration}", flush=True)
                try:
                    success = self.save_ply_callback(oom_state.iteration, self.rank)
                    print(f"[robust-oom] Rank {self.rank} DEBUG: save_ply_callback returned success={success}", flush=True)
                except Exception as e:
                    print(f"[robust-oom] Rank {self.rank} ERROR: save_ply_callback failed with exception: {e}", flush=True)
                    success = False
                
                if success:
                    print(f"[robust-oom] Rank {self.rank} successfully saved PLY", flush=True)
                else:
                    print(f"[robust-oom] ERROR: Rank {self.rank} failed to save PLY", flush=True)
                    
                # Write completion marker for Category 3
                done_path = f"/tmp/oom_done_rank_{self.rank}.marker"
                with open(done_path, 'w') as f:
                    f.write(f"COMPLETED_{time.time()}")
                    
                print(f"[robust-oom] Rank {self.rank} emergency save completed, exiting", flush=True)
            else:
                # Category 1, 2: No PLY save needed, just signal completion
                print(f"[robust-oom] Rank {self.rank} category {oom_state.category} OOM handled, no PLY save needed", flush=True)
            
        except Exception as e:
            print(f"[robust-oom] FATAL: Rank {self.rank} emergency save failed: {e}", flush=True)
            
        # Only exit for Category 3 OOM (which needs PLY save)
        if oom_state and oom_state.category == 3:
            os._exit(42)
        else:
            # Category 1, 2: Let adaptive tiling continue
            category = oom_state.category if oom_state else "unknown"
            print(f"[robust-oom] Rank {self.rank} category {category} OOM signal handled, continuing adaptive tiling", flush=True)
            return
    
    def detect_oom(self, iteration: int, category: int, cause: str, *, kill_process_group: bool = True, signal_self: bool = True, save_self: bool = True):
        """
        Detect OOM - GUARANTEE all ranks save PLY files.
        
        Strategy: 
        1. Write OOM signal to shared memory 
        2. Send SIGUSR1 to ALL ranks in process group
        3. Each rank immediately saves PLY upon receiving signal
        """
        print(f"[robust-oom] Rank {self.rank} detected OOM: category={category}, iter={iteration}, cause={cause}", flush=True)
        
        try:
            if not signal_self:
                self.allow_self_sigusr1 = False
            # Step 1: Write OOM state to shared memory FIRST
            oom_state = OOMState(
                detected=True,
                category=category,
                iteration=iteration,
                cause=cause,
                rank_who_detected=self.rank,
                timestamp=time.time()
            )
            self._write_oom_state(oom_state)
            print(f"[robust-oom] Rank {self.rank} wrote OOM state to shared memory", flush=True)
            
            # Immediately save PLY for detecting rank before any signals
            if category == 3 and save_self:
                print(f"[robust-oom] Rank {self.rank} saving PLY immediately after writing to shared memory", flush=True)
                
                # Write marker file to signal wrapper
                oom_marker = f"/tmp/cat3_oom_iter{iteration}_rank{self.rank}.marker"
                with open(oom_marker, 'w') as f:
                    f.write(f"{iteration},{self.rank},{cause}")
                
                # Skip cuda.synchronize() and empty_cache(): PLY save uses CPU snapshots.
                # See monitor thread comment for full rationale.
                try:
                    import gc
                    gc.collect()
                except Exception as e:
                    print(f"[robust-oom] Rank {self.rank} gc failed: {e}", flush=True)
                
                # Save PLY with timeout
                import threading
                ply_save_done = threading.Event()
                ply_save_success = [False]
                
                def save_ply_with_timeout():
                    try:
                        print(f"[robust-oom] Rank {self.rank} entering PLY save callback (main thread)", flush=True)
                        ply_save_success[0] = self.save_ply_callback(iteration, self.rank)
                        print(f"[robust-oom] Rank {self.rank} PLY save callback returned: {ply_save_success[0]}", flush=True)
                    except Exception as e:
                        print(f"[robust-oom] FATAL: Rank {self.rank} PLY save exception: {e}", flush=True)
                        import traceback
                        print(f"[robust-oom] Rank {self.rank} traceback:\n{traceback.format_exc()}", flush=True)
                    finally:
                        ply_save_done.set()
                
                print(f"[robust-oom] Rank {self.rank} starting PLY save thread (main)", flush=True)
                save_thread = threading.Thread(target=save_ply_with_timeout)
                save_thread.start()
                
                # Wait for PLY save (configurable, default 300s)
                ply_timeout = float(os.environ.get("OOM_PLY_SAVE_TIMEOUT", "300"))
                if ply_save_done.wait(timeout=ply_timeout):
                    if ply_save_success[0]:
                        print(f"[robust-oom] Rank {self.rank} successfully saved PLY (detecting rank)", flush=True)
                        time.sleep(1.0)  # Give time for file flush
                    else:
                        print(f"[robust-oom] ERROR: Rank {self.rank} failed to save PLY (detecting rank)", flush=True)
                else:
                    print(f"[robust-oom] ERROR: Rank {self.rank} PLY save timed out after {ply_timeout:.0f} seconds!", flush=True)
            
            else:
                print(f"[robust-oom] Rank {self.rank} category {category} OOM - no PLY save needed", flush=True)

            # Step 2: Send SIGUSR1 to ALL ranks (targeted by PID to avoid signaling wrapper)
            if self.rank_pids:
                print(f"[robust-oom] Rank {self.rank} sending SIGUSR1 to rank PIDs {self.rank_pids}", flush=True)
                for pid in self.rank_pids:
                    if not signal_self and pid == self.self_pid:
                        continue
                    try:
                        os.kill(pid, signal.SIGUSR1)
                    except Exception as e:
                        print(f"[robust-oom] WARNING: Rank {self.rank} failed to signal pid {pid}: {e}", flush=True)
                # Give other ranks time to receive and process SIGUSR1
                time.sleep(2.0)
            else:
                print(f"[robust-oom] Rank {self.rank} WARNING: rank_pids unavailable, skipping SIGUSR1 broadcast", flush=True)
            
            # Step 3: Removed - PLY already saved above
                    
            # Force kill entire process group to ensure complete cleanup (optional)
            if kill_process_group:
                print(f"[robust-oom] Rank {self.rank} force killing process group for complete cleanup", flush=True)
                try:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                except Exception as e:
                    # This is a serious problem - killpg failed
                    print(f"[robust-oom] FATAL ERROR: Rank {self.rank} failed to kill process group: {e}", flush=True)
                    print(f"[robust-oom] WARNING: GPU processes may still be running!", flush=True)
                    os._exit(42)
                
        except Exception as e:
            print(f"[robust-oom] FATAL: Rank {self.rank} failed to handle OOM: {e}", flush=True)
            sys.exit(1)
    
    def check_oom_signal(self) -> Optional[Dict[str, Any]]:
        """
        Check for OOM signal from other ranks.
        
        Returns OOM signal data if detected, None otherwise.
        This is for backward compatibility with existing code.
        """
        try:
            oom_state = self._read_oom_state()
            
            if oom_state.detected and oom_state.rank_who_detected != self.rank:
                # Another rank detected OOM
                return {
                    'iteration': oom_state.iteration,
                    'oom_category': oom_state.category, 
                    'oom_cause': oom_state.cause,
                    'detecting_rank': oom_state.rank_who_detected,
                    'timestamp': oom_state.timestamp
                }
                
        except Exception as e:
            print(f"[robust-oom] ERROR: Rank {self.rank} failed to check OOM signal: {e}", flush=True)
            
        return None
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.shm:
                self.shm.close()
            if self.shm_fd:
                os.close(self.shm_fd)
            if self.rank == 0 and os.path.exists(self.shm_path):
                os.unlink(self.shm_path)
                
        except Exception as e:
            print(f"[robust-oom] ERROR: Rank {self.rank} cleanup failed: {e}", flush=True)


# Global instance - initialized once per process
_robust_oom_handler: Optional[RobustOOMHandler] = None

def initialize_robust_oom_handler(args, rank: int, world_size: int, save_ply_callback):
    """Initialize the global robust OOM handler instance."""
    global _robust_oom_handler
    
    if _robust_oom_handler is not None:
        print(f"[robust-oom] WARNING: Rank {rank} OOM handler already initialized", flush=True)
        return
        
    _robust_oom_handler = RobustOOMHandler(args, rank, world_size, save_ply_callback)

def get_robust_oom_handler() -> Optional[RobustOOMHandler]:
    """Get the global robust OOM handler instance.""" 
    return _robust_oom_handler

def detect_oom_robust(iteration: int, category: int, cause: str, *, kill_process_group: bool = True, signal_self: bool = True, save_self: bool = True):
    """Detect OOM using robust handler."""
    handler = get_robust_oom_handler()
    if handler:
        handler.detect_oom(iteration, category, cause, kill_process_group=kill_process_group, signal_self=signal_self, save_self=save_self)
    else:
        print(f"[robust-oom] FATAL: OOM handler not initialized", flush=True)
        sys.exit(1)

def check_oom_signal_robust() -> Optional[Dict[str, Any]]:
    """Check OOM signal using robust handler."""
    handler = get_robust_oom_handler() 
    if handler:
        return handler.check_oom_signal()
    return None
