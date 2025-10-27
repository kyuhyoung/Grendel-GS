import os
import torch
import json
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
from utils.system_utils import mkdir_p
from utils.general_utils import prepare_output_and_logger, globally_sync_for_timer
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
import torch.distributed as dist
from densification import densification, gsplat_densification
import torchvision.transforms.functional as tvf


def training_refactored_main(dataset_args, opt_args, pipe_args, args, log_file):
    # Refactored training function
    gaussians, timers, background = _initialize_training_components(dataset_args, opt_args, pipe_args, args, log_file)
    scene, start_from_this_iteration, saved_removed_path = _setup_training_scene(args, gaussians, opt_args, log_file)
    _training_loop(gaussians, scene, opt_args, pipe_args, args, timers, background, start_from_this_iteration, log_file)
    _finalize_training(args, opt_args, gaussians, log_file)
    return saved_removed_path


def _initialize_training_components(dataset_args, opt_args, pipe_args, args, log_file):
    """Initialize training components: gaussians, scene, timers, background"""
    # Process progressive training state if available
    previous_state = getattr(args, 'previous_state_data', None)
    is_progressive_training = getattr(args, 'is_progressive_training', False)

    if is_progressive_training:
        utils.print_rank_0(f"🔄 Progressive Training Mode Enabled")
        if previous_state:
            utils.print_rank_0(f"   Window: {previous_state.get('iteration_name', 'unknown')}")
            utils.print_rank_0(f"   Window number: {previous_state.get('window_number', 0)}")
            utils.print_rank_0(f"   Points processed so far: {len(previous_state.get('processed_points', []))}")
            utils.print_rank_0(f"   Trained gaussians: {len(previous_state.get('trained_gaussians', []))}")
        else:
            utils.print_rank_0(f"   Initial window (no previous state)")
    else:
        utils.print_rank_0(f"🚀 Standard Training Mode")

    # Debug: Check progressive training status
    checkpoint_available = False
    if previous_state:
        checkpoint_available = ('all_checkpoint_paths' in previous_state and previous_state['all_checkpoint_paths'])
    else:
        checkpoint_available = getattr(args, 'start_checkpoint', '') != ''

    utils.print_rank_0(f"🔍 DEBUG: previous_state exists: {previous_state is not None}")
    utils.print_rank_0(f"🔍 DEBUG: args.start_checkpoint: '{getattr(args, 'start_checkpoint', 'NOT_SET')}'")
    utils.print_rank_0(f"🔍 DEBUG: checkpoint available: {checkpoint_available}")

    # Init auxiliary tools
    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")

    # Init parameterized scene
    gaussians = GaussianModel(dataset_args.sh_degree)

    # Init background
    if args.backend == "gsplat":
        bg_color = [1, 1, 1] if dataset_args.white_background else None
    else:
        bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]

    background = None
    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    return gaussians, timers, background


def _filter_scene_cameras(scene, camera_ids):
    """Filter scene train_cameras by camera IDs (using colmap_id)"""
    if not camera_ids:
        return scene

    # Filter train_cameras by COLMAP camera IDs
    filtered_cameras = []
    for cam in scene.train_cameras:
        if cam.colmap_id in camera_ids:
            filtered_cameras.append(cam)
    scene.train_cameras = filtered_cameras
    utils.print_rank_0(f"📊 Filtered to {len(filtered_cameras)} cameras from COLMAP IDs: {camera_ids}")
    return scene


def _setup_training_scene(args, gaussians, opt_args, log_file):
    """Setup training scene: load checkpoints/COLMAP, create scene, add gaussians"""
    previous_state = getattr(args, 'previous_state_data', None)
    start_from_this_iteration = 1
    saved_removed_path = None  # Path to saved removed gaussians ply file

    with torch.no_grad():
        # Check for progressive training mode first
        if previous_state:
            #i_pre = previous_state['window_number']
            utils.print_rank_0("🔄 PROGRESSIVE TRAINING: Using progressive state")

            # Check for checkpoint directory from JSON
            checkpoint_dir = previous_state.get('checkpoint_dir', None) if previous_state else None

            # Skip checkpoint loading only if we're running the initial window itself
            # For window 1+, we should load checkpoints from previous windows
            i_cur = previous_state.get('window_number', 0) + 1
            if i_cur == 0:  # This should never happen in practice
                checkpoint_dir = None
                utils.print_rank_0("📁 INITIAL WINDOW: Skipping checkpoint loading")

            if checkpoint_dir:
                utils.print_rank_0(f"🔄 CHECKPOINT FROM JSON: Loading from {checkpoint_dir}")
                utils.print_rank_0(f"   GPU {utils.GLOBAL_RANK} will load: chkpnt_ws={utils.WORLD_SIZE}_rk={utils.GLOBAL_RANK}.pth")

                # Debug info
                if i_cur > 0:
                    utils.print_rank_0(f"   args.cams_prev: {args.cams_prev}")
                    utils.print_rank_0(f"   args.cams_2_add: {args.cams_2_add}")
                    utils.print_rank_0(f"   args.cams_2_delete: {args.cams_2_delete}")

                # Load checkpoint from directory (load_checkpoint will find the rank-specific file)
                original_checkpoint = getattr(args, 'start_checkpoint', '')
                args.start_checkpoint = checkpoint_dir
                model_params, _ = utils.load_checkpoint(args)
                # Progressive mode: each window starts from iteration 1
                start_from_this_iteration = 1
                args.start_checkpoint = original_checkpoint

                # Prepare train/test view IDs for Scene initialization
                train_view_ids = None
                test_view_ids = None

                # Calculate current window cameras: prev - delete + add
                if hasattr(args, 'cams_prev') and args.cams_prev:
                    prev_cameras = set([int(x) for x in args.cams_prev.split(",")])

                    # Remove cameras to delete
                    if hasattr(args, 'cams_2_delete') and args.cams_2_delete:
                        cams_to_delete = set([int(x) for x in args.cams_2_delete.split(",")])
                        prev_cameras -= cams_to_delete
                        utils.print_rank_0(f"📊 Deleting cameras: {cams_to_delete}")

                    # Add cameras to add
                    if hasattr(args, 'cams_2_add') and args.cams_2_add:
                        cams_to_add = set([int(x) for x in args.cams_2_add.split(",")])
                        prev_cameras |= cams_to_add
                        utils.print_rank_0(f"📊 Adding cameras: {cams_to_add}")

                    train_view_ids = list(prev_cameras)
                    utils.print_rank_0(f"📊 Current window cameras: {sorted(train_view_ids)}")
                    utils.print_rank_0(f"📊 train_view_ids type: {type(train_view_ids)}, value: {train_view_ids}")
                # TODO: Add test_view_ids logic if needed (e.g., from args.cams_test)

                # Create scene from COLMAP data with filtered cameras
                utils.print_rank_0("📊 CREATING SCENE with COLMAP data")
                utils.print_rank_0(f"📊 Passing train_view_ids={train_view_ids} to Scene.__init__")
                scene = Scene(args, gaussians, load_from_checkpoint = False, skip_gaussian_init = True, train_view_ids = train_view_ids, test_view_ids = test_view_ids)

                # Note: scene.all_cameras is already populated in Scene.__init__ with ALL cameras from COLMAP
                utils.print_rank_0(f"📊 Scene has {len(scene.all_cameras)} total cameras for visibility checks")

                gaussians.training_setup(opt_args)
                gaussians.restore(model_params, opt_args)

                # Print Gaussian count after restoration
                n_gaussians_restored = len(gaussians.get_xyz)
                utils.print_rank_0(f"✅ PROGRESSIVE CHECKPOINT RESTORED: {n_gaussians_restored} Gaussians loaded, starting from iteration {start_from_this_iteration}")

                # Handle progressive gaussian processing after checkpoint restoration
                if hasattr(args, 'cams_prev') and args.cams_prev:
                    # Progressive window - calculate new/removed cameras and process gaussians
                    prev_cameras = [int(x) for x in args.cams_prev.split(",")]
                    current_cameras = [cam.colmap_id for cam in scene.train_cameras]
                    new_cameras = set(current_cameras) - set(prev_cameras)
                    removed_cameras = set(prev_cameras) - set(current_cameras)

                    # Check if we should use all_processed_cameras instead of just prev_cameras
                    cameras_for_visibility_check = prev_cameras

                    # Log GPU memory before visibility check decision
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        utils.print_rank_0(f"🧠 [MEMORY BEFORE VISIBILITY CHECK] GPU 0: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Total={mem_total:.2f}GB")

                    if hasattr(args, 'cams_all_processed') and args.cams_all_processed:
                        cameras_for_visibility_check = [int(x) for x in args.cams_all_processed.split(",")]
                        utils.print_rank_0(f"🔄 Using all_processed_cameras for gaussian addition visibility check ({len(cameras_for_visibility_check)} cameras)")

                        # Show breakdown: prev_cameras vs removed_cameras
                        removed_from_window = sorted(set(cameras_for_visibility_check) - set(prev_cameras))
                        utils.print_rank_0(f"   📋 Breakdown:")
                        utils.print_rank_0(f"      - Previous window cameras (현재 윈도우): {sorted(prev_cameras)} ({len(prev_cameras)} cameras)")
                        utils.print_rank_0(f"      - Removed cameras (제거된 카메라): {removed_from_window} ({len(removed_from_window)} cameras)")
                        utils.print_rank_0(f"      - Total for visibility check: {sorted(cameras_for_visibility_check)} ({len(cameras_for_visibility_check)} cameras)")
                    else:
                        utils.print_rank_0(f"🔄 Using prev_cameras for gaussian addition visibility check ({len(cameras_for_visibility_check)} cameras)")

                    # Logging
                    utils.print_rank_0(f"🔄 PROGRESSIVE CHECKPOINT - Previous cameras: {prev_cameras}")
                    utils.print_rank_0(f"🔄 PROGRESSIVE CHECKPOINT - Current cameras: {current_cameras}")
                    utils.print_rank_0(f"🔄 PROGRESSIVE CHECKPOINT - New cameras: {list(new_cameras)}")
                    utils.print_rank_0(f"🔄 PROGRESSIVE CHECKPOINT - Removed cameras: {list(removed_cameras)}")

                    # Process gaussian removal for removed cameras (after checkpoint loading)
                    n_removed_gaussians = 0
                    saved_removed_path = None

                    if removed_cameras:
                        # Generate save path for removed gaussians
                        import os
                        removed_ply_path = None
                        if args.model_path and 'window_' in args.model_path:
                            window_num = args.model_path.split('window_')[-1]
                            # Format window number with zero-padding (e.g., "003")
                            window_num_padded = f"{int(window_num):03d}"
                            removed_ply_path = os.path.join(args.model_path, f"removed_gaussians_window_{window_num_padded}.ply")

                        n_removed_gaussians, saved_removed_path = _remove_gaussians_only_visible_to_removed_cameras(
                            gaussians, scene, removed_cameras, current_cameras, save_path=removed_ply_path
                        )
                    else:
                        utils.print_rank_0("📊 No cameras to remove, skipping gaussian removal")

                    n_gaussians_after_removal = len(gaussians.get_xyz)

                    # Log GPU memory after gaussian removal
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                        utils.print_rank_0(f"🧠 [MEMORY AFTER REMOVAL] GPU 0: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Gaussians={n_gaussians_after_removal}")

                    # Process gaussian addition for new cameras (after checkpoint loading)
                    # Use cameras_for_visibility_check which is either prev_cameras or all_processed_cameras
                    _add_gaussians_only_visible_to_new_cameras(gaussians, scene, new_cameras, cameras_for_visibility_check, opt_args)
                    n_gaussians_after_addition = len(gaussians.get_xyz)

                    # Log GPU memory after gaussian addition
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                        utils.print_rank_0(f"🧠 [MEMORY AFTER ADDITION] GPU 0: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Gaussians={n_gaussians_after_addition}")

                    # Calculate counts
                    n_removed = n_gaussians_restored - n_gaussians_after_removal
                    n_added = n_gaussians_after_addition - n_gaussians_after_removal

                    # Log gaussian removal/addition
                    utils.print_rank_0("=" * 80)
                    utils.print_rank_0(f"📊 [GAUSSIAN COUNT SUMMARY]")
                    utils.print_rank_0(f"   Restored from checkpoint: {n_gaussians_restored}")
                    utils.print_rank_0(f"   After removal: {n_gaussians_after_removal} (removed: {n_removed})")
                    utils.print_rank_0(f"   After addition: {n_gaussians_after_addition} (added: {n_added})")
                    utils.print_rank_0(f"   Net change: {n_gaussians_after_addition - n_gaussians_restored:+d}")
                    utils.print_rank_0("=" * 80)
                    utils.print_rank_0(f"📊 Progressive checkpoint loaded: {n_gaussians_restored} gaussians")
                    utils.print_rank_0(f"   Gaussians removed: {n_removed}")
                    utils.print_rank_0(f"   Gaussians added: {n_added}")
                    utils.print_rank_0(f"   Gaussians after removal/addition: {n_gaussians_after_addition}")
                    if saved_removed_path:
                        utils.print_rank_0(f"   Removed gaussians saved to: {saved_removed_path}")
            else:
                # Progressive training without checkpoint
                utils.print_rank_0("📁 PROGRESSIVE WITHOUT CHECKPOINT: Loading from COLMAP")

                # Prepare train/test view IDs for Scene initialization
                train_view_ids = None
                test_view_ids = None

                # Handle progressive training camera filtering and gaussian processing
                if hasattr(args, 'cams_init') and args.cams_init:
                    # Initial window - filter to specified cameras
                    init_cameras = [int(x) for x in args.cams_init.split(",")]
                    utils.print_rank_0(f"📁 PROGRESSIVE INITIAL - Filtering to cameras: {init_cameras}")
                    train_view_ids = init_cameras

                elif hasattr(args, 'cams_prev') and args.cams_prev:
                    # Non-initial window without checkpoint - THIS IS AN ERROR!
                    utils.print_rank_0("\n" + "="*80)
                    utils.print_rank_0("❌ FATAL ERROR: Non-initial progressive window requires checkpoint!")
                    utils.print_rank_0("="*80)
                    utils.print_rank_0(f"  cams_prev: {args.cams_prev}")
                    utils.print_rank_0(f"  previous_state: {args.previous_state}")
                    utils.print_rank_0(f"  start_checkpoint: {args.start_checkpoint}")
                    utils.print_rank_0("\nThis should not happen in progressive training mode.")
                    utils.print_rank_0("Each non-initial window MUST start from the previous window's checkpoint.")
                    utils.print_rank_0("="*80)
                    import sys
                    sys.exit(1)

                # Create scene from COLMAP data with filtered cameras (only for initial window)
                utils.print_rank_0(f"📊 Creating Scene with train_view_ids: {train_view_ids}")
                scene = Scene(args, gaussians, train_view_ids=train_view_ids, test_view_ids=test_view_ids)

                # Note: scene.all_cameras is already populated in Scene.__init__ from COLMAP
                utils.print_rank_0(f"📊 Using {len(scene.all_cameras)} cameras for visibility checks (from Scene.__init__)")

                gaussians.training_setup(opt_args)

                # Print Gaussian count after COLMAP initialization
                n_gaussians_colmap = len(gaussians.get_xyz)
                utils.print_rank_0(f"✅ PROGRESSIVE LOADED: {n_gaussians_colmap} Gaussians initialized")

        elif args.start_checkpoint != "":
            # Load from checkpoint first, then create scene without COLMAP initialization
            utils.print_rank_0(f"🔄 CHECKPOINT FOUND: Loading from {args.start_checkpoint}")
            log_file.write(f"Loading from checkpoint: {args.start_checkpoint}\n")
            model_params, start_from_this_iteration = utils.load_checkpoint(args)

            # Create scene but skip COLMAP initialization since we have checkpoint
            utils.print_rank_0("🚫 SKIPPING COLMAP - Creating scene from checkpoint")
            scene = Scene(args, gaussians, load_from_checkpoint=True)
            gaussians.training_setup(opt_args)
            gaussians.restore(model_params, opt_args)

            # Print Gaussian count after restoration
            n_gaussians_restored = len(gaussians.get_xyz)
            utils.print_rank_0(f"✅ CHECKPOINT RESTORED: {n_gaussians_restored} Gaussians loaded, starting from iteration {start_from_this_iteration}")
            log_file.write(f"Restored from checkpoint: {args.start_checkpoint}\n")
        else:
            # Normal initialization from COLMAP
            utils.print_rank_0("📁 NO CHECKPOINT - Loading from COLMAP")
            #exit(1)
            scene = Scene(args, gaussians)
            gaussians.training_setup(opt_args)

            # Handle progressive training camera filtering and gaussian processing
            if hasattr(args, 'cams_init') and args.cams_init:
                # Initial window - filter to specified cameras
                init_cameras = [int(x) for x in args.cams_init.split(",")]
                utils.print_rank_0(f"📁 INITIAL WINDOW - Filtering to cameras: {init_cameras}")
                scene = _filter_scene_cameras(scene, init_cameras)

            elif hasattr(args, 'cams_prev') and args.cams_prev:
                # Progressive window - calculate new/removed cameras and process gaussians
                prev_cameras = [int(x) for x in args.cams_prev.split(",")]
                current_cameras = [cam.colmap_id for cam in scene.train_cameras]

                new_cameras = set(current_cameras) - set(prev_cameras)
                removed_cameras = set(prev_cameras) - set(current_cameras)

                utils.print_rank_0(f"🔄 PROGRESSIVE WINDOW - Previous cameras: {prev_cameras}")
                utils.print_rank_0(f"🔄 PROGRESSIVE WINDOW - Current cameras: {current_cameras}")
                utils.print_rank_0(f"🔄 PROGRESSIVE WINDOW - New cameras: {list(new_cameras)}")
                utils.print_rank_0(f"🔄 PROGRESSIVE WINDOW - Removed cameras: {list(removed_cameras)}")

                # Process gaussian removal for removed cameras
                _remove_gaussians_only_visible_to_removed_cameras(gaussians, scene, removed_cameras, current_cameras)

                # Process gaussian addition for new cameras
                _add_gaussians_only_visible_to_new_cameras(gaussians, scene, new_cameras, prev_cameras, opt_args)

            # Print Gaussian count after COLMAP initialization
            n_gaussians_colmap = len(gaussians.get_xyz)
            utils.print_rank_0(f"✅ COLMAP LOADED: {n_gaussians_colmap} Gaussians initialized")

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")

    utils.check_initial_gpu_memory_usage("after init and before training loop")
    return scene, start_from_this_iteration, saved_removed_path


def _training_loop(gaussians, scene, opt_args, pipe_args, args, timers, background, start_from_this_iteration, log_file):
    """Main training loop"""
    n_g_max = args.n_g_per_proc

    # Init dataset
    train_dataset = SceneDataset(scene.getTrainCameras())
    if args.adjust_strategy_warmp_iterations == -1:
        args.adjust_strategy_warmp_iterations = len(train_dataset.cameras)

    # Init distribution strategy history
    strategy_history = DivisionStrategyHistoryFinal(train_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank())

    # Training Loop
    end2end_timers = End2endTimer(args)
    end2end_timers.start()

    # Reset peak memory stats to track this window's training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Set progress bar description based on progressive state
    previous_state = getattr(args, 'previous_state_data', None)
    progress_desc = "Training progress"
    if previous_state:
        # We're training the next window after the one saved in previous_state
        prev_window = previous_state.get('window_number', -1)
        current_window = prev_window + 1
        progress_desc = f"Window {current_window} progress"

    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc=progress_desc,
        disable=(utils.LOCAL_RANK != 0),
        bar_format='{desc}:{percentage:3.0f}%|{bar:2}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{postfix}]'
    )
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0

    ema_loss_for_log = 0
    debug_info_printed = False
    print(f'start_from_this_iteration : {start_from_this_iteration}, opt_args.iterations + 1 : {opt_args.iterations + 1}, args.bsz : {args.bsz}')

    # Log initial memory before training loop
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        utils.print_rank_0(f"🧠 [MEMORY BEFORE TRAINING LOOP] GPU 0: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB")

    for iteration in range(start_from_this_iteration, opt_args.iterations + 1, args.bsz):
        # Log memory at iteration start
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            utils.print_rank_0(f"🧠 [ITER {iteration} START] Alloc={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Free={mem_free:.2f}GB")

        ema_loss_for_log = _process_iteration(iteration, gaussians, scene, args, timers, strategy_history, train_dataset, background, pipe_args, progress_bar, ema_loss_for_log, debug_info_printed, end2end_timers, log_file, n_g_max)

        # Log memory at iteration end
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            utils.print_rank_0(f"🧠 [ITER {iteration} END] Alloc={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Free={mem_free:.2f}GB")

    '''
    if previous_state:
        if 1 == current_window:
            exit(1)
    '''

    # Finish training
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)

    # Report peak memory usage
    if torch.cuda.is_available():
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        peak_usage_ratio = peak_memory_bytes / total_memory_bytes

        memory_report = f"\n{'='*80}\n"
        memory_report += f"GPU Memory Report:\n"
        memory_report += f"  Peak Memory Used: {peak_memory_gb:.2f} GB / {total_memory_gb:.2f} GB ({peak_usage_ratio*100:.1f}%)\n"
        memory_report += f"{'='*80}\n"

        print(memory_report)
        log_file.write(memory_report)
        log_file.write(f"Max Memory usage: {peak_memory_gb:.2f} GB.\n")

    progress_bar.close()


def _process_iteration(iteration, gaussians, scene, args, timers, strategy_history, train_dataset,
                      background, pipe_args, progress_bar, ema_loss_for_log, debug_info_printed,
                      end2end_timers, log_file, n_g_max):
    """Process single iteration"""
    # Setup iteration
    _setup_iteration(iteration, gaussians, args, progress_bar, ema_loss_for_log, timers)

    # Prepare camera data and strategies
    batched_cameras, batched_strategies, gpuid2tasks = _prepare_camera_data(
        train_dataset, strategy_history, args, timers, gaussians, iteration)

    # Execute rendering pipeline
    batched_image, batched_compute_locally, batch_statistic_collector, batched_screenspace_pkg = _execute_rendering(
        batched_cameras, gaussians, pipe_args, background, batched_strategies, args)

    # Debug: Check Gaussian projection and colors for first iteration
    '''
    if iteration == 1:
        _debug_gaussian_projection_check(gaussians, scene, args, batched_cameras)
        _debug_colmap_vs_gaussian_colors_check(gaussians, scene, args)
    '''
    # Compute loss and execute backward pass
    loss_sum, batched_losses, ema_loss_for_log = _compute_loss_and_backward(
        batched_image, batched_cameras, batched_compute_locally, batched_strategies,
        batch_statistic_collector, args, timers, ema_loss_for_log, train_dataset, log_file, iteration, strategy_history)

    # Handle iteration tasks (evaluation, saving, densification, checkpoints)
    _handle_iteration_tasks(iteration, scene, gaussians, args, batched_cameras, batched_image,
                           batched_strategies, batch_statistic_collector, strategy_history,
                           batched_screenspace_pkg, end2end_timers, log_file, pipe_args,
                           background, n_g_max, ema_loss_for_log)

    # Optimize and cleanup
    _optimize_and_cleanup(iteration, gaussians, args, batched_cameras, timers, log_file)

    return ema_loss_for_log


def _setup_iteration(iteration, gaussians, args, progress_bar, ema_loss_for_log, timers):
    """Setup iteration: progress bar, learning rate, profiling"""
    n_gauss = len(gaussians.get_xyz)

    if iteration > 0:
        progress_bar.set_postfix({
            "#G": f"{n_gauss}/{args.n_g_per_proc}",
            "Loss": f"{ema_loss_for_log:.{3}f}"
        })
    progress_bar.update(args.bsz)
    utils.set_cur_iter(iteration)
    gaussians.update_learning_rate(iteration)
    timers.clear()

    if args.nsys_profile:
        nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")

    # Every 1000 its we increase the levels of SH up to a maximum degree
    if utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
        gaussians.oneupSHdegree()


def _prepare_camera_data(train_dataset, strategy_history, args, timers, gaussians, iteration):
    """Prepare camera data and workload division strategies"""
    # Log number of cameras in dataset
    utils.print_rank_0(f"📷 [CAMERA COUNT] Total cameras in dataset: {train_dataset.camera_size}")

    # Prepare data: Pick random Cameras for training
    if args.local_sampling:
        assert args.bsz % utils.WORLD_SIZE == 0, "Batch size should be divisible by the number of GPUs."
        batched_cameras_idx = train_dataset.get_batched_cameras_idx(args.bsz // utils.WORLD_SIZE)
        batched_all_cameras_idx = torch.zeros((utils.WORLD_SIZE, len(batched_cameras_idx)), device="cuda", dtype=int)
        batched_cameras_idx = torch.tensor(batched_cameras_idx, device="cuda", dtype=int)
        torch.distributed.all_gather_into_tensor(batched_all_cameras_idx, batched_cameras_idx, group=utils.DEFAULT_GROUP)
        batched_all_cameras_idx = batched_all_cameras_idx.cpu().numpy().squeeze()
        batched_cameras = train_dataset.get_batched_cameras_from_idx(batched_all_cameras_idx)
    else:
        batched_cameras = train_dataset.get_batched_cameras(args.bsz)

    # Log selected camera UIDs and image sizes
    camera_info = []
    for cam in batched_cameras:
        camera_info.append(f"UID={cam.uid}, size={cam.image_height}x{cam.image_width}")
    utils.print_rank_0(f"📷 [SELECTED CAMERAS] {', '.join(camera_info)}")

    # Log Gaussian count and memory before rendering
    gaussian_count = gaussians.get_xyz.shape[0]
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    utils.print_rank_0(f"📊 [PRE-RENDER] Iter {iteration}, Gaussians: {gaussian_count}, GPU0 Memory: {allocated:.2f}GB")

    with torch.no_grad():
        # Prepare Workload division strategy
        timers.start("prepare_strategies")
        batched_strategies, gpuid2tasks = start_strategy_final(batched_cameras, strategy_history)
        timers.stop("prepare_strategies")

        # Load ground-truth images to GPU
        timers.start("load_cameras")
        load_camera_from_cpu_to_all_gpu(batched_cameras, batched_strategies, gpuid2tasks)
        timers.stop("load_cameras")

    return batched_cameras, batched_strategies, gpuid2tasks


def _execute_rendering(batched_cameras, gaussians, pipe_args, background, batched_strategies, args):
    """Execute rendering pipeline"""
    # Rendering
    if args.backend == "gsplat":
        batched_screenspace_pkg = gsplat_distributed_preprocess3dgs_and_all2all_final(
            batched_cameras, gaussians, pipe_args, background, batched_strategies=batched_strategies, mode="train",)
        batched_image, batched_compute_locally = gsplat_render_final(batched_screenspace_pkg, batched_strategies)
        batch_statistic_collector = [cuda_args["stats_collector"] for cuda_args in batched_screenspace_pkg["batched_cuda_args"]]
    else:
        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
            batched_cameras, gaussians, pipe_args, background, batched_strategies=batched_strategies, mode="train",)
        batched_image, batched_compute_locally = render_final(batched_screenspace_pkg, batched_strategies)
        batch_statistic_collector = [cuda_args["stats_collector"] for cuda_args in batched_screenspace_pkg["batched_cuda_args"]]

    return batched_image, batched_compute_locally, batch_statistic_collector, batched_screenspace_pkg


def _debug_gaussian_projection_check(gaussians, scene, args, batched_cameras):
    """Debug function to check Gaussian projection using current 3DGS camera settings"""
    try:
        print("🔍 DEBUG: Checking Gaussian projection for current iteration cameras...")

        # Get current Gaussian positions
        gaussian_positions = gaussians.get_xyz.detach().cpu().numpy()  # Shape: (N, 3)
        print(f"📊 Total Gaussians: {len(gaussian_positions)}")

        # Project to current iteration cameras only
        import cv2
        import numpy as np

        for cam in batched_cameras:
            try:
                # Get camera parameters from current 3DGS camera object
                fx = cam.FoVx  # This needs to be converted to focal length
                fy = cam.FoVy  # This needs to be converted to focal length
                width = cam.image_width
                height = cam.image_height

                # Convert FoV to focal length (same as fov2focal in graphics_utils)
                fx_focal = width / (2 * np.tan(fx / 2))
                fy_focal = height / (2 * np.tan(fy / 2))
                cx = width / 2
                cy = height / 2

                from utils.projection_utils import project_points_to_camera

                # Camera matrix using current 3DGS settings
                camera_matrix = np.array([
                    [fx_focal, 0, cx],
                    [0, fy_focal, cy],
                    [0, 0, 1]
                ], dtype=np.float64)

                # Rotation and translation from current 3DGS camera
                R = cam.R if isinstance(cam.R, np.ndarray) else cam.R.detach().cpu().numpy()  # 3x3 rotation matrix
                T = cam.T if isinstance(cam.T, np.ndarray) else cam.T.detach().cpu().numpy()  # 3x1 translation vector

                # Use common projection utility (no distortion for 3DGS cameras)
                result = project_points_to_camera(
                    gaussian_positions, R, T, camera_matrix, dist_coeffs=None,
                    check_behind_camera=True,
                    image_width=width,
                    image_height=height,
                    margin_pixels=0
                )

                visible_mask = result['visible_mask']
                visible_count = np.sum(visible_mask)

                print(f"  📷 Camera {cam.colmap_id} (3DGS): {visible_count} / {len(gaussian_positions)} Gaussians visible")

            except Exception as e:
                print(f"❌ Error projecting to camera {cam.colmap_id}: {e}")

        print("🔍 DEBUG: 3DGS Gaussian projection check complete")

    except Exception as e:
        print(f"❌ Debug projection failed: {e}")


def _debug_colmap_vs_gaussian_colors_check(gaussians, scene, args):
    """Debug function to compare COLMAP original colors vs Gaussian colors"""
    try:
        from utils.debug_color_comparison import debug_colmap_vs_gaussian_colors
        debug_colmap_vs_gaussian_colors(gaussians, scene, args, num_samples=10)
    except Exception as e:
        print(f"❌ Debug COLMAP vs Gaussian color comparison failed: {e}")
    #exit(1)


def _compute_loss_and_backward(batched_image, batched_cameras, batched_compute_locally, batched_strategies,
                               batch_statistic_collector, args, timers, ema_loss_for_log, train_dataset, log_file, iteration, strategy_history):
    """Compute loss and execute backward pass"""
    # Loss computation and backward
    loss_sum, batched_losses = batched_loss_computation(
        batched_image, batched_cameras, batched_compute_locally, batched_strategies, batch_statistic_collector, args.use_chunk)

    timers.start("backward")
    loss_sum.backward()
    timers.stop("backward")
    utils.check_initial_gpu_memory_usage("after backward")

    # Log memory after backward
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        utils.print_rank_0(f"🧠 [ITER {iteration} AFTER BACKWARD] Alloc={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Free={mem_free:.2f}GB")

    with torch.no_grad():
        # Adjust workload division strategy
        globally_sync_for_timer()
        timers.start("finish_strategy_final")
        finish_strategy_final(batched_cameras, strategy_history, batched_strategies, batch_statistic_collector)
        timers.stop("finish_strategy_final")

        # Sync losses in the batch
        timers.start("sync_loss_and_log")
        batched_losses = torch.tensor(batched_losses, device="cuda")
        if utils.DEFAULT_GROUP.size() > 1:
            dist.all_reduce(batched_losses, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
        batched_loss = (1.0 - args.lambda_dssim) * batched_losses[:, 0] + args.lambda_dssim * (1.0 - batched_losses[:, 1])
        batched_loss_cpu = batched_loss.cpu().numpy()
        ema_loss_for_log = (batched_loss_cpu.mean() if ema_loss_for_log is None else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean())

        # Update Epoch Statistics
        train_dataset.update_losses(batched_loss_cpu)

        # Logging
        batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
        log_string = "iteration[{},{}) loss: {} image: {}\n".format(
            iteration, iteration + args.bsz, batched_loss_cpu, [viewpoint_cam.image_name for viewpoint_cam in batched_cameras])
        log_file.write(log_string)
        timers.stop("sync_loss_and_log")

    return loss_sum, batched_losses, ema_loss_for_log


def _handle_iteration_tasks(iteration, scene, gaussians, args, batched_cameras, batched_image,
                           batched_strategies, batch_statistic_collector, strategy_history,
                           batched_screenspace_pkg, end2end_timers, log_file, pipe_args,
                           background, n_g_max, ema_loss_for_log):
    """Handle iteration tasks: evaluation, saving, densification, checkpoints"""
    with torch.no_grad():
        # Evaluation
        end2end_timers.stop()
        training_report(iteration, l1_loss, args.test_iterations, scene, pipe_args, background, args.backend)
        end2end_timers.start()

        # Save Gaussians BEFORE densification
        '''
        t0 = [iteration <= save_iteration < iteration + args.bsz for save_iteration in args.save_iterations]
        print(f'\n\niteration : {iteration}, t0 : {t0}\n\n');
        '''
        if any([iteration <= save_iteration < iteration + args.bsz for save_iteration in args.save_iterations]):
            _handle_saving(iteration, scene, batched_cameras, batched_image, args, end2end_timers, log_file, strategy_history, ema_loss_for_log)

        # Densification AFTER saving
        gaussians_before = gaussians.get_xyz.shape[0]
        if args.backend == "gsplat":
            gsplat_densification(iteration, scene, gaussians, n_g_max, batched_screenspace_pkg)
        else:
            densification(iteration, scene, gaussians, n_g_max, batched_screenspace_pkg)
        gaussians_after = gaussians.get_xyz.shape[0]
        if gaussians_after != gaussians_before:
            utils.print_rank_0(f"🔬 [DENSIFICATION] Iter {iteration}: Gaussians {gaussians_before} → {gaussians_after} (Δ{gaussians_after - gaussians_before:+d})")

        # Save Checkpoints
        checkpoint_condition = any([iteration <= checkpoint_iteration < iteration + args.bsz for checkpoint_iteration in args.checkpoint_iterations])
        if checkpoint_condition:
            checkpoint_dir = _handle_checkpoints(iteration, scene, gaussians, args, end2end_timers, log_file)
            # Store checkpoint directory for progressive training (all GPUs share same directory)
            args.checkpoint_dir = checkpoint_dir


def _optimize_and_cleanup(iteration, gaussians, args, batched_cameras, timers, log_file):
    """Execute optimizer step and cleanup"""
    # Optimizer step
    if iteration < args.iterations:
        timers.start("optimizer_step")
        if args.lr_scale_mode != "accumu":
            for param in gaussians.all_parameters():
                if param.grad is not None:
                    param.grad /= args.bsz
        if not args.stop_update_param:
            gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        timers.stop("optimizer_step")
        utils.check_initial_gpu_memory_usage("after optimizer step")

        # Log memory after optimizer step
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            utils.print_rank_0(f"🧠 [ITER {iteration} AFTER OPTIMIZER] Alloc={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Free={mem_free:.2f}GB")

    # Cleanup
    torch.cuda.synchronize()
    for viewpoint_cam in batched_cameras:
        viewpoint_cam.original_image = None
    if args.nsys_profile:
        nvtx.range_pop()
    if utils.check_enable_python_timer():
        timers.printTimers(iteration, mode="sum")
    log_file.flush()


def _handle_saving(iteration, scene, batched_cameras, batched_image, args, end2end_timers, log_file, strategy_history, ema_loss_for_log):
    """Handle saving gaussians and debug images"""
    end2end_timers.stop()
    end2end_timers.print_time(log_file, iteration + args.bsz)
    utils.print_rank_0(f"\n[ITER {iteration}] Saving Gaussians")
    log_file.write(f"[ITER {iteration}] Saving Gaussians\n")
    scene.save(iteration, ema_loss_for_log, utils.WORLD_SIZE)

    # Save debug images
    neim = batched_cameras[0].image_name
    fn_gt = f'{iteration:06d}_l_{ema_loss_for_log:.3f}_{neim}_rank{utils.LOCAL_RANK}_gt.png'
    fn_rd = f'{iteration:06d}_l_{ema_loss_for_log:.3f}_{neim}_rank{utils.LOCAL_RANK}_rd.png'
    path_gt = os.path.join(scene.model_path, 'im_dbg', fn_gt)
    path_rd = os.path.join(scene.model_path, 'im_dbg', fn_rd)
    mkdir_p(os.path.dirname(path_gt))

    # Debug info (only once)
    debug_info_printed = getattr(_handle_saving, 'debug_printed', False)
    if args.show_memory_debug_info and not debug_info_printed and utils.LOCAL_RANK == 0:
        _handle_saving.debug_printed = True
        print(f"\n=== Debug Info at iteration {iteration} (Rank {utils.LOCAL_RANK}) ===")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")

    torch.cuda.synchronize()
    tvf.to_pil_image(batched_cameras[0].original_image.cpu()).save(path_gt)
    tvf.to_pil_image(batched_image[0].detach().cpu()).save(path_rd)

    if args.save_strategy_history:
        with open(args.log_folder + f"/strategy_history_ws={utils.WORLD_SIZE}_rk={utils.GLOBAL_RANK}.json", "w") as f:
            json.dump(strategy_history.to_json(), f)
    end2end_timers.start()


def _handle_checkpoints(iteration, scene, gaussians, args, end2end_timers, log_file):
    """Handle saving checkpoints"""
    end2end_timers.stop()
    utils.print_rank_0(f"\n[ITER {iteration}] Saving Checkpoint")
    log_file.write(f"[ITER {iteration}] Saving Checkpoint\n")
    save_folder = scene.model_path + "/checkpoints/" + str(iteration) + "/"
    if utils.DEFAULT_GROUP.rank() == 0:
        os.makedirs(save_folder, exist_ok=True)
        if utils.DEFAULT_GROUP.size() > 1:
            torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    elif utils.DEFAULT_GROUP.size() > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    path_ckpt = save_folder + f"/chkpnt_ws={utils.WORLD_SIZE}_rk={utils.GLOBAL_RANK}.pth"
    torch.save((gaussians.capture(), iteration + args.bsz), path_ckpt)

    # Store actual checkpoint paths for progressive training
    if not hasattr(args, 'actual_checkpoint_paths'):
        args.actual_checkpoint_paths = []
    args.actual_checkpoint_paths.append(path_ckpt)

    # Save latest checkpoint directory to file for progressive trainer
    latest_checkpoint_file = os.path.join(scene.model_path, "latest_checkpoint.txt")
    with open(latest_checkpoint_file, 'w') as f:
        f.write(save_folder)
    end2end_timers.start()

    return save_folder


def _finalize_training(args, opt_args, gaussians, log_file):
    """Finalize training and print summary"""
    previous_state = getattr(args, 'previous_state_data', None)

    # Save tile distribution statistics if enabled
    if hasattr(args, 'enable_tile_distribution_stats') and args.enable_tile_distribution_stats:
        from gaussian_renderer.workload_division import get_tile_distribution_stats
        stats = get_tile_distribution_stats()

        # Only rank 0 saves the statistics
        if utils.GLOBAL_RANK == 0:
            stats_file = os.path.join(args.model_path, "tile_distribution_stats.json")
            stats_data = {
                'heuristic_times': stats['heuristic'],
                'uniform_times': stats['uniform']
            }
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
            utils.print_rank_0(f"📊 Saved tile distribution statistics to: {stats_file}")

    # Save checkpoint directory for progressive training (temporary file)
    if getattr(args, 'is_progressive_training', False):
        # Calculate current window number (same logic as line 348-349)
        prev_window = previous_state.get('window_number', -1) if previous_state else -1
        window_num = prev_window + 1
        checkpoint_info_file = os.path.join(args.model_path, f"window_{window_num}_checkpoint.json")

        # Check if checkpoint was saved
        checkpoint_dir = getattr(args, 'checkpoint_dir', None)

        # Collect GPU memory metrics (gather max across all GPUs)
        gpu_metrics = {}
        if torch.cuda.is_available():
            # Get local GPU's peak memory
            local_peak_memory_bytes = torch.cuda.max_memory_allocated()
            total_memory_bytes = torch.cuda.get_device_properties(0).total_memory

            # Gather peak memory from all GPUs to find the maximum
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                # Create tensor for all-gather
                local_peak_tensor = torch.tensor([local_peak_memory_bytes], dtype=torch.int64, device="cuda")
                world_size = torch.distributed.get_world_size()
                all_peaks = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world_size)]
                torch.distributed.all_gather(all_peaks, local_peak_tensor)

                # Find maximum peak memory across all GPUs
                peak_memory_bytes = max([t.item() for t in all_peaks])

                utils.print_rank_0(f"📊 GPU memory across all ranks:")
                for rank, peak_tensor in enumerate(all_peaks):
                    peak_gb = peak_tensor.item() / (1024 ** 3)
                    peak_ratio = peak_tensor.item() / total_memory_bytes
                    utils.print_rank_0(f"   Rank {rank}: {peak_gb:.2f} GB ({peak_ratio*100:.1f}%)")
                utils.print_rank_0(f"   Using maximum: {peak_memory_bytes / (1024 ** 3):.2f} GB for camera removal decision")
            else:
                peak_memory_bytes = local_peak_memory_bytes

            peak_memory_gb = peak_memory_bytes / (1024 ** 3)
            total_memory_gb = total_memory_bytes / (1024 ** 3)
            peak_usage_ratio = peak_memory_bytes / total_memory_bytes

            gpu_metrics = {
                'peak_memory_bytes': int(peak_memory_bytes),
                'total_memory_bytes': int(total_memory_bytes),
                'peak_memory_gb': float(peak_memory_gb),
                'total_memory_gb': float(total_memory_gb),
                'peak_usage_ratio': float(peak_usage_ratio)
            }

        if checkpoint_dir is None:
            utils.print_rank_0(f"⚠️  No checkpoints were saved in this window")
            checkpoint_info = {
                "checkpoint_dir": None,
                "gpu_metrics": gpu_metrics
            }
        else:
            checkpoint_info = {
                "checkpoint_dir": checkpoint_dir,
                "gpu_metrics": gpu_metrics
            }
            utils.print_rank_0(f"💾 Checkpoint directory: {checkpoint_dir}")

        # Only rank 0 saves the checkpoint info file (all GPUs share same directory)
        if utils.GLOBAL_RANK == 0:
            with open(checkpoint_info_file, 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            utils.print_rank_0(f"💾 Saved checkpoint info to: {checkpoint_info_file}")
            utils.print_rank_0(f"   Checkpoint directory: {checkpoint_dir}")
            if gpu_metrics:
                utils.print_rank_0(f"   Peak GPU memory: {gpu_metrics['peak_memory_gb']:.2f} GB / {gpu_metrics['total_memory_gb']:.2f} GB ({gpu_metrics['peak_usage_ratio']*100:.1f}%)")

    #print(f'previous_state : {previous_state}');  exit(1)
    # Progressive training completion summary
    if previous_state:
        # Get gaussian count per GPU
        n_gaussians_local = len(gaussians.get_xyz)

        # Gather gaussian counts from all GPUs
        if utils.DEFAULT_GROUP.size() > 1:
            gaussian_counts = torch.tensor([n_gaussians_local], dtype=torch.int64, device="cuda")
            all_gaussian_counts = torch.zeros(utils.DEFAULT_GROUP.size(), dtype=torch.int64, device="cuda")
            torch.distributed.all_gather_into_tensor(all_gaussian_counts, gaussian_counts, group=utils.DEFAULT_GROUP)
            all_gaussian_counts = all_gaussian_counts.cpu().numpy()
            total_gaussians = all_gaussian_counts.sum()
        else:
            all_gaussian_counts = [n_gaussians_local]
            total_gaussians = n_gaussians_local

        utils.print_rank_0(f"\n🎉 Progressive Window Training Complete!")
        utils.print_rank_0(f"   Window: {previous_state.get('iteration_name', 'unknown')}")
        utils.print_rank_0(f"   Window number: {previous_state.get('window_number', 0)}")

        # Print per-GPU gaussian counts
        utils.print_rank_0(f"   Gaussian counts per GPU:")
        for gpu_id, count in enumerate(all_gaussian_counts):
            utils.print_rank_0(f"     GPU {gpu_id}: {count:,} gaussians")
        utils.print_rank_0(f"   Total gaussians across all GPUs: {total_gaussians:,}")

        log_file.write(f"\nProgressive Window Training Complete:\n")
        log_file.write(f"  Window: {previous_state.get('iteration_name', 'unknown')}\n")
        log_file.write(f"  Gaussian counts per GPU: {all_gaussian_counts.tolist() if utils.DEFAULT_GROUP.size() > 1 else all_gaussian_counts}\n")
        log_file.write(f"  Total gaussians: {total_gaussians:,}\n")
        #exit(1)

def training_report(
    iteration, l1_loss, testing_iterations, scene: Scene, pipe_args, background, backend
):

    if scene.getTestCameras() is None:
        return
 

    #print(f'backend : {backend}');  exit(1)
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
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras(), "num_cameras": len(scene.getTestCameras())},
            {
                "name": "train",
                "cameras": scene.getTrainCameras(),
                "num_cameras": max(len(scene.getTrainCameras()) // args.llffhold, args.bsz),
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


def _get_points_only_visible_to_new_camera(scene, new_camera_id, current_cameras):
    """
    Get COLMAP points that are only visible to the new camera (not visible to current cameras)

    Args:
        scene: Scene object containing COLMAP data
        new_camera_id: ID of the newly added camera
        current_cameras: List of current window camera IDs (excluding new camera)

    Returns:
        List of 3D points only visible to new camera
    """
    import numpy as np

    utils.print_rank_0(f"\n🔍 [DEBUG] _get_points_only_visible_to_new_camera called:")
    utils.print_rank_0(f"  - new_camera_id: {new_camera_id}")
    utils.print_rank_0(f"  - current_cameras: {list(current_cameras) if current_cameras else 'None'}")

    # Check scene attributes
    utils.print_rank_0(f"🔍 [DEBUG] Scene attributes:")
    utils.print_rank_0(f"  - has point_cloud: {hasattr(scene, 'point_cloud')}")
    #utils.print_rank_0(f"  - has cameras: {hasattr(scene, 'cameras')}")
    utils.print_rank_0(f"  - has all_cameras: {hasattr(scene, 'all_cameras')}")

    # Get COLMAP points from scene
    if hasattr(scene, 'point_cloud') and hasattr(scene.point_cloud, 'points'):
        colmap_points = scene.point_cloud.points  # [N, 3] array
        utils.print_rank_0(f"✅ [DEBUG] Found {len(colmap_points)} COLMAP points")
        utils.print_rank_0(f"  - Points shape: {colmap_points.shape}")
        utils.print_rank_0(f"  - Points dtype: {colmap_points.dtype}")
    else:
        utils.print_rank_0("⚠️ [DEBUG] No COLMAP points found in scene")
        utils.print_rank_0(f"  - has point_cloud: {hasattr(scene, 'point_cloud')}")
        if hasattr(scene, 'point_cloud'):
            utils.print_rank_0(f"  - point_cloud has points: {hasattr(scene.point_cloud, 'points')}")
        return []

    # Get camera data
    utils.print_rank_0(f"🔍 [DEBUG] Checking camera data:")
    '''
    if hasattr(scene, 'cameras'):
        utils.print_rank_0(f"  - scene.cameras has {len(scene.cameras)} cameras")
        utils.print_rank_0(f"  - scene.cameras keys: {list(scene.cameras.keys())[:10]}...")
        utils.print_rank_0(f"  - new_camera_id {new_camera_id} in scene.cameras: {new_camera_id in scene.cameras}")
    else:
        utils.print_rank_0(f"  - scene.cameras does not exist")
    '''
    if hasattr(scene, 'all_cameras'):
        utils.print_rank_0(f"  - scene.all_cameras has {len(scene.all_cameras)} cameras")
        utils.print_rank_0(f"  - scene.all_cameras keys: {list(scene.all_cameras.keys())[:10]}...")
        utils.print_rank_0(f"  - new_camera_id {new_camera_id} in scene.all_cameras: {new_camera_id in scene.all_cameras}")
    else:
        utils.print_rank_0(f"  - scene.all_cameras does not exist")

    if not hasattr(scene, 'all_cameras') or new_camera_id not in scene.all_cameras:
        utils.print_rank_0(f"⚠️ [DEBUG] Camera {new_camera_id} not found in scene.all_cameras")
        return []

    utils.print_rank_0(f"✅ [DEBUG] Got camera {new_camera_id}")
    utils.print_rank_0(f"🔍 [DEBUG] Starting batched visibility check for {len(colmap_points)} points...")

    # Step 1: Check which points are visible to new camera (batched)
    utils.print_rank_0(f"  - Checking visibility to new camera {new_camera_id}...")
    visible_to_new = _check_points_visibility_batch(colmap_points, new_camera_id, scene, margin_pixels=0)
    visible_to_new_count = np.sum(visible_to_new)
    utils.print_rank_0(f"  - {visible_to_new_count} points visible to new camera")

    if visible_to_new_count == 0:
        utils.print_rank_0(f"⚠️ [DEBUG] No points visible to new camera {new_camera_id}")
        return []

    # Step 2: Check which points are visible to ANY current camera (batched)
    visible_to_any_current = np.zeros(len(colmap_points), dtype=bool)

    utils.print_rank_0(f"  - Checking visibility to {len(current_cameras)} current cameras...")
    for cam_id in current_cameras:
        if cam_id not in scene.all_cameras:
            utils.print_rank_0(f"    ⚠️ Camera {cam_id} not found in scene.all_cameras")
            continue

        cam_visible = _check_points_visibility_batch(colmap_points, cam_id, scene, margin_pixels=0)
        visible_to_any_current |= cam_visible  # Logical OR
        utils.print_rank_0(f"    - {np.sum(cam_visible)} points visible to camera {cam_id}")

    visible_to_current_count = np.sum(visible_to_any_current)
    utils.print_rank_0(f"  - Total {visible_to_current_count} points visible to any current camera")

    # Step 3: Find points visible ONLY to new camera (not to any current camera)
    visible_only_to_new = visible_to_new & ~visible_to_any_current
    new_only_indices = np.where(visible_only_to_new)[0]
    new_only_points = colmap_points[new_only_indices].tolist()

    visible_to_both_count = np.sum(visible_to_new & visible_to_any_current)

    utils.print_rank_0(f"\n📊 [DEBUG] Point visibility summary for camera {new_camera_id}:")
    utils.print_rank_0(f"  - Total COLMAP points: {len(colmap_points)}")
    utils.print_rank_0(f"  - Visible to new camera: {visible_to_new_count}")
    utils.print_rank_0(f"  - Visible to both new and current: {visible_to_both_count}")
    utils.print_rank_0(f"  - Only visible to new camera: {len(new_only_points)}")

    return new_only_points


def _check_points_visibility_batch(points_3d, camera_id, scene, margin_pixels=0):
    """
    Check visibility of multiple 3D points to a camera using batched cv2.projectPoints

    Args:
        points_3d: Numpy array of 3D points [N, 3]
        camera_id: Camera ID to check visibility
        scene: Scene object containing camera data
        margin_pixels: Margin in pixels for visibility check
                      - Positive value: shrinks valid region (stricter, excludes boundary points)
                      - Negative value: expands valid region (more lenient, includes nearly-visible points)
                      - Zero: exact image boundary

    Returns:
        np.ndarray: Boolean array [N] indicating visibility for each point
    """
    import numpy as np
    import cv2

    from utils.projection_utils import project_points_to_camera

    n_points = len(points_3d)
    visible = np.zeros(n_points, dtype=bool)

    try:
        # Get camera data from scene.all_cameras
        if not hasattr(scene, 'all_cameras') or camera_id not in scene.all_cameras:
            utils.print_rank_0(f"⚠️  Camera {camera_id} not found in scene.all_cameras")
            return visible

        camera = scene.all_cameras[camera_id]

        # Get camera parameters
        R = camera.R  # Rotation matrix [3, 3]
        T = camera.T  # Translation vector [3]

        # Get camera intrinsics - compute from FOV
        from utils.graphics_utils import fov2focal

        # Handle both Camera and CameraInfo objects
        if hasattr(camera, 'FoVx'):
            # Camera object
            fov_x = camera.FoVx
            fov_y = camera.FoVy
            width = camera.image_width
            height = camera.image_height
        elif hasattr(camera, 'FovX'):
            # CameraInfo object
            fov_x = camera.FovX
            fov_y = camera.FovY
            width = camera.width
            height = camera.height
        else:
            raise AttributeError(f"Camera object has neither FoVx nor FovX attribute")

        fx = fov2focal(fov_x, width)
        fy = fov2focal(fov_y, height)
        cx = width / 2.0
        cy = height / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Get distortion coefficients from camera model
        # COLMAP camera models: PINHOLE (no distortion), RADIAL (k1, k2), etc.
        dist_coeffs = None
        if hasattr(camera, 'distortion_params') and camera.distortion_params is not None:
            dist_coeffs = np.array(camera.distortion_params, dtype=np.float64)
        elif hasattr(camera, 'k1') and hasattr(camera, 'k2'):
            # RADIAL model: k1, k2
            dist_coeffs = np.array([camera.k1, camera.k2], dtype=np.float64)

        # Use common projection utility
        result = project_points_to_camera(
            points_3d, R, T, K, dist_coeffs,
            check_behind_camera=False,  # Only check image bounds (matching original behavior)
            image_width=width,
            image_height=height,
            margin_pixels=margin_pixels
        )

        visible = result['in_bounds_mask']
        return visible

    except Exception as e:
        utils.print_rank_0(f"⚠️  Error checking visibility for camera {camera_id}: {e}")
        import traceback
        traceback.print_exc()
        return visible


def _is_point_visible_to_camera(point_3d, camera_id, scene, margin_pixels=0):
    """
    Check if a 3D point is visible from a specific camera using projection

    Args:
        point_3d: 3D point coordinates [x, y, z]
        camera_id: Camera ID to check visibility
        scene: Scene object containing camera data
        margin_pixels: Margin in pixels for visibility check (see _check_points_visibility_batch)

    Returns:
        bool: True if point is visible from camera
    """
    # Use batch function for single point
    result = _check_points_visibility_batch(np.array([point_3d]), camera_id, scene, margin_pixels)
    return result[0] if len(result) > 0 else False


def _remove_gaussians_only_visible_to_removed_cameras(gaussians, scene, removed_cameras, current_cameras, margin_pixels=10, save_path=None):
    """
    Remove gaussians that are only visible to removed cameras (not visible to current cameras)
    Uses batched projection for efficiency and prunes only once.

    Args:
        gaussians: GaussianModel to remove gaussians from
        scene: Scene object containing camera data
        removed_cameras: List of removed camera IDs
        current_cameras: List of current window camera IDs (excluding removed cameras)
        margin_pixels: Margin in pixels for visibility check
                      - Positive: excludes boundary points (conservative removal)
                      - Negative: includes nearly-visible points (aggressive removal)
                      - Default 10: reasonable safety margin
        save_path: Optional path to save removed gaussians as .ply file

    Returns:
        tuple: (int: Number of gaussians removed, str: Path to saved ply file or None)
    """
    import numpy as np
    import torch

    if not removed_cameras:
        # This should never be called with empty removed_cameras
        # Caller should check before calling this function
        utils.print_rank_0("⚠️  Warning: _remove_gaussians_only_visible_to_removed_cameras called with empty removed_cameras")
        return 0, None

    # Get current gaussian positions
    gaussian_xyz = gaussians.get_xyz.detach().cpu().numpy()  # [N, 3]
    n_gaussians = len(gaussian_xyz)

    if n_gaussians == 0:
        utils.print_rank_0("⚠️  No gaussians to process for removal")
        return 0, None

    utils.print_rank_0(f"📊 Checking visibility for {n_gaussians} gaussians against {len(removed_cameras)} removed cameras (margin: {margin_pixels}px)...")

    # Step 1: Check which gaussians are visible to ANY removed camera
    visible_to_any_removed = np.zeros(n_gaussians, dtype=bool)

    for removed_cam_id in removed_cameras:
        if removed_cam_id not in scene.all_cameras:
            utils.print_rank_0(f"⚠️  Camera {removed_cam_id} not found in scene.all_cameras")
            continue

        cam_visible = _check_points_visibility_batch(gaussian_xyz, removed_cam_id, scene, margin_pixels)
        visible_to_any_removed |= cam_visible  # Logical OR - visible to at least one removed camera
        utils.print_rank_0(f"📊 {np.sum(cam_visible)} gaussians visible to removed camera {removed_cam_id}")

    n_visible_to_removed = np.sum(visible_to_any_removed)
    utils.print_rank_0(f"📊 Total {n_visible_to_removed} gaussians visible to any removed camera")

    # Step 2: Check if gaussians are visible to any current camera
    visible_to_current = np.zeros(n_gaussians, dtype=bool)

    for cam_id in current_cameras:
        if cam_id not in scene.all_cameras:
            utils.print_rank_0(f"⚠️  Camera {cam_id} not found in scene.all_cameras")
            continue

        # Check visibility for all gaussians to this camera (batched)
        cam_visible = _check_points_visibility_batch(gaussian_xyz, cam_id, scene, margin_pixels)
        visible_to_current |= cam_visible  # Logical OR - visible to at least one current camera

    n_visible_to_current = np.sum(visible_to_current)
    utils.print_rank_0(f"📊 {n_visible_to_current} gaussians visible to current cameras")

    # Step 3: Determine which gaussians to remove
    # Remove if: visible to any removed camera AND NOT visible to any current camera
    gaussians_to_remove_mask = visible_to_any_removed & ~visible_to_current
    n_removed_local = np.sum(gaussians_to_remove_mask)

    # Gather total count from all ranks for accurate logging
    n_removed_tensor = torch.tensor([n_removed_local], dtype=torch.int64, device=gaussians.get_xyz.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(n_removed_tensor, op=torch.distributed.ReduceOp.SUM)
    n_removed_total = n_removed_tensor.item()

    saved_ply_path = None

    # Remove gaussians using boolean mask (True = remove, False = keep)
    if n_removed_local > 0:
        # Save removed gaussians to ply file before pruning
        # All ranks must participate in save_ply() for collective communication
        if save_path:
            try:
                from scene.gaussian_model import GaussianModel

                # Create a temporary GaussianModel with only the removed gaussians
                # All ranks create this to participate in collective ops
                removed_gaussians = GaussianModel(gaussians.max_sh_degree)

                # Copy only the gaussians that will be removed (each rank has its own subset)
                prune_mask_torch = torch.tensor(gaussians_to_remove_mask, dtype=torch.bool, device=gaussians.get_xyz.device)

                # Extract removed gaussian data from this rank
                removed_gaussians._xyz = gaussians._xyz[prune_mask_torch].clone()
                removed_gaussians._features_dc = gaussians._features_dc[prune_mask_torch].clone()
                removed_gaussians._features_rest = gaussians._features_rest[prune_mask_torch].clone()
                removed_gaussians._scaling = gaussians._scaling[prune_mask_torch].clone()
                removed_gaussians._rotation = gaussians._rotation[prune_mask_torch].clone()
                removed_gaussians._opacity = gaussians._opacity[prune_mask_torch].clone()

                # Save to ply file - all ranks must call this for collective communication
                # save_ply() internally gathers to rank 0 and only rank 0 writes the file
                removed_gaussians.save_ply(save_path)
                saved_ply_path = save_path
                utils.print_rank_0(f"💾 Saved {n_removed_total} removed gaussians to: {save_path}")

                # Verify the saved file (only rank 0 checks file system)
                import os
                if utils.GLOBAL_RANK == 0:
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path)
                        utils.print_rank_0(f"   ✅ Removed gaussians file verified:")
                        utils.print_rank_0(f"      Path: {save_path}")
                        utils.print_rank_0(f"      Size: {file_size:,} bytes")
                    else:
                        utils.print_rank_0(f"   ⚠️  Warning: Removed gaussians file not found: {save_path}")
                #exit(1)
            except Exception as e:
                utils.print_rank_0(f"⚠️  Failed to save removed gaussians: {e}")
                exit(1)
        prune_mask = torch.tensor(gaussians_to_remove_mask, dtype=torch.bool, device=gaussians.get_xyz.device)
        gaussians.prune_points(prune_mask)
        utils.print_rank_0(f"✅ Removed {n_removed_total} gaussians only visible to removed cameras {list(removed_cameras)}")
    else:
        utils.print_rank_0(f"📊 No gaussians found only visible to removed cameras")

    return n_removed_total, saved_ply_path


def _add_gaussians_only_visible_to_new_cameras(gaussians, scene, new_cameras, prev_cameras, opt_args):
    """
    Add gaussians from points visible to new cameras but not to previous cameras
    Collects all new points and adds them in a single operation for efficiency.

    Args:
        gaussians: GaussianModel to add gaussians to
        scene: Scene object containing camera and point data
        new_cameras: List of new camera IDs to process
        prev_cameras: List of previous camera IDs (or all_processed_cameras if USE_ALL_PROCESSED_CAMERAS=true)
        opt_args: Optimization arguments
    """
    import numpy as np

    new_cams_list = list(new_cameras) if new_cameras else []
    prev_cams_list = list(prev_cameras) if prev_cameras else []

    utils.print_rank_0("=" * 80)
    utils.print_rank_0("🔍 [GAUSSIAN ADDITION DEBUG] _add_gaussians_only_visible_to_new_cameras STARTED")
    utils.print_rank_0(f"🔍 Input parameters:")
    utils.print_rank_0(f"  - new_cameras: {new_cams_list} ({len(new_cams_list)} cameras)")
    utils.print_rank_0(f"  - prev_cameras (for visibility check): {prev_cams_list} ({len(prev_cams_list)} cameras)")
    utils.print_rank_0(f"  - Current gaussian count: {len(gaussians.get_xyz)}")

    if not new_cameras:
        # utils.print_rank_0("⚠️ [DEBUG] No new cameras to add - returning early")
        # utils.print_rank_0("=" * 80)
        return

    # Check if scene has COLMAP point cloud data
    # utils.print_rank_0(f"🔍 [DEBUG] Checking scene COLMAP data:")
    # utils.print_rank_0(f"  - scene has point_cloud attribute: {hasattr(scene, 'point_cloud')}")

    # Get COLMAP points from scene
    if not hasattr(scene, 'point_cloud') or not hasattr(scene.point_cloud, 'points'):
        # utils.print_rank_0("⚠️ [DEBUG] No COLMAP point cloud found in scene")
        return

    colmap_points = scene.point_cloud.points  # [N, 3] array
    colmap_colors = scene.point_cloud.colors  # [N, 3] array (RGB, 0-1 range)
    n_points = len(colmap_points)
    # utils.print_rank_0(f"✅ [DEBUG] Found {n_points} COLMAP points with colors")

    # Step 1: Check which points are visible to ANY new camera (batched)
    utils.print_rank_0(f"\n🔍 Step 1: Checking visibility to {len(new_cameras)} NEW cameras...")
    visible_to_any_new = np.zeros(n_points, dtype=bool)

    for new_camera_id in new_cameras:
        if new_camera_id not in scene.all_cameras:
            utils.print_rank_0(f"  ⚠️ Camera {new_camera_id} not found in scene.all_cameras")
            continue

        cam_visible = _check_points_visibility_batch(colmap_points, new_camera_id, scene, margin_pixels=0)
        visible_to_any_new |= cam_visible  # Logical OR - 합집합

    n_visible_to_new = np.sum(visible_to_any_new)
    utils.print_rank_0(f"  → {n_visible_to_new} points visible to ANY new camera")

    if n_visible_to_new == 0:
        # utils.print_rank_0("⚠️ [DEBUG] No points visible to any new camera")
        return

    # Step 2: Check which points are visible to ANY prev camera (batched)
    utils.print_rank_0(f"\n🔍 Step 2: Checking visibility to {len(prev_cameras)} PREV/ALL_PROCESSED cameras...")
    visible_to_any_prev = np.zeros(n_points, dtype=bool)

    for prev_camera_id in prev_cameras:
        if prev_camera_id not in scene.all_cameras:
            utils.print_rank_0(f"  ⚠️ Camera {prev_camera_id} not found in scene.all_cameras")
            continue

        cam_visible = _check_points_visibility_batch(colmap_points, prev_camera_id, scene, margin_pixels=0)
        visible_to_any_prev |= cam_visible  # Logical OR - 합집합
        utils.print_rank_0(f"  ✓ Camera {prev_camera_id}: {np.sum(cam_visible)} points visible")

    n_visible_to_prev = np.sum(visible_to_any_prev)
    utils.print_rank_0(f"📊 Total {n_visible_to_prev} points visible to ANY prev/all_processed camera")

    # Step 3: Find points visible ONLY to new cameras (not to any prev camera)
    visible_only_to_new = visible_to_any_new & ~visible_to_any_prev
    n_only_to_new = np.sum(visible_only_to_new)

    visible_to_both = np.sum(visible_to_any_new & visible_to_any_prev)

    utils.print_rank_0(f"\n📊 [FINAL RESULT] Point visibility summary:")
    utils.print_rank_0(f"  - Total COLMAP points: {n_points}")
    utils.print_rank_0(f"  - Visible to ANY new camera: {n_visible_to_new}")
    utils.print_rank_0(f"  - Visible to ANY prev/all_processed camera: {n_visible_to_prev} (cameras: {prev_cams_list})")
    utils.print_rank_0(f"  - Visible to BOTH (filtered out): {visible_to_both}")
    utils.print_rank_0(f"  - ONLY visible to new cameras (will add as Gaussians): {n_only_to_new}")
    utils.print_rank_0("=" * 80)

    # Verification complete - USE_ALL_PROCESSED_CAMERAS is working correctly
    utils.print_rank_0("\n✅ USE_ALL_PROCESSED_CAMERAS verification:")
    utils.print_rank_0(f"   Used {len(prev_cams_list)} cameras for visibility check: {prev_cams_list}")
    utils.print_rank_0(f"   New cameras being added: {new_cams_list}")
    utils.print_rank_0(f"   Filtered out {visible_to_both} duplicate points")
    # Extract points
    new_only_indices = np.where(visible_only_to_new)[0]
    all_new_points = colmap_points[new_only_indices].tolist()

    # utils.print_rank_0(f"\n🔍 [DEBUG] Point collection completed:")
    # utils.print_rank_0(f"  - Total new points collected: {len(all_new_points)}")

    # Add all collected points in a single operation
    if all_new_points:
        all_new_colors = colmap_colors[new_only_indices].tolist()
        # utils.print_rank_0(f"\n✅ [DEBUG] Adding {len(all_new_points)} gaussians from {len(new_cameras)} new cameras")
        # utils.print_rank_0(f"  - Gaussian count before: {len(gaussians.get_xyz)}")

        _add_gaussians_from_points(gaussians, all_new_points, all_new_colors, opt_args)

        # utils.print_rank_0(f"  - Gaussian count after: {len(gaussians.get_xyz)}")
        # utils.print_rank_0(f"  - Net increase: {len(gaussians.get_xyz) - (len(gaussians.get_xyz) - len(all_new_points))}")
    else:
        pass
        # utils.print_rank_0(f"⚠️ [DEBUG] No new points found for any of the {len(new_cameras)} new cameras")

    # utils.print_rank_0("🔍 [DEBUG] _add_gaussians_only_visible_to_new_cameras COMPLETED")
    # utils.print_rank_0("=" * 80)


def _add_gaussians_from_points(gaussians, new_points, new_colors, opt_args):
    """
    Add new gaussians initialized from 3D points

    Args:
        gaussians: GaussiarModel to add gaussians to
        new_points: List of 3D points to create gaussians from
        new_colors: List of RGB colors from COLMAP
        opt_args: Optimization arguments
    """
    import torch
    import numpy as np
    from utils.sh_utils import RGB2SH
    from simple_knn._C import distCUDA2

    # utils.print_rank_0(f"\n🔍 [DEBUG] _add_gaussians_from_points called:")
    # utils.print_rank_0(f"  - Number of points: {len(new_points)}")
    # utils.print_rank_0(f"  - Current gaussian count: {len(gaussians.get_xyz)}")

    if len(new_points) == 0:
        # utils.print_rank_0("⚠️ [DEBUG] No points to add - returning early")
        return

    try:
        # Check input data
        # utils.print_rank_0(f"🔍 [DEBUG] Input points info:")
        # utils.print_rank_0(f"  - Type: {type(new_points)}")
        # if isinstance(new_points, list) and len(new_points) > 0:
        #     utils.print_rank_0(f"  - First point type: {type(new_points[0])}")
        #     utils.print_rank_0(f"  - First point: {new_points[0]}")

        # Convert points to tensor
        # utils.print_rank_0(f"🔍 [DEBUG] Converting points to tensor...")
        new_points_tensor = torch.tensor(new_points, dtype=torch.float32, device="cuda")
        # utils.print_rank_0(f"  - Tensor shape: {new_points_tensor.shape}")
        # utils.print_rank_0(f"  - Tensor dtype: {new_points_tensor.dtype}")
        # utils.print_rank_0(f"  - Tensor device: {new_points_tensor.device}")

        # Initialize basic gaussian parameters for new points
        N = len(new_points)
        # utils.print_rank_0(f"🔍 [DEBUG] Initializing parameters for {N} gaussians...")

        # Convert RGB colors to SH coefficients
        colors_tensor = torch.tensor(new_colors, dtype=torch.float32, device="cuda")
        # utils.print_rank_0(f"  - Colors tensor shape: {colors_tensor.shape}, dtype: {colors_tensor.dtype}")
        # utils.print_rank_0(f"  - First 5 colors: {colors_tensor[:min(5, len(colors_tensor))].tolist()}")
        # exit(1)
        fused_color = RGB2SH(colors_tensor)  # [N, 3]
        # utils.print_rank_0(f"  - RGB colors converted to SH: shape={fused_color.shape}")
        # utils.print_rank_0(f"  - First 5 SH colors: {fused_color[:min(5, len(fused_color))].tolist()}")

        # Create features array with SH coefficients
        features = torch.zeros((N, 3, (gaussians.max_sh_degree + 1) ** 2), dtype=torch.float32, device="cuda")
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # Split into DC and rest components (matching gaussian_model.py:220-224)
        new_features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()  # [N, 1, 3]
        new_features_rest = features[:, :, 1:].transpose(1, 2).contiguous()  # [N, num_rest, 3]
        # utils.print_rank_0(f"  - Features DC: shape={new_features_dc.shape}")
        # utils.print_rank_0(f"  - Features rest: shape={new_features_rest.shape}")

        # Compute adaptive scales from nearest neighbor distances (like create_from_pcd)
        dist2 = torch.clamp_min(distCUDA2(new_points_tensor), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # utils.print_rank_0(f"  - Scales: shape={scales.shape}, computed from nearest neighbor distances")
        # utils.print_rank_0(f"  - Scale range: min={scales.min().item():.4f}, max={scales.max().item():.4f}, mean={scales.mean().item():.4f}")

        # Identity rotations (like create_from_pcd)
        rotations = torch.zeros(N, 4, dtype=torch.float32, device="cuda")
        rotations[:, 0] = 1  # w=1, x=y=z=0 (identity quaternion)
        # utils.print_rank_0(f"  - Rotations: shape={rotations.shape} (identity quaternion)")

        # Small initial opacity (inverse sigmoid of 0.1)
        from utils.general_utils import inverse_sigmoid
        opacities = inverse_sigmoid(0.1 * torch.ones(N, 1, dtype=torch.float32, device="cuda"))
        # utils.print_rank_0(f"  - Opacities: shape={opacities.shape}, value=inverse_sigmoid(0.1)")

        # Send to GPU count (matching gaussian_model.py:251-253)
        shard_world_size = gaussians.group_for_redistribution().size()
        new_send_to_gpui_cnt = torch.zeros((N, shard_world_size), dtype=torch.int, device="cuda")
        # utils.print_rank_0(f"  - Send to GPU count: shape={new_send_to_gpui_cnt.shape}")

        # Add gaussians to the model
        # utils.print_rank_0(f"🔍 [DEBUG] Calling gaussians.densification_postfix...")
        # utils.print_rank_0(f"  - Method exists: {hasattr(gaussians, 'densification_postfix')}")

        before_count = len(gaussians.get_xyz)
        gaussians.densification_postfix(
            new_points_tensor,
            new_features_dc,
            new_features_rest,
            opacities,
            scales,
            rotations,
            new_send_to_gpui_cnt,
        )
        after_count = len(gaussians.get_xyz)

        # utils.print_rank_0(f"✅ [DEBUG] Successfully added {N} new gaussians")
        # utils.print_rank_0(f"  - Gaussian count: {before_count} -> {after_count} (delta: {after_count - before_count})")

    except Exception as e:
        utils.print_rank_0(f"❌ [DEBUG] Error adding gaussians from points: {e}")
        import traceback
        utils.print_rank_0(traceback.format_exc())
