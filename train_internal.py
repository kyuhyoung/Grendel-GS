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
    gaussians, timers, background, start_from_this_iteration = _initialize_training_components(dataset_args, opt_args, pipe_args, args, log_file)
    scene, start_from_this_iteration = _setup_training_scene(args, gaussians, opt_args, log_file)
    _training_loop(gaussians, scene, opt_args, pipe_args, args, timers, background, start_from_this_iteration, log_file)
    _finalize_training(args, opt_args, gaussians, log_file)


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
            utils.print_rank_0(f"   Current window cameras: {len(previous_state.get('current_window_cameras', []))}")
            utils.print_rank_0(f"   Unprocessed cameras remaining: {len(previous_state.get('unprocessed_cameras', []))}")
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
    start_from_this_iteration = 1

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

    return gaussians, timers, background, start_from_this_iteration


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

    with torch.no_grad():
        # Check for progressive training mode first
        if previous_state:
            #i_pre = previous_state['window_number']
            utils.print_rank_0("🔄 PROGRESSIVE TRAINING: Using progressive state")

            # Check for checkpoint from JSON or command line
            checkpoint_path = None

            if 'all_checkpoint_paths' in previous_state and previous_state['all_checkpoint_paths']:
                # Each GPU loads its own checkpoint
                all_paths = previous_state['all_checkpoint_paths']
                if str(utils.GLOBAL_RANK) in all_paths:
                    checkpoint_path = all_paths[str(utils.GLOBAL_RANK)]
                    utils.print_rank_0(f"🔄 CHECKPOINT FROM JSON: GPU {utils.GLOBAL_RANK} loading from {checkpoint_path}")
                else:
                    utils.print_rank_0(f"⚠️  No checkpoint found for GPU {utils.GLOBAL_RANK} in JSON")

            # Skip checkpoint loading only if we're running the initial window itself
            # For window 1+, we should load checkpoints from previous windows
            i_cur = previous_state.get('window_number', 0) + 1
            #print(f'i_win : {i_win}, current_window_num : {current_window_num}');   exit(1)
            if i_cur == 0:  # This should never happen in practice
                checkpoint_path = None
                utils.print_rank_0("📁 INITIAL WINDOW: Skipping checkpoint loading")
            #'''
            if i_cur > 0:
                print(f'checkpoint_path : {checkpoint_path}');  #exit(1) 
                #checkpoint_path : output/progressive_test/model_initial/checkpoints/25//chkpnt_ws=3_rk=0.pth [29/09 10:58:46]
                print(f'args.cams_prev : {args.cams_prev}');    #exit(1)
                #args.cams_prev : 36,7
                print(f'args.cams_2_add : {args.cams_2_add}');    #exit(1)
                #args.cams_2_add : 46
                print(f'args.cams_2_delete : {args.cams_2_delete}');    #exit(1)
                #args.cams_2_delete : 7
            #'''
            if checkpoint_path:
                # Load checkpoint first
                import os
                checkpoint_dir = os.path.dirname(checkpoint_path)
                #print(f'checkpoint_dir : {checkpoint_dir}')
                #checkpoint_dir : output/progressive_test/model_initial/checkpoints/25
                original_checkpoint = getattr(args, 'start_checkpoint', '')
                #print(f'args.start_checkpoint b4 : {original_checkpoint}')
                #args.start_checkpoint b4 :
                args.start_checkpoint = checkpoint_dir
                #print(f'args.start_checkpoint after : {args.start_checkpoint}');    exit(1)
                #args.start_checkpoint after : output/progressive_test/model_initial/checkpoints/25
                model_params, start_from_this_iteration = utils.load_checkpoint(args)
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
                    #exit(1)
                # TODO: Add test_view_ids logic if needed (e.g., from args.cams_test)

                # Create scene from COLMAP data with filtered cameras
                utils.print_rank_0("📊 CREATING SCENE with COLMAP data")
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

                    # Logging
                    utils.print_rank_0(f"🔄 PROGRESSIVE CHECKPOINT - Previous cameras: {prev_cameras}")
                    utils.print_rank_0(f"🔄 PROGRESSIVE CHECKPOINT - Current cameras: {current_cameras}")
                    utils.print_rank_0(f"🔄 PROGRESSIVE CHECKPOINT - New cameras: {list(new_cameras)}")
                    utils.print_rank_0(f"🔄 PROGRESSIVE CHECKPOINT - Removed cameras: {list(removed_cameras)}")

                    # Process gaussian removal for removed cameras (after checkpoint loading)
                    for removed_cam in removed_cameras:
                        _remove_gaussians_only_visible_to_removed_camera(gaussians, scene, removed_cam, current_cameras)
                    exit(1)
                    # Process gaussian addition for new cameras (after checkpoint loading)
                    for new_cam in new_cameras:
                        new_points = _get_points_only_visible_to_new_camera(scene, new_cam, prev_cameras)
                        if new_points:
                            _add_gaussians_from_points(gaussians, new_points, opt_args)
            else:
                # Progressive training without checkpoint (should not happen with new design)
                utils.print_rank_0("📁 PROGRESSIVE WITHOUT CHECKPOINT: Loading from COLMAP")
                scene = Scene(args, gaussians)

                # Handle progressive training camera filtering and gaussian processing
                if hasattr(args, 'cams_init') and args.cams_init:
                    # Initial window - filter to specified cameras
                    init_cameras = [int(x) for x in args.cams_init.split(",")]
                    utils.print_rank_0(f"📁 PROGRESSIVE INITIAL - Filtering to cameras: {init_cameras}")
                    scene = _filter_scene_cameras(scene, init_cameras)

                elif hasattr(args, 'cams_prev') and args.cams_prev:
                    # Progressive window - calculate new/removed cameras and process gaussians
                    prev_cameras = [int(x) for x in args.cams_prev.split(",")]
                    current_cameras = [cam.colmap_id for cam in scene.train_cameras]

                    new_cameras = set(current_cameras) - set(prev_cameras)
                    removed_cameras = set(prev_cameras) - set(current_cameras)

                    utils.print_rank_0(f"🔄 PROGRESSIVE NO-CHECKPOINT - Previous cameras: {prev_cameras}")
                    utils.print_rank_0(f"🔄 PROGRESSIVE NO-CHECKPOINT - Current cameras: {current_cameras}")
                    utils.print_rank_0(f"🔄 PROGRESSIVE NO-CHECKPOINT - New cameras: {list(new_cameras)}")
                    utils.print_rank_0(f"🔄 PROGRESSIVE NO-CHECKPOINT - Removed cameras: {list(removed_cameras)}")

                    # Filter scene to current cameras
                    scene = _filter_scene_cameras(scene, current_cameras)

                    # Process gaussian removal for removed cameras
                    for removed_cam in removed_cameras:
                        _remove_gaussians_only_visible_to_removed_camera(gaussians, scene, removed_cam, current_cameras)

                    # Process gaussian addition for new cameras
                    for new_cam in new_cameras:
                        new_points = _get_points_only_visible_to_new_camera(scene, new_cam, prev_cameras)
                        if new_points:
                            _add_gaussians_from_points(gaussians, new_points, opt_args)

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
                exit(1)
                # Process gaussian removal for removed cameras
                for removed_cam in removed_cameras:
                    _remove_gaussians_only_visible_to_removed_camera(gaussians, scene, removed_cam, current_cameras)

                # Process gaussian addition for new cameras
                for new_cam in new_cameras:
                    new_points = _get_points_only_visible_to_new_camera(scene, new_cam, prev_cameras)
                    if new_points:
                        _add_gaussians_from_points(gaussians, new_points, opt_args)

            # Print Gaussian count after COLMAP initialization
            n_gaussians_colmap = len(gaussians.get_xyz)
            utils.print_rank_0(f"✅ COLMAP LOADED: {n_gaussians_colmap} Gaussians initialized")

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")

    utils.check_initial_gpu_memory_usage("after init and before training loop")
    return scene, start_from_this_iteration


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

    # Set progress bar description based on progressive state
    previous_state = getattr(args, 'previous_state_data', None)
    progress_desc = "Training progress"
    if previous_state:
        progress_desc = f"Window {previous_state.get('window_number', 0)} progress"

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

    for iteration in range(start_from_this_iteration, opt_args.iterations + 1, args.bsz):
        ema_loss_for_log = _process_iteration(iteration, gaussians, scene, args, timers, strategy_history, train_dataset,
                                            background, pipe_args, progress_bar, ema_loss_for_log, debug_info_printed,
                                            end2end_timers, log_file, n_g_max)

    # Finish training
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)
    log_file.write(f"Max Memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB.\n")
    progress_bar.close()


def _process_iteration(iteration, gaussians, scene, args, timers, strategy_history, train_dataset,
                      background, pipe_args, progress_bar, ema_loss_for_log, debug_info_printed,
                      end2end_timers, log_file, n_g_max):
    """Process single iteration"""
    # Setup iteration
    _setup_iteration(iteration, gaussians, args, progress_bar, ema_loss_for_log, timers)

    # Prepare camera data and strategies
    batched_cameras, batched_strategies, gpuid2tasks = _prepare_camera_data(
        train_dataset, strategy_history, args, timers)

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


def _prepare_camera_data(train_dataset, strategy_history, args, timers):
    """Prepare camera data and workload division strategies"""
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
        if args.backend == "gsplat":
            gsplat_densification(iteration, scene, gaussians, n_g_max, batched_screenspace_pkg)
        else:
            densification(iteration, scene, gaussians, n_g_max, batched_screenspace_pkg)

        # Save Checkpoints
        checkpoint_condition = any([iteration <= checkpoint_iteration < iteration + args.bsz for checkpoint_iteration in args.checkpoint_iterations])
        if checkpoint_condition:
            last_checkpoint_path = _handle_checkpoints(iteration, scene, gaussians, args, end2end_timers, log_file)
            # Store for progressive training JSON state
            if not hasattr(args, 'all_checkpoint_paths'):
                args.all_checkpoint_paths = {}
            args.all_checkpoint_paths[utils.GLOBAL_RANK] = last_checkpoint_path


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

    # Save latest checkpoint path to file for progressive trainer
    latest_checkpoint_file = os.path.join(scene.model_path, "latest_checkpoint.txt")
    with open(latest_checkpoint_file, 'w') as f:
        f.write(path_ckpt)
    end2end_timers.start()

    return path_ckpt


def _finalize_training(args, opt_args, gaussians, log_file):
    """Finalize training and print summary"""
    previous_state = getattr(args, 'previous_state_data', None)


    # Save checkpoint paths for progressive training (temporary file)
    if getattr(args, 'is_progressive_training', False):
        window_num = previous_state.get('window_number', 0) if previous_state else 0
        checkpoint_paths_file = os.path.join(args.model_path, f"window_{window_num}_checkpoint_paths.json")

        # Create empty dict if no checkpoints were saved
        if not hasattr(args, 'all_checkpoint_paths'):
            args.all_checkpoint_paths = {}
            utils.print_rank_0(f"⚠️  No checkpoints were saved in this window")

        # Collect all checkpoint paths from all GPUs using distributed communication
        if utils.WORLD_SIZE > 1:
            # Gather all checkpoint paths from all ranks
            import torch.distributed as dist

            # Each rank prepares its checkpoint path data
            local_checkpoint_data = {str(utils.GLOBAL_RANK): args.all_checkpoint_paths.get(utils.GLOBAL_RANK, None)}

            # Gather all checkpoint data on rank 0
            all_checkpoint_data = [None] * utils.WORLD_SIZE
            dist.all_gather_object(all_checkpoint_data, local_checkpoint_data)

            if utils.GLOBAL_RANK == 0:
                # Combine all checkpoint paths from all ranks
                combined_checkpoint_paths = {}
                for rank_data in all_checkpoint_data:
                    if rank_data:
                        combined_checkpoint_paths.update(rank_data)

                # Save combined checkpoint paths (temporary - will be moved to state.json)
                with open(checkpoint_paths_file, 'w') as f:
                    json.dump(combined_checkpoint_paths, f, indent=2)
                utils.print_rank_0(f"💾 Saved checkpoint paths to: {checkpoint_paths_file}")
                utils.print_rank_0(f"   Saved paths: {combined_checkpoint_paths}")
        else:
            # Single GPU case
            with open(checkpoint_paths_file, 'w') as f:
                json.dump(args.all_checkpoint_paths, f, indent=2)
            utils.print_rank_0(f"💾 Saved checkpoint paths to: {checkpoint_paths_file}")
            utils.print_rank_0(f"   Saved paths: {args.all_checkpoint_paths}")

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

        utils.print_rank_0(f"   Cameras in this window: {len(previous_state.get('current_window_cameras', []))}")
        utils.print_rank_0(f"   Global progress: {len(previous_state.get('processed_cameras', [])) + len(previous_state.get('current_window_cameras', []))} / {previous_state.get('total_cameras', 0)} cameras")

        log_file.write(f"\nProgressive Window Training Complete:\n")
        log_file.write(f"  Window: {previous_state.get('iteration_name', 'unknown')}\n")
        log_file.write(f"  Gaussian counts per GPU: {all_gaussian_counts.tolist() if utils.DEFAULT_GROUP.size() > 1 else all_gaussian_counts}\n")
        log_file.write(f"  Total gaussians: {total_gaussians:,}\n")
        log_file.write(f"  Global camera progress: {len(previous_state.get('processed_cameras', [])) + len(previous_state.get('current_window_cameras', []))} / {previous_state.get('total_cameras', 0)}\n")
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

    # Get COLMAP points from scene
    if hasattr(scene, 'point_cloud') and hasattr(scene.point_cloud, 'points'):
        colmap_points = scene.point_cloud.points  # [N, 3] array
    else:
        utils.print_rank_0("⚠️  No COLMAP points found in scene")
        return []

    # Get camera data
    if not hasattr(scene, 'cameras') or new_camera_id not in scene.cameras:
        utils.print_rank_0(f"⚠️  Camera {new_camera_id} not found in scene")
        return []

    new_camera = scene.cameras[new_camera_id]

    # Check visibility for each point
    new_only_points = []

    for i, point_3d in enumerate(colmap_points):
        # Check if point is visible to new camera
        visible_to_new = _is_point_visible_to_camera(point_3d, new_camera_id, scene)

        if not visible_to_new:
            continue

        # Check if point is visible to any current camera
        visible_to_current = False
        for cam_id in current_cameras:
            if cam_id in scene.cameras and _is_point_visible_to_camera(point_3d, cam_id, scene):
                visible_to_current = True
                break

        # If visible to new camera but not to any current camera, include it
        if not visible_to_current:
            new_only_points.append(point_3d)

    utils.print_rank_0(f"📊 Found {len(new_only_points)} points only visible to camera {new_camera_id}")
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


# Legacy function - no longer used in new cams_init/cams_prev design
# def _handle_progressive_gaussian_removal(gaussians, scene, args):
#     """
#     Handle progressive gaussian removal based on removed camera
#     """
#     if args.progressive_removed_camera != -1:
#         n_gaussians_before_removal = len(gaussians.get_xyz)
#         current_cameras = []
#         if args.progressive_current_cameras:
#             current_cameras = [int(x) for x in args.progressive_current_cameras.split(",")]
#         utils.print_rank_0(f"📊 Gaussians before removal: {n_gaussians_before_removal}")
#         utils.print_rank_0(f"📊 Current cameras: {current_cameras}")
#         n_removed = _remove_gaussians_only_visible_to_removed_camera(
#             gaussians, scene, args.progressive_removed_camera, current_cameras
#         )
#         n_gaussians_after_removal = len(gaussians.get_xyz)
#         utils.print_rank_0(f"✅ PROGRESSIVE GAUSSIAN REMOVAL: Removed {n_removed} gaussians only visible to camera {args.progressive_removed_camera}")


# Legacy function - no longer used in new cams_init/cams_prev design
# def _handle_progressive_gaussian_addition(gaussians, scene, args, opt_args):
#     """
#     Handle progressive gaussian addition based on new camera
#     """
#     if args.progressive_new_camera != -1:
#         n_gaussians_before = len(gaussians.get_xyz)
#         current_cameras = []
#         if args.progressive_current_cameras:
#             current_cameras = [int(x) for x in args.progressive_current_cameras.split(",")]
#         new_points = _get_points_only_visible_to_new_camera(scene, args.progressive_new_camera, current_cameras)
#         if len(new_points) > 0:
#             _add_gaussians_from_points(gaussians, new_points, opt_args)
#             n_gaussians_after = len(gaussians.get_xyz)
#             n_added = n_gaussians_after - n_gaussians_before
#             utils.print_rank_0(f"✅ PROGRESSIVE GAUSSIAN ADDITION: Added {n_added} gaussians for camera {args.progressive_new_camera} (only visible points)")
#         else:
#             utils.print_rank_0(f"⚠️  PROGRESSIVE GAUSSIAN ADDITION: No points found only visible to camera {args.progressive_new_camera}")


def _remove_gaussians_only_visible_to_removed_camera(gaussians, scene, removed_camera_id, current_cameras, margin_pixels=10):
    """
    Remove gaussians that are only visible to the removed camera (not visible to current cameras)
    Uses batched projection for efficiency.

    Args:
        gaussians: GaussianModel to remove gaussians from
        scene: Scene object containing camera data
        removed_camera_id: ID of the removed camera
        current_cameras: List of current window camera IDs (excluding removed camera)
        margin_pixels: Margin in pixels for visibility check
                      - Positive: excludes boundary points (conservative removal)
                      - Negative: includes nearly-visible points (aggressive removal)
                      - Default 10: reasonable safety margin

    Returns:
        int: Number of gaussians removed
    """
    import numpy as np
    import torch

    # Get current gaussian positions
    gaussian_xyz = gaussians.get_xyz.detach().cpu().numpy()  # [N, 3]
    n_gaussians = len(gaussian_xyz)

    if n_gaussians == 0:
        utils.print_rank_0("⚠️  No gaussians to process for removal")
        return 0

    utils.print_rank_0(f"📊 Checking visibility for {n_gaussians} gaussians (margin: {margin_pixels}px)...")

    # Step 1: Check which gaussians are visible to removed camera (batched)
    visible_to_removed = _check_points_visibility_batch(gaussian_xyz, removed_camera_id, scene, margin_pixels)
    n_visible_to_removed = np.sum(visible_to_removed)
    utils.print_rank_0(f"📊 {n_visible_to_removed} gaussians visible to removed camera {removed_camera_id}")

    # Step 2: For gaussians visible to removed camera, check if they're visible to any current camera
    # Create visibility matrix: [N_visible, M_cameras]
    visible_to_current = np.zeros(n_gaussians, dtype=bool)

    for cam_id in current_cameras:
        if cam_id not in scene.all_cameras:
            utils.print_rank_0(f"⚠️  Camera {cam_id} not found in scene.all_cameras")
            continue

        # Check visibility for all gaussians to this camera (batched)
        cam_visible = _check_points_visibility_batch(gaussian_xyz, cam_id, scene, margin_pixels)
        visible_to_current |= cam_visible  # Logical OR - visible to at least one current camera

    # Step 3: Determine which gaussians to remove
    # Remove if: visible to removed camera AND NOT visible to any current camera
    gaussians_to_remove_mask = visible_to_removed & ~visible_to_current
    n_removed = np.sum(gaussians_to_remove_mask)

    # Remove gaussians using boolean mask (True = remove, False = keep)
    if n_removed > 0:
        prune_mask = torch.tensor(gaussians_to_remove_mask, dtype=torch.bool, device=gaussians.get_xyz.device)
        gaussians.prune_points(prune_mask)
        utils.print_rank_0(f"✅ Removed {n_removed} gaussians only visible to camera {removed_camera_id}")
    else:
        utils.print_rank_0(f"📊 No gaussians found only visible to camera {removed_camera_id}")

    return n_removed


def _add_gaussians_from_points(gaussians, new_points, opt_args):
    """
    Add new gaussians initialized from 3D points

    Args:
        gaussians: GaussianModel to add gaussians to
        new_points: List of 3D points to create gaussians from
        opt_args: Optimization arguments
    """
    import torch
    import numpy as np

    if len(new_points) == 0:
        return

    try:
        # Convert points to tensor
        new_points_tensor = torch.tensor(new_points, dtype=torch.float32, device="cuda")

        # Initialize basic gaussian parameters for new points
        N = len(new_points)

        # Random colors (will be optimized during training)
        colors = torch.rand(N, 3, dtype=torch.float32, device="cuda")

        # Small initial scales
        scales = torch.ones(N, 3, dtype=torch.float32, device="cuda") * 0.01

        # Random rotations (quaternions)
        rotations = torch.randn(N, 4, dtype=torch.float32, device="cuda")
        rotations = rotations / rotations.norm(dim=-1, keepdim=True)  # Normalize quaternions

        # Small initial opacity
        opacities = torch.ones(N, 1, dtype=torch.float32, device="cuda") * 0.1

        # Add gaussians to the model
        # This is a simplified version - actual implementation depends on GaussianModel structure
        gaussians.densification_postfix(
            new_points_tensor, colors, opacities, scales, rotations
        )

        utils.print_rank_0(f"✅ Successfully added {N} new gaussians")

    except Exception as e:
        utils.print_rank_0(f"⚠️  Error adding gaussians from points: {e}")
        import traceback
        utils.print_rank_0(traceback.format_exc())
