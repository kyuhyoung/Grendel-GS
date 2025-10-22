import torch
import numpy as np
import utils.general_utils as utils


def compute_visibility_prune_mask(gaussians, scene, margin_pixels=20):
    """
    Compute mask of gaussians that are outside ALL camera frustums.

    Args:
        gaussians: GaussianModel containing gaussians to check
        scene: Scene object containing camera data
        margin_pixels: Margin in pixels for visibility check (positive = stricter)

    Returns:
        torch.Tensor: Boolean mask where True = should be pruned (not visible to any camera)
    """
    from train_internal import _check_points_visibility_batch

    gaussian_xyz = gaussians.get_xyz.detach().cpu().numpy()  # [N, 3]
    n_gaussians = len(gaussian_xyz)

    utils.print_rank_0(f"🔍 [compute_visibility_prune_mask] Starting with {n_gaussians} gaussians, margin={margin_pixels}px")

    if n_gaussians == 0:
        utils.print_rank_0("🔍 [compute_visibility_prune_mask] No gaussians to check")
        return torch.zeros(0, dtype=torch.bool, device=gaussians.get_xyz.device)

    # Get current training cameras
    train_cameras = scene.getTrainCameras()
    camera_ids = [cam.uid for cam in train_cameras]
    utils.print_rank_0(f"🔍 [compute_visibility_prune_mask] Checking against training cameras of {camera_ids}")

    if len(train_cameras) == 0:
        utils.print_rank_0("⚠️  No training cameras found, skipping visibility pruning")
        return torch.zeros(n_gaussians, dtype=torch.bool, device=gaussians.get_xyz.device)

    # Check visibility for each camera
    visible_to_any_camera = np.zeros(n_gaussians, dtype=bool)

    for camera in train_cameras:
        cam_visible = _check_points_visibility_batch(
            gaussian_xyz, camera.uid, scene, margin_pixels
        )
        visible_to_any_camera |= cam_visible  # OR operation

    # Prune gaussians NOT visible to any camera
    gaussians_to_prune = ~visible_to_any_camera
    n_to_prune = np.sum(gaussians_to_prune)
    n_visible = np.sum(visible_to_any_camera)

    utils.print_rank_0(f"🔍 [compute_visibility_prune_mask] Result: {n_visible} visible, {n_to_prune} to prune (outside all frustums)")

    # Convert to torch tensor
    prune_mask = torch.tensor(gaussians_to_prune, dtype=torch.bool, device=gaussians.get_xyz.device)
    return prune_mask


def densification(iteration, scene, gaussians, n_g_max, batched_screenspace_pkg):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()

    n_gauss = len(gaussians.get_xyz)
    # Densification
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
    #if not args.disable_auto_densification and iteration <= args.densify_until_iter and n_gauss <= n_g_max:
        # Keep track of max radii in image-space for pruning
        timers.start("densification")

        timers.start("densification_update_stats")
        for radii, visibility_filter, screenspace_mean2D in zip(
            batched_screenspace_pkg["batched_locally_preprocessed_radii"],
            batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"],
            batched_screenspace_pkg["batched_locally_preprocessed_mean2D"],
        ):
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(screenspace_mean2D, visibility_filter)
        timers.stop("densification_update_stats")

        should_densify = iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        )

        if should_densify:
            assert (
                args.stop_update_param == False
            ), "stop_update_param must be false for densification; because it is a flag for debugging."
            # utils.print_rank_0("iteration: {}, bsz: {}, update_interval: {}, update_residual: {}".format(iteration, args.bsz, args.densification_interval, 0))

            timers.start("densify_and_prune")
            size_threshold = 20 if iteration > args.opacity_reset_interval else None
            num_gaussians_before = gaussians.get_xyz.shape[0]

            # Visibility pruning is now handled inside densify_and_prune (after densification)
            gaussians.densify_and_prune(
                args.densify_grad_threshold,
                args.min_opacity,
                scene.cameras_extent,
                size_threshold,
                visibility_prune_mask=None,  # Let densify_and_prune compute it after densification
                scene=scene,  # Pass scene so it can compute visibility mask
            )
            num_gaussians_after = gaussians.get_xyz.shape[0]
            timers.stop("densify_and_prune")
            
            # print(f"[DENSIFY] Iteration {iteration}: {num_gaussians_before} -> {num_gaussians_after} gaussians")

            # redistribute after densify_and_prune, because we have new gaussians to distribute evenly.
            if utils.get_denfify_iter() % args.redistribute_gaussians_frequency == 0:
                num_3dgs_before_redistribute = gaussians.get_xyz.shape[0]
                timers.start("redistribute_gaussians")
                gaussians.redistribute_gaussians()
                timers.stop("redistribute_gaussians")
                num_3dgs_after_redistribute = gaussians.get_xyz.shape[0]

                log_file.write(
                    "iteration[{},{}) redistribute. Now num of 3dgs before redistribute: {}. Now num of 3dgs after redistribute: {}. \n".format(
                        iteration,
                        iteration + args.bsz,
                        num_3dgs_before_redistribute,
                        num_3dgs_after_redistribute,
                    )
                )

            utils.check_memory_usage(log_file, args, iteration, gaussians, n_g_max, before_densification_stop = True)

            utils.inc_densify_iter()

        if (
            utils.check_update_at_this_iter(
                iteration, args.bsz, args.opacity_reset_interval, 0
            )
            and iteration + args.bsz <= args.opacity_reset_until_iter
        ):
            timers.start("reset_opacity")
            gaussians.reset_opacity()
            timers.stop("reset_opacity")

        timers.stop("densification")
    else:
        should_densify = iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        )
        
        if should_densify:
            utils.check_memory_usage(log_file, args, iteration, gaussians, n_g_max, before_densification_stop = False)


def gsplat_densification(iteration, scene, gaussians, n_g_max, batched_screenspace_pkg):
    args = utils.get_args()
    timers = utils.get_timers()
    log_file = utils.get_log_file()
    '''
    print(f'args.disable_auto_densification : {args.disable_auto_densification}')  
    print(f'iteration : {iteration} / {args.densify_from_iter} ~ {args.densify_until_iter}');  exit(1)
    '''
    # Densification
    n_gauss = len(gaussians.get_xyz)
    if not args.disable_auto_densification and iteration <= args.densify_until_iter:
    #if not args.disable_auto_densification and iteration <= args.densify_until_iter and n_gauss <= n_g_max:
        # Keep track of max radii in image-space for pruning
        timers.start("densification")

        timers.start("densification_update_stats")
        image_width = batched_screenspace_pkg["image_width"]
        image_height = batched_screenspace_pkg["image_height"]
        batched_screenspace_mean2D_grad = batched_screenspace_pkg[
            "batched_locally_preprocessed_mean2D"
        ].grad
        for i, (radii, visibility_filter) in enumerate(
            zip(
                batched_screenspace_pkg["batched_locally_preprocessed_radii"],
                batched_screenspace_pkg[
                    "batched_locally_preprocessed_visibility_filter"
                ],
            )
        ):
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.gsplat_add_densification_stats(
                batched_screenspace_mean2D_grad[i],
                visibility_filter,
                image_width,
                image_height,
            )
        timers.stop("densification_update_stats")

        should_densify = iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        )

        if should_densify:
            assert (
                args.stop_update_param == False
            ), "stop_update_param must be false for densification; because it is a flag for debugging."
            # utils.print_rank_0("iteration: {}, bsz: {}, update_interval: {}, update_residual: {}".format(iteration, args.bsz, args.densification_interval, 0))

            timers.start("densify_and_prune")
            size_threshold = 20 if iteration > args.opacity_reset_interval else None
            num_gaussians_before = gaussians.get_xyz.shape[0]

            # Visibility pruning is now handled inside densify_and_prune (after densification)
            gaussians.densify_and_prune(
                args.densify_grad_threshold,
                args.min_opacity,
                scene.cameras_extent,
                size_threshold,
                visibility_prune_mask=None,  # Let densify_and_prune compute it after densification
                scene=scene,  # Pass scene so it can compute visibility mask
            )
            num_gaussians_after = gaussians.get_xyz.shape[0]
            timers.stop("densify_and_prune")
            
            # print(f"[DENSIFY] Iteration {iteration}: {num_gaussians_before} -> {num_gaussians_after} gaussians")

            # redistribute after densify_and_prune, because we have new gaussians to distribute evenly.
            if utils.get_denfify_iter() % args.redistribute_gaussians_frequency == 0:
                num_3dgs_before_redistribute = gaussians.get_xyz.shape[0]
                timers.start("redistribute_gaussians")
                gaussians.redistribute_gaussians()
                timers.stop("redistribute_gaussians")
                num_3dgs_after_redistribute = gaussians.get_xyz.shape[0]

                log_file.write(
                    "iteration[{},{}) redistribute. Now num of 3dgs before redistribute: {}. Now num of 3dgs after redistribute: {}. \n".format(
                        iteration,
                        iteration + args.bsz,
                        num_3dgs_before_redistribute,
                        num_3dgs_after_redistribute,
                    )
                )

            utils.check_memory_usage(log_file, args, iteration, gaussians, n_g_max, before_densification_stop=True)

            utils.inc_densify_iter()

        if (
            utils.check_update_at_this_iter(
                iteration, args.bsz, args.opacity_reset_interval, 0
            )
            and iteration + args.bsz <= args.opacity_reset_until_iter
        ):
            timers.start("reset_opacity")
            gaussians.reset_opacity()
            timers.stop("reset_opacity")

        timers.stop("densification")
    else:
        should_densify = iteration > args.densify_from_iter and utils.check_update_at_this_iter(
            iteration, args.bsz, args.densification_interval, 0
        )
        
        if should_densify:
            utils.check_memory_usage(log_file, args, iteration, gaussians, n_g_max,before_densification_stop = False)
