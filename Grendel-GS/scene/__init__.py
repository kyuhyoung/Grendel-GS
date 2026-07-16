#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from random import randint
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import BasicPointCloud
import utils.general_utils as utils
import torch
import numpy as np

# Adaptive tile imports
from scene.adaptive_tile_utils import (
    TileBBox,
    filter_point_cloud,
    compute_visible_caminfos,
    compute_visible_cameras_and_crops,
    ProjectionDebugInfo,
    apply_crop_to_camera,
)


def check_gaussians_visibility(gaussian_xyz, cameras, verbose=True):
    """
    Check how many gaussians project into at least one camera's view.

    Args:
        gaussian_xyz: Tensor or array of shape (N, 3) - gaussian 3D positions
        cameras: List of Camera objects with projection matrices
        verbose: Print debug info

    Returns:
        (num_visible, total): Number of visible gaussians and total gaussians
    """
    if isinstance(gaussian_xyz, torch.Tensor):
        xyz = gaussian_xyz.detach().cpu().numpy()
    else:
        xyz = np.asarray(gaussian_xyz)

    N = len(xyz)
    visible_mask = np.zeros(N, dtype=bool)

    for cam in cameras:
        # Get camera matrices
        W = cam.image_width
        H = cam.image_height

        # World to camera transformation
        R = cam.R.T  # Camera rotation (transpose for world-to-cam)
        T = cam.T    # Camera translation

        # Project points to camera space
        # cam_xyz = R @ (world_xyz - cam_pos)
        # But in gaussian splatting convention: cam_xyz = R @ world_xyz + T
        xyz_cam = xyz @ R.T + T.reshape(1, 3)

        # Filter by depth (must be in front of camera)
        valid_depth = xyz_cam[:, 2] > 0.1  # Positive z = in front

        if not np.any(valid_depth):
            continue

        # Get intrinsics from FoV
        fx = W / (2 * np.tan(cam.FoVx / 2))
        fy = H / (2 * np.tan(cam.FoVy / 2))
        # Use actual principal point if available (for off-center/cropped cameras)
        cx = cam._cx if hasattr(cam, '_cx') and cam._cx is not None else W / 2
        cy = cam._cy if hasattr(cam, '_cy') and cam._cy is not None else H / 2

        if verbose and utils.GLOBAL_RANK == 0:
            is_offcenter = (hasattr(cam, '_cx') and cam._cx is not None)
            if is_offcenter:
                print(f"[visibility-check] Camera {cam.image_name}: W={W}, H={H}, cx={cx:.1f}, cy={cy:.1f} (off-center)", flush=True)

        # Project to 2D
        x_2d = (xyz_cam[:, 0] / xyz_cam[:, 2]) * fx + cx
        y_2d = (xyz_cam[:, 1] / xyz_cam[:, 2]) * fy + cy

        # Check if within image bounds (the camera has already been cropped)
        in_bounds = (
            valid_depth &
            (x_2d >= 0) & (x_2d < W) &
            (y_2d >= 0) & (y_2d < H)
        )

        visible_mask |= in_bounds

        if verbose:
            num_in_this_cam = np.sum(in_bounds)
            if num_in_this_cam > 0:
                utils.print_rank_0(
                    f"[visibility-check] Camera {cam.image_name}: {num_in_this_cam} gaussians visible"
                )

    num_visible = np.sum(visible_mask)
    return num_visible, N


class Scene:

    gaussians: GaussianModel

    def __init__(
        self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        log_file = utils.get_log_file()

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        utils.log_cpu_memory_usage("before loading images meta data")

        if os.path.exists(
            os.path.join(args.source_path, "sparse")
        ):  # This is the format from colmap.
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, args.llffhold
            )
        elif "matrixcity" in args.source_path:  # This is for matrixcity
            scene_info = sceneLoadTypeCallbacks["City"](
                args.source_path,
                args.random_background,
                args.white_background,
                llffhold=args.llffhold,
            )
        else:
            raise ValueError("No valid dataset found in the source path")

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        utils.log_cpu_memory_usage("before decoding images")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # Set image size to global variable
        orig_w, orig_h = (
            scene_info.train_cameras[0].width,
            scene_info.train_cameras[0].height,
        )
        utils.set_img_size(orig_h, orig_w)
        # Dataset size in GB
        dataset_size_in_GB = (
            1.0
            * (len(scene_info.train_cameras) + len(scene_info.test_cameras))
            * orig_w
            * orig_h
            * 3
            / 1e9
        )
        log_file.write(f"Dataset size: {dataset_size_in_GB} GB\n")
        if (
            dataset_size_in_GB < args.preload_dataset_to_gpu_threshold
        ):  # 10GB memory limit for dataset
            log_file.write(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable local_sampling and distributed_dataset_storage.\n"
            )
            print(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable local_sampling and distributed_dataset_storage."
            )
            args.preload_dataset_to_gpu = True
            args.local_sampling = False  # TODO: Preloading dataset to GPU is not compatible with local_sampling and distributed_dataset_storage for now. Fix this.
            args.distributed_dataset_storage = False

        # Train on original resolution, no downsampling in our implementation.
        utils.print_rank_0("Decoding Training Cameras")
        self.train_cameras = None
        self.test_cameras = None
        if args.num_train_cameras >= 0:
            train_cam_infos = scene_info.train_cameras[: args.num_train_cameras]
        else:
            train_cam_infos = scene_info.train_cameras

        # ============================================
        # Adaptive tile: filter cameras BEFORE loading images
        # ============================================
        tile_bbox = None
        self.camera_crops = {}  # camera_idx -> CropRegion
        visible_cam_crops = None  # List of (idx, crop, debug_info) for visible cameras

        if getattr(args, "adaptive_tile_enabled", False) and getattr(args, "tile_bbox", ""):
            tile_bbox = TileBBox.from_string(args.tile_bbox)

            # Print full scene extent vs tile extent
            pcd = scene_info.point_cloud
            pcd_points = np.asarray(pcd.points)

            # Full scene extent (min/max)
            scene_min = pcd_points.min(axis=0)
            scene_max = pcd_points.max(axis=0)
            scene_size = scene_max - scene_min

            # Outlier-free extent (0.1% ~ 99.9% percentile)
            pct_low, pct_high = 0.1, 99.9
            clean_min = np.percentile(pcd_points, pct_low, axis=0)
            clean_max = np.percentile(pcd_points, pct_high, axis=0)
            clean_size = clean_max - clean_min

            # Current tile extent
            tile_size = np.array([
                tile_bbox.x_max - tile_bbox.x_min,
                tile_bbox.y_max - tile_bbox.y_min,
                tile_bbox.z_max - tile_bbox.z_min
            ])

            # Calculate margin percentage
            margin_pct = ((tile_size / clean_size) - 1.0) * 100 / 2  # divide by 2 since margin is on both sides

            utils.print_rank_0(f"[adaptive-tile] ========== Tile Info ==========")
            utils.print_rank_0(f"[adaptive-tile] Tile ID: {getattr(args, 'tile_id', 'unknown')}")
            utils.print_rank_0(f"[adaptive-tile] Full scene extent (min/max):")
            utils.print_rank_0(f"[adaptive-tile]   X: {scene_min[0]:.3f} ~ {scene_max[0]:.3f} (size: {scene_size[0]:.3f})")
            utils.print_rank_0(f"[adaptive-tile]   Y: {scene_min[1]:.3f} ~ {scene_max[1]:.3f} (size: {scene_size[1]:.3f})")
            utils.print_rank_0(f"[adaptive-tile]   Z: {scene_min[2]:.3f} ~ {scene_max[2]:.3f} (size: {scene_size[2]:.3f})")
            utils.print_rank_0(f"[adaptive-tile] Outlier-free extent ({pct_low}%-{pct_high}% percentile):")
            utils.print_rank_0(f"[adaptive-tile]   X: {clean_min[0]:.3f} ~ {clean_max[0]:.3f} (size: {clean_size[0]:.3f})")
            utils.print_rank_0(f"[adaptive-tile]   Y: {clean_min[1]:.3f} ~ {clean_max[1]:.3f} (size: {clean_size[1]:.3f})")
            utils.print_rank_0(f"[adaptive-tile]   Z: {clean_min[2]:.3f} ~ {clean_max[2]:.3f} (size: {clean_size[2]:.3f})")
            utils.print_rank_0(f"[adaptive-tile] Current tile extent (~{margin_pct[0]:.0f}% margin):")
            utils.print_rank_0(f"[adaptive-tile]   X: {tile_bbox.x_min:.3f} ~ {tile_bbox.x_max:.3f} (size: {tile_size[0]:.3f})")
            utils.print_rank_0(f"[adaptive-tile]   Y: {tile_bbox.y_min:.3f} ~ {tile_bbox.y_max:.3f} (size: {tile_size[1]:.3f})")
            utils.print_rank_0(f"[adaptive-tile]   Z: {tile_bbox.z_min:.3f} ~ {tile_bbox.z_max:.3f} (size: {tile_size[2]:.3f})")
            utils.print_rank_0(f"[adaptive-tile] ===============================")

            crop_margin = getattr(args, "tile_crop_margin", 100)
            ndc_limit = getattr(args, "ndc_limit", 1.0)
            utils.print_rank_0(f"[adaptive-tile] Crop margin: {crop_margin}px, NDC limit: {ndc_limit}")
            total_cams = len(train_cam_infos)

            # Check if visible_cameras was pre-computed by train_adaptive.py
            precomputed_visible = getattr(args, "visible_cameras", "")
            if precomputed_visible:
                # Use pre-computed visible camera list from train_adaptive.py
                visible_camera_names = set(precomputed_visible.split(","))
                utils.print_rank_0(f"[adaptive-tile] Using pre-computed visible cameras: {len(visible_camera_names)}")

                # Filter by name and compute crops for remaining cameras
                # IMPORTANT: Only include cameras that have valid crops!
                filtered_cam_infos = []
                visible_cam_crops = []  # List of (idx, crop, debug_info)
                for idx, cam_info in enumerate(train_cam_infos):
                    if cam_info.image_name in visible_camera_names:
                        # Compute crop for this camera (using point cloud for accuracy)
                        crop_result = compute_visible_caminfos(
                            tile_bbox, [cam_info], margin=crop_margin,
                            points=pcd_points, return_debug_info=True,
                            ndc_limit=ndc_limit
                        )
                        if crop_result:
                            # crop_result is [(0, crop, debug_info)]
                            # Only add camera if crop was successfully computed
                            new_idx = len(filtered_cam_infos)
                            filtered_cam_infos.append(cam_info)
                            visible_cam_crops.append((new_idx, crop_result[0][1], crop_result[0][2]))
                        else:
                            utils.print_rank_0(f"[adaptive-tile] Skipping camera {cam_info.image_name}: no valid crop computed")

                train_cam_infos = filtered_cam_infos
                visible_indices = list(range(len(train_cam_infos)))
            else:
                # Compute visibility here (fallback for direct torchrun invocation)
                crop_results = compute_visible_caminfos(
                    tile_bbox, train_cam_infos, margin=crop_margin,
                    points=pcd_points, return_debug_info=True,
                    ndc_limit=ndc_limit
                )
                visible_indices = [idx for idx, crop, debug in crop_results]

                # Filter train_cam_infos to only visible cameras
                train_cam_infos = [train_cam_infos[idx] for idx in visible_indices]

                # Re-index crops for filtered list (keep debug info)
                visible_cam_crops = [(new_idx, crop, debug) for new_idx, (_, crop, debug) in enumerate(crop_results)]

            # Filter out crops that are too small for distributed rendering
            # Minimum size = num_gpus * BLOCK_SIZE (each GPU needs at least 1 tile)
            # Combined with division_pos_heuristic fix, this ensures no GPU gets 0 tiles
            BLOCK_SIZE = 16  # BLOCK_X = BLOCK_Y = 16
            num_gpus = utils.WORLD_SIZE if hasattr(utils, 'WORLD_SIZE') and utils.WORLD_SIZE > 0 else 4
            MIN_CROP_SIZE = num_gpus * BLOCK_SIZE  # e.g., 8 GPUs → 128 pixels
            if visible_cam_crops:
                original_count = len(visible_cam_crops)
                # Filter crops and track which indices to keep
                filtered_crops = []
                filtered_indices = []
                for idx, crop, debug in visible_cam_crops:
                    if crop.width >= MIN_CROP_SIZE and crop.height >= MIN_CROP_SIZE:
                        filtered_indices.append(idx)
                        filtered_crops.append((len(filtered_crops), crop, debug))
                    else:
                        cam_name = train_cam_infos[idx].image_name
                        utils.print_rank_0(f"[adaptive-tile] Skipping camera {cam_name}: crop {crop.width}x{crop.height} < {MIN_CROP_SIZE}x{MIN_CROP_SIZE}")

                if len(filtered_crops) < original_count:
                    # Re-filter train_cam_infos
                    train_cam_infos = [train_cam_infos[idx] for idx in filtered_indices]
                    visible_cam_crops = filtered_crops
                    utils.print_rank_0(f"[adaptive-tile] Filtered out {original_count - len(filtered_crops)} cameras with small crops")

            utils.print_rank_0(f"[adaptive-tile] Camera visibility (before image loading):")
            utils.print_rank_0(f"[adaptive-tile]   Total cameras: {total_cams}")
            utils.print_rank_0(f"[adaptive-tile]   Visible cameras: {len(train_cam_infos)}")

            # If no cameras remain after filtering, this tile is too small/problematic
            if len(train_cam_infos) == 0:
                utils.print_rank_0(f"[adaptive-tile] ERROR: No cameras with valid crops for this tile!")
                utils.print_rank_0(f"[adaptive-tile] Tile is likely too small or outside camera frustums.")
                # Raise OOM-like error to trigger tile split or skip
                raise RuntimeError("No cameras with valid crops - tile too small")

            if train_cam_infos:
                # Print which cameras are visible
                visible_names = [cam.image_name for cam in train_cam_infos]
                utils.print_rank_0(f"[adaptive-tile]   Visible camera names: {visible_names[:10]}{'...' if len(visible_names) > 10 else ''}")

            utils.print_rank_0(f"[adaptive-tile] Loading only {len(train_cam_infos)} visible camera images (skipping {total_cams - len(train_cam_infos)})")

            # Print crop info BEFORE loading
            if visible_cam_crops:
                max_width = max(crop.width for _, crop, _ in visible_cam_crops)
                max_height = max(crop.height for _, crop, _ in visible_cam_crops)
                utils.print_rank_0(f"[adaptive-tile] Crop regions calculated (uniform size: {max_width}x{max_height})")
                # Print all crop regions
                for idx, crop, debug in visible_cam_crops:
                    cam_name = train_cam_infos[idx].image_name
                    utils.print_rank_0(f"[adaptive-tile]   {idx+1}: {cam_name}: ({crop.x_min},{crop.y_min}) - ({crop.x_max},{crop.y_max}) = {crop.width}x{crop.height}")

            log_file.write(f"[adaptive-tile] Visible cameras: {total_cams} -> {len(train_cam_infos)}\n")

        # Build crop dict for loading (crop during image load, not after)
        load_crops = None
        if visible_cam_crops:
            # Print point cloud info
            points_in_tile_mask = tile_bbox.contains_points(pcd_points)
            num_points_in_tile = np.sum(points_in_tile_mask)
            utils.print_rank_0(f"[adaptive-tile] ========== Point Cloud Info ==========")
            utils.print_rank_0(f"[adaptive-tile] PLY file: {scene_info.ply_path}")
            utils.print_rank_0(f"[adaptive-tile] Total points: {len(pcd_points)}")
            utils.print_rank_0(f"[adaptive-tile] Points in tile: {num_points_in_tile}")
            utils.print_rank_0(f"[adaptive-tile] ======================================")

            load_crops = {idx: crop for idx, crop, _ in visible_cam_crops}
            utils.print_rank_0(f"[adaptive-tile] Applying crop DURING image loading (memory efficient)")
            utils.print_rank_0(f"[adaptive-tile] Crop info per camera ({len(load_crops)} cameras):")
            for idx, crop, debug in visible_cam_crops:
                cam_info = train_cam_infos[idx]
                cam_name = cam_info.image_name
                # Camera intrinsics and pose
                fx = cam_info.width / (2 * np.tan(cam_info.FovX / 2))
                fy = cam_info.height / (2 * np.tan(cam_info.FovY / 2))
                T = np.array(cam_info.T)
                R = np.array(cam_info.R)
                # Extract camera axes in world coordinates
                # R[:, 0] = image right direction, R[:, 1] = image down direction, R[:, 2] = viewing direction (before negation)
                img_right = R[:, 0]  # image X axis in world coords
                img_down = R[:, 1]   # image Y axis in world coords
                # Debug info: raw projected range before margin/clamp
                if debug:
                    raw_x = f"raw_x=[{debug.raw_x_min:.0f}, {debug.raw_x_max:.0f}]"
                    raw_y = f"raw_y=[{debug.raw_y_min:.0f}, {debug.raw_y_max:.0f}]"
                else:
                    raw_x = raw_y = "no_debug"
                utils.print_rank_0(
                    f"[adaptive-tile]   {idx}: {cam_name} | "
                    f"T=[{T[0]:.1f}, {T[1]:.1f}, {T[2]:.1f}] | "
                    f"crop: x={crop.x_min}, y={crop.y_min}, w={crop.width}, h={crop.height} | "
                    f"{raw_x} | {raw_y}"
                )

        # Now load images only for visible/selected cameras
        self.train_cameras = cameraList_from_camInfos(train_cam_infos, args, crops=load_crops)

        # Off-center projection is already applied by adjust_camera_for_crop() in loadCam().
        # Do NOT call apply_crop_to_camera() here — it uses a different (buggy) convention
        # and overwrites the correct projection matrix with wrong values.
        if visible_cam_crops and self.train_cameras:
            for idx, crop, debug in visible_cam_crops:
                if idx < len(self.train_cameras):
                    camera = self.train_cameras[idx]
                    cam_info = train_cam_infos[idx]
                    camera._original_width = cam_info.width
                    camera._original_height = cam_info.height
                    camera._crop_region = crop

        # output the number of cameras in the training set and image size to the log file
        log_file.write(
            "Number of local training cameras: {}\n".format(len(self.train_cameras))
        )
        if len(self.train_cameras) > 0:
            log_file.write(
                "Image size: {}x{}\n".format(
                    self.train_cameras[0].image_height,
                    self.train_cameras[0].image_width,
                )
            )

        if args.eval:
            utils.print_rank_0("Decoding Test Cameras")
            if args.num_test_cameras >= 0:
                test_cameras = scene_info.test_cameras[: args.num_test_cameras]
            else:
                test_cameras = scene_info.test_cameras
            self.test_cameras = cameraList_from_camInfos(test_cameras, args)
            # output the number of cameras in the training set and image size to the log file
            log_file.write(
                "Number of local test cameras: {}\n".format(len(self.test_cameras))
            )
            if len(self.test_cameras) > 0:
                log_file.write(
                    "Image size: {}x{}\n".format(
                        self.test_cameras[0].image_height,
                        self.test_cameras[0].image_width,
                    )
                )

        utils.check_initial_gpu_memory_usage("after Loading all images")
        utils.log_cpu_memory_usage("after decoding images")

        # Store crop info and update global image size (crop already applied during load)
        if visible_cam_crops and self.train_cameras:
            crop_sizes = [(crop.width, crop.height) for _, crop, _ in visible_cam_crops]
            min_w, min_h = min(c[0] for c in crop_sizes), min(c[1] for c in crop_sizes)
            max_w, max_h = max(c[0] for c in crop_sizes), max(c[1] for c in crop_sizes)

            utils.print_rank_0(
                f"[adaptive-tile] Crop sizes applied: min={min_w}x{min_h}, max={max_w}x{max_h}"
            )

            # Store crop info for reference (crop already applied during loading)
            for idx, crop, _ in visible_cam_crops:
                self.camera_crops[idx] = crop

            # Print actual loaded camera sizes
            for idx, cam in enumerate(self.train_cameras):
                utils.print_rank_0(
                    f"[adaptive-tile]   Camera {idx+1} ({cam.image_name}): loaded as {cam.image_width}x{cam.image_height}"
                )

            # Set global image size to max crop size (for any code that needs it)
            utils.set_img_size(max_h, max_w)
            utils.print_rank_0(
                f"[adaptive-tile] Global image size set to max: {max_w}x{max_h}"
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter)
                )
            )
        elif hasattr(args, "load_ply_path") and args.load_ply_path:
            self.gaussians.load_ply(args.load_ply_path)
        elif getattr(args, "pretrained_ply", "") and os.path.exists(args.pretrained_ply):
            # Load pre-trained gaussians from PLY file (for Category 3 OOM resume)
            print(f"\n{'='*60}", flush=True)
            print(f"[Category 3 Resume] Loading pre-trained gaussians", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"  PLY file: {args.pretrained_ply}", flush=True)
            import os as _os
            import json as _json
            ply_size = _os.path.getsize(args.pretrained_ply) / 1024 / 1024
            print(f"  File size: {ply_size:.2f} MB", flush=True)
            self.gaussians.load_ply(args.pretrained_ply, init_training_tensors=True)
            num_gaussians_local = self.gaussians.get_xyz.shape[0]
            # Compute total across all GPUs
            local_count = torch.tensor([num_gaussians_local], device="cuda", dtype=torch.long)
            torch.distributed.all_reduce(local_count, op=torch.distributed.ReduceOp.SUM)
            num_gaussians_total = local_count.item()
            print(f"  Loaded gaussians: TOTAL {num_gaussians_total:,} (local: {num_gaussians_local:,})", flush=True)

            # Validate against expected count from merge validation file
            val_file = args.pretrained_ply.replace('.ply', '.validation.json')
            if _os.path.exists(val_file):
                try:
                    with open(val_file) as f:
                        val_data = _json.load(f)
                    expected_count = val_data.get('expected_count', 0)
                    if expected_count > 0:
                        if num_gaussians_total == expected_count:
                            print(f"  [Validation] ✓ Loaded count matches expected: {num_gaussians_total:,}", flush=True)
                        else:
                            print(f"  [Validation] ✗ Count mismatch! Loaded: {num_gaussians_total:,}, Expected: {expected_count:,}", flush=True)
                            print(f"  [Validation]   Difference: {num_gaussians_total - expected_count:+,}", flush=True)
                except Exception as e:
                    print(f"  [Validation] Warning: Could not read validation file: {e}", flush=True)

            # Check if any gaussians are visible in the camera crop regions
            use_pretrained = True
            if self.train_cameras and num_gaussians_local > 0:
                num_visible, total = check_gaussians_visibility(
                    self.gaussians.get_xyz, self.train_cameras, verbose=True
                )
                print(f"  Visibility check: {num_visible}/{total} gaussians visible in crop regions", flush=True)

                if num_visible == 0:
                    error_msg = (
                        f"\n{'!'*60}\n"
                        f"  FATAL ERROR: NO gaussians visible in camera crop regions!\n"
                        f"  Loaded {total:,} pre-trained gaussians but 0 are visible.\n"
                        f"  This indicates a bug in visibility check or projection matrix.\n"
                        f"  (Likely off-center projection not handled correctly)\n"
                        f"{'!'*60}\n"
                    )
                    print(error_msg, flush=True)
                    raise RuntimeError(
                        f"Visibility check failed: 0/{total} pre-trained gaussians visible. "
                        f"This is a bug - check projection matrix handling for cropped cameras."
                    )

            print(f"  (Training will RESUME from pre-trained state, NOT from scratch)", flush=True)
            print(f"{'='*60}\n", flush=True)

            # Resume 검증 시각화 (best-effort, 자식 bbox + 로드된 가우시안 + visible cameras)
            try:
                from pathlib import Path
                from utils.oom_viz import save_resume_viz
                from scene.adaptive_tile_utils import TileBBox as _RTBB
                _tile_bbox_str = getattr(args, "tile_bbox", None)
                if _tile_bbox_str:
                    _tile_bbox_obj = (_RTBB.from_string(_tile_bbox_str)
                                       if isinstance(_tile_bbox_str, str)
                                       else _tile_bbox_str)
                    _xyz_np = self.gaussians._xyz.detach().cpu().numpy()
                    _cam_pos = []
                    for _c in (self.train_cameras or []):
                        try:
                            _cc = getattr(_c, "camera_center", None)
                            if _cc is None:
                                continue
                            if hasattr(_cc, "detach"):
                                _cc = _cc.detach().cpu().numpy()
                            _cam_pos.append(np.asarray(_cc, dtype=np.float32).reshape(3))
                        except Exception:
                            continue
                    _cam_pos_np = np.stack(_cam_pos, axis=0) if _cam_pos else np.zeros((0, 3), dtype=np.float32)
                    _tile_id = getattr(args, "tile_id", "tile")
                    _viz_dir = Path(getattr(args, "tile_output_dir", "") or args.model_path).parent / "visualizations" / "resume"
                    _rank = utils.GLOBAL_RANK
                    _resume_path = save_resume_viz(
                        _xyz_np,
                        rank=_rank,
                        tile_id=_tile_id,
                        tile_bbox=_tile_bbox_obj,
                        camera_positions=_cam_pos_np,
                        out_dir=_viz_dir,
                        pretrained_ply_path=args.pretrained_ply,
                    )
                    # 검증 모드: resume_viz png 만들고 자식 프로세스 즉시 종료 (학습 시작 안 함)
                    print("\n" + "=" * 60, flush=True)
                    print(f"[resume_viz] STOP: tile={_tile_id} rank={_rank}", flush=True)
                    print(f"  -> {_resume_path}", flush=True)
                    print("=" * 60, flush=True)
                    import sys as _sys
                    _sys.exit(0)
            except SystemExit:
                raise
            except Exception as _viz_e:
                print(f"  [resume_viz] skipped: {_viz_e}", flush=True)
        elif getattr(args, "tile_scene_root", ""):
            utils.print_rank_0(
                "[tile-ooc] Skipping initial point cloud loading; tiles will be streamed on demand"
            )
        else:
            # Filter point cloud if tile_bbox is specified
            pcd = scene_info.point_cloud
            if tile_bbox is not None:
                filtered_points, filtered_colors, filtered_normals = filter_point_cloud(
                    np.asarray(pcd.points),
                    np.asarray(pcd.colors),
                    np.asarray(pcd.normals),
                    tile_bbox
                )
                utils.print_rank_0(
                    f"[adaptive-tile] Filtered points: {len(pcd.points)} -> {len(filtered_points)}"
                )
                pcd = BasicPointCloud(
                    points=filtered_points,
                    colors=filtered_colors,
                    normals=filtered_normals
                )
            self.gaussians.create_from_pcd(pcd, self.cameras_extent)

        # Store tile_bbox for later use
        self.tile_bbox = tile_bbox

        utils.check_initial_gpu_memory_usage("after initializing point cloud")
        utils.log_cpu_memory_usage("after loading initial 3dgs points")

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

    def log_scene_info_to_file(self, log_file, prefix_str=""):

        # Print shape of gaussians parameters.
        log_file.write("xyz shape: {}\n".format(self.gaussians._xyz.shape))
        log_file.write("f_dc shape: {}\n".format(self.gaussians._features_dc.shape))
        log_file.write("f_rest shape: {}\n".format(self.gaussians._features_rest.shape))
        log_file.write("opacity shape: {}\n".format(self.gaussians._opacity.shape))
        log_file.write("scaling shape: {}\n".format(self.gaussians._scaling.shape))
        log_file.write("rotation shape: {}\n".format(self.gaussians._rotation.shape))


class SceneDataset:
    def __init__(self, cameras):
        self.cameras = cameras
        self.camera_size = len(self.cameras)
        self.sample_camera_idx = []
        for i in range(self.camera_size):
            if self.cameras[i].original_image_backup is not None:
                self.sample_camera_idx.append(i)
        # print("Number of cameras with sample images: ", len(self.sample_camera_idx))

        self.cur_epoch_cameras = []
        self.cur_iteration = 0

        self.iteration_loss = []
        self.epoch_loss = []

        self.log_file = utils.get_log_file()
        self.args = utils.get_args()

        self.last_time_point = None
        self.epoch_time = []
        self.epoch_n_sample = []

    @property
    def cur_epoch(self):
        return len(self.epoch_loss)

    @property
    def cur_iteration_in_epoch(self):
        return len(self.iteration_loss)

    def get_one_camera(self, batched_cameras_uid):
        args = utils.get_args()
        if len(self.cur_epoch_cameras) == 0:
            # start a new epoch
            if args.local_sampling:
                self.cur_epoch_cameras = self.sample_camera_idx.copy()
            else:
                self.cur_epoch_cameras = list(range(self.camera_size))
            # random.shuffle(self.cur_epoch_cameras)
            indices = torch.randperm(len(self.cur_epoch_cameras))
            self.cur_epoch_cameras = [self.cur_epoch_cameras[i] for i in indices]

        self.cur_iteration += 1

        idx = 0
        while self.cameras[self.cur_epoch_cameras[idx]].uid in batched_cameras_uid:
            idx += 1
        camera_idx = self.cur_epoch_cameras.pop(idx)
        viewpoint_cam = self.cameras[camera_idx]
        return camera_idx, viewpoint_cam

    def get_batched_cameras(self, batch_size):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras = []
        batched_cameras_uid = []
        for i in range(batch_size):
            _, camera = self.get_one_camera(batched_cameras_uid)
            batched_cameras.append(camera)
            batched_cameras_uid.append(camera.uid)

        return batched_cameras

    def get_batched_cameras_idx(self, batch_size):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras_idx = []
        batched_cameras_uid = []
        for i in range(batch_size):
            idx, camera = self.get_one_camera(batched_cameras_uid)
            batched_cameras_uid.append(camera.uid)
            batched_cameras_idx.append(idx)

        return batched_cameras_idx

    def get_batched_cameras_from_idx(self, idx_list):
        return [self.cameras[i] for i in idx_list]

    def update_losses(self, losses):
        for loss in losses:
            self.iteration_loss.append(loss)
            if len(self.iteration_loss) % self.camera_size == 0:
                self.epoch_loss.append(
                    sum(self.iteration_loss[-self.camera_size :]) / self.camera_size
                )
                self.log_file.write(
                    "epoch {} loss: {}\n".format(
                        len(self.epoch_loss), self.epoch_loss[-1]
                    )
                )
                self.iteration_loss = []
