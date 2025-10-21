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
from utils.graphics_utils import filter_pc_by_visibility
import utils.general_utils as utils
import torch


class Scene:

    gaussians: GaussianModel

    '''
    def __init__(
        self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True, load_from_checkpoint=False, progressive_dataset=None
    ):
        """
        Initialize Scene object.
        :param args: Arguments object containing configuration
        :param gaussians: GaussianModel instance
        :param load_iteration: Iteration to load from checkpoint
        :param shuffle: Whether to shuffle cameras
        :param load_from_checkpoint: Whether loading from checkpoint
        :param progressive_dataset: Progressive dataset for training
        """
        self._initialize_basic_attributes(args, gaussians, load_iteration)

        if load_from_checkpoint:
            self._handle_checkpoint_loading(args, progressive_dataset)
            return

        scene_info = self._load_scene_data(args)
        self._initialize_cameras(args, scene_info, shuffle)
        self._initialize_point_cloud(args, scene_info)
    '''

    # ===== ORIGINAL __init__ METHOD (BACKUP) =====
    #'''
    def __init__(
        self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True, load_from_checkpoint=False, progressive_dataset=None, skip_gaussian_init=False, train_view_ids=None, test_view_ids=None
    ):
        """b
        :param path: Path to colmap scene main folder.
        :param train_view_ids: List of camera IDs to use for training (for progressive mode)
        :param test_view_ids: List of camera IDs to use for testing (for progressive mode)
        """
        init_cameras = None
        if hasattr(args, 'cams_init') and args.cams_init:
            # Initial window - filter to specified cameras
            init_cameras = [int(x) for x in args.cams_init.split(",")]
            utils.print_rank_0(f"🔍 Scene.__init__: Got init_cameras from args.cams_init: {init_cameras}")

        # Override init_cameras with train_view_ids if provided (for progressive mode)
        if train_view_ids is not None:
            init_cameras = train_view_ids
            utils.print_rank_0(f"🔍 Scene.__init__: Overriding with train_view_ids: {train_view_ids}")

        utils.print_rank_0(f"🔍 Scene.__init__: Final init_cameras: {init_cameras}")

        self.model_path = args.model_path
        #print(f'self.model_path : {self.model_path}');  exit(1)
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

        # Store args for later use in progressive training
        self.args = args

        utils.print_rank_0(f"🔍 Scene.__init__: load_from_checkpoint={load_from_checkpoint}, skip_gaussian_init={skip_gaussian_init}")

        if load_from_checkpoint:
            # Skip COLMAP loading when loading from checkpoint
            # Cameras and scene info will be restored from checkpoint
            self.train_cameras = {}
            self.test_cameras = {}
            self.all_cameras = {}  # Initialize empty for checkpoint loading
            self.cameras_extent = 1.0  # Will be updated when checkpoint is loaded
            utils.set_img_size(800, 800)  # Temporary values, will be updated from checkpoint
            utils.print_rank_0("🚫 Scene.__init__: SKIPPING COLMAP LOADING - will use checkpoint data")

            # If progressive dataset is provided, add cameras from it
            if progressive_dataset:
                print("📊 Scene.__init__: ADDING CAMERAS from progressive dataset")
                self._add_cameras_from_progressive_dataset(progressive_dataset)

            return  # Skip the rest of initialization
        else:
            # Normal COLMAP loading
            print("📁 Scene.__init__: NORMAL COLMAP LOADING")
            # Check if we're using Colmap format
            has_colmap = False

            # Case 1: Direct directories provided
            if args.dir_sparse and args.dir_images:
                has_colmap = True
            # Case 2: Traditional source_path structure
            elif args.source_path and os.path.exists(os.path.join(args.source_path, "sparse")):
                has_colmap = True

            if has_colmap:  # This is the format from colmap.
                track_by_projection = getattr(args, 'track_by_projection', False)
                scene_info = sceneLoadTypeCallbacks["Colmap"](
                    args.source_path, args.images, args.eval, args.llffhold,
                    args.dir_images, args.dir_sparse, track_by_projection
                )

                # Store all camera infos from COLMAP for progressive training (without loading images)
                self.all_cameras = {}
                for cam_info in scene_info.train_cameras:
                    self.all_cameras[cam_info.uid] = cam_info
                if args.eval:
                    for cam_info in scene_info.test_cameras:
                        self.all_cameras[cam_info.uid] = cam_info

                utils.print_rank_0(f"📊 Stored {len(self.all_cameras)} total cameras from COLMAP in Scene constructor")

            elif "matrixcity" in args.source_path:  # This is for matrixcity
                scene_info = sceneLoadTypeCallbacks["City"](
                    args.source_path,
                    args.random_background,
                    args.white_background,
                    llffhold=args.llffhold,
                )
            else:
                raise ValueError("No valid dataset found in the source path")

        # Store point cloud for progressive training (to add gaussians from new cameras)
        self.point_cloud = scene_info.point_cloud
        if hasattr(scene_info.point_cloud, 'points'):
            utils.print_rank_0(f"✅ Scene.__init__: Stored point_cloud with {len(scene_info.point_cloud.points)} points")
        else:
            utils.print_rank_0(f"⚠️  Scene.__init__: point_cloud has no 'points' attribute")

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

        # Only shuffle if not in deterministic mode
        if shuffle and not args.deterministic:
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

        print(f'dataset_size_in_GB : {dataset_size_in_GB}, args.preload_dataset_to_gpu_threshold : {args.preload_dataset_to_gpu_threshold}'); #exit(1)

        if dataset_size_in_GB < args.preload_dataset_to_gpu_threshold:  # 10GB memory limit for dataset
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
        # Note: self.all_cameras already initialized and populated from COLMAP loading above
        if args.num_train_cameras >= 0:
            train_cameras = scene_info.train_cameras[: args.num_train_cameras]
        else:
            train_cameras = scene_info.train_cameras
        #print(f'type(train_cameras[0].uid)) : {type(train_cameras[0].keys())}');  exit(1)
        if init_cameras:
            utils.print_rank_0(f'🔍 Scene.__init__: Filtering train_cameras')
            utils.print_rank_0(f'  - Before: {len(train_cameras)} cameras')
            utils.print_rank_0(f'  - Filter IDs (init_cameras): {init_cameras}')
            utils.print_rank_0(f'  - Available UIDs: {[cam.uid for cam in train_cameras[:10]]}...')
            train_cameras = [cam for cam in train_cameras if cam.uid in init_cameras]
            utils.print_rank_0(f'  - After: {len(train_cameras)} cameras')
            if len(train_cameras) > 0:
                utils.print_rank_0(f'  - Filtered camera UIDs: {[cam.uid for cam in train_cameras]}')
        #print(f'args.normalize : {args.normalize}')
        if args.normalize:
            self.train_cameras = cameraList_from_camInfos(train_cameras, scene_info.nerf_normalization, args)
        else:
            self.train_cameras = cameraList_from_camInfos(train_cameras, None, args)
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

            # Filter test cameras based on test_view_ids or init_cameras
            test_filter_ids = test_view_ids if test_view_ids is not None else init_cameras
            if test_filter_ids:
                #print(f'len(test_cameras) b4 : {len(test_cameras)}');
                test_cameras = [cam for cam in test_cameras if cam.uid in test_filter_ids]
                #print(f'len(test_cameras) after : {len(test_cameras)}');  exit(1)

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

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter)
                )
            )
        elif hasattr(args, "load_ply_path"):
            self.gaussians.load_ply(args.load_ply_path)
        elif not skip_gaussian_init:
            if init_cameras:
                #print(f"Original point cloud size: {len(scene_info.point_cloud.points)}")
                all_cameras = self.train_cameras[:]
                if self.test_cameras is not None:
                    all_cameras.extend(self.test_cameras)
                pc_sub = filter_pc_by_visibility(scene_info.point_cloud, all_cameras)
                #print(f"Filtered point cloud size: {len(pc_sub.points)}")
                #exit(1)
                self.gaussians.create_from_pcd(pc_sub, self.cameras_extent)

            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        else:
            utils.print_rank_0("🚫 SKIPPING GAUSSIAN INITIALIZATION - Using existing gaussians from checkpoint")
            
            
        utils.check_initial_gpu_memory_usage("after initializing point cloud")
        utils.log_cpu_memory_usage("after loading initial 3dgs points")

        # Debug: Print self.all_cameras status at end of constructor
        if hasattr(self, 'all_cameras') and self.all_cameras:
            utils.print_rank_0(f"✅ Scene constructor completed: {len(self.all_cameras)} cameras in self.all_cameras")
            #sample_ids = list(self.all_cameras.keys())[:5]  # Show first 5 camera IDs
            utils.print_rank_0(f"   self.all_cameras IDs: {self.all_cameras.keys()}")
        else:
            utils.print_rank_0("⚠️  Scene constructor completed: self.all_cameras is empty or not initialized")
        i_win = -1
        # model_path가 "output/model_window_3" 형태인 경우
        if 'window_' in self.model_path:
            i_win = int(self.model_path.split('window_')[-1])
        elif 'initial' in self.model_path:
            i_win = 0
        print(f'i_win : {i_win}');
        # Removed window limit check to allow all windows to train
        # if 2 * 2 * 2 * 2 < i_win:
        #     exit(1)
            
    #'''

    def _initialize_basic_attributes(self, args, gaussians, load_iteration):
        """Initialize basic scene attributes."""
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args  # Store args for later use in progressive training

        # For progressive training: store all cameras from COLMAP
        self.all_cameras = None  # Will store all camera objects from COLMAP

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        utils.log_cpu_memory_usage("before loading images meta data")

    def _handle_checkpoint_loading(self, args, progressive_dataset):
        """Handle scene initialization when loading from checkpoint."""
        # Skip COLMAP loading when loading from checkpoint
        # Cameras and scene info will be restored from checkpoint
        self.train_cameras = {}
        self.test_cameras = {}
        self.cameras_extent = 1.0  # Will be updated when checkpoint is loaded
        utils.set_img_size(800, 800)  # Temporary values, will be updated from checkpoint
        print("🚫 Scene.__init__: SKIPPING COLMAP LOADING - will use checkpoint data")

        # If progressive dataset is provided, add cameras from it
        if progressive_dataset:
            print("📊 Scene.__init__: ADDING CAMERAS from progressive dataset")
            self._add_cameras_from_progressive_dataset(progressive_dataset)

    def _load_scene_data(self, args):
        """Load scene data from COLMAP or other formats."""
        print("📁 Scene.__init__: NORMAL COLMAP LOADING")

        # Check if we're using Colmap format
        has_colmap = False

        # Case 1: Direct directories provided
        if args.dir_sparse and args.dir_images:
            has_colmap = True
        # Case 2: Traditional source_path structure
        elif args.source_path and os.path.exists(os.path.join(args.source_path, "sparse")):
            has_colmap = True

        if has_colmap:  # This is the format from colmap.
            track_by_projection = getattr(args, 'track_by_projection', False)
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, args.llffhold,
                args.dir_images, args.dir_sparse, track_by_projection
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

        # Save input.ply and cameras.json if not loading from iteration
        if not self.loaded_iter:
            self._save_initial_files(scene_info)

        return scene_info

    def _save_initial_files(self, scene_info):
        """Save initial PLY and camera JSON files."""
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

    def _initialize_cameras(self, args, scene_info, shuffle):
        """Initialize training and test cameras."""
        log_file = utils.get_log_file()

        # Parse initial cameras if specified
        init_cameras = None
        if hasattr(args, 'cams_init') and args.cams_init:
            init_cameras = [int(x) for x in args.cams_init.split(",")]

        # Only shuffle if not in deterministic mode
        if shuffle and not args.deterministic:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        utils.log_cpu_memory_usage("before decoding images")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # Set image size to global variable
        orig_w, orig_h = (
            scene_info.train_cameras[0].width,
            scene_info.train_cameras[0].height,
        )
        utils.set_img_size(orig_h, orig_w)

        # Calculate and log dataset size
        self._handle_dataset_preloading(args, scene_info, orig_w, orig_h, log_file)

        # Note: self.all_cameras already populated in constructor from COLMAP data
        utils.print_rank_0(f"📊 Using {len(self.all_cameras)} total cameras already loaded from COLMAP")

        # Initialize training cameras (filtered or full)
        self._setup_train_cameras(args, scene_info, init_cameras, log_file)

        # Initialize test cameras if evaluation is enabled
        if args.eval:
            self._setup_test_cameras(args, scene_info, init_cameras, log_file)

        utils.check_initial_gpu_memory_usage("after Loading all images")
        utils.log_cpu_memory_usage("after decoding images")

    def _handle_dataset_preloading(self, args, scene_info, orig_w, orig_h, log_file):
        """Handle dataset size calculation and GPU preloading logic."""
        dataset_size_in_GB = (
            1.0
            * (len(scene_info.train_cameras) + len(scene_info.test_cameras))
            * orig_w
            * orig_h
            * 3
            / 1e9
        )
        log_file.write(f"Dataset size: {dataset_size_in_GB} GB\n")
        print(f'dataset_size_in_GB : {dataset_size_in_GB}, args.preload_dataset_to_gpu_threshold : {args.preload_dataset_to_gpu_threshold}')

        if dataset_size_in_GB < args.preload_dataset_to_gpu_threshold:
            log_file.write(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable local_sampling and distributed_dataset_storage.\n"
            )
            print(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable local_sampling and distributed_dataset_storage."
            )
            args.preload_dataset_to_gpu = True
            args.local_sampling = False
            args.distributed_dataset_storage = False

    def _setup_train_cameras(self, args, scene_info, init_cameras, log_file):
        """Setup training cameras."""
        utils.print_rank_0("Decoding Training Cameras")

        if args.num_train_cameras >= 0:
            train_cameras = scene_info.train_cameras[:args.num_train_cameras]
        else:
            train_cameras = scene_info.train_cameras

        if init_cameras:
            train_cameras = [cam for cam in train_cameras if cam.uid in init_cameras]

        if args.normalize:
            self.train_cameras = cameraList_from_camInfos(train_cameras, scene_info.nerf_normalization, args)
        else:
            self.train_cameras = cameraList_from_camInfos(train_cameras, None, args)

        log_file.write("Number of local training cameras: {}\n".format(len(self.train_cameras)))
        if len(self.train_cameras) > 0:
            log_file.write(
                "Image size: {}x{}\n".format(
                    self.train_cameras[0].image_height,
                    self.train_cameras[0].image_width,
                )
            )

    def _setup_test_cameras(self, args, scene_info, init_cameras, log_file):
        """Setup test cameras."""
        utils.print_rank_0("Decoding Test Cameras")

        if args.num_test_cameras >= 0:
            test_cameras = scene_info.test_cameras[:args.num_test_cameras]
        else:
            test_cameras = scene_info.test_cameras

        if init_cameras:
            test_cameras = [cam for cam in test_cameras if cam.uid in init_cameras]

        self.test_cameras = cameraList_from_camInfos(test_cameras, args)

        log_file.write("Number of local test cameras: {}\n".format(len(self.test_cameras)))
        if len(self.test_cameras) > 0:
            log_file.write(
                "Image size: {}x{}\n".format(
                    self.test_cameras[0].image_height,
                    self.test_cameras[0].image_width,
                )
            )

    def _initialize_point_cloud(self, args, scene_info):
        """Initialize point cloud from scene data or checkpoint."""
        # Parse initial cameras if specified
        init_cameras = None
        if hasattr(args, 'cams_init') and args.cams_init:
            init_cameras = [int(x) for x in args.cams_init.split(",")]

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter)
                )
            )
        elif hasattr(args, "load_ply_path"):
            self.gaussians.load_ply(args.load_ply_path)
        else:
            if init_cameras:
                #print(f"Original point cloud size: {len(scene_info.point_cloud.points)}")
                all_cameras = self.train_cameras[:]
                if self.test_cameras is not None:
                    all_cameras.extend(self.test_cameras)
                pc_sub = filter_pc_by_visibility(scene_info.point_cloud, all_cameras)
                #print(f"Filtered point cloud size: {len(pc_sub.points)}")
                #exit(1)
                #self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
                self.gaussians.create_from_pcd(pc_sub, self.cameras_extent)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        utils.check_initial_gpu_memory_usage("after initializing point cloud")
        utils.log_cpu_memory_usage("after loading initial 3dgs points")

    def save(self, iteration, loss, n_gpu):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{:06d}".format(iteration)
        )
        n_gauss = self.gaussians.get_xyz.shape[0] * n_gpu
        
        # Save main combined PLY file
        main_ply_path = os.path.join(point_cloud_path, f'point_cloud_i_{iteration:05d}_g_{n_gauss:08d}_l_{loss:.3f}.ply')
        
        # Track save count to decide between save_ply and save_ply_debug
        if not hasattr(self, '_gaussian_save_count'):
            self._gaussian_save_count = 0
        self._gaussian_save_count += 1
        
        if self._gaussian_save_count == 1 or self._gaussian_save_count == 5:
            # Use save_ply_debug to create both main PLY and GPU-specific highlighted PLYs
            if utils.LOCAL_RANK == 0:
                print(f"Saving with GPU highlighting (save #{self._gaussian_save_count})")
            self.gaussians.save_ply_debug(main_ply_path)
        else:
            # Use normal save_ply for regular saves
            self.gaussians.save_ply(main_ply_path)


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


    def _add_cameras_from_progressive_dataset(self, progressive_dataset):
        """
        Add cameras from progressive dataset (memory data, not files)
        """
        try:
            from scene.cameras import Camera
            from scene.dataset_readers import fetchPly
            from utils.graphics_utils import fov2focal, focal2fov
            from PIL import Image
            import numpy as np

            all_cameras = progressive_dataset['cameras']  # All camera info
            selected_images = progressive_dataset['images']  # Only selected camera images

            print(f"📊 Adding {len(selected_images)} cameras from progressive dataset")

            for img_id, img_data in selected_images.items():
                # Get corresponding camera info
                cam_id = img_data['camera_id']
                if cam_id not in all_cameras:
                    print(f"❌ Camera {cam_id} not found for image {img_id}")
                    continue

                cam_data = all_cameras[cam_id]

                # Convert quaternion to rotation matrix
                qw, qx, qy, qz = img_data['qw'], img_data['qx'], img_data['qy'], img_data['qz']
                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                ])

                # Translation vector
                T = np.array([img_data['tx'], img_data['ty'], img_data['tz']])

                # Camera intrinsics
                width, height = cam_data['width'], cam_data['height']

                # Get focal length from camera parameters (assuming PINHOLE model)
                if cam_data['model'] == 'PINHOLE':
                    fx = cam_data['params']['fx']
                    fy = cam_data['params']['fy']
                else:
                    # Fallback for other models
                    fx = fy = cam_data['raw_params'][0] if cam_data['raw_params'] else width

                # Calculate field of view
                FoVx = focal2fov(fx, width)
                FoVy = focal2fov(fy, height)

                # Load image
                image_path = self.args.source_path / "images" / img_data['name'] if hasattr(self.args, 'source_path') else None
                if image_path and image_path.exists():
                    image = Image.open(image_path)
                else:
                    # Create dummy image if not found
                    image = Image.new('RGB', (width, height), color='black')

                # Convert PIL image to tensor
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                image_tensor = transform(image)

                # Create Camera object
                camera = Camera(
                    colmap_id=img_id,  # Use image ID as colmap_id
                    R=R,
                    T=T,
                    FoVx=FoVx,
                    FoVy=FoVy,
                    image=image_tensor,
                    gt_alpha_mask=None,
                    image_name=img_data['name'],
                    uid=len(self.train_cameras),  # New unique ID
                    data_device=self.args.data_device if hasattr(self.args, 'data_device') else "cuda"
                )

                # Add to train_cameras dict
                self.train_cameras[camera.uid] = camera

            print(f"✅ Added {len(selected_images)} cameras to scene")
            print(f"📊 Total train cameras: {len(self.train_cameras)}")

        except Exception as e:
            print(f"❌ Error adding cameras from progressive dataset: {e}")
            import traceback
            traceback.print_exc()


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
            if not self.args.deterministic:
                indices = torch.randperm(len(self.cur_epoch_cameras))
                self.cur_epoch_cameras = [self.cur_epoch_cameras[i] for i in indices]
            # In deterministic mode, keep original order (no shuffling)

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
