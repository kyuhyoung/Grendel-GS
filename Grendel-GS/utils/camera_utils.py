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

from scene.cameras import Camera
import numpy as np
import sys
from utils.general_utils import PILtoTorch, get_args, get_log_file
import utils.general_utils as utils
from tqdm import tqdm
from utils.graphics_utils import fov2focal
import time
import multiprocessing
from multiprocessing import shared_memory
import torch
from PIL import Image


def loadCam(args, id, cam_info, decompressed_image=None, return_image=False, crop=None):
    """
    Load a camera with its ground truth image.

    Args:
        args: Arguments
        id: Camera ID
        cam_info: CameraInfo object
        decompressed_image: Optional pre-decompressed image
        return_image: If True, return only the image tensor
        crop: Optional CropRegion to apply during loading (saves memory)
    """
    import math

    orig_w, orig_h = cam_info.width, cam_info.height

    # With crop, we allow different image sizes (will be cropped to different sizes)
    if crop is None:
        assert (
            orig_w == utils.get_img_width() and orig_h == utils.get_img_height()
        ), "All images should have the same size (unless using adaptive tile crop)."

    args = get_args()
    log_file = get_log_file()

    # Determine final resolution (after crop if applicable)
    if crop:
        resolution = crop.width, crop.height
    else:
        resolution = orig_w, orig_h

    # Compute FoV (will be adjusted if cropped)
    FovX = cam_info.FovX
    FovY = cam_info.FovY

    if crop:
        # Adjust FoV to maintain correct focal length after crop
        # focal_x = orig_w / (2 * tan(FoVx/2))
        # new_FoVx = 2 * atan(crop_width / (2 * focal_x))
        tanfovx = math.tan(FovX / 2)
        tanfovy = math.tan(FovY / 2)
        focal_x = orig_w / (2 * tanfovx)
        focal_y = orig_h / (2 * tanfovy)
        FovX = 2 * math.atan(crop.width / (2 * focal_x))
        FovY = 2 * math.atan(crop.height / (2 * focal_y))

    # may use cam_info.uid
    if (
        (
            args.local_sampling
            and args.distributed_dataset_storage
            and utils.GLOBAL_RANK == id % utils.WORLD_SIZE
        )
        or (
            not args.local_sampling
            and args.distributed_dataset_storage
            and utils.LOCAL_RANK == 0
        )
        or (not args.distributed_dataset_storage)
    ):
        if args.time_image_loading:
            start_time = time.time()

        image = Image.open(cam_info.image_path)
        orig_size = image.size  # (width, height)

        # Apply crop during loading (saves memory - don't decode full image)
        if crop:
            # PIL crop: (left, upper, right, lower)
            image = image.crop((crop.x_min, crop.y_min, crop.x_max, crop.y_max))
            worker_name = multiprocessing.current_process().name
            worker_num = worker_name.split("-")[-1] if "-" in worker_name else worker_name
            utils.print_rank_0(f"[loadCam] Camera {id+1}: orig={orig_size}, crop=({crop.x_min},{crop.y_min})-({crop.x_max},{crop.y_max}), after_crop={image.size}, resolution={resolution} (worker {worker_num})")

        resized_image_rgb = PILtoTorch(
            image, resolution, args, log_file, decompressed_image=decompressed_image
        )
        if args.time_image_loading:
            log_file.write(f"PILtoTorch image in {time.time() - start_time} seconds\n")

        # assert resized_image_rgb.shape[0] == 3, "Image should have exactly 3 channels!"
        gt_image = resized_image_rgb[:3, ...].contiguous()
        loaded_mask = None

        # Free the memory: because the PIL image has been converted to torch tensor, we don't need it anymore. And it takes up lots of cpu memory.
        image.close()
        image = None
    else:
        gt_image = None
        loaded_mask = None

    if return_image:
        return gt_image

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=FovX,
        FoVy=FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        image_width=resolution[0],  # Pass explicit size for distributed storage
        image_height=resolution[1],
    )


def load_decompressed_image(params):
    args, id, cam_info, crop = params
    worker_name = multiprocessing.current_process().name  # e.g., "ForkPoolWorker-1"
    result = loadCam(args, id, cam_info, decompressed_image=None, return_image=True, crop=crop)
    return (result, worker_name)


# Modify this code to support shared_memory.SharedMemory to make inter-process communication faster
def decompressed_images_from_camInfos_multiprocess(cam_infos, args, crops=None):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Create a pool of processes (auto-detect based on CPU cores, max 8)
    import os
    num_workers = min(os.cpu_count() or 2, 8)
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Prepare data for processing (include crop for each camera)
        tasks = [
            (args, id, cam_info, crops.get(id) if crops else None)
            for id, cam_info in enumerate(cam_infos)
        ]

        # Process images and print progress
        utils.print_rank_0(f"Loading {total_cameras} images (multiprocess, {num_workers} workers)...")
        for id, (result, worker_name) in enumerate(pool.imap(load_decompressed_image, tasks)):
            crop = crops.get(id) if crops else None
            # Extract worker number from name like "ForkPoolWorker-1" or "SpawnPoolWorker-1"
            worker_num = worker_name.split("-")[-1] if "-" in worker_name else worker_name
            if crop:
                utils.print_rank_0(f"  [{id+1}/{total_cameras}] {cam_infos[id].image_name}: {crop.width}x{crop.height} (worker {worker_num})")
            else:
                utils.print_rank_0(f"  [{id+1}/{total_cameras}] {cam_infos[id].image_name}: full image (worker {worker_num})")
            decompressed_images.append(result)

    return decompressed_images


def decompress_and_scale_image(cam_info):
    pil_image = cam_info.image
    resolution = cam_info.image.size  # (w, h)
    # print("cam_info.image.size: ", cam_info.image.size)
    pil_image.load()
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = np.array(resized_image_PIL)  # (h, w, 3)
    # print("resized_image.shape: ", resized_image.shape)
    if len(resized_image.shape) == 3:
        return resized_image.transpose(2, 0, 1)
    else:
        return resized_image[..., np.newaxis].transpose(2, 0, 1)


def load_decompressed_image_shared(params):
    shared_mem_name, args, id, cam_info, resolution_scale = params
    # Retrieve the shared memory block
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)

    # Assume each image will be stored as a flat array in shared memory
    # Example: using numpy for manipulation; adjust size and dtype as needed
    resolution_width, resolution_height = cam_info.image.size
    image_shape = (3, resolution_height, resolution_width)  # Set appropriate values
    dtype = np.uint8  # Adjust as per your image data type

    # Calculate the offset for this particular image
    offset = id * np.prod(image_shape)
    np_image_array = np.ndarray(
        image_shape, dtype=dtype, buffer=existing_shm.buf, offset=offset
    )

    # Decompress image into the numpy array directly
    decompressed_image = decompress_and_scale_image(cam_info)  # Implement this
    np_image_array[:] = decompressed_image

    # Clean up
    existing_shm.close()


def decompressed_images_from_camInfos_multiprocess_sharedmem(
    cam_infos, resolution_scale, args
):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Assume each image shape and dtype
    resolution_width, resolution_height = cam_infos[0].image.size
    image_shape = (
        3,
        resolution_height,
        resolution_width,
    )  # Define these as per your data
    dtype = np.uint8
    image_size = np.prod(image_shape) * np.dtype(dtype).itemsize

    # Create shared memory
    total_size = image_size * total_cameras
    shm = shared_memory.SharedMemory(create=True, size=total_size)

    # Create a pool of processes
    with multiprocessing.Pool(16) as pool:
        # Prepare data for processing
        tasks = [
            (shm.name, args, id, cam_info, resolution_scale)
            for id, cam_info in enumerate(cam_infos)
        ]

        # print("Start Parallel loading...")
        # Map load_camera_data to the tasks
        list(
            tqdm(pool.imap(load_decompressed_image_shared, tasks), total=total_cameras,
                 desc="Loading images (shared mem)")
        )

    # Read images from shared memory
    decompressed_images = []
    for id in range(total_cameras):
        offset = id * np.prod(image_shape)
        np_image_array = np.ndarray(
            image_shape, dtype=dtype, buffer=shm.buf, offset=offset
        )
        decompressed_images.append(
            torch.from_numpy(np_image_array)
        )  # Make a copy if necessary

    # Clean up shared memory
    shm.close()
    shm.unlink()

    return decompressed_images


def cameraList_from_camInfos(cam_infos, args, crops=None):
    """
    Load cameras from camera infos.

    Args:
        cam_infos: List of CameraInfo objects
        args: Arguments
        crops: Optional dict mapping camera index to CropRegion.
               If provided, images will be cropped during loading (saves memory).
    """
    args = get_args()

    if args.multiprocesses_image_loading:
        # Multiprocess loading already applies crops and returns image tensors
        decompressed_images = decompressed_images_from_camInfos_multiprocess(
            cam_infos, args, crops=crops
        )
        # decompressed_images = decompressed_images_from_camInfos_multiprocess_sharedmem(cam_infos, resolution_scale, args)
    else:
        decompressed_images = [None for _ in cam_infos]

    camera_list = []
    for id, c in tqdm(
        enumerate(cam_infos), total=len(cam_infos), disable=(utils.LOCAL_RANK != 0),
        desc="Creating Camera objects"
    ):
        # Always pass crop info so that loadCam computes correct resolution
        # (even if multiprocess loading already applied the crop to the image)
        crop = crops.get(id) if crops else None
        camera_list.append(
            loadCam(
                args,
                id,
                c,
                decompressed_image=decompressed_images[id],
                return_image=False,
                crop=crop,
            )
        )

    if utils.DEFAULT_GROUP.size() > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
