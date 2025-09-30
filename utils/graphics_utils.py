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

import torch
import math
import numpy as np
from typing import NamedTuple, List, Set


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array
    tracks: List[Set[int]] = None  # 각 점이 보이는 camera ID들의 집합


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def filter_pc_by_visibility(point_cloud: BasicPointCloud, camera_list):
    """
    Filter BasicPointCloud to only include points visible in the given cameras.

    Args:
        point_cloud: BasicPointCloud with tracks information
        camera_list: List of cameras with uid attribute

    Returns:
        BasicPointCloud containing only points visible in the specified cameras
    """
    if point_cloud.tracks is None:
        # If no track information, return original point cloud
        return point_cloud

    # Get camera IDs from the camera list
    camera_ids = set(cam.uid for cam in camera_list)
    #print(f"🔍 Filtering points for cameras: {sorted(camera_ids)}")

    # Find points that are visible in at least one of the specified cameras
    visible_indices = []
    for i, track in enumerate(point_cloud.tracks):
        if track.intersection(camera_ids):  # If track has common camera IDs
            visible_indices.append(i)

    print(f"✅ Filter result: {len(visible_indices)} / {len(point_cloud.tracks)} points visible in cameras of {camera_ids}")

    if not visible_indices:
        # Return empty point cloud if no points are visible
        return BasicPointCloud(
            points=np.empty((0, 3)),
            colors=np.empty((0, 3)),
            normals=np.empty((0, 3)),
            tracks=[]
        )

    # Filter the point cloud data
    filtered_points = point_cloud.points[visible_indices]
    filtered_colors = point_cloud.colors[visible_indices]
    filtered_normals = point_cloud.normals[visible_indices]
    filtered_tracks = [point_cloud.tracks[i] for i in visible_indices]

    return BasicPointCloud(
        points=filtered_points,
        colors=filtered_colors,
        normals=filtered_normals,
        tracks=filtered_tracks
    )
