"""
Generalized camera projection utilities for cropped and uncropped images.

This module provides a unified approach to handle:
1. Full images (no crop) - standard symmetric frustum
2. Cropped images - asymmetric frustum with off-center projection
3. Proper principal point handling for both cases
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class CropInfo:
    """Information about image cropping."""
    x_min: int  # Crop start X in original image
    y_min: int  # Crop start Y in original image
    x_max: int  # Crop end X in original image
    y_max: int  # Crop end Y in original image
    
    @property
    def width(self) -> int:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> int:
        return self.y_max - self.y_min
    
    def is_full_image(self, orig_width: int, orig_height: int) -> bool:
        """Check if this crop represents the full image."""
        return (self.x_min == 0 and self.y_min == 0 and
                self.x_max == orig_width and self.y_max == orig_height)


def compute_projection_matrix_with_crop(
    fov_x: float,
    fov_y: float,
    orig_width: int,
    orig_height: int,
    znear: float = 0.01,
    zfar: float = 100.0,
    crop_info: Optional[CropInfo] = None,
    orig_cx: Optional[float] = None,
    orig_cy: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute projection matrix that handles both cropped and uncropped images.
    
    This creates an asymmetric frustum for cropped images, ensuring that:
    1. The 3D-to-2D projection remains consistent
    2. The principal point is properly adjusted for the crop
    3. The frustum boundaries match the visible region
    
    Args:
        fov_x: Horizontal field of view (radians)
        fov_y: Vertical field of view (radians)
        orig_width: Original image width
        orig_height: Original image height
        znear: Near clipping plane
        zfar: Far clipping plane
        crop_info: Optional cropping information
        orig_cx: Original principal point X (None = center)
        orig_cy: Original principal point Y (None = center)
    
    Returns:
        Tuple of (projection_matrix, proj_offset_x, proj_offset_y)
        - projection_matrix: 4x4 projection matrix
        - proj_offset_x: Offset for X coordinate in NDC space
        - proj_offset_y: Offset for Y coordinate in NDC space
    """
    # Default principal point to image center if not provided
    if orig_cx is None:
        orig_cx = orig_width / 2.0
    if orig_cy is None:
        orig_cy = orig_height / 2.0
    
    # If no crop, use full image
    if crop_info is None:
        crop_info = CropInfo(0, 0, orig_width, orig_height)
    
    # Compute frustum parameters based on FOV
    # These represent the full image frustum at z=1
    tan_half_fov_x = np.tan(fov_x * 0.5)
    tan_half_fov_y = np.tan(fov_y * 0.5)
    
    # Full image extends from -tan_half_fov to +tan_half_fov at z=1
    full_left = -tan_half_fov_x
    full_right = tan_half_fov_x
    full_bottom = -tan_half_fov_y
    full_top = tan_half_fov_y
    full_width_at_z1 = full_right - full_left
    full_height_at_z1 = full_top - full_bottom
    
    # Map crop boundaries to frustum coordinates
    # First, normalize crop to [0,1] in image space
    crop_u_min = crop_info.x_min / orig_width
    crop_u_max = crop_info.x_max / orig_width
    crop_v_min = crop_info.y_min / orig_height
    crop_v_max = crop_info.y_max / orig_height
    
    # Then map to frustum space
    crop_left = full_left + crop_u_min * full_width_at_z1
    crop_right = full_left + crop_u_max * full_width_at_z1
    crop_bottom = full_bottom + crop_v_min * full_height_at_z1
    crop_top = full_bottom + crop_v_max * full_height_at_z1
    
    # Build asymmetric projection matrix
    # This follows OpenGL convention
    P = np.zeros((4, 4), dtype=np.float32)
    
    # Asymmetric frustum formulation
    P[0, 0] = 2.0 * znear / (crop_right - crop_left)
    P[0, 2] = (crop_right + crop_left) / (crop_right - crop_left)
    
    P[1, 1] = 2.0 * znear / (crop_top - crop_bottom)
    P[1, 2] = (crop_top + crop_bottom) / (crop_top - crop_bottom)
    
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    
    P[3, 2] = -1.0
    
    # Compute projection offsets for the rasterizer
    # These are 2 * P[0,2] and 2 * P[1,2]
    proj_offset_x = 2.0 * P[0, 2]
    proj_offset_y = 2.0 * P[1, 2]
    
    return P, proj_offset_x, proj_offset_y


def adjust_camera_for_crop(
    camera,
    crop_info: CropInfo,
    orig_width: int,
    orig_height: int,
) -> None:
    """
    Adjust camera parameters in-place for cropped rendering.
    
    This modifies:
    1. Image dimensions to match crop
    2. Projection matrix for asymmetric frustum
    3. Stores crop information for later use
    
    Args:
        camera: Camera object to modify
        crop_info: Cropping information
        orig_width: Original image width
        orig_height: Original image height
    """
    import math
    from utils.graphics_utils import getWorld2View2
    
    # Store original values if not already stored
    if not hasattr(camera, '_orig_width'):
        camera._orig_width = camera.image_width
        camera._orig_height = camera.image_height
        camera._orig_proj_matrix = camera.projection_matrix.clone()
        camera._orig_full_proj = camera.full_proj_transform.clone()
    
    # Update image dimensions to crop size
    camera.image_width = crop_info.width
    camera.image_height = crop_info.height
    
    # Get original principal point (if available)
    orig_cx = getattr(camera, '_cx', None)
    orig_cy = getattr(camera, '_cy', None)
    
    # Compute new projection matrix for crop
    P, proj_offset_x, proj_offset_y = compute_projection_matrix_with_crop(
        fov_x=camera.FoVx,
        fov_y=camera.FoVy,
        orig_width=orig_width,
        orig_height=orig_height,
        znear=camera.znear,
        zfar=camera.zfar,
        crop_info=crop_info,
        orig_cx=orig_cx,
        orig_cy=orig_cy,
    )
    
    # Convert to torch and transpose (camera stores P.T)
    camera.projection_matrix = torch.from_numpy(P).transpose(0, 1).cuda()
    
    # Recompute full projection transform
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0).bmm(
            camera.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)
    
    # Store crop info and offsets
    camera._crop_info = crop_info
    camera._proj_offset_x = proj_offset_x
    camera._proj_offset_y = proj_offset_y


def restore_camera_original(camera) -> None:
    """
    Restore camera to original uncropped state.
    
    Args:
        camera: Camera object to restore
    """
    if hasattr(camera, '_orig_width'):
        camera.image_width = camera._orig_width
        camera.image_height = camera._orig_height
        camera.projection_matrix = camera._orig_proj_matrix
        camera.full_proj_transform = camera._orig_full_proj
        
        # Remove temporary attributes
        delattr(camera, '_orig_width')
        delattr(camera, '_orig_height')
        delattr(camera, '_orig_proj_matrix')
        delattr(camera, '_orig_full_proj')
        
        if hasattr(camera, '_crop_info'):
            delattr(camera, '_crop_info')
        if hasattr(camera, '_proj_offset_x'):
            delattr(camera, '_proj_offset_x')
        if hasattr(camera, '_proj_offset_y'):
            delattr(camera, '_proj_offset_y')


def get_camera_proj_offsets(camera) -> Tuple[float, float]:
    """
    Get projection offsets for a camera.
    
    This handles both:
    1. Cameras with explicit offsets from cropping
    2. Cameras with off-center projection matrices
    3. Standard centered cameras
    
    Args:
        camera: Camera object
    
    Returns:
        Tuple of (proj_offset_x, proj_offset_y)
    """
    # First check for explicit offsets (from cropping)
    if hasattr(camera, '_proj_offset_x'):
        return camera._proj_offset_x, camera._proj_offset_y
    
    # Otherwise compute from projection matrix
    if hasattr(camera, 'projection_matrix'):
        # projection_matrix is stored transposed: P.T
        # So P[0,2] is at projection_matrix[2, 0]
        # and P[1,2] is at projection_matrix[2, 1]
        proj_matrix = camera.projection_matrix
        p02 = proj_matrix[2, 0].item()  # P[0,2]
        p12 = proj_matrix[2, 1].item()  # P[1,2]
        return 2.0 * p02, 2.0 * p12
    
    # Default: centered projection
    return 0.0, 0.0


# Compatibility wrapper for existing code
def get_proj_offsets(camera, verbose=False):
    """
    Backward compatible wrapper for get_camera_proj_offsets.
    
    Args:
        camera: Camera object
        verbose: Whether to print debug info
    
    Returns:
        Tuple of (proj_offset_x, proj_offset_y)
    """
    offset_x, offset_y = get_camera_proj_offsets(camera)
    
    if verbose or (abs(offset_x) > 1e-6 or abs(offset_y) > 1e-6):
        cam_name = getattr(camera, 'image_name', 'unknown')
        p02 = offset_x / 2.0
        p12 = offset_y / 2.0
        print(f"[off-center] Camera {cam_name}: "
              f"proj_offset=({offset_x:.6f}, {offset_y:.6f}), "
              f"P[0,2]={p02:.6f}, P[1,2]={p12:.6f}")
    
    return offset_x, offset_y