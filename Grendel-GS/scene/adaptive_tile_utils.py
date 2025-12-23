"""
Adaptive Tile Utilities for OOM-aware 3DGS Training

This module provides utilities for:
1. Filtering point clouds by tile bounding box
2. Computing camera visibility for tiles
3. Computing crop regions for visible cameras
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class TileBBox:
    """3D bounding box for a tile."""
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @classmethod
    def from_string(cls, bbox_str: str) -> "TileBBox":
        """Parse bbox from string format 'x_min,y_min,z_min,x_max,y_max,z_max'."""
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) != 6:
            raise ValueError(f"Invalid bbox string: {bbox_str}")
        return cls(*parts)

    def get_corners(self) -> np.ndarray:
        """Get 8 corners of the bounding box."""
        return np.array([
            [self.x_min, self.y_min, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_min, self.y_max, self.z_min],
            [self.x_min, self.y_max, self.z_max],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_min],
            [self.x_max, self.y_max, self.z_max],
        ], dtype=np.float32)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Check which points are inside the bounding box."""
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        )
        return mask

    def split(self) -> Tuple["TileBBox", "TileBBox"]:
        """Split the tile along the longest axis."""
        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        dz = self.z_max - self.z_min

        if dx >= dy and dx >= dz:
            mid = (self.x_min + self.x_max) / 2
            return (
                TileBBox(self.x_min, self.y_min, self.z_min, mid, self.y_max, self.z_max),
                TileBBox(mid, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max),
            )
        elif dy >= dz:
            mid = (self.y_min + self.y_max) / 2
            return (
                TileBBox(self.x_min, self.y_min, self.z_min, self.x_max, mid, self.z_max),
                TileBBox(self.x_min, mid, self.z_min, self.x_max, self.y_max, self.z_max),
            )
        else:
            mid = (self.z_min + self.z_max) / 2
            return (
                TileBBox(self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, mid),
                TileBBox(self.x_min, self.y_min, mid, self.x_max, self.y_max, self.z_max),
            )

    def to_string(self) -> str:
        """Convert to string format."""
        return f"{self.x_min},{self.y_min},{self.z_min},{self.x_max},{self.y_max},{self.z_max}"


@dataclass
class CropRegion:
    """2D crop region in image coordinates."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def is_valid(self, img_width: int, img_height: int) -> bool:
        """Check if crop region overlaps with image."""
        return (
            self.x_min < img_width and self.x_max > 0 and
            self.y_min < img_height and self.y_max > 0 and
            self.width > 0 and self.height > 0
        )

    def clamp(self, img_width: int, img_height: int) -> "CropRegion":
        """Clamp crop region to image bounds."""
        return CropRegion(
            x_min=max(0, self.x_min),
            y_min=max(0, self.y_min),
            x_max=min(img_width, self.x_max),
            y_max=min(img_height, self.y_max),
        )


def expand_crop_to_size(crop: CropRegion, target_width: int, target_height: int,
                         img_width: int, img_height: int) -> CropRegion:
    """
    Expand a crop region to a target size while staying within image bounds.

    The expansion tries to center the original crop within the new larger region.
    If the expanded region would exceed image bounds, it shifts to stay within bounds.

    Args:
        crop: Original crop region
        target_width: Target width for the expanded crop
        target_height: Target height for the expanded crop
        img_width: Original image width (max bound)
        img_height: Original image height (max bound)

    Returns:
        Expanded CropRegion with target dimensions (or clamped to image bounds)
    """
    # Calculate how much to expand on each side
    expand_x = target_width - crop.width
    expand_y = target_height - crop.height

    # Try to expand equally on both sides
    expand_left = expand_x // 2
    expand_right = expand_x - expand_left
    expand_top = expand_y // 2
    expand_bottom = expand_y - expand_top

    new_x_min = crop.x_min - expand_left
    new_x_max = crop.x_max + expand_right
    new_y_min = crop.y_min - expand_top
    new_y_max = crop.y_max + expand_bottom

    # Shift if we went out of bounds on the left/top
    if new_x_min < 0:
        new_x_max -= new_x_min  # shift right
        new_x_min = 0
    if new_y_min < 0:
        new_y_max -= new_y_min  # shift down
        new_y_min = 0

    # Shift if we went out of bounds on the right/bottom
    if new_x_max > img_width:
        new_x_min -= (new_x_max - img_width)  # shift left
        new_x_max = img_width
    if new_y_max > img_height:
        new_y_min -= (new_y_max - img_height)  # shift up
        new_y_max = img_height

    # Final clamp to ensure we're within bounds
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(img_width, new_x_max)
    new_y_max = min(img_height, new_y_max)

    return CropRegion(
        x_min=int(new_x_min),
        y_min=int(new_y_min),
        x_max=int(new_x_max),
        y_max=int(new_y_max)
    )


def filter_point_cloud(points: np.ndarray, colors: np.ndarray, normals: np.ndarray,
                       tile_bbox: TileBBox) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter point cloud to keep only points inside the tile bounding box.

    Args:
        points: (N, 3) array of point positions
        colors: (N, 3) array of point colors
        normals: (N, 3) array of point normals
        tile_bbox: Bounding box to filter by

    Returns:
        Filtered (points, colors, normals)
    """
    mask = tile_bbox.contains_points(points)
    return points[mask], colors[mask], normals[mask]


def project_points_to_camera(points_3d: np.ndarray,
                             view_matrix: np.ndarray,
                             proj_matrix: np.ndarray,
                             img_width: int,
                             img_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array of 3D points
        view_matrix: (4, 4) world-to-camera matrix
        proj_matrix: (4, 4) projection matrix
        img_width: Image width
        img_height: Image height

    Returns:
        Tuple of (2D points, valid mask)
    """
    # Convert to homogeneous coordinates
    N = points_3d.shape[0]
    points_h = np.concatenate([points_3d, np.ones((N, 1))], axis=1)  # (N, 4)

    # Transform to camera space
    points_cam = points_h @ view_matrix.T  # (N, 4)

    # Project to clip space
    points_clip = points_cam @ proj_matrix.T  # (N, 4)

    # Perspective divide (check for points behind camera)
    w = points_clip[:, 3:4]
    valid_mask = (w[:, 0] > 0.001)  # Points in front of camera

    # Normalize to NDC
    points_ndc = np.zeros((N, 2))
    points_ndc[valid_mask] = points_clip[valid_mask, :2] / w[valid_mask]

    # Convert to pixel coordinates
    points_2d = np.zeros((N, 2))
    points_2d[:, 0] = (points_ndc[:, 0] + 1.0) * 0.5 * img_width
    points_2d[:, 1] = (points_ndc[:, 1] + 1.0) * 0.5 * img_height

    return points_2d, valid_mask


def compute_tile_crop_for_camera(tile_bbox: TileBBox,
                                  view_matrix: np.ndarray,
                                  proj_matrix: np.ndarray,
                                  img_width: int,
                                  img_height: int,
                                  margin: int = 100) -> Optional[CropRegion]:
    """
    Compute the crop region for a tile in a camera's view.

    Args:
        tile_bbox: 3D bounding box of the tile
        view_matrix: (4, 4) world-to-camera matrix
        proj_matrix: (4, 4) projection matrix
        img_width: Image width
        img_height: Image height
        margin: Extra margin around the projected bbox (in pixels)

    Returns:
        CropRegion if tile is visible, None otherwise
    """
    # Get 8 corners of the bbox
    corners = tile_bbox.get_corners()

    # Project to 2D
    corners_2d, valid_mask = project_points_to_camera(
        corners, view_matrix, proj_matrix, img_width, img_height
    )

    # Need at least one valid corner
    if not np.any(valid_mask):
        return None

    # Get bounding box of valid projected corners
    valid_corners = corners_2d[valid_mask]
    x_min = int(np.floor(valid_corners[:, 0].min())) - margin
    y_min = int(np.floor(valid_corners[:, 1].min())) - margin
    x_max = int(np.ceil(valid_corners[:, 0].max())) + margin
    y_max = int(np.ceil(valid_corners[:, 1].max())) + margin

    crop = CropRegion(x_min, y_min, x_max, y_max)

    # Check if crop overlaps with image
    if not crop.is_valid(img_width, img_height):
        return None

    # Clamp to image bounds
    return crop.clamp(img_width, img_height)


def compute_visible_cameras_and_crops(tile_bbox: TileBBox,
                                       cameras: List,
                                       margin: int = 100) -> List[Tuple[int, CropRegion]]:
    """
    Compute which cameras can see the tile and their crop regions.

    Args:
        tile_bbox: 3D bounding box of the tile
        cameras: List of camera objects with view_matrix, proj_matrix, width, height
        margin: Extra margin around the projected bbox (in pixels)

    Returns:
        List of (camera_index, crop_region) for visible cameras
    """
    visible = []

    for idx, cam in enumerate(cameras):
        # Get camera matrices (convert from torch if needed)
        if hasattr(cam, 'world_view_transform'):
            view_matrix = cam.world_view_transform.cpu().numpy()
        else:
            view_matrix = np.array(cam.view_matrix)

        if hasattr(cam, 'full_proj_transform'):
            proj_matrix = cam.full_proj_transform.cpu().numpy()
        else:
            proj_matrix = np.array(cam.proj_matrix)

        img_width = cam.image_width if hasattr(cam, 'image_width') else cam.width
        img_height = cam.image_height if hasattr(cam, 'image_height') else cam.height

        crop = compute_tile_crop_for_camera(
            tile_bbox, view_matrix, proj_matrix, img_width, img_height, margin
        )

        if crop is not None:
            visible.append((idx, crop))

    return visible


def apply_crop_to_camera(camera, crop: CropRegion):
    """
    Modify camera to use crop region instead of full image.

    This modifies the camera's:
    - image dimensions
    - FoVx/FoVy (adjusted to maintain correct focal length)
    - ground truth image (if loaded)

    The key insight: when cropping, we must maintain the same focal length.
    Original: focal_x = orig_width / (2 * tan(FoVx/2))
    After crop: focal_x stays same, so new_FoVx = 2 * atan(crop_width / (2 * focal_x))

    Args:
        camera: Camera object to modify
        crop: Crop region to apply
    """
    import math

    # Store original dimensions
    if not hasattr(camera, '_original_width'):
        camera._original_width = camera.image_width
        camera._original_height = camera.image_height
        camera._crop_region = crop

    orig_width = camera._original_width
    orig_height = camera._original_height

    # Compute original focal lengths from FoV
    if hasattr(camera, 'FoVx') and hasattr(camera, 'FoVy'):
        tanfovx = math.tan(camera.FoVx / 2)
        tanfovy = math.tan(camera.FoVy / 2)
        focal_x = orig_width / (2 * tanfovx)
        focal_y = orig_height / (2 * tanfovy)

        # Compute new FoV for cropped dimensions (maintaining same focal length)
        new_tanfovx = crop.width / (2 * focal_x)
        new_tanfovy = crop.height / (2 * focal_y)
        camera.FoVx = 2 * math.atan(new_tanfovx)
        camera.FoVy = 2 * math.atan(new_tanfovy)

    # Update dimensions
    camera.image_width = crop.width
    camera.image_height = crop.height

    # Crop the ground truth image if loaded
    if hasattr(camera, 'original_image') and camera.original_image is not None:
        img = camera.original_image
        if isinstance(img, torch.Tensor):
            # Assuming CHW format
            camera.original_image = img[:, crop.y_min:crop.y_max, crop.x_min:crop.x_max]
        else:
            camera.original_image = img[crop.y_min:crop.y_max, crop.x_min:crop.x_max]

    # Also crop backup image if exists
    if hasattr(camera, 'original_image_backup') and camera.original_image_backup is not None:
        img = camera.original_image_backup
        if isinstance(img, torch.Tensor):
            camera.original_image_backup = img[:, crop.y_min:crop.y_max, crop.x_min:crop.x_max]
        else:
            camera.original_image_backup = img[crop.y_min:crop.y_max, crop.x_min:crop.x_max]
