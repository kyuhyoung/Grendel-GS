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

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a single point is inside the bounding box."""
        return (
            self.x_min <= point[0] <= self.x_max and
            self.y_min <= point[1] <= self.y_max and
            self.z_min <= point[2] <= self.z_max
        )

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


def project_points_to_camera_raw(points_3d: np.ndarray,
                                 full_proj_transform: np.ndarray,
                                 img_width: int,
                                 img_height: int) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates WITHOUT any filtering.
    
    Returns ALL projected points, even if they're behind camera or outside image.
    This is useful for debugging where you want to see all corner projections.
    """
    # Convert to homogeneous coordinates
    N = points_3d.shape[0]
    points_h = np.concatenate([points_3d, np.ones((N, 1))], axis=1)  # (N, 4)
    
    # Project to clip space
    points_clip = points_h @ full_proj_transform  # (N, 4)
    
    # Perspective divide (even if w is negative - behind camera)
    w = points_clip[:, 3:4]
    
    # Handle points behind camera (w <= 0) by using a small epsilon
    w_safe = np.where(np.abs(w) < 0.001, np.sign(w) * 0.001, w)
    
    # Normalize to NDC
    points_ndc = points_clip[:, :2] / w_safe
    
    # Convert to pixel coordinates
    points_2d = np.zeros((N, 2))
    points_2d[:, 0] = (points_ndc[:, 0] + 1.0) * 0.5 * img_width
    points_2d[:, 1] = (points_ndc[:, 1] + 1.0) * 0.5 * img_height
    
    return points_2d


def project_points_to_camera(points_3d: np.ndarray,
                             full_proj_transform: np.ndarray,
                             img_width: int,
                             img_height: int,
                             ndc_limit: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array of 3D points
        full_proj_transform: (4, 4) combined world-view-projection matrix
        img_width: Image width
        img_height: Image height
        ndc_limit: Maximum allowed NDC coordinate (points outside are invalid).
                   NDC of 1.0 = edge of image, 3.0 = 3x image size away (generous margin)

    Returns:
        Tuple of (2D points, valid mask)
    """
    # Convert to homogeneous coordinates
    N = points_3d.shape[0]
    points_h = np.concatenate([points_3d, np.ones((N, 1))], axis=1)  # (N, 4)

    # Project directly to clip space using full_proj_transform
    # full_proj_transform = Rt.T @ P.T, so:
    # points_h @ full_proj_transform = points_h @ Rt.T @ P.T (correct order)
    points_clip = points_h @ full_proj_transform  # (N, 4)

    # Perspective divide (check for points behind camera)
    w = points_clip[:, 3:4]
    valid_mask = (w[:, 0] > 0.001)  # Points in front of camera

    # Normalize to NDC (only for valid points)
    points_ndc = np.zeros((N, 2))
    points_ndc[valid_mask] = points_clip[valid_mask, :2] / w[valid_mask]

    # Convert to pixel coordinates (no NDC filtering - just project and clamp later)
    points_2d = np.zeros((N, 2))
    points_2d[:, 0] = (points_ndc[:, 0] + 1.0) * 0.5 * img_width
    points_2d[:, 1] = (points_ndc[:, 1] + 1.0) * 0.5 * img_height

    return points_2d, valid_mask


@dataclass
class ProjectionDebugInfo:
    """Debug info for point projection."""
    num_valid_points: int
    raw_x_min: float
    raw_x_max: float
    raw_y_min: float
    raw_y_max: float


def get_camera_position_from_RT(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Get camera position in world coordinates from R and T.

    The world-view transform Rt has:
    - Rt[:3, :3] = R.T
    - Rt[:3, 3] = T

    Camera position in world space = -R @ T

    Args:
        R: (3, 3) rotation matrix
        T: (3,) translation vector

    Returns:
        (3,) camera position in world coordinates
    """
    return -R @ T


def save_projection_debug_visualization(tile_bbox: TileBBox,
                                       camera_info,
                                       crop: CropRegion,
                                       full_proj_transform: np.ndarray,
                                       output_dir: str,
                                       tile_id: str,
                                       cam_idx: int,
                                       full_scene_bbox: TileBBox = None,
                                       all_cameras=None,
                                       fixed_image_bounds=None):
    """Save a debug visualization of tile projection and crop."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    
    # Define corner colors for identification (8 corners of a box)
    corner_colors = [
        'red',      # (min, min, min) - front bottom left
        'blue',     # (min, min, max) - front bottom right  
        'green',    # (min, max, min) - front top left
        'orange',   # (min, max, max) - front top right
        'purple',   # (max, min, min) - back bottom left
        'brown',    # (max, min, max) - back bottom right
        'pink',     # (max, max, min) - back top left
        'cyan'      # (max, max, max) - back top right
    ]
    
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(16, 8))
    
    # Extract camera position for 3D view
    # Handle both dict and object formats for camera_info
    if isinstance(camera_info, dict):
        camera_R = np.array(camera_info['R'])
        camera_T = np.array(camera_info['T'])
        camera_image_name = camera_info.get('image_name', f'cam_{cam_idx}')
    else:
        camera_R = np.array(camera_info.R)
        camera_T = np.array(camera_info.T)
        camera_image_name = getattr(camera_info, 'image_name', f'cam_{cam_idx}')
    
    # Camera position in world coordinates: -R^T * T
    camera_pos = -camera_R.T @ camera_T
    
    # ============ LEFT: 3D Scene View (Perspective) ============
    ax = fig.add_subplot(121, projection='3d')
    
    # Check if camera is inside tile bounding box
    camera_inside_tile = tile_bbox.contains_point(camera_pos)
    
    title_color = 'red' if camera_inside_tile else 'green'
    ax.set_title(f"3D Scene View - Camera {'INSIDE' if camera_inside_tile else 'OUTSIDE'} Tile", 
                fontsize=14, fontweight='bold', color=title_color)
    
    # Draw 3D bounding box using all 8 corners
    corners_3d = tile_bbox.get_corners()
    
    # Define the 12 edges of the bounding box
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Draw edges
    for edge in edges:
        points = [corners_3d[edge[0]], corners_3d[edge[1]]]
        ax.plot3D([points[0][0], points[1][0]], 
                  [points[0][1], points[1][1]], 
                  [points[0][2], points[1][2]], 
                  'b-', linewidth=2)
    
    # Draw corners with colors  
    for i, corner in enumerate(corners_3d):
        ax.scatter(corner[0], corner[1], corner[2], 
                  color=corner_colors[i], s=100, edgecolor='black', linewidth=1)
        # Add corner numbers with Z value for debugging
        ax.text(corner[0], corner[1], corner[2], f'  {i+1}\n  Z:{corner[2]:.1f}', 
                color=corner_colors[i], fontsize=8, fontweight='bold')
    
    # Draw all cameras if provided
    if all_cameras:
        current_camera_labeled = False
        other_cameras_labeled = False
        
        for i, cam in enumerate(all_cameras):
            try:
                # Handle both dict and object formats
                if isinstance(cam, dict):
                    cam_R = np.array(cam['R'])
                    cam_T = np.array(cam['T'])
                    cam_name = cam.get('image_name', f'cam_{i}')
                else:
                    cam_R = np.array(cam.R)
                    cam_T = np.array(cam.T)
                    cam_name = getattr(cam, 'image_name', f'cam_{i}')
                
                # Calculate camera position
                cam_pos = -cam_R.T @ cam_T
                
                # Check if this is the current camera (use name matching primarily)
                is_current = (cam_name == camera_image_name)
                
                if is_current:
                    # Current camera - highlighted
                    label = f'Current Camera ({"INSIDE" if camera_inside_tile else "OUTSIDE"} tile)' if not current_camera_labeled else None
                    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], 
                              color='red' if camera_inside_tile else 'orange', 
                              s=300, marker='^', edgecolor='yellow', linewidth=3,
                              label=label)
                    current_camera_labeled = True
                else:
                    # Other cameras - smaller and gray
                    label = 'Other Cameras' if not other_cameras_labeled else None
                    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], 
                              color='lightgray', s=50, marker='o', alpha=0.6,
                              label=label)
                    other_cameras_labeled = True
            except Exception as e:
                print(f"Warning: Failed to process camera {i}: {e}")
                continue
    else:
        # Fallback: draw only current camera if all_cameras not provided
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], 
                  color='red' if camera_inside_tile else 'black', 
                  s=200, marker='^', edgecolor='white', linewidth=2,
                  label=f'Camera ({"INSIDE" if camera_inside_tile else "OUTSIDE"} tile)')
    
    # Draw camera frustum (pyramid shape) for current camera only
    # In COLMAP/3DGS coordinate system, camera looks along +Z axis
    forward = camera_R.T @ np.array([0, 0, 1])  # Looking along +Z
    right = camera_R.T @ np.array([1, 0, 0])
    up = camera_R.T @ np.array([0, -1, 0])  # Y-axis is inverted in image space
    
    # Create frustum corners at a distance from camera
    frustum_distance = 300  # Distance to far plane
    frustum_width = 150
    frustum_height = 112
    
    frustum_center = camera_pos + forward * frustum_distance
    
    # Four corners of the far plane
    frustum_corners = [
        frustum_center + right * frustum_width + up * frustum_height,
        frustum_center - right * frustum_width + up * frustum_height,
        frustum_center - right * frustum_width - up * frustum_height,
        frustum_center + right * frustum_width - up * frustum_height,
    ]
    
    # Draw frustum pyramid edges from camera to corners
    for corner in frustum_corners:
        ax.plot3D([camera_pos[0], corner[0]], 
                 [camera_pos[1], corner[1]], 
                 [camera_pos[2], corner[2]], 
                 'g-', linewidth=1.5, alpha=0.6)
    
    # Draw frustum far plane rectangle
    for i in range(4):
        next_i = (i + 1) % 4
        ax.plot3D([frustum_corners[i][0], frustum_corners[next_i][0]], 
                 [frustum_corners[i][1], frustum_corners[next_i][1]], 
                 [frustum_corners[i][2], frustum_corners[next_i][2]], 
                 'g-', linewidth=1.5, alpha=0.6)
    
    # Draw viewing direction arrow
    ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
             forward[0]*200, forward[1]*200, forward[2]*200,
             color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.8)
    
    # Set 3D view limits with equal aspect ratio
    all_x = [camera_pos[0]] + [c[0] for c in corners_3d] + [fc[0] for fc in frustum_corners]
    all_y = [camera_pos[1]] + [c[1] for c in corners_3d] + [fc[1] for fc in frustum_corners]
    all_z = [camera_pos[2]] + [c[2] for c in corners_3d] + [fc[2] for fc in frustum_corners]
    
    # Calculate ranges for each axis
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    z_range = max(all_z) - min(all_z)
    
    # Find the maximum range to use for all axes (equal aspect)
    max_range = max(x_range, y_range, z_range)
    
    # Calculate center points
    x_center = (max(all_x) + min(all_x)) / 2
    y_center = (max(all_y) + min(all_y)) / 2
    z_center = (max(all_z) + min(all_z)) / 2
    
    # Set equal limits for all axes with some margin
    half_range = max_range / 2 * 1.1  # 10% margin
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)
    ax.set_zlim(z_center - half_range, z_center + half_range)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add labels and grid for 3D
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set a good viewing angle (bird's eye but angled to see depth)
    ax.view_init(elev=25, azim=45)
    
    # Invert Z axis if needed (if Z represents depth/height downward)
    # ax.invert_zaxis()  # Uncomment if Z should point downward
    
    # Add legend
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    # ============ RIGHT: 2D Image Projection ============
    ax = fig.add_subplot(122)
    ax.set_title(f"Image Projection ({camera_info.image_name})", fontsize=14, fontweight='bold')
    
    # Draw image boundary
    ax.add_patch(Rectangle((0, 0), camera_info.width, camera_info.height,
                           fill=False, edgecolor='black', linewidth=2, label='Image Bounds'))
    
    # Project tile corners to image (without any filtering)
    corners_2d_raw = project_points_to_camera_raw(
        corners_3d, full_proj_transform, camera_info.width, camera_info.height
    )
    
    # Draw ALL projected corners with same colors as 3D view
    for i, corner_2d in enumerate(corners_2d_raw):
        # Always draw the corner, even if it's outside image bounds
        ax.plot(corner_2d[0], corner_2d[1], 'o', 
               color=corner_colors[i], markersize=10, 
               markeredgecolor='black', markeredgewidth=0.5,
               label=f'Corner {i+1}' if i < 4 else None)  # Only label first few
        
        # Add corner number text with offset to avoid overlap
        text_offset_x = 80 if corner_2d[0] > camera_info.width/2 else -80
        text_offset_y = 80 if corner_2d[1] > camera_info.height/2 else -80
        ax.text(corner_2d[0] + text_offset_x, corner_2d[1] + text_offset_y, f'{i+1}', 
               color=corner_colors[i], fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Draw crop region if computed
    if crop:
        ax.add_patch(Rectangle((crop.x_min, crop.y_min), 
                               crop.width, crop.height,
                               fill=False, edgecolor='lime', linewidth=3,
                               label='Crop Region'))
    
    # Draw principal point if available  
    if hasattr(camera_info, 'cx') and camera_info.cx is not None:
        ax.plot(camera_info.cx, camera_info.cy, '+', color='red', 
               markersize=20, markeredgewidth=3, label='Principal Point')
    
    # Add info text
    info_lines = [
        f"Tile: {tile_id}",
        f"Image: {camera_info.width}×{camera_info.height}",
    ]
    if crop:
        info_lines.append(f"Crop: {crop.width}×{crop.height}")
        if hasattr(camera_info, 'cx') and camera_info.cx is not None:
            pp_in_crop = (crop.x_min <= camera_info.cx <= crop.x_max and 
                         crop.y_min <= camera_info.cy <= crop.y_max)
            info_lines.append(f"PP in crop: {'Yes' if pp_in_crop else 'No'}")
    
    # Use fixed image bounds if provided, otherwise calculate dynamically
    if fixed_image_bounds:
        x_min_view, y_min_view, x_max_view, y_max_view = fixed_image_bounds
    else:
        # Calculate view range to include all projected corners (fallback)
        x_coords = corners_2d_raw[:, 0]
        y_coords = corners_2d_raw[:, 1]
        margin = max(camera_info.width, camera_info.height) * 0.1
        x_min_view = min(x_coords.min(), -margin)
        x_max_view = max(x_coords.max(), camera_info.width + margin)
        y_min_view = min(y_coords.min(), -margin)
        y_max_view = max(y_coords.max(), camera_info.height + margin)
    
    # Count corners inside current image bounds (not fixed bounds)
    x_coords = corners_2d_raw[:, 0]
    y_coords = corners_2d_raw[:, 1]
    inside_x = (x_coords >= 0) & (x_coords <= camera_info.width)
    inside_y = (y_coords >= 0) & (y_coords <= camera_info.height)
    inside_count = np.sum(inside_x & inside_y)
    info_lines.append(f"Corners in image: {inside_count}/8")
    
    ax.text(0.02, 0.98, '\n'.join(info_lines), transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlim(x_min_view, x_max_view)
    ax.set_ylim(y_min_view, y_max_view)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    import os
    os.makedirs(output_dir, exist_ok=True)
    # Use actual camera image name if available, otherwise use cam_idx
    camera_name = getattr(camera_info, 'image_name', f"{cam_idx:04d}")
    if camera_name.endswith('.jpg') or camera_name.endswith('.png'):
        camera_name = camera_name.rsplit('.', 1)[0]  # Remove extension
    
    filename = os.path.join(output_dir, f"debug_{tile_id}_{camera_name}.png")
    plt.suptitle(f"Projection Debug: {tile_id} - Camera {camera_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return filename


def compute_tile_crop_for_camera(tile_bbox: TileBBox,
                                  full_proj_transform: np.ndarray,
                                  img_width: int,
                                  img_height: int,
                                  margin: int = 100,
                                  points_in_tile: np.ndarray = None,
                                  return_debug_info: bool = False,
                                  ndc_limit: float = 1.0,
                                  camera_position: np.ndarray = None) -> Optional[CropRegion]:
    """
    Compute the crop region for a tile in a camera's view.

    NEW ALGORITHM (efficient):
    1. Project only the 8 corners of the tile bounding box
    2. Use min/max of projected corners as crop region
    3. Error if camera is inside the bounding box

    Args:
        tile_bbox: 3D bounding box of the tile
        full_proj_transform: (4, 4) combined world-view-projection matrix
        img_width: Image width
        img_height: Image height
        margin: Extra margin around the projected bbox (in pixels)
        points_in_tile: DEPRECATED - no longer used, kept for API compatibility
        return_debug_info: If True, return (crop, debug_info) tuple
        ndc_limit: Maximum allowed NDC coordinate
        camera_position: (3,) camera position in world coordinates (required for inside check)

    Returns:
        CropRegion if tile is visible, None otherwise
        If return_debug_info is True, returns (CropRegion, ProjectionDebugInfo) or (None, None)
    """
    # Check if camera is inside the bounding box
    if camera_position is not None:
        if tile_bbox.contains_point(camera_position):
            raise RuntimeError(
                f"Camera is inside the tile bounding box! "
                f"Camera position: ({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}), "
                f"Tile bbox: X[{tile_bbox.x_min:.2f}, {tile_bbox.x_max:.2f}], "
                f"Y[{tile_bbox.y_min:.2f}, {tile_bbox.y_max:.2f}], "
                f"Z[{tile_bbox.z_min:.2f}, {tile_bbox.z_max:.2f}]. "
                f"This case is not supported."
            )

    # Get 8 corners of the bounding box
    corners = tile_bbox.get_corners()  # (8, 3)

    # Project corners to 2D
    corners_2d, valid_mask = project_points_to_camera(
        corners, full_proj_transform, img_width, img_height, ndc_limit=ndc_limit
    )

    # Need at least one valid corner
    if not np.any(valid_mask):
        if return_debug_info:
            return None, None
        return None

    # Get bounding box of valid projected corners
    valid_corners = corners_2d[valid_mask]
    raw_x_min = valid_corners[:, 0].min()
    raw_y_min = valid_corners[:, 1].min()
    raw_x_max = valid_corners[:, 0].max()
    raw_y_max = valid_corners[:, 1].max()

    x_min = int(np.floor(raw_x_min)) - margin
    y_min = int(np.floor(raw_y_min)) - margin
    x_max = int(np.ceil(raw_x_max)) + margin
    y_max = int(np.ceil(raw_y_max)) + margin

    crop = CropRegion(x_min, y_min, x_max, y_max)

    debug_info = ProjectionDebugInfo(
        num_valid_points=int(np.sum(valid_mask)),
        raw_x_min=float(raw_x_min),
        raw_x_max=float(raw_x_max),
        raw_y_min=float(raw_y_min),
        raw_y_max=float(raw_y_max),
    )

    # Check if crop overlaps with image
    if not crop.is_valid(img_width, img_height):
        if return_debug_info:
            return None, debug_info
        return None

    # Clamp to image bounds
    clamped = crop.clamp(img_width, img_height)
    
    if return_debug_info:
        return clamped, debug_info
    return clamped


def build_full_proj_transform(R: np.ndarray, T: np.ndarray,
                               fov_x: float, fov_y: float,
                               width: int, height: int,
                               znear: float = 0.01, zfar: float = 100.0) -> np.ndarray:
    """
    Build full_proj_transform from camera parameters (R, T, FoV).

    Uses the exact same formula as Grendel-GS's getWorld2View2 and getProjectionMatrix.

    Args:
        R: (3, 3) rotation matrix (camera orientation)
        T: (3,) translation vector (camera position)
        fov_x: horizontal field of view in radians
        fov_y: vertical field of view in radians
        width: image width
        height: image height
        znear: near clipping plane
        zfar: far clipping plane

    Returns:
        (4, 4) full projection transform matrix
    """
    import math

    # Build world_view_transform using getWorld2View2 formula
    # (with translate=[0,0,0] and scale=1.0)
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T  # transpose of R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    # Note: getWorld2View2 does C2W = inv(Rt), cam_center = (cam_center + translate) * scale,
    # C2W[:3, 3] = cam_center, Rt = inv(C2W). With translate=0 and scale=1, this is identity transform.
    # So Rt stays as is.

    # Build projection matrix using getProjectionMatrix formula (EXACTLY)
    tanHalfFovY = math.tan(fov_y / 2)
    tanHalfFovX = math.tan(fov_x / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    z_sign = 1.0

    P = np.zeros((4, 4))
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    # Match cameras.py convention:
    # world_view_transform = Rt.T
    # projection_matrix = P.T
    # full_proj_transform = world_view_transform @ projection_matrix = Rt.T @ P.T
    world_view_T = Rt.T
    proj_T = P.T
    full_proj = world_view_T @ proj_T

    return full_proj


def compute_visible_caminfos(tile_bbox: TileBBox,
                              cam_infos: List,
                              margin: int = 100,
                              points: np.ndarray = None,
                              return_debug_info: bool = False,
                              ndc_limit: float = 1.0,
                              visual_debug: bool = False,
                              visual_debug_dir: str = None,
                              tile_id: str = None) -> List[Tuple[int, CropRegion]]:
    """
    Compute which CamInfos can see the tile (without loading images).

    NEW ALGORITHM:
    - Projects only the 8 corners of the tile bounding box (O(1) per camera)
    - Uses min/max of projected corners as crop region
    - Errors if camera is inside the bounding box

    Args:
        tile_bbox: 3D bounding box of the tile
        cam_infos: List of CamInfo objects (R, T, FovX, FovY, width, height)
        margin: Extra margin around the projected bbox (in pixels)
        points: DEPRECATED - no longer used, kept for API compatibility
        return_debug_info: If True, include ProjectionDebugInfo in results.
        ndc_limit: Maximum allowed NDC coordinate
        visual_debug: If True, save visualization images
        visual_debug_dir: Directory to save debug visualizations
        tile_id: Tile ID for debug output naming

    Returns:
        List of (camera_index, crop_region) for visible cameras
        If return_debug_info is True, returns List of (camera_index, crop_region, debug_info)
    """
    visible = []
    
    # Calculate fixed image bounds for consistent visualization (always calculate for all visual debug)
    fixed_image_bounds = None
    
    # Always calculate fixed bounds when we have cam_infos
    if cam_infos:
        # Find max image dimensions across all cameras
        max_width = 0
        max_height = 0
        for cam in cam_infos:
            if isinstance(cam, dict):
                width = cam.get('width', 0)
                height = cam.get('height', 0)
            else:
                width = getattr(cam, 'width', 0)
                height = getattr(cam, 'height', 0)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        
        # Debug print to verify we found the dimensions
        if visual_debug:
            print(f"[Visual Debug] Max image dimensions: {max_width}x{max_height}", flush=True)
        
        # Set fixed bounds with margin - use larger margin to ensure all corners fit
        margin_ratio = 0.5  # 50% margin around largest image
        margin_x = max_width * margin_ratio
        margin_y = max_height * margin_ratio
        fixed_image_bounds = (-margin_x, -margin_y, max_width + margin_x, max_height + margin_y)
        
        if visual_debug:
            print(f"[Visual Debug] Fixed bounds: x:[{fixed_image_bounds[0]:.0f}, {fixed_image_bounds[2]:.0f}], y:[{fixed_image_bounds[1]:.0f}, {fixed_image_bounds[3]:.0f}]", flush=True)

    for idx, cam_info in enumerate(cam_infos):
        R = np.array(cam_info.R)
        T = np.array(cam_info.T)

        # Compute camera position in world coordinates
        camera_position = get_camera_position_from_RT(R, T)

        # Build full_proj_transform from CamInfo
        full_proj = build_full_proj_transform(
            R=R,
            T=T,
            fov_x=cam_info.FovX,
            fov_y=cam_info.FovY,
            width=cam_info.width,
            height=cam_info.height
        )

        if return_debug_info:
            # Remove arbitrary max_crop_size limit - let tile bbox naturally determine crop size
            crop, debug = compute_tile_crop_for_camera(
                tile_bbox, full_proj, cam_info.width, cam_info.height, margin,
                return_debug_info=True,
                ndc_limit=ndc_limit,
                camera_position=camera_position
            )
            if crop is not None:
                visible.append((idx, crop, debug))
                # Save visualization if requested
                if visual_debug and visual_debug_dir and tile_id:
                    # For now, use tile_bbox as full_scene_bbox (can be improved later)
                    save_projection_debug_visualization(
                        tile_bbox, cam_info, crop, full_proj,
                        visual_debug_dir, tile_id, idx, full_scene_bbox=tile_bbox,
                        all_cameras=cam_infos, fixed_image_bounds=fixed_image_bounds
                    )
        else:
            # Remove arbitrary max_crop_size limit - let tile bbox naturally determine crop size
            crop = compute_tile_crop_for_camera(
                tile_bbox, full_proj, cam_info.width, cam_info.height, margin,
                ndc_limit=ndc_limit,
                camera_position=camera_position
            )
            if crop is not None:
                visible.append((idx, crop))
                # Save visualization if requested
                if visual_debug and visual_debug_dir and tile_id:
                    # For now, use tile_bbox as full_scene_bbox (can be improved later)
                    save_projection_debug_visualization(
                        tile_bbox, cam_info, crop, full_proj,
                        visual_debug_dir, tile_id, idx, full_scene_bbox=tile_bbox,
                        all_cameras=cam_infos, fixed_image_bounds=fixed_image_bounds
                    )

    return visible


def compute_visible_cameras_and_crops(tile_bbox: TileBBox,
                                       cameras: List,
                                       margin: int = 100,
                                       tile_id: str = None) -> List[Tuple[int, CropRegion]]:
    """
    Compute which cameras can see the tile and their crop regions.

    Args:
        tile_bbox: 3D bounding box of the tile
        cameras: List of camera objects with full_proj_transform, width, height
        margin: Extra margin around the projected bbox (in pixels)

    Returns:
        List of (camera_index, crop_region) for visible cameras
    """
    visible = []

    for idx, cam in enumerate(cameras):
        # Get full_proj_transform (combined world-view-projection matrix)
        # full_proj_transform = world_view_transform @ projection_matrix
        if hasattr(cam, 'full_proj_transform'):
            full_proj_transform = cam.full_proj_transform.cpu().numpy()
        else:
            raise ValueError(f"Camera {idx} does not have full_proj_transform attribute")

        img_width = cam.image_width if hasattr(cam, 'image_width') else cam.width
        img_height = cam.image_height if hasattr(cam, 'image_height') else cam.height

        crop = compute_tile_crop_for_camera(
            tile_bbox, full_proj_transform, img_width, img_height, margin
        )

        if crop is not None:
            visible.append((idx, crop))

    return visible


def apply_crop_to_camera(camera, crop: CropRegion):
    """
    Update camera's projection matrix for off-center principal point after cropping.

    IMPORTANT: This function assumes the camera was already loaded with cropped image
    and adjusted FoV by loadCam(). It ONLY updates the projection matrix to account
    for the shifted principal point.

    When an image is cropped, the principal point (which was at the center of the
    original image) shifts relative to the cropped region. This shift must be
    reflected in the projection matrix, otherwise gaussians will project to
    completely wrong pixel locations!

    Args:
        camera: Camera object (already has cropped FoV and dimensions from loadCam)
        crop: Crop region that was applied

    Requires:
        camera._original_width, camera._original_height: Original image dimensions
        (must be set before calling this function)
    """
    import math

    # Get original dimensions (must be set by caller before calling this function)
    if not hasattr(camera, '_original_width') or not hasattr(camera, '_original_height'):
        print(f"[crop-projection] WARNING: _original_width/_original_height not set for camera {camera.image_name}!")
        print(f"[crop-projection] Skipping projection matrix update - this may cause rendering issues!")
        return

    orig_width = camera._original_width
    orig_height = camera._original_height
    camera._crop_region = crop

    # NOTE: FoV and image dimensions are already adjusted by loadCam(), don't modify again!
    # camera.FoVx, camera.FoVy = crop-adjusted FoV
    # camera.image_width, camera.image_height = crop dimensions

    # ============================================
    # CRITICAL: Update projection matrix for off-center principal point
    # ============================================
    # When cropping, the principal point shifts relative to the cropped image.
    # Original principal point: (orig_width/2, orig_height/2)
    # New principal point in crop coords: (orig_cx - crop.x_min, orig_cy - crop.y_min)
    #
    # The projection matrix must account for this offset, otherwise gaussians
    # will project to completely wrong pixel locations!

    if hasattr(camera, 'projection_matrix'):
        # Get original principal point from camera intrinsics (from COLMAP)
        # If not available, fall back to image center
        orig_cx = getattr(camera, '_cx', None)
        orig_cy = getattr(camera, '_cy', None)
        if orig_cx is None:
            orig_cx = orig_width / 2.0
        if orig_cy is None:
            orig_cy = orig_height / 2.0

        # New principal point in cropped image coordinates
        # Formula from diagram: Cx_new = Cx - crop.x_min, Cy_new = Cy - crop.y_min
        new_cx = orig_cx - crop.x_min
        new_cy = orig_cy - crop.y_min

        # CRITICAL: Update camera._cx and _cy with adjusted values for visibility check
        camera._cx = new_cx
        camera._cy = new_cy

        # Offset from center of cropped image (in pixels)
        offset_x = new_cx - (crop.width / 2.0)
        offset_y = new_cy - (crop.height / 2.0)

        # Compute focal lengths from FoV (for debug output)
        tanfovx = math.tan(camera.FoVx / 2)
        tanfovy = math.tan(camera.FoVy / 2)
        # Focal length in pixels: f = (size/2) / tan(fov/2)
        focal_x = (crop.width / 2.0) / tanfovx
        focal_y = (crop.height / 2.0) / tanfovy

        # Compute frustum bounds for off-center projection
        # Using OpenGL-style frustum: the principal point determines where the
        # optical axis intersects the image plane
        znear = camera.znear
        zfar = camera.zfar

        # For off-center projection, we need asymmetric frustum bounds
        # The principal point (new_cx, new_cy) should map to NDC (0, 0)
        # Standard: image center maps to NDC (0, 0)
        # Off-center: shift the frustum so that (new_cx, new_cy) maps to NDC (0, 0)

        # Frustum bounds at znear plane (for off-center projection):
        #
        # Pinhole model: pixel_x = fx * X/Z + cx
        # NDC: ndc_x = 2 * pixel_x / width - 1 = (2*fx/width) * X/Z + (2*cx - width)/width
        #
        # For P[0,2] = (right + left) / (right - left) = (2*cx - width) / width:
        #   right = cx / fx * znear
        #   left = -(width - cx) / fx * znear
        #
        # Verification (centered case, cx = width/2):
        #   right = (width/2) / fx * znear = tan(fovX/2) * znear  ✓
        #   left = -(width/2) / fx * znear = -tan(fovX/2) * znear  ✓
        #
        right = new_cx / focal_x * znear
        left = -(crop.width - new_cx) / focal_x * znear
        top = new_cy / focal_y * znear
        bottom = -(crop.height - new_cy) / focal_y * znear

        # Debug print for principal point adjustment
        cx_source = "COLMAP" if getattr(camera, '_cx', None) is not None else "center"
        cy_source = "COLMAP" if getattr(camera, '_cy', None) is not None else "center"
        print(f"[crop-projection] Camera {camera.image_name}:", flush=True)
        print(f"  Original image: {orig_width}x{orig_height}", flush=True)
        print(f"  Principal point: ({orig_cx:.1f}, {orig_cy:.1f}) [cx:{cx_source}, cy:{cy_source}]", flush=True)
        print(f"  Crop: ({crop.x_min}, {crop.y_min}) - ({crop.x_max}, {crop.y_max}) = {crop.width}x{crop.height}", flush=True)
        print(f"  New principal point (in crop coords): ({new_cx:.1f}, {new_cy:.1f})", flush=True)
        print(f"  Focal length: fx={focal_x:.1f}, fy={focal_y:.1f}", flush=True)
        print(f"  Frustum bounds: L={left:.4f}, R={right:.4f}, B={bottom:.4f}, T={top:.4f}", flush=True)

        # Note: We allow extreme off-center projections now that the rasterizer
        # properly supports proj_offset_x and proj_offset_y parameters.
        # The old check was too conservative and prevented off-center projection
        # from working in many valid cases.

        # Build projection matrix using exact getProjectionMatrix formula
        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0

        print(f"  Projection: P[0,2]={P[0,2]:.4f}, P[1,2]={P[1,2]:.4f}", flush=True)

        camera.projection_matrix = P.transpose(0, 1).cuda()

        # Update full_proj_transform
        camera.full_proj_transform = (
            camera.world_view_transform.unsqueeze(0).bmm(
                camera.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
