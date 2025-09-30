"""
Projection utilities for 3D points to camera views
Common functions for cv2.projectPoints usage across the codebase
"""

import numpy as np
import cv2


def project_points_to_camera(points_3d, camera_R, camera_t, camera_K, dist_coeffs=None,
                             check_behind_camera=True, image_width=None, image_height=None,
                             margin_pixels=0):
    """
    Project 3D world points to a single camera view using cv2.projectPoints

    This is a common utility function used across colmap_visualizer.py, colmap_loader.py,
    and train_internal.py to ensure consistent projection behavior.

    Args:
        points_3d: numpy array of 3D points in world coordinates, shape (N, 3)
        camera_R: 3x3 rotation matrix (world -> camera)
        camera_t: 3x1 or (3,) translation vector (world -> camera)
        camera_K: 3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        dist_coeffs: distortion coefficients (5,) or None. If None, uses zero distortion
        check_behind_camera: if True, returns mask for points in front of camera
        image_width: image width in pixels (for bounds checking if provided)
        image_height: image height in pixels (for bounds checking if provided)
        margin_pixels: margin in pixels for bounds checking
                      - Positive: shrinks valid region (stricter)
                      - Negative: expands valid region (more lenient)
                      - Zero: exact image boundary

    Returns:
        dict with keys:
            'points_2d': numpy array of projected 2D points, shape (N, 2)
            'in_front_mask': boolean array (N,) - True if point is in front of camera (if check_behind_camera=True)
            'in_bounds_mask': boolean array (N,) - True if point is within image bounds (if image_width/height provided)
            'visible_mask': boolean array (N,) - True if point is in front AND in bounds (if all checks enabled)

    Example:
        >>> R = np.eye(3)
        >>> t = np.array([0, 0, 0])
        >>> K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        >>> result = project_points_to_camera(points_3d, R, t, K,
        ...                                   image_width=1920, image_height=1080)
        >>> visible_points = points_3d[result['visible_mask']]
    """
    # Ensure inputs are numpy arrays with correct dtypes
    points_3d = np.asarray(points_3d, dtype=np.float64)
    camera_R = np.asarray(camera_R, dtype=np.float64)
    camera_t = np.asarray(camera_t, dtype=np.float64).flatten()
    camera_K = np.asarray(camera_K, dtype=np.float64)

    # Handle distortion coefficients
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)
    else:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        # Pad or truncate to 5 elements as expected by cv2.projectPoints
        if len(dist_coeffs) < 5:
            dist_coeffs = np.pad(dist_coeffs, (0, 5 - len(dist_coeffs)), 'constant')
        elif len(dist_coeffs) > 5:
            dist_coeffs = dist_coeffs[:5]

    # Convert rotation matrix to rotation vector for cv2.projectPoints
    rvec, _ = cv2.Rodrigues(camera_R)
    tvec = camera_t.reshape(3, 1)

    # Project all points at once using cv2.projectPoints
    # Input shape: (N, 1, 3) for world coordinates
    object_points = points_3d.reshape(-1, 1, 3)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_K, dist_coeffs)

    # Reshape to (N, 2)
    points_2d = image_points.reshape(-1, 2)

    # Prepare result dictionary
    result = {
        'points_2d': points_2d
    }

    # Check which points are in front of camera (positive Z in camera coordinates)
    if check_behind_camera:
        # Transform points to camera coordinates: P_cam = R * P_world + t
        points_camera = (camera_R @ points_3d.T).T + camera_t
        in_front_mask = points_camera[:, 2] > 0
        result['in_front_mask'] = in_front_mask
    else:
        in_front_mask = np.ones(len(points_3d), dtype=bool)
        result['in_front_mask'] = in_front_mask

    # Check which points are within image bounds
    if image_width is not None and image_height is not None:
        in_bounds_mask = (
            (points_2d[:, 0] >= margin_pixels) &
            (points_2d[:, 0] < image_width - margin_pixels) &
            (points_2d[:, 1] >= margin_pixels) &
            (points_2d[:, 1] < image_height - margin_pixels)
        )
        result['in_bounds_mask'] = in_bounds_mask

        # Combine both checks for final visibility
        result['visible_mask'] = in_front_mask & in_bounds_mask
    else:
        result['in_bounds_mask'] = None
        result['visible_mask'] = in_front_mask

    return result