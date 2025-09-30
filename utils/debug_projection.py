import torch
import cv2
import numpy as np
from utils.camera_param_parser import parse_camera_parameters_heuristic

def debug_gaussian_projection(gaussians, scene, cameras_dict, images_dict):
    """
    Debug function to check how many Gaussians project into each camera view.
    This uses the same projection method as colmap_loader.py for consistency.

    Args:
        gaussians: GaussianModel with initialized points
        scene: Scene object with cameras
        cameras_dict: COLMAP cameras dictionary
        images_dict: COLMAP images dictionary
    """
    print("🔍 DEBUG: Checking Gaussian projection into camera views...")

    # Get current Gaussian positions
    gaussian_positions = gaussians.get_xyz.detach().cpu().numpy()  # Shape: (N, 3)
    print(f"📊 Total Gaussians: {len(gaussian_positions)}")

    # Project to each camera
    for cam in scene.train_cameras:
        camera_id = cam.colmap_id  # Use original COLMAP ID

        if camera_id not in images_dict or camera_id not in cameras_dict:
            print(f"⚠️  Camera {camera_id} not found in COLMAP data")
            continue

        camera = cameras_dict[camera_id]
        image = images_dict[camera_id]

        # Use same projection method as colmap_loader.py
        try:
            # Parse camera parameters using heuristic
            parsed_params = parse_camera_parameters_heuristic(
                camera.params, camera.width, camera.height, camera.model
            )

            fx = parsed_params['fx']
            fy = parsed_params['fy']
            cx = parsed_params['cx']
            cy = parsed_params['cy']
            distortion = parsed_params['distortion']

            # Camera matrix (same as colmap_loader.py)
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)

            # Distortion coefficients (same as colmap_loader.py)
            dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)
            if len(distortion) > 0:
                num_dist_params = min(len(distortion), 5)
                dist_coeffs[:num_dist_params] = distortion[:num_dist_params]

            # Rotation and translation (same as colmap_loader.py)
            # Convert quaternion to rotation matrix
            qw, qx, qy, qz = image.qvec
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])
            T = np.array(image.tvec)

            # For cv2.projectPoints
            rvec, _ = cv2.Rodrigues(R)
            tvec = T.reshape(3, 1)

            # Project all Gaussians
            points_2d, _ = cv2.projectPoints(
                gaussian_positions.reshape(-1, 1, 3),  # World coordinates
                rvec,  # Rotation vector
                tvec,  # Translation vector
                camera_matrix,
                dist_coeffs
            )

            points_2d = points_2d.reshape(-1, 2)  # Shape: (N, 2)

            # Check which points are in front of camera
            camera_points = (R.T @ (gaussian_positions - T).T).T  # Nx3
            valid_mask = camera_points[:, 2] > 0

            # Check which points are within image bounds
            in_bounds_mask = (
                (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.width) &
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.height)
            )

            # Combine masks
            visible_mask = valid_mask & in_bounds_mask
            visible_count = np.sum(visible_mask)

            print(f"  📷 Camera {camera_id}: {visible_count} / {len(gaussian_positions)} Gaussians visible")

        except Exception as e:
            print(f"❌ Error projecting to camera {camera_id}: {e}")

    print("🔍 DEBUG: Gaussian projection check complete")