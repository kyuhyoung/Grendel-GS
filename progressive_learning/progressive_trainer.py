"""
Progressive Training Controller for Grendel-GS
Main control logic for DTM-based progressive learning
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import json
import sys

# Add scripts/experiments to path
scripts_path = Path(__file__).parent.parent / "scripts" / "experiments"
sys.path.insert(0, str(scripts_path))

from colmap_visualizer import COLMAPVisualizer


def calculate_balanced_smooth_score(candidate_idx, D_indices, positions, F, prev_window_center,
                                   prev_movement, window_size, outward_weight, compact_weight,
                                   smooth_window_weight, smooth_camera_weight, distance_weight,
                                   directional_weight, last_added_idx, second_last_added_idx,
                                   initial_window_indices=None, debug=False, candidate_cam_id=None,
                                   D_cam_ids=None, idx_to_cam_id=None):
    """
    Calculate score balancing six forces:
    1. Outward: Window center moves away from F
    2. Compact: Window variance is minimized
    3. Smooth Window: Window center trajectory is smooth (velocity continuity)
    4. Smooth Camera: Added camera trajectory is smooth (directional continuity)
    5. Distance: Candidate is close to current window center
    6. Directional Alignment: Candidate aligns with window movement direction

    Args:
        candidate_idx: Index of candidate camera
        D_indices: List of camera indices in current window D
        positions: Array of all camera positions [N, 2]
        F: Global mean position [2]
        prev_window_center: Previous window center [2] or None
        prev_movement: Previous window center movement [2] or None
        window_size: Maximum window size
        outward_weight: Weight for outward score
        compact_weight: Weight for compact score
        smooth_window_weight: Weight for smooth window score
        smooth_camera_weight: Weight for smooth camera score
        distance_weight: Weight for distance score
        directional_weight: Weight for directional alignment score
        last_added_idx: Index of last added camera or None
        second_last_added_idx: Index of second-to-last added camera or None

    Returns:
        Tuple of (total_score, outward_score, compact_score, smooth_window_score,
                 smooth_camera_score, distance_score, directional_score, D_prime_center, variance)
    """

    # Create new window D' with candidate, considering FIFO removal
    D_prime_indices = list(D_indices)

    # FIFO removal logic (matches line 2720-2732 in main loop)
    if initial_window_indices is not None and len(D_prime_indices) >= window_size:
        # Check how many initial window cameras remain in D
        remaining_initial_in_D = [idx for idx in D_prime_indices if idx in initial_window_indices]

        if len(remaining_initial_in_D) >= 2:
            # 2개 이상 남아있으면 → candidate와 가장 먼 initial camera 제거
            candidate_pos = positions[candidate_idx]
            max_distance = -1
            farthest_idx = None

            for idx in remaining_initial_in_D:
                dist = np.linalg.norm(positions[idx] - candidate_pos)
                if dist > max_distance:
                    max_distance = dist
                    farthest_idx = idx

            if farthest_idx is not None:
                D_prime_indices.remove(farthest_idx)
        else:
            # 1개 이하면 → 진짜 FIFO (D_indices[0] 제거)
            oldest_cam_idx = D_prime_indices[0]
            D_prime_indices.remove(oldest_cam_idx)

    # Add candidate
    D_prime_indices.append(candidate_idx)

    D_prime_positions = positions[D_prime_indices]
    D_prime_center = np.mean(D_prime_positions, axis=0)

    # 1. Outward score: Candidate camera's distance from F (normalized)
    # Reward cameras farther from global center F
    candidate_pos = positions[candidate_idx]
    candidate_dist_to_F = np.linalg.norm(candidate_pos - F)

    # Normalize by mean distance from F to all cameras for scale independence
    F_mean_radius = np.mean(np.linalg.norm(positions - F, axis=1)) + 1e-6
    normalized_dist_to_F = candidate_dist_to_F / F_mean_radius
    outward_score = normalized_dist_to_F  # Typically 0.5 ~ 1.5 range

    # 2. Compact score: Window radius (max distance from center) should be small
    distances_from_center = np.linalg.norm(D_prime_positions - D_prime_center, axis=1)
    variance = np.var(distances_from_center)  # Still keep for logging
    max_distance = np.max(distances_from_center) + 1e-6

    # Normalize by current window D's mean radius for scale independence
    D_positions = positions[D_indices]
    if len(D_indices) == 1:
        D_positions = D_positions.reshape(1, -1)
    D_center = np.mean(D_positions, axis=0)
    D_mean_radius = np.mean(np.linalg.norm(D_positions - D_center, axis=1)) + 1e-6

    # Normalized max distance (relative to current window size)
    normalized_max_distance = max_distance / D_mean_radius
    # Lower normalized distance = higher score (exponential decay)
    compact_score = np.exp(-normalized_max_distance)

    # Debug logging for compact score calculation
    if debug and candidate_cam_id is not None:
        print(f"\n{'='*80}")
        print(f"🔍 COMPACT SCORE DEBUG for Camera {candidate_cam_id}")
        print(f"{'='*80}")

        # Build D' camera IDs list
        if D_cam_ids is not None:
            D_prime_cam_ids = D_cam_ids + [candidate_cam_id]
            if len(D_prime_cam_ids) > window_size:
                D_prime_cam_ids = D_prime_cam_ids[-window_size:]
            print(f"D' = D ∪ {{candidate {candidate_cam_id}}} = Camera IDs: {D_prime_cam_ids}")
        else:
            print(f"D' = D ∪ {{candidate {candidate_cam_id}}}")
        print(f"D' array indices: {D_prime_indices}")

        if len(D_prime_center) == 2:
            print(f"D'_center: ({D_prime_center[0]:.2f}, {D_prime_center[1]:.2f})")
        else:
            print(f"D'_center: {D_prime_center}")
        print(f"\nDistances from D'_center to each camera in D':")
        for i, (idx, dist) in enumerate(zip(D_prime_indices, distances_from_center)):
            marker = " ← candidate" if idx == candidate_idx else ""
            cam_id_str = ""
            if idx_to_cam_id is not None and idx in idx_to_cam_id:
                cam_id_str = f" (Camera {idx_to_cam_id[idx]})"
            pos_str = f"({positions[idx][0]:.2f}, {positions[idx][1]:.2f})" if len(positions[idx]) == 2 else str(positions[idx])
            print(f"  D'[{i}]{cam_id_str} at {pos_str}: {dist:.3f}m{marker}")
        print(f"Max distance in D': {max_distance:.3f}m")
        print(f"Variance of D': {variance:.3f}")

        print(f"\nCurrent window D:")
        if D_cam_ids is not None:
            print(f"D Camera IDs: {D_cam_ids}")
        print(f"D array indices: {D_indices}")
        if len(D_center) == 2:
            print(f"D_center: ({D_center[0]:.2f}, {D_center[1]:.2f})")
        else:
            print(f"D_center: {D_center}")
        D_distances = np.linalg.norm(D_positions - D_center, axis=1)
        print(f"D distances from D_center: {[f'{d:.3f}' for d in D_distances]}")
        print(f"D_mean_radius: {D_mean_radius:.3f}m")
        print(f"\n📐 Calculation:")
        print(f"  Normalized max distance = {max_distance:.3f} / {D_mean_radius:.3f} = {normalized_max_distance:.6f}")
        print(f"  Compact score = exp(-{normalized_max_distance:.6f}) = {compact_score:.6f}")
        print(f"{'='*80}\n")

    # 3. Smooth Window score: Window center movement direction should be continuous
    smooth_window_score = 0.0
    if prev_window_center is not None and prev_movement is not None:
        # Current movement vector
        current_movement = D_prime_center - prev_window_center
        current_norm = np.linalg.norm(current_movement)
        prev_norm = np.linalg.norm(prev_movement)

        if prev_norm > 1e-6 and current_norm > 1e-6:
            # Normalize vectors
            prev_movement_unit = prev_movement / prev_norm
            current_movement_unit = current_movement / current_norm

            # Cosine similarity between movements
            cos_similarity = np.dot(prev_movement_unit, current_movement_unit)
            cos_similarity = np.clip(cos_similarity, -1.0, 1.0)

            # Score: high when directions are similar (smooth trajectory)
            # Range: [-1, 1] → [0, 1], where 1 = same direction, 0 = opposite
            smooth_window_score = (1 + cos_similarity) / 2.0

    # 4. Smooth Camera score: Added camera direction should be continuous
    smooth_camera_score = 0.0
    if last_added_idx is not None and second_last_added_idx is not None:
        # Previous camera movement: second_last → last
        prev_camera_movement = positions[last_added_idx] - positions[second_last_added_idx]
        # Current camera movement: last → candidate
        current_camera_movement = positions[candidate_idx] - positions[last_added_idx]

        prev_cam_norm = np.linalg.norm(prev_camera_movement)
        current_cam_norm = np.linalg.norm(current_camera_movement)

        if prev_cam_norm > 1e-6 and current_cam_norm > 1e-6:
            # Normalize vectors
            prev_cam_unit = prev_camera_movement / prev_cam_norm
            current_cam_unit = current_camera_movement / current_cam_norm

            # Cosine similarity
            cos_similarity_cam = np.dot(prev_cam_unit, current_cam_unit)
            cos_similarity_cam = np.clip(cos_similarity_cam, -1.0, 1.0)

            # Score: high when directions are similar
            smooth_camera_score = (1 + cos_similarity_cam) / 2.0

    # 5. Distance score: Candidate should be close to FUTURE window center
    # Determine which cameras will remain after FIFO removal
    D_for_distance_indices = D_indices

    if initial_window_indices is not None and len(D_indices) >= window_size:
        # Check how many initial cameras remain in D
        remaining_initial_in_D = [idx for idx in D_indices if idx in initial_window_indices]

        if len(remaining_initial_in_D) <= 1:
            # FIFO will definitely remove D_indices[0]
            # Use only cameras that will remain in future window
            D_for_distance_indices = D_indices[1:]

    # Calculate future window center
    if len(D_for_distance_indices) == 0:
        # Fallback: use full D if something goes wrong
        D_for_distance_indices = D_indices

    D_positions = positions[D_for_distance_indices]
    if len(D_for_distance_indices) == 1:
        D_positions = D_positions.reshape(1, -1)
    D_center = np.mean(D_positions, axis=0)

    # Distance from candidate to future window center
    candidate_pos = positions[candidate_idx]
    dist_to_D_center = np.linalg.norm(candidate_pos - D_center)

    # Calculate normalization factor (mean radius of future window)
    D_mean_radius = np.mean(np.linalg.norm(D_positions - D_center, axis=1)) + 1e-6

    # Exponential decay score (closer = higher score)
    distance_score = np.exp(-dist_to_D_center / (2 * D_mean_radius))

    # 6. Directional Alignment score: Candidate should align with window movement direction
    directional_score = 0.0
    if prev_window_center is not None:
        # Calculate current window center (before adding candidate)
        D_positions_current = positions[D_indices]
        if len(D_indices) == 1:
            D_positions_current = D_positions_current.reshape(1, -1)
        current_window_center = np.mean(D_positions_current, axis=0)

        # Window movement direction: from current window center to future window center (D')
        # Both vectors start from the same point (current_window_center)
        window_movement = D_prime_center - current_window_center

        # Direction from current window center to candidate
        candidate_direction = candidate_pos - current_window_center

        # Calculate cosine similarity
        window_norm = np.linalg.norm(window_movement)
        candidate_norm = np.linalg.norm(candidate_direction)

        if window_norm > 1e-6 and candidate_norm > 1e-6:
            window_unit = window_movement / window_norm
            candidate_unit = candidate_direction / candidate_norm

            cos_similarity = np.dot(window_unit, candidate_unit)
            cos_similarity = np.clip(cos_similarity, -1.0, 1.0)

            # Score: high when candidate is in the direction of window movement
            directional_score = (1 + cos_similarity) / 2.0

    # Weighted sum
    total_score = (outward_weight * outward_score +
                   compact_weight * compact_score +
                   smooth_window_weight * smooth_window_score +
                   smooth_camera_weight * smooth_camera_score +
                   distance_weight * distance_score +
                   directional_weight * directional_score)

    return total_score, outward_score, compact_score, smooth_window_score, smooth_camera_score, distance_score, directional_score, D_prime_center, variance


class ProgressiveTrainer:
    """
    Main controller for progressive training with DTM-based camera selection
    
    Algorithm Flow:
    1. Load COLMAP data (N images)
    2. Create DTM mesh from 3D point cloud
    3. Extract (x,y) coordinates set W from points
    4. Compute camera footprints F_c and centers S_c
    5. Select initial m cameras based on median position
    6. Train initial Gaussian set V
    7. Progressively add cameras until GPU memory limit
    8. Swap Gaussians for remaining cameras
    """
    
    def __init__(
        self,
        colmap_path: str,
        output_path: str,
        dtm_module=None,
        initial_cameras: int = 2,
        camera_removal_margin: float = 0.15,
        debug: bool = False,
        only_positive_z: bool = True,
        only_actually_visible: bool = True
    ):
        """
        Initialize progressive trainer

        Args:
            colmap_path: Path to COLMAP reconstruction
            output_path: Path for output files
            dtm_module: External DTM mesh module (if available)
            initial_cameras: Number of cameras to start with
            camera_removal_margin: Margin below densify_memory_limit_percentage for camera removal decision (default: 0.1)
            debug: Enable debug output
        """
        self.colmap_path = Path(colmap_path)
        self.output_path = Path(output_path)

        # Remove existing output directory and create fresh one
        if self.output_path.exists():
            import shutil
            shutil.rmtree(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.dtm_module = dtm_module
        self.initial_cameras = initial_cameras
        self.only_positive_z = only_positive_z
        self.camera_removal_margin = camera_removal_margin
        self.debug = debug
        # Skip sliding window visualization for fast debugging (controlled by env variable or default True)
        import os
        self.skip_4_fast_debug = bool(int(os.environ.get('SKIP_4_FAST_DEBUG', '1')))
        self.only_actually_visible = only_actually_visible
        self.densify_memory_limit_percentage = None  # Will be set from run_progressive.py
        self.exit_after_first_removal = False  # Will be set from run_progressive.py
        
        # Data containers
        self.cameras = {}
        self.images = {}
        self.points3D = {}
        
        # Computed data
        self.W = None  # (x,y) coordinates of all 3D points
        self.camera_footprints = {}  # F_c for each camera
        self.points_in_view = {}  # Camera ID -> List of 3D point IDs visible through projection
        self.cumulative_removed_regions = None  # Cumulative union of removed footprint regions (Shapely Polygon/MultiPolygon)
        
        # Training state
        self.current_window_cameras = set()  # Current sliding window cameras
        self.H = None  # Union of selected footprints
        self.V = None  # Current Gaussian set
        self.I = []  # Removed Gaussians (saved to PLY)

        # Sliding window global state
        self.unprocessed_cameras = set()  # Cameras remaining to be processed
        self.processed_cameras = set()  # Cameras that have been fully processed
        self.all_processed_cameras = set()  # All cameras that have ever been in any window (including removed ones)
        self.unprocessed_points = set()  # 3D points not yet covered by any processed camera
        self.trained_gaussians = set()  # Gaussians that have completed training and been saved
        # Note: We only track processed_gaussians, not processed_points

        # Checkpoint management
        
        # Find COLMAP files and image directory
        self._find_colmap_files()
        
        # COLMAP data will be loaded later via COLMAPVisualizer with filtering
        
        # Initialize visualizer with external data
        # only_actually_visible 플래그 전달
        #self.visualizer = COLMAPVisualizer(only_actually_visible=self.only_actually_visible)
        #self.visualizer.set_external_data(self.cameras, self.images, self.points3D)
        
    def _find_colmap_files(self):
        """Find COLMAP files and image directory"""
        print(f"Searching for COLMAP files in {self.colmap_path}")
        
        # Search for COLMAP files (cameras.txt, images.txt, points3D.txt)
        possible_paths = [
            self.colmap_path,
            self.colmap_path / "sparse",
            self.colmap_path / "sparse" / "0",
        ]
        
        self.colmap_files_path = None
        for path in possible_paths:
            if (path / "cameras.txt").exists() and (path / "images.txt").exists() and (path / "points3D.txt").exists():
                self.colmap_files_path = path
                print(f"Found COLMAP files in: {path}")
                break
        
        #print(f'self.colmap_files_path : {self.colmap_files_path}');    exit(1)
        if self.colmap_files_path is None:
            raise FileNotFoundError(f"Could not find COLMAP files (cameras.txt, images.txt, points3D.txt) in {self.colmap_path} or its sparse/ subdirectories")
        
        # Find image directory by checking image names from images.txt
        print("Searching for image directory...")
        
        # First, read a few image names from images.txt
        images_file = self.colmap_files_path / "images.txt"
        sample_image_names = []
        
        with open(images_file, 'r') as f:
            lines = f.readlines()
            count = 0
            for i in range(0, len(lines), 2):
                if lines[i].startswith('#') or not lines[i].strip():
                    continue
                parts = lines[i].strip().split()
                if len(parts) >= 10:
                    image_name = parts[9]
                    sample_image_names.append(image_name)
                    count += 1
                    if count >= 5:  # Check first 5 images
                        break
        
        #print(f'sample_image_names : {sample_image_names}');    exit(1)
        if not sample_image_names:
            raise ValueError("No valid image names found in images.txt")
        
        # Search for image directory
        possible_image_paths = [
            self.colmap_path / "images",
            self.colmap_path / "input",
            self.colmap_path,
            self.colmap_path.parent / "images",
            self.colmap_path.parent / "input",
        ]
        
        self.image_path = None
        for img_path in possible_image_paths:
            if img_path.exists() and img_path.is_dir():
                # Check if sample images exist in this directory
                found_count = 0
                for img_name in sample_image_names:
                    if (img_path / img_name).exists():
                        found_count += 1
                
                if found_count >= len(sample_image_names) * 0.8:  # At least 80% of sample images found
                    self.image_path = img_path
                    print(f"Found image directory: {img_path}")
                    break
        
        #print(f'self.image_path : {self.image_path}');    exit(1)
        if self.image_path is None:
            print(f"Warning: Could not find image directory. Searched in:")
            for path in possible_image_paths:
                print(f"  {path}")
            print(f"Sample image names: {sample_image_names[:3]}...")
            self.image_path = self.colmap_path / "images"  # Default fallback
            print(f"Using default image path: {self.image_path}")
        
    def _load_colmap_data(self):
        """Load COLMAP reconstruction data"""
        print(f"Loading COLMAP data from {self.colmap_path}")
        
        # Load cameras.txt
        cameras_file = self.colmap_files_path / "cameras.txt"
        if cameras_file.exists():
            with open(cameras_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    cam_id = int(parts[0])
                    params_list = [float(x) for x in parts[4:]]

                    # Convert params to dictionary format for compatibility with colmap_visualizer
                    if parts[1] == 'PINHOLE':
                        params_dict = {
                            'fx': params_list[0],
                            'fy': params_list[1],
                            'cx': params_list[2],
                            'cy': params_list[3],
                            'distortion': []
                        }
                    elif parts[1] == 'RADIAL':
                        # RADIAL: fx, cx, cy, k1, k2
                        params_dict = {
                            'fx': params_list[0],
                            'fy': params_list[0],  # fx == fy for RADIAL
                            'cx': params_list[1],
                            'cy': params_list[2],
                            'distortion': params_list[3:] if len(params_list) > 3 else []
                        }
                    else:
                        # For other camera models, use indices
                        params_dict = {f'param_{i}': p for i, p in enumerate(params_list)}

                    self.cameras[cam_id] = {
                        'id': cam_id,
                        'model': parts[1],
                        'width': int(parts[2]),
                        'height': int(parts[3]),
                        'params': params_dict,  # Use dictionary format consistently
                        'raw_params': params_list  # Keep original list for backward compatibility
                    }
        
        # Load images.txt
        images_file = self.colmap_files_path / "images.txt"
        if images_file.exists():
            with open(images_file, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 2):
                    if lines[i].startswith('#'):
                        continue
                    parts = lines[i].strip().split()
                    if len(parts) >= 10:
                        img_id = int(parts[0])
                        # Read quaternion and translation
                        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])

                        # Convert quaternion to rotation matrix
                        # R is world-to-camera rotation
                        R = np.array([
                            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
                        ])

                        # Camera center in world coordinates: C = -R^T * t
                        t = np.array([tx, ty, tz])
                        camera_center = -R.T @ t

                        self.images[img_id] = {
                            'id': img_id,
                            'qw': qw,
                            'qx': qx,
                            'qy': qy,
                            'qz': qz,
                            'tx': tx,
                            'ty': ty,
                            'tz': tz,
                            'camera_id': int(parts[8]),
                            'name': parts[9],
                            'position': camera_center  # Camera center in world coordinates
                        }
        
        # Load points3D.txt with track information
        points_file = self.colmap_files_path / "points3D.txt"
        if points_file.exists():
            print(f"📖 Loading points3D from: {points_file}")
            total_points_in_file = 0
            points_after_positive_z = 0
            points_after_track_filter = 0

            with open(points_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        total_points_in_file += 1
                        pt_id = int(parts[0])
                        xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

                        # only_positive_z 플래그 체크
                        if self.only_positive_z and xyz[2] < 0:
                            continue  # 음수 Z값 포인트는 제외
                        points_after_positive_z += 1

                        rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
                        error = float(parts[7])

                        # Parse track information (IMAGE_ID, POINT2D_IDX pairs)
                        track = []
                        track_tokens = parts[8:]
                        if len(track_tokens) % 2 == 0:  # Should be pairs
                            for i in range(0, len(track_tokens), 2):
                                img_id = int(track_tokens[i])
                                point2d_idx = int(track_tokens[i + 1])
                                track.append((img_id, point2d_idx))

                        # Skip points with no track (only_actually_visible - now default)
                        if len(track) == 0:
                            continue

                        # Skip points with non-positive Z (only_positive_z - now default)
                        if xyz[2] <= 0:
                            continue

                        points_after_track_filter += 1
                        self.points3D[pt_id] = {
                            'id': pt_id,
                            'xyz': xyz,
                            'rgb': rgb,
                            'error': error,
                            'track': track
                        }

            print(f"📊 Points3D filtering results:")
            print(f"  📄 Total points in file: {total_points_in_file}")
            print(f"  ➕ After positive Z filter: {points_after_positive_z}")
            print(f"  👀 After track filter: {points_after_track_filter}")
            print(f"  ✅ Final loaded points: {len(self.points3D)}")

        print(f"Loaded: {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} 3D points")
        
        # Extract W: (x,y) coordinates of all points
        self.W = np.array([[pt['xyz'][0], pt['xyz'][1]] for pt in self.points3D.values()])
        self.points_3d = np.array([pt['xyz'] for pt in self.points3D.values()])  # Full 3D coordinates for visualization
        print(f"Extracted {len(self.W)} (x,y) coordinates")
        
        # Print 3D bounding box of point cloud
        if len(self.points_3d) > 0:
            x_min, x_max = np.min(self.points_3d[:, 0]), np.max(self.points_3d[:, 0])
            y_min, y_max = np.min(self.points_3d[:, 1]), np.max(self.points_3d[:, 1])
            z_min, z_max = np.min(self.points_3d[:, 2]), np.max(self.points_3d[:, 2])
            
            print(f"Point cloud 3D bounding box:")
            print(f"  X: [{x_min:.2f}, {x_max:.2f}] (range: {x_max-x_min:.2f}m)")
            print(f"  Y: [{y_min:.2f}, {y_max:.2f}] (range: {y_max-y_min:.2f}m)")
            print(f"  Z: [{z_min:.2f}, {z_max:.2f}] (range: {z_max-z_min:.2f}m)")
            
            # Z값 분포 분석
            z_values = self.points_3d[:, 2]
            z_mean = np.mean(z_values)
            z_median = np.median(z_values)
            z_std = np.std(z_values)
            
            # Z값 히스토그램 (간단한 분포 확인)
            z_negative_count = np.sum(z_values < 0)
            z_positive_count = np.sum(z_values >= 0)
            
            print(f"Z값 분석:")
            print(f"  평균: {z_mean:.2f}m, 중앙값: {z_median:.2f}m, 표준편차: {z_std:.2f}m")
            print(f"  음수 포인트: {z_negative_count:,}개 ({z_negative_count/len(z_values)*100:.1f}%)")
            print(f"  양수 포인트: {z_positive_count:,}개 ({z_positive_count/len(z_values)*100:.1f}%)")
            
            # 카메라 위치도 확인
            if self.images:
                camera_positions = np.array([img['position'] for img in self.images.values()])
                cam_z_min, cam_z_max = np.min(camera_positions[:, 2]), np.max(camera_positions[:, 2])
                cam_z_mean = np.mean(camera_positions[:, 2])
                print(f"카메라 Z 위치: [{cam_z_min:.2f}, {cam_z_max:.2f}] (평균: {cam_z_mean:.2f}m)")

            #exit(1)

    def _rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        import numpy as np

        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        return qw, qx, qy, qz

    def compute_camera_footprints(self):
        """
        Step 4: Compute camera footprints and centers
        For each camera, compute the footprint on DTM and its center
        """
        print("Computing camera footprints...")
        
        # Use verified colmap_visualizer directly
        if self.dtm_module is None:
            try:
                import sys
                import os
                # COLMAPVisualizer already imported at top
                print("Loading built-in DTM module...")
                
                # Find actual COLMAP files location
                colmap_path = self.colmap_path
                if (colmap_path / "cameras.txt").exists():
                    actual_path = colmap_path
                elif (colmap_path / "sparse" / "0" / "cameras.txt").exists():
                    actual_path = colmap_path / "sparse" / "0"
                elif (colmap_path / "sparse" / "cameras.txt").exists():
                    actual_path = colmap_path / "sparse"
                else:
                    raise FileNotFoundError(f"Could not find COLMAP files in {colmap_path}")
                
                # only_actually_visible 플래그는 kwargs에서 저장한 값 사용
                self.dtm_module = COLMAPVisualizer(str(actual_path), only_actually_visible=self.only_actually_visible)
                self.dtm_module.read_cameras_txt()
                self.dtm_module.read_images_txt()
                self.dtm_module.read_points3d_txt()  # analyze_unobserved_points() is called inside this method
                self.dtm_module.create_dtm(resolution=2.0)

                # Update all COLMAP data with filtered data from COLMAPVisualizer
                if hasattr(self.dtm_module, 'cameras') and self.dtm_module.cameras:
                    print(f"🔄 Updating COLMAP data from COLMAPVisualizer")

                    # Copy cameras
                    self.cameras = self.dtm_module.cameras.copy()
                    print(f"   Cameras: {len(self.cameras)}")

                    # Copy and convert images
                    self.images = {}
                    for img_id, img_data in self.dtm_module.images.items():
                        # Convert from COLMAPVisualizer format to ProgressiveTrainer format
                        # Extract position from camera_center or compute from R,t
                        if 'camera_center' in img_data:
                            position = img_data['camera_center']
                        else:
                            # Compute camera center: C = -R.T @ t
                            R = img_data['R']
                            t = img_data['t']
                            position = -R.T @ t

                        # Convert rotation matrix back to quaternion
                        qw, qx, qy, qz = self._rotation_matrix_to_quaternion(img_data['R'])

                        self.images[img_id] = {
                            'id': img_id,
                            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                            'tx': img_data['t'][0], 'ty': img_data['t'][1], 'tz': img_data['t'][2],
                            'camera_id': img_data['camera_id'],
                            'name': img_data['name'],
                            'position': position
                        }
                    print(f"   Images: {len(self.images)}")

                    # Copy and convert points3D
                    if hasattr(self.dtm_module, 'points3d') and self.dtm_module.points3d:
                        # Get original count from DTM module (before filtering)
                        original_count = getattr(self.dtm_module, '_original_points_count', len(self.dtm_module.points3d))
                        filtered_points3D = {}
                        for pt_id, pt_data in self.dtm_module.points3d.items():
                            filtered_points3D[pt_id] = {
                                'xyz': pt_data['xyz'],
                                'rgb': pt_data['rgb'],
                                'error': pt_data['error'],
                                'track': pt_data.get('track', [])
                            }

                        self.points3D = filtered_points3D
                        removed_count = original_count - len(self.points3D)
                        if removed_count > 0:
                            print(f"   Points3D: {len(self.points3D)} (removed {removed_count} always_outside points)")
                        else:
                            print(f"   Points3D: {len(self.points3D)} (from DTM module)")


                    # Generate W and points_3d arrays from loaded data
                    if self.points3D:
                        self.W = np.array([[pt['xyz'][0], pt['xyz'][1]] for pt in self.points3D.values()])
                        self.points_3d = np.array([pt['xyz'] for pt in self.points3D.values()])
                        self.point_colors = np.array([pt['rgb'] for pt in self.points3D.values()])
                        print(f"   Generated W array: {len(self.W)} points")

                print("DTM module loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load DTM module: {e}")
                print("Using simplified footprint computation")
        
        for img_id, img_data in self.images.items():
            print(f'img_id : {img_id}')
            if self.dtm_module:
                # Use verified colmap_visualizer for accurate computation
                try:
                    # Get camera corners using verified method
                    camera_center, ray_dirs = self.dtm_module.get_camera_corners(img_id)

                    # Compute footprint from ray-DTM intersections
                    corners_3d = []
                    for i in range(4):  # 4 corners
                        ray_dir = ray_dirs[:, i]

                        # For aerial photos: if ray points upward, flip Z direction
                        if ray_dir[2] > 0:
                            ray_dir = ray_dir.copy()
                            ray_dir[2] = -ray_dir[2]  # Flip Z component to point downward

                        result, status = self.dtm_module.raycast_to_dtm(camera_center, ray_dir)
                        if result is not None:
                            corners_3d.append(result[:2])  # Only (x,y)
                    
                    if len(corners_3d) == 4:
                        # Valid rectangular footprint
                        F_c = np.array(corners_3d)
                        S_c = np.mean(F_c, axis=0)

                        # Filter points within footprint and get their IDs
                        from matplotlib.path import Path
                        poly_path = Path(F_c)
                        mask = poly_path.contains_points(self.W)

                        # Get point IDs that are within the footprint
                        point_ids = []
                        all_point_ids = list(self.points3D.keys())
                        for i, is_inside in enumerate(mask):
                            if is_inside:
                                point_ids.append(all_point_ids[i])
                        W_c = point_ids  # Store IDs instead of coordinates
                    else:
                        # DTM ray-casting failed to get exactly 4 corners
                        raise RuntimeError(
                            f"❌ Camera {img_id}: Failed to compute footprint from DTM ray-casting. "
                            f"Got {len(corners_3d)} corners but need exactly 4 for rectangular footprint. "
                            f"This should not happen with valid aerial imagery."
                        )
                        
                except Exception as e:
                    # Re-raise the error instead of using fallback
                    raise RuntimeError(
                        f"❌ Camera {img_id}: Error computing footprint from DTM: {e}"
                    ) from e
            else:
                # DTM module not available
                raise RuntimeError(
                    f"❌ Camera {img_id}: DTM module is required for footprint computation. "
                    f"Cannot proceed without DTM ray-casting."
                )

            # Store footprint rectangle (4 corners in (x,y))
            self.camera_footprints[img_id] = F_c

            # Store point IDs (W_c is now always a list of point IDs)
            # Get points visible through projection for this camera from DTM module
            if hasattr(self.dtm_module, 'camera_visible_points') and img_id in self.dtm_module.camera_visible_points:
                visible_points = self.dtm_module.camera_visible_points[img_id]
            else:
                # Fallback: use all points in footprint
                visible_points = W_c if 'W_c' in locals() else []

            self.points_in_view[img_id] = visible_points
            
        #print(f"Computed {len(self.camera_footprints)} footprints")
        avg_points = np.mean([len(pts) for pts in self.points_in_view.values()])
        print(f"Average points per footprint: {avg_points:.1f}")
        #print(f'self.debug : {self.debug}');    exit(1)    
        if self.debug and self.dtm_module is not None:
            # Use existing verified visualization from colmap_visualizer
            if not self.skip_4_fast_debug:
                try:
                    #print('111')
                    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                    #print('222')
                    scene_path = os.path.join(self.output_path, f"progressive_3d_scene_{timestamp}.png")
                    #print('333')
                    ortho_path = os.path.join(self.output_path, f"progressive_ortho_view_{timestamp}.png")

                    #print('444')
                    scene_center = self.dtm_module.visualize_3d_scene(save_path=scene_path)
                    print(f'ortho_path : {ortho_path}')
                    self.dtm_module.render_orthographic_view(scene_center, save_path=ortho_path)

                    print(f"Visualizations saved:")
                    print(f"  - 3D scene: {scene_path}")
                    print(f"  - Orthographic: {ortho_path}")
                except Exception as e:
                    print(f"Visualization failed: {e}");
                    exit(1)
            else:
                print("⚠️  SKIP_4_FAST_DEBUG: Skipping 3D scene and orthographic view visualization")

    def select_initial_cameras(self) -> List[int]:
        """
        Steps 5-7: Select initial m cameras based on median position
        """
        print(f"Selecting initial {self.initial_cameras} cameras...")
        
        # Step 5: Compute median of camera positions
        positions = np.array([img['position'] for img in self.images.values()])
        self.cam_pos_med = np.median(positions, axis=0)
        M = self.cam_pos_med[:2]  # (x,y) coordinate of median
        
        if self.debug:
            print(f'positions : \n{positions}')
            print(f"Median camera position: {self.cam_pos_med}")
        #exit(1)
        # Step 6-7: Find m cameras closest to 3D median
        distances = []
        for img_id, img_data in self.images.items():
            camera_pos = img_data['position']  # 3D position
            dist = np.linalg.norm(camera_pos - self.cam_pos_med)  # 3D distance
            distances.append((img_id, dist))
        
        distances.sort(key=lambda x: x[1])
        selected = [img_id for img_id, _ in distances[:self.initial_cameras]]
        
        print(f"Selected initial cameras: {selected}")

        # Print camera IDs with their 3D positions
        for img_id in selected:
            pos = self.images[img_id]['position']
            print(f"  Camera {img_id}: position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        #exit(1)
        return selected
    
    def create_initial_dataset(self, selected_cameras: List[int]) -> Dict:
        """
        Step 8: Create subset B from selected cameras P using track information
        """
        print("Creating initial dataset...")
        print(f"Filtering points observed in selected cameras: {selected_cameras}")

        included_points = {}

        for pt_id, pt_data in self.points3D.items():
            # Check if this point is observed by any of the selected cameras
            observed_in_selected = False
            filtered_track = []

            # Check the track information
            if 'track' in pt_data:
                for img_id, point2d_idx in pt_data['track']:
                    if img_id in selected_cameras:
                        observed_in_selected = True
                        filtered_track.append((img_id, point2d_idx))

            # Only include points that are observed in at least one selected camera
            if observed_in_selected:
                included_points[pt_id] = {
                    'id': pt_id,
                    'xyz': pt_data['xyz'],
                    'rgb': pt_data['rgb'],
                    'error': pt_data['error'],
                    'track': filtered_track
                }

        B = {
            'cameras': self.cameras,  # Keep all camera intrinsics
            'images': {img_id: self.images[img_id] for img_id in selected_cameras},
            'points3D': included_points
        }
        
        # Create visualizer for DTM and ray casting (for visualization)
        if self.debug:
            print("Debug mode is enabled, creating visualization...")
            print(f"self.dtm_module exists: {self.dtm_module is not None}")
            if self.dtm_module:
                print(f"create_nadir_view_multi exists: {hasattr(self.dtm_module, 'create_nadir_view_multi')}")
            self._visualize_original_vs_subset(B, selected_cameras, only_selected=False)  # Show all cameras for comparison

        print(f"Initial dataset: {len(B['images'])} images, {len(B['points3D'])} / {len(self.points3D)}  points")
        # Update H for future use (union of footprints)
        H_points = set()
        for pt_data in included_points.values():
            H_points.add(tuple(pt_data['xyz'][:2]))
        self.H = H_points


        return B

    def _visualize_original_vs_subset(self, subset_dataset: Dict, selected_cameras: List[int], only_selected: bool = False):
        """Create nadir view comparing original dataset vs subset using create_nadir_view_multi"""
        print("_visualize_original_vs_subset called")
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = self.output_path / f"progressive_nadir_comparison_{timestamp}.png"
            print(f"Attempting to create nadir comparison at: {comparison_path}")

            # Prepare subset info for create_nadir_view_multi
            subsets_info = [
                {
                    'name': 'Initial Selection',
                    'camera_ids': selected_cameras,
                    'color': 'red'
                }
            ]

            print(f"Selected cameras for visualization: {selected_cameras}")
            print("Calling dtm_module.create_nadir_view_multi...")

            # Use the new create_nadir_view_multi function
            self.dtm_module.create_nadir_view_multi(subsets_info, save_path=str(comparison_path), only_selected=only_selected)
            print(f"✓ Nadir comparison saved to: {comparison_path}")
            '''
            if hasattr(self, 'visualizer') and hasattr(self.visualizer, 'create_nadir_view_multi'):
                # Ensure DTM is created if not already
                if not hasattr(self.visualizer, 'dtm'):
                    print("Creating DTM for visualization...")
                    self.visualizer.create_dtm(resolution=2.0)

                # Call the new multi-subset visualization function
                #self.visualizer.create_nadir_view_multi(subsets_info, save_path=str(comparison_path))
            else:
                print("Warning: create_nadir_view_multi not available, skipping visualization")
            '''
        except Exception as e:
            print(f"Warning: Could not create dataset comparison visualization: {e}")

    def _visualize_sliding_window_coverage(self, dataset: Dict, current_cameras: List[int],
                                         window_num: int, removed_camera: int = None, added_camera: int = None):
        """Create nadir view showing current sliding window coverage"""
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            window_path = self.output_path / f"sliding_window_{window_num:03d}_{timestamp}.png"

            print(f"   Creating sliding window visualization at: {window_path}")

            # Prepare subset info with detailed naming
            window_name = f"Window {window_num}"
            if removed_camera is not None and added_camera is not None:
                window_name += f" (-{removed_camera}, +{added_camera})"

            subsets_info = [
                {
                    'name': window_name,
                    'camera_ids': current_cameras,
                    'color': 'blue'  # Different color from initial selection
                }
            ]

            print(f"   Current cameras in window: {current_cameras}")
            if removed_camera is not None:
                print(f"   Removed camera: {removed_camera}")
            if added_camera is not None:
                print(f"   Added camera: {added_camera}")

            # Calculate median position for visualization
            #positions = np.array([img['position'] for img in self.images.values()])
            #median_position = np.median(positions, axis=0)

            # Create visualization - show only selected cameras in sliding window
            self.dtm_module.create_nadir_view_multi(subsets_info, save_path=str(window_path),
                                                  only_selected=True, median_point=self.cam_pos_med)
            print(f"   ✓ Sliding window coverage saved to: {window_path}")

        except Exception as e:
            print(f"   Warning: Could not create sliding window visualization: {e}")

    def get_previous_iteration_name(self, current_name: str) -> str:
        """Get the previous iteration name for checkpoint loading"""
        if current_name == "initial":
            return None
        # Parse iteration number
        if current_name.startswith("iter_"):
            current_num = int(current_name.split("_")[1])
            if current_num == 1:
                return "initial"
            return f"iter_{current_num - 1:03d}"
        elif current_name.startswith("window_"):
            current_num = int(current_name.split("_")[1])
            if current_num == 1:
                return "initial"
            return f"window_{current_num - 1:03d}"
        return "initial"

    def train_grendel_gs(self, dataset: Dict, cam_id_2_delete, new_camera_id, total_iterations, iteration_name: str = "initial", sliding_window: bool = False) -> None:
        """
        Step 9: Train Grendel-GS on dataset

        TODO: Currently saves to disk then reloads. Could be optimized to pass
        dataset directly in memory if train.py is modified to accept it.

        Args:
            dataset: COLMAP dataset dictionary
            iteration_name: Name for this training iteration
            total_iterations: Override default iterations (for sliding window)
            sliding_window: If True, use shorter iterations for testing
        """
        print(f"   Training Grendel-GS ({iteration_name})...")

        '''
        # For sliding window testing, use shorter iterations
        if sliding_window and total_iterations is None:
            total_iterations = 60  # Quick test iterations
        '''

        # Use original COLMAP data instead of creating temporary files

        # Prepare model output path
        model_output = self.output_path / f"model_{iteration_name}"
        model_output.mkdir(exist_ok=True)

        # Build training command for all training (including progressive)
        import subprocess
        import sys

        # Get current working directory (should be Grendel-GS root)
        grendel_root = Path(__file__).parent.parent
        train_script = grendel_root / "train.py"

        if not train_script.exists():
            raise FileNotFoundError(f"train.py not found at {train_script}")

        # Check GPU count for multi-GPU training
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Prepare paths for the new directory structure
        sparse_dir = self.colmap_path / "sparse" / "0"  # Use original COLMAP sparse directory
        #print(f"🔍 Using sparse_dir: {sparse_dir}");    exit(1)
        images_dir = self.image_path  # Use the original image path
        if gpu_count > 1:
            # Multi-GPU training with torchrun
            cmd = [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={gpu_count}",
                str(train_script),
                "--dir_images", str(images_dir),
                "--dir_sparse", str(sparse_dir),
                "-m", str(model_output),
                "--iterations", str(total_iterations),
                "--sh_degree", str(self.sh_degree),
                "--bsz", "1"  # Batch size for multi-GPU
            ]
        else:
            # Single GPU or CPU training
            cmd = [
                sys.executable, str(train_script),
                "--dir_images", str(images_dir),
                "--dir_sparse", str(sparse_dir),
                "-m", str(model_output),
                "--iterations", str(total_iterations if total_iterations else self.iterations),
                "--sh_degree", str(self.sh_degree)
            ]

        # Progressive training checkpoint will be loaded via JSON file
        # No need to pass --start_checkpoint option
        
        if self.backend == "gsplat":
            cmd.extend(["--backend", "gsplat"])
        
        if self.deterministic:
            cmd.append("--deterministic")

        # Add use_chunk option if set
        if hasattr(self, 'use_chunk') and self.use_chunk:
            cmd.append("--use_chunk")

        # Add densification_interval if set
        if hasattr(self, 'densification_interval'):
            cmd.extend(["--densification_interval", str(self.densification_interval)])

        # Add densify_from_iter if set
        if hasattr(self, 'densify_from_iter'):
            cmd.extend(["--densify_from_iter", str(self.densify_from_iter)])

        # Add densify_memory_limit_percentage if set
        if hasattr(self, 'densify_memory_limit_percentage'):
            cmd.extend(["--densify_memory_limit_percentage", str(self.densify_memory_limit_percentage)])

        # Add show_memory_debug_info if set
        if hasattr(self, 'show_memory_debug_info') and self.show_memory_debug_info:
            cmd.append("--show_memory_debug_info")

        # Progressive training should not auto-save final iteration by default
        if not hasattr(self, 'auto_save_final_iteration') or not self.auto_save_final_iteration:
            cmd.append("--no_auto_save_final_iteration")

        # Add track_by_projection flag if set
        if hasattr(self, 'track_by_projection') and self.track_by_projection:
            cmd.append("--track_by_projection")

        # Add prune_by_visibility flag if set
        if hasattr(self, 'prune_by_visibility') and self.prune_by_visibility:
            cmd.append("--prune_by_visibility")

        # Add visibility_prune_margin if set
        if hasattr(self, 'visibility_prune_margin'):
            cmd.extend(["--visibility_prune_margin", str(self.visibility_prune_margin)])

        # Force checkpoint save at final iteration for progressive training continuity
        final_iter = total_iterations if total_iterations else self.iterations
        cmd.extend(["--checkpoint_iterations", str(final_iter)])
        print(f"   DEBUG: Forcing checkpoint save at iteration {final_iter}")

        # Progressive state will be saved after training completion
        # Add previous state file to training command (for progressive training)
        if iteration_name == "initial":
            # Initial window gets empty previous_state to indicate progressive training start
            cmd.extend(["--previous_state", ""])
        else:
            # For window 1+, load state from previous window
            prev_window_num = int(iteration_name.split("_")[-1]) - 1 if "window_" in iteration_name else 0
            if prev_window_num == 0:
                prev_model_name = "initial"
            else:
                prev_model_name = f"window_{prev_window_num:03d}"

            prev_state_file = self.output_path / f"model_{prev_model_name}" / "state.json"
            if prev_state_file.exists():
                cmd.extend(["--previous_state", str(prev_state_file)])
            else:
                error_msg = f"❌ FATAL ERROR: Previous state file not found: {prev_state_file}\n"
                error_msg += f"   Progressive training requires checkpoint from previous window.\n"
                error_msg += f"   Cannot continue without previous state."
                print(error_msg)
                raise FileNotFoundError(error_msg)

        # Add camera parameters based on window type
        if iteration_name == "initial":
            # Initial window - pass cameras to use for initialization
            init_cameras = list(dataset['images'].keys())
            cmd.extend(["--cams_init", ",".join(map(str, init_cameras))])
        else:
            # Progressive window - pass previous window cameras and changes
            prev_iteration = self.get_previous_iteration_name(iteration_name)
            if prev_iteration:
                # Get previous window cameras from state
                # For now, we'll extract from the current sliding window logic
                # This should be improved to read from previous state.json
                prev_cameras = getattr(self, 'prev_window_cameras', [])
                if prev_cameras:
                    cmd.extend(["--cams_prev", ",".join(map(str, prev_cameras))])
                else:
                    utils.print_rank_0("⚠️  Warning: No previous cameras found for progressive window")

                # Add all processed cameras if use_all_processed_cameras flag is set
                if hasattr(self, 'use_all_processed_cameras') and self.use_all_processed_cameras:
                    all_proc_cams = list(self.all_processed_cameras)
                    cmd.extend(["--cams_all_processed", ",".join(map(str, all_proc_cams))])
                    print(f"   Passing all_processed_cameras ({len(all_proc_cams)} cameras) to train.py")

                # Add cameras to delete and add
                if cam_id_2_delete is not None:
                    cmd.extend(["--cams_2_delete", str(cam_id_2_delete)])
                if new_camera_id is not None:
                    cmd.extend(["--cams_2_add", str(new_camera_id)])


        if self.debug:
            print(f"   Training command: {' '.join(cmd)}")
        
        try:
            # Run training
            result = subprocess.run(cmd, 
                                  cwd=str(grendel_root),
                                  capture_output=not self.debug,
                                  text=True,
                                  timeout=3600  # 1 hour timeout
                                  )
            
            if result.returncode == 0:
                print(f"✓ Training completed successfully for {iteration_name}")
                if self.debug and result.stdout:
                    print(f"Training output: {result.stdout}")

                # Store checkpoint path for next window BEFORE saving state
                final_iter = total_iterations if total_iterations else self.iterations

                # Update trained gaussians after successful training
                self._update_trained_gaussians_after_training(model_output, iteration_name)

                # Update processed points after successful training (this calls _save_progressive_state)
                self._update_processed_points_after_training(dataset, iteration_name)

                # Clear GPU memory cache to prevent memory accumulation across windows
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"🧹 Cleared GPU memory cache after {iteration_name}")
            else:
                error_msg = f"Training failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nError: {result.stderr}"
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Training timed out for {iteration_name}")
        except Exception as e:
            raise RuntimeError(f"Training failed for {iteration_name}: {e}")
        
        #exit(1)
        # Load resulting Gaussian set V
        # self.V = load_gaussians(...)
    '''        
    def _save_colmap_format(self, dataset: Dict, path: Path):
        """Save dataset in COLMAP format"""
        print(f"💾 Saving COLMAP format to: {path}")
        print(f"💾 Dataset cameras: {len(dataset['cameras'])}")
        print(f"💾 Dataset images: {len(dataset['images'])}")
        print(f"💾 Dataset points3D: {len(dataset['points3D'])}")

        sparse_path = path / "sparse" / "0"
        sparse_path.mkdir(parents=True, exist_ok=True)

        # Save cameras.txt
        with open(sparse_path / "cameras.txt", 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            for cam_id, cam_data in dataset['cameras'].items():
                f.write(f"{cam_id} {cam_data['model']} {cam_data['width']} {cam_data['height']}")

                # Use raw_params if available, otherwise reconstruct from params dict
                if 'raw_params' in cam_data:
                    for param in cam_data['raw_params']:
                        f.write(f" {param}")
                else:
                    # Params is a dictionary, extract values based on model type
                    params = cam_data['params']
                    if cam_data['model'] == 'PINHOLE':
                        f.write(f" {params['fx']} {params['fy']} {params['cx']} {params['cy']}")
                    elif cam_data['model'] == 'RADIAL':
                        # RADIAL: fx, cx, cy, k1, k2, ...
                        f.write(f" {params['fx']} {params['cx']} {params['cy']}")
                        if 'distortion' in params and params['distortion']:
                            for d in params['distortion']:
                                f.write(f" {d}")
                    else:
                        # For other models, output values (not keys)
                        for key in sorted(params.keys()):
                            f.write(f" {params[key]}")

                f.write("\n")

        # Save images.txt
        with open(sparse_path / "images.txt", 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for img_id, img_data in dataset['images'].items():
                f.write(f"{img_id} {img_data['qw']} {img_data['qx']} {img_data['qy']} {img_data['qz']}")
                f.write(f" {img_data['tx']} {img_data['ty']} {img_data['tz']}")
                f.write(f" {img_data['camera_id']} {img_data['name']}\n")
                f.write("\n")  # Empty line for POINTS2D

        # Save points3D.txt with track information
        with open(sparse_path / "points3D.txt", 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            for pt_id, pt_data in dataset['points3D'].items():
                f.write(f"{pt_id} {pt_data['xyz'][0]} {pt_data['xyz'][1]} {pt_data['xyz'][2]}")
                f.write(f" {pt_data['rgb'][0]} {pt_data['rgb'][1]} {pt_data['rgb'][2]} {pt_data['error']}")

                # Add track information if available
                if 'track' in pt_data and pt_data['track']:
                    for img_id, point2d_idx in pt_data['track']:
                        f.write(f" {img_id} {point2d_idx}")

                f.write("\n")

        # Remove existing PLY file so Scene will use our new points3D.txt
        ply_path = sparse_path / "points3D.ply"
        if ply_path.exists():
            ply_path.unlink()
            print(f"🗑️ Removed existing PLY file: {ply_path}")

        print(f"✓ COLMAP format saved to: {sparse_path}")
        print(f"  - cameras.txt: {len(dataset['cameras'])} cameras")
        print(f"  - images.txt: {len(dataset['images'])} images")
        print(f"  - points3D.txt: {len(dataset['points3D'])} points")
    '''

    def _calculate_window_mean_3d(self, camera_ids):
        """Calculate mean position of camera window in 3D (x, y, z)"""
        if not camera_ids:
            return None

        positions = []
        for cam_id in camera_ids:
            if cam_id in self.images:
                pos = self.images[cam_id]['position']
                positions.append(pos)

        if positions:
            positions = np.array(positions)
            mean_pos = np.mean(positions, axis=0).tolist()
            print(f"   📍 Calculated window mean: {mean_pos}")
            return mean_pos
        return None

    def _calculate_window_mean_2d(self, camera_ids):
        """Calculate mean position of camera window in 2D (x, y)"""
        if not camera_ids:
            return None

        positions = []
        for cam_id in camera_ids:
            if cam_id in self.images:
                pos = self.images[cam_id]['position']
                positions.append([pos[0], pos[1]])  # Only x, y

        if positions:
            positions = np.array(positions)
            mean_pos = np.mean(positions, axis=0)
            return mean_pos
        return None

    def _calculate_window_mean(self, camera_ids):
        """Deprecated: Use _calculate_window_mean_2d or _calculate_window_mean_3d"""
        return self._calculate_window_mean_3d(camera_ids)

    def _find_farthest_camera_from_new(self, window_cameras, newly_added_camera):
        """Find camera in window that is farthest from newly added camera"""
        if newly_added_camera not in self.images:
            print(f"⚠️  Newly added camera {newly_added_camera} not found in images")
            return window_cameras[0]  # Fallback to first camera

        new_camera_pos = np.array(self.images[newly_added_camera]['position'])

        farthest_camera = None
        max_distance = -1

        for cam_id in window_cameras:
            if cam_id == newly_added_camera:
                continue  # Skip the newly added camera itself

            if cam_id in self.images:
                cam_pos = np.array(self.images[cam_id]['position'])
                distance = np.linalg.norm(cam_pos - new_camera_pos)

                if distance > max_distance:
                    max_distance = distance
                    farthest_camera = cam_id

        if farthest_camera is None:
            print(f"⚠️  Could not find farthest camera, using first camera")
            farthest_camera = window_cameras[0]
        else:
            print(f"🎯 Camera {farthest_camera} is farthest from new camera {newly_added_camera} (distance: {max_distance:.2f})")

        return farthest_camera

    def _find_closest_camera_to_reference(self, reference_mean):
        """Find closest unprocessed camera to reference mean position"""
        best_camera = None
        best_distance = float('inf')

        for cam_id in self.unprocessed_cameras:
            if cam_id in self.images:
                pos = self.images[cam_id]['position']
                distance = np.linalg.norm(pos - reference_mean)

                if distance < best_distance:
                    best_distance = distance
                    best_camera = cam_id

        if best_camera is not None:
            print(f"🎯 Selected camera {best_camera} closest to reference (distance: {best_distance:.2f})")
        else:
            print(f"⚠️  No unprocessed cameras available")

        return best_camera

    def _remove_camera_from_window(self, window_cameras, newly_added_camera):
        """
        Remove camera that is farthest from newly added camera

        Args:
            window_cameras: Current list of cameras in window
            newly_added_camera: ID of newly added camera

        Returns:
            int: ID of removed camera
        """
        # Find camera farthest from newly added camera
        farthest_camera = self._find_farthest_camera_from_new(window_cameras, newly_added_camera)

        # Remove from window
        window_cameras.remove(farthest_camera)

        # Update global state
        self.processed_cameras.add(farthest_camera)

        print(f"🗑️  Removed camera {farthest_camera} from window")
        print(f"📊 Processed cameras: {len(self.processed_cameras)}")
        print(f"📊 Unprocessed cameras: {len(self.unprocessed_cameras)}")

        return farthest_camera

    # DELETED: _remove_gaussians_outside_camera_view
    # This function was never called and has been replaced by
    # _remove_gaussians_only_visible_to_removed_camera in train_internal.py
    # which properly handles gaussian removal during progressive training

    def _add_camera_to_window(self, window_cameras, window_num):
        """Select and add new camera to window after cleaning up invisible gaussians"""
        # STEP 1: Select next camera based on previous window mean
        previous_mean = None
        if window_num >= 1:
            # Load previous window's progressive state to get mean
            prev_iteration_name = self.get_previous_iteration_name(f"window_{window_num:03d}")
            print(f'prev_iteration_name : {prev_iteration_name}')
            if prev_iteration_name:
                prev_model_path = self.output_path / f"model_{prev_iteration_name}"
                prev_state_file = prev_model_path / "state.json"
                if prev_state_file.exists():
                    try:
                        with open(prev_state_file, 'r') as f:
                            prev_state = json.load(f)
                        previous_mean = prev_state.get('current_window_mean')
                        if previous_mean:
                            print(f"📍 Using previous window mean: {previous_mean}")

                        # Load checkpoint path from previous state
                    except Exception as e:
                        print(f"⚠️  Could not load previous state: {e}")

        # Use previous mean to find best camera
        reference_mean = np.array(previous_mean)
        best_camera = self._find_closest_camera_to_reference(reference_mean)

        if best_camera is None:
            return None, window_cameras  # No more cameras available

        # STEP 2: Store new camera info for training process
        # (Gaussian removal will be done inside training after checkpoint loading)

        # STEP 3: Add camera to window
        window_cameras.append(best_camera)

        # STEP 4: Update global state
        self.unprocessed_cameras.remove(best_camera)  # Remove from unprocessed
        print(f"✅ Added camera {best_camera} to window")

        return best_camera, window_cameras

    def _save_progressive_state(self, iteration_name: str, model_output: Path):
        #print(f'iteration_name : {iteration_name}, model_output : {model_output}'); exit(1)
        """Save progressive training state to JSON for subprocess communication"""
        # Use previously calculated window mean
        current_window_mean = getattr(self, 'current_window_mean', None)
        window_number = int(iteration_name.split("_")[-1]) if "window_" in iteration_name else 0

        # Get current window's checkpoint directory after training completion
        checkpoint_dir = None
        gpu_metrics = {}

        # Try to load checkpoint directory and GPU metrics from temporary file created by train_internal.py
        temp_checkpoint_file = model_output / f"window_{window_number}_checkpoint.json"
        if temp_checkpoint_file.exists():
            try:
                with open(temp_checkpoint_file, 'r') as f:
                    checkpoint_info = json.load(f)
                    checkpoint_dir = checkpoint_info.get('checkpoint_dir', None)
                    gpu_metrics = checkpoint_info.get('gpu_metrics', {})
                print(f"📂 Collected checkpoint directory: {checkpoint_dir}")
                if gpu_metrics:
                    print(f"📊 Collected GPU metrics: {gpu_metrics.get('peak_memory_gb', 0):.2f} GB / {gpu_metrics.get('total_memory_gb', 0):.2f} GB ({gpu_metrics.get('peak_usage_ratio', 0)*100:.1f}%)")
                # Remove temporary file since we'll store in state.json
                temp_checkpoint_file.unlink()
                print(f"🗑️  Removed temporary checkpoint file: {temp_checkpoint_file}")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint directory: {e}")

        state = {
            "iteration_name": iteration_name,
            "window_number": window_number,
            "processed_cameras": list(self.processed_cameras),
            "all_processed_cameras": list(self.all_processed_cameras),
            "unprocessed_points": list(self.unprocessed_points),
            # "processed_points": removed - only track processed_gaussians
            "trained_gaussians": list(self.trained_gaussians),
            "current_window_mean": current_window_mean,
            "checkpoint_dir": checkpoint_dir,
            "gpu_metrics": gpu_metrics,
            "total_cameras": len(self.images),
            "total_points": len(self.points3D),
            "camera_removal_margin": self.camera_removal_margin,
            "debug": self.debug
        }

        state_file = model_output / "state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"💾 Progressive state saved to: {state_file}")
        print(f"   iteration_name: {state['iteration_name']}")
        print(f"   window_number: {state['window_number']}")
        print(f"   processed_cameras: {state['processed_cameras']}")
        print(f"   unprocessed_points: {len(state['unprocessed_points'])} points")
        # print(f"   processed_points: removed - only track processed_gaussians")
        print(f"   trained_gaussians: {len(state['trained_gaussians'])} files")
        print(f"   current_window_mean: {state['current_window_mean']}")
        print(f"   checkpoint_dir: {state['checkpoint_dir']}")
        print(f"   total_cameras: {state['total_cameras']}")
        print(f"   total_points: {state['total_points']}")
        print(f"   camera_removal_margin: {state.get('camera_removal_margin', state.get('gpu_memory_threshold', 0.1))}")
        print(f"   debug: {state['debug']}")
        #exit(1)
    
    def _load_progressive_state(self, state_file: Path) -> Dict:
        """Load progressive training state from JSON"""
        if not state_file.exists():
            return {}

        with open(state_file, 'r') as f:
            state = json.load(f)

        print(f"📖 Progressive state loaded from: {state_file}")
        return state

    def _update_trained_gaussians_after_training(self, model_output: Path, iteration_name: str):
        """Update trained_gaussians set after successful training completion"""
        try:
            # Look for saved gaussians (PLY files, checkpoints, etc.)
            gaussian_files = []

            # Check for final PLY file
            final_ply = model_output / "point_cloud" / "iteration_final" / "point_cloud.ply"
            if final_ply.exists():
                gaussian_files.append(str(final_ply))

            # Check for checkpoint gaussians
            checkpoint_dir = model_output / "checkpoints"
            if checkpoint_dir.exists():
                for ckpt_dir in checkpoint_dir.iterdir():
                    if ckpt_dir.is_dir() and ckpt_dir.name.isdigit():
                        ply_file = ckpt_dir / "point_cloud.ply"
                        if ply_file.exists():
                            gaussian_files.append(str(ply_file))

            # Add to trained gaussians set (using file paths as identifiers)
            for gaussian_file in gaussian_files:
                self.trained_gaussians.add(gaussian_file)

            print(f"📊 Updated trained_gaussians after {iteration_name}:")
            print(f"   Added {len(gaussian_files)} gaussian files")
            print(f"   Total trained gaussians: {len(self.trained_gaussians)}")

        except Exception as e:
            print(f"Warning: Could not update trained_gaussians: {e}")

    def _update_processed_points_after_training(self, dataset: Dict, iteration_name: str):
        """Update unprocessed points after successful training completion"""
        try:
            # Get points from the current dataset that were just trained
            current_points = set(dataset['points3D'].keys())

            # Move points from unprocessed (no longer need to track as processed)
            points_to_remove = current_points & self.unprocessed_points
            self.unprocessed_points -= points_to_remove

            print(f"📊 Updated point tracking after {iteration_name}:")
            print(f"   Points trained in this window: {len(current_points)}")
            print(f"   Points removed from unprocessed: {len(points_to_remove)}")
            print(f"   Total unprocessed points remaining: {len(self.unprocessed_points)}")

            # Save updated state after training completion
            print(f"💾 Saving updated progressive state after training...")
            model_output = self.output_path / f"model_{iteration_name}"
            self._save_progressive_state(iteration_name, model_output)

        except Exception as e:
            print(f"Warning: Could not update point tracking: {e}")

    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage percentage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            usage = allocated / total
            if self.debug:
                print(f"GPU Memory: {allocated:.2f}/{total:.2f} GB ({usage*100:.1f}%)")
            return usage
        return 0.0
    
    def find_next_closest_camera(self) -> Optional[int]:
        """Find next closest camera to median that's not in P"""
        available = set(self.images.keys()) - self.current_window_cameras
        if not available:
            return None
        
        positions = np.array([img['position'] for img in self.images.values()])
        median_pos = np.median(positions, axis=0)  # 3D median

        best_id = None
        best_dist = float('inf')

        for img_id in available:
            camera_pos = self.images[img_id]['position']  # 3D position
            dist = np.linalg.norm(camera_pos - median_pos)  # 3D distance
            if dist < best_dist:
                best_dist = dist
                best_id = img_id
        
        return best_id
    
    def expand_until_memory_limit(self):
        """
        Step 10: Expand camera set until GPU memory limit
        """
        print("Expanding camera set until GPU memory limit...")
        
        while self.get_gpu_memory_usage() < (1.0 - self.camera_removal_margin):
            # Step 10.1: Find next closest camera
            j = self.find_next_closest_camera()
            if j is None:
                print("No more cameras to add")
                break
            
            self.current_window_cameras.add(j)
            print(f"Adding camera {j} (total: {len(self.current_window_cameras)})")
            
            # Step 10.2: Find new points R_j not in H
            W_j = self.points_in_view[j]  # List of point IDs
            R_j = []
            for point_id in W_j:
                # Get coordinate from point ID and check if it's in H
                point_coord = self.points3D[point_id]['xyz'][:2]
                if tuple(point_coord) not in self.H:
                    R_j.append(point_coord)  # Store coordinates for processing
                    self.H.add(tuple(point_coord))
            
            if R_j:
                print(f"Found {len(R_j)} new points for camera {j}")
                
                # Create initial Gaussians L from R_j
                # L = create_gaussians_from_points(R_j)
                
                # Step 10.3: Add L to V and retrain
                # self.V = self.V + L
                # self.train_grendel_gs(...)
            
            # Check memory again
            if self.get_gpu_memory_usage() >= (1.0 - self.camera_removal_margin):
                print(f"GPU memory threshold reached ({(1.0 - self.camera_removal_margin)*100}%)")
                break
    
    def swap_gaussians_for_remaining(self):
        """
        Steps 11-12: Process remaining cameras with Gaussian swapping
        Using sliding window approach with checkpoint loading
        """
        print("Processing remaining cameras with Gaussian swapping...")

        # Step 11: Create empty PLY file for removed Gaussians
        removed_gaussians_path = self.output_path / "removed_gaussians.ply"

        total_cameras = len(self.images)
        
        # Step 12: Process remaining cameras
        while len(self.current_window_cameras) < total_cameras:
            # Step 12.1: Find next closest camera
            j = self.find_next_closest_camera()
            if j is None:
                break
            
            self.current_window_cameras.add(j)
            print(f"Processing camera {j} ({len(self.current_window_cameras)}/{total_cameras})")
            
            # Step 12.2: Find new points R_j
            W_j = self.points_in_view[j]  # List of point IDs
            R_j = []
            for point_id in W_j:
                # Get coordinate from point ID and check if it's in H
                point_coord = self.points3D[point_id]['xyz'][:2]
                if tuple(point_coord) not in self.H:
                    R_j.append(point_coord)  # Store coordinates for processing
                    self.H.add(tuple(point_coord))
            
            if not R_j:
                print(f"No new points for camera {j}")
                continue
            
            print(f"Found {len(R_j)} new points")
            
            # Step 12.3: Compute centroid of R_j
            T_j = np.mean(R_j, axis=0)
            print(f"Centroid of new points: {T_j}")
            
            # Step 12.4: Find furthest Gaussians from T_j
            # E = find_furthest_gaussians(self.V, T_j, len(R_j))
            
            # Step 12.5: Remove E from V, add to I
            # self.V = self.V - E
            # self.I.extend(E)
            
            # Step 12.6: Add L to V and retrain
            # L = create_gaussians_from_points(R_j)
            # self.V = self.V + L
            # self.train_grendel_gs(...)
            
        # Save removed Gaussians
        if self.I:
            print(f"Saving {len(self.I)} removed Gaussians to {removed_gaussians_path}")
            # save_gaussians_to_ply(self.I, removed_gaussians_path)
    
    
    def create_sliding_dataset(self, camera_ids: List[int]) -> Dict:
        print(f"   📊 Create dataset for sliding window cameras: {camera_ids}")
        #print(f"📊 Total 3D points in memory: {len(self.points3D)}")
        #print(f'📊 Camera visible points available: {list(self.points_in_view.keys())}')

        # Include points visible from selected cameras
        included_points = {}
        total_footprint_points = 0

        for cam_id in camera_ids:
            if cam_id in self.points_in_view:
                visible_points = self.points_in_view[cam_id]
                print(f"   📊 Camera {cam_id} visible points: {len(visible_points)}")
                total_footprint_points += len(visible_points)

                valid_points_in_view = 0
                for pt_idx in visible_points:
                    if pt_idx in self.points3D:
                        included_points[pt_idx] = self.points3D[pt_idx]
                        valid_points_in_view += 1

                print(f"   📊 Camera {cam_id}: {valid_points_in_view} valid points added to dataset")
            else:
                print(f"   ⚠️  Camera {cam_id} has no footprint data")

        print(f"   📊 Total footprint points from all cameras: {total_footprint_points}")
        print(f"   📊 Unique points included in dataset: {len(included_points)}")

        final_points = included_points if included_points else self.points3D
        print(f"   📊 Final dataset points count: {len(final_points)}")

        return {
            'cameras': self.cameras,
            'images': {img_id: self.images[img_id] for img_id in camera_ids},
            'points3D': final_points
        }

    def _visualize_steps_1_7(self, camera_positions, window_cam_ids, P, Z):
        """
        Visualize Steps 1-7: Initial window selection, mean, and footprint union

        Args:
            camera_positions: Dict {cam_id: [x, y]}
            window_cam_ids: List of camera IDs in initial window
            P: Window mean position
            Z: Footprint union polygon
        """
        print("\n📊 Creating Steps 1-7 visualization...")

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
            from scipy.spatial import ConvexHull as SciPyConvexHull

            fig, ax = plt.subplots(figsize=(14, 12))

            # Convert camera_positions dict to arrays for plotting
            all_cam_ids = list(camera_positions.keys())
            all_positions = np.array([camera_positions[cam_id] for cam_id in all_cam_ids])

            # Plot all camera footprints first (as background)
            print("  Plotting footprints...")
            footprint_count = 0
            for i, cam_id in enumerate(all_cam_ids):
                if cam_id in self.camera_footprints:
                    footprint = self.camera_footprints[cam_id]
                    if isinstance(footprint, np.ndarray) and footprint.shape[0] == 4:
                        label = 'Camera Footprints' if i == 0 else None
                        poly = MplPolygon(footprint, closed=True,
                                        edgecolor='gray', facecolor='lightgray',
                                        alpha=0.2, linewidth=0.5, zorder=1,
                                        label=label)
                        ax.add_patch(poly)
                        footprint_count += 1
            print(f"    Plotted {footprint_count} footprints")

            # Plot selected window footprints union (reuse Z)
            if Z is not None:
                if Z.geom_type == 'Polygon':
                    coords = np.array(Z.exterior.coords)
                    poly = MplPolygon(coords, closed=True,
                                    edgecolor='none', facecolor='red',
                                    alpha=0.15, linewidth=0, zorder=2,
                                    label='Selected Window Footprint Union')
                    ax.add_patch(poly)
                elif Z.geom_type == 'MultiPolygon':
                    for geom in Z.geoms:
                        coords = np.array(geom.exterior.coords)
                        poly = MplPolygon(coords, closed=True,
                                        edgecolor='none', facecolor='red',
                                        alpha=0.15, linewidth=0, zorder=2)
                        ax.add_patch(poly)
                    ax.plot([], [], 'r-', linewidth=0, alpha=0, label='Selected Window Footprint Union')

            # Plot all cameras
            ax.scatter(all_positions[:, 0], all_positions[:, 1],
                      c='dimgray', s=100, alpha=0.7, label='All Cameras', zorder=3)

            # Annotate all camera IDs
            for i, cam_id in enumerate(all_cam_ids):
                pos = all_positions[i]
                ax.annotate(f'{cam_id}',
                           (pos[0], pos[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.6)

            # Plot convex hull
            if len(all_positions) >= 3:
                try:
                    hull = SciPyConvexHull(all_positions)
                    hull_points = all_positions[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax.plot(hull_points[:, 0], hull_points[:, 1],
                           'b--', linewidth=1, label='Convex Hull', zorder=4)
                    ax.fill(hull_points[:, 0], hull_points[:, 1],
                           'blue', alpha=0.05, zorder=0)

                    ax.scatter(all_positions[hull.vertices, 0],
                              all_positions[hull.vertices, 1],
                              c='blue', s=10, marker='s',
                              label='Convex Hull Vertices', zorder=5)
                except Exception as e:
                    print(f"  Warning: Could not plot convex hull: {e}")

            # Plot selected cameras (initial window)
            selected_pos = np.array([camera_positions[cam_id] for cam_id in window_cam_ids])
            ax.scatter(selected_pos[:, 0], selected_pos[:, 1],
                      c='red', s=300, marker='*',
                      label=f'Selected Window (n={len(window_cam_ids)})',
                      zorder=6, edgecolors='darkred', linewidths=2)

            # Annotate selected camera IDs in bold
            for cam_id in window_cam_ids:
                pos = camera_positions[cam_id]
                ax.annotate(f'CAM {cam_id}',
                           (pos[0], pos[1]),
                           xytext=(10, -15), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            # Plot mean F
            f_mode = getattr(self, 'f_mode', 'global')
            if f_mode == 'remaining' and hasattr(self, 'current_F_remaining'):
                # Use stored F from remaining cameras
                F = self.current_F_remaining
                A_count = len(self.window_selector.A)
                ax.scatter(F[0], F[1],
                          c='green', s=200, marker='x', linewidths=3,
                          label=f'Remaining Mean (F, n={A_count})', zorder=7)
                ax.annotate(f'F (Remaining Mean, n={A_count})',
                           (F[0], F[1]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, color='green', fontweight='bold')
            elif f_mode == 'global':
                # Use pre-calculated global mean
                F = self.window_selector.F
                if F is not None:
                    ax.scatter(F[0], F[1],
                              c='green', s=200, marker='x', linewidths=3,
                              label='Global Mean (F)', zorder=7)
                    ax.annotate('F (Global Mean)',
                               (F[0], F[1]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, color='green', fontweight='bold')

            # Plot window mean P
            ax.scatter(P[0], P[1],
                      c='orange', s=200, marker='+', linewidths=3,
                      label='Window Mean (P)', zorder=7)

            # Calculate statistics
            selected_mean = np.mean(selected_pos, axis=0)
            selected_std = np.std(selected_pos, axis=0).mean()

            ax.set_xlabel('X Position (m)', fontsize=12)
            ax.set_ylabel('Y Position (m)', fontsize=12)
            ax.set_title(f'Steps 1-7: Initial Window Selection + Mean + Union\n'
                        f'Dataset: {self.colmap_path}\n'
                        f'Selected: {window_cam_ids}, Std: {selected_std:.3f}',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Save figure
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "steps1-7_initial_window.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {output_file}")

            # Also save high-res version
            output_file_hires = output_dir / "steps1-7_initial_window_hires.png"
            plt.savefig(output_file_hires, dpi=300, bbox_inches='tight')
            print(f"✅ High-res version saved to: {output_file_hires}")

            plt.close()

        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_iteration(self, camera_positions, D_cam_ids, P, Z, R, iteration_count, Q, G_cam_id=None, current_removed_region=None, E_cam_id=None, current_added_region=None):
        """
        Visualize each iteration of Step 16.

        Args:
            camera_positions: Dict of {cam_id: np.array([x, y])}
            D_cam_ids: Current window camera IDs
            P: Window mean position
            Z: Footprint union (Shapely Polygon/MultiPolygon)
            R: Direction vector
            iteration_count: Current iteration number
            Q: Window number
            G_cam_id: ID of camera being removed (optional)
            current_removed_region: Current removed region (G - Z) (Shapely Polygon/MultiPolygon, optional)
            E_cam_id: ID of camera being added (optional)
            current_added_region: Current added region (E - Z) (Shapely Polygon/MultiPolygon, optional)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
            from shapely.geometry import Polygon, MultiPolygon
            import numpy as np

            print(f"\n📊 Creating visualization for iteration {iteration_count} (Q={Q})...")

            fig, ax = plt.subplots(figsize=(16, 12))

            # Convert dict to arrays
            all_cam_ids = list(camera_positions.keys())
            all_positions = np.array([camera_positions[cam_id] for cam_id in all_cam_ids])

            # Plot individual camera footprints for all cameras
            # First plot non-D cameras (background), then D cameras (foreground)
            if hasattr(self, 'camera_footprints'):
                from matplotlib.patches import Polygon as MplPolygon

                # First pass: plot non-D camera footprints
                for cam_id in all_cam_ids:
                    if cam_id not in D_cam_ids and cam_id in self.camera_footprints:
                        footprint_coords = self.camera_footprints[cam_id]
                        if len(footprint_coords) >= 3:
                            poly = MplPolygon(footprint_coords, closed=True,
                                            edgecolor='none', facecolor='lightgray',
                                            alpha=0.2, linewidth=0, zorder=1)
                            ax.add_patch(poly)

                # Second pass: plot D camera footprints (on top)
                for cam_id in D_cam_ids:
                    if cam_id in self.camera_footprints:
                        footprint_coords = self.camera_footprints[cam_id]
                        if len(footprint_coords) >= 3:
                            poly = MplPolygon(footprint_coords, closed=True,
                                            edgecolor='none', facecolor='red',
                                            alpha=0.15, linewidth=0, zorder=1)
                            ax.add_patch(poly)

            # Plot footprint union Z if available
            if Z is not None:
                try:
                    if Z.geom_type == 'Polygon':
                        polygons = [Z]
                    elif Z.geom_type == 'MultiPolygon':
                        polygons = list(Z.geoms)
                    else:
                        polygons = []

                    for poly in polygons:
                        if not poly.is_empty:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, 'lightblue', alpha=0.2, zorder=2)
                            ax.plot(x, y, 'red', linewidth=2, zorder=3)
                            if poly == polygons[0]:
                                ax.plot([], [], 'red', linewidth=2, label='Footprint Union (Z)')
                except Exception as e:
                    print(f"  Warning: Could not plot footprint union: {e}")

            # Plot cumulative removed regions
            if self.cumulative_removed_regions is not None:
                try:
                    if self.cumulative_removed_regions.geom_type == 'Polygon':
                        cumulative_polys = [self.cumulative_removed_regions]
                    elif self.cumulative_removed_regions.geom_type == 'MultiPolygon':
                        cumulative_polys = list(self.cumulative_removed_regions.geoms)
                    else:
                        cumulative_polys = []

                    for poly in cumulative_polys:
                        if not poly.is_empty:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, 'yellowgreen', alpha=0.15, zorder=2)
                            ax.plot(x, y, 'yellowgreen', linewidth=2, zorder=2)
                            if poly == cumulative_polys[0]:
                                ax.plot([], [], 'yellowgreen', linewidth=0, alpha=0, label='Cumulative Removed Regions')
                except Exception as e:
                    print(f"  Warning: Could not plot cumulative removed regions: {e}")

            # Plot current removed region (G - Z)
            if current_removed_region is not None:
                try:
                    if current_removed_region.geom_type == 'Polygon':
                        current_polys = [current_removed_region]
                    elif current_removed_region.geom_type == 'MultiPolygon':
                        current_polys = list(current_removed_region.geoms)
                    else:
                        current_polys = []

                    for poly in current_polys:
                        if not poly.is_empty:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, 'orange', alpha=0.3, zorder=3)
                            if poly == current_polys[0]:
                                ax.plot([], [], 'orange', linewidth=0, alpha=0, label=f'Current Removed Region (Cam {G_cam_id})')
                except Exception as e:
                    print(f"  Warning: Could not plot current removed region: {e}")

            # Plot current added region (E - Z)
            if current_added_region is not None:
                try:
                    if current_added_region.geom_type == 'Polygon':
                        current_added_polys = [current_added_region]
                    elif current_added_region.geom_type == 'MultiPolygon':
                        current_added_polys = list(current_added_region.geoms)
                    else:
                        current_added_polys = []

                    for poly in current_added_polys:
                        if not poly.is_empty:
                            x, y = poly.exterior.xy
                            ax.fill(x, y, 'cyan', alpha=0.3, zorder=3)
                            if poly == current_added_polys[0]:
                                ax.plot([], [], 'cyan', linewidth=0, alpha=0, label=f'Current Added Region (Cam {E_cam_id})')
                except Exception as e:
                    print(f"  Warning: Could not plot current added region: {e}")

            # Plot ALL cameras (not just A)
            ax.scatter(all_positions[:, 0], all_positions[:, 1],
                      c='dimgray', s=100, alpha=0.7,
                      label=f'All Cameras (n={len(all_cam_ids)})', zorder=3)

            # Annotate all camera IDs
            for cam_id in all_cam_ids:
                pos = camera_positions[cam_id]
                ax.annotate(f'{cam_id}',
                           (pos[0], pos[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12, alpha=0.6)

            # Plot cameras in A (remaining) with different marker
            A_cam_ids = list(self.window_selector.A)
            if len(A_cam_ids) > 0:
                A_positions = np.array([camera_positions[cam_id] for cam_id in A_cam_ids])
                ax.scatter(A_positions[:, 0], A_positions[:, 1],
                          c='blue', s=100, alpha=0.3, marker='o',
                          label=f'Remaining Cameras (A, n={len(A_cam_ids)})', zorder=5)

            # Plot current window D cameras
            D_positions = np.array([camera_positions[cam_id] for cam_id in D_cam_ids])
            ax.scatter(D_positions[:, 0], D_positions[:, 1],
                      c='red', s=300, marker='*',
                      label=f'Current Window D (n={len(D_cam_ids)})',
                      zorder=6, edgecolors='darkred', linewidths=2)

            # Plot convex hull of window D cameras
            if len(D_positions) == 2:
                # Two cameras: draw a line between them
                ax.plot(D_positions[:, 0], D_positions[:, 1],
                       'c--', linewidth=2, alpha=0.6,
                       label='Window D Extent', zorder=5)
            elif len(D_positions) >= 3:
                # Three or more cameras: draw convex hull
                from scipy.spatial import ConvexHull
                hull = ConvexHull(D_positions)
                # Plot hull polygon
                for simplex in hull.simplices:
                    ax.plot(D_positions[simplex, 0], D_positions[simplex, 1],
                           'c--', linewidth=2, alpha=0.6, zorder=5)
                # Fill hull area
                hull_points = D_positions[hull.vertices]
                ax.fill(hull_points[:, 0], hull_points[:, 1],
                       color='cyan', alpha=0.15, label='Window D Convex Hull', zorder=4)

            # Annotate D camera IDs
            for cam_id in D_cam_ids:
                pos = camera_positions[cam_id]
                ax.annotate(f'{cam_id}',
                           (pos[0], pos[1]),
                           xytext=(10, -15), textcoords='offset points',
                           fontsize=9, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            # Plot mean F
            f_mode = getattr(self, 'f_mode', 'global')
            if f_mode == 'remaining' and hasattr(self, 'current_F_remaining'):
                # Use stored F from remaining cameras
                F = self.current_F_remaining
                A_count = len(self.window_selector.A)
                ax.scatter(F[0], F[1],
                          c='green', s=200, marker='x', linewidths=3,
                          label=f'Remaining Mean (F, n={A_count})', zorder=7)
            elif f_mode == 'global':
                # Use pre-calculated global mean
                F = self.window_selector.F
                if F is not None:
                    ax.scatter(F[0], F[1],
                              c='green', s=200, marker='x', linewidths=3,
                              label='Global Mean (F)', zorder=7)

            # Plot window mean P
            ax.scatter(P[0], P[1],
                      c='orange', s=200, marker='+', linewidths=3,
                      label='Window Mean (P)', zorder=7)

            # Plot window center trajectory (for balanced_smooth_trajectory strategy)
            if hasattr(self, 'balanced_window_center_history') and len(self.balanced_window_center_history) > 1:
                window_centers = np.array(self.balanced_window_center_history)
                ax.plot(window_centers[:, 0], window_centers[:, 1],
                       c='green', linewidth=3, alpha=0.7, linestyle='-',
                       marker='o', markersize=6, markerfacecolor='lightgreen',
                       markeredgecolor='darkgreen', markeredgewidth=1.5,
                       label='Window Center Trajectory', zorder=8)

            # Plot added camera trajectory (for balanced_smooth_trajectory strategy)
            if hasattr(self, 'balanced_camera_history') and len(self.balanced_camera_history) >= 1:
                camera_ids = self.balanced_camera_history
                camera_trajectory = np.array([camera_positions[cam_id] for cam_id in camera_ids])
                if len(camera_trajectory) == 1:
                    # Single camera: plot as point only
                    ax.scatter(camera_trajectory[0, 0], camera_trajectory[0, 1],
                             c='red', s=100, marker='o', alpha=0.6,
                             edgecolors='darkred', linewidths=2,
                             label='Added Camera (1st)', zorder=8)
                else:
                    # Multiple cameras: plot as trajectory
                    ax.plot(camera_trajectory[:, 0], camera_trajectory[:, 1],
                           c='red', linewidth=2, alpha=0.6, linestyle='--',
                           marker='o', markersize=5, markerfacecolor='pink',
                           markeredgecolor='darkred', markeredgewidth=1,
                           label='Added Camera Trajectory', zorder=8)

            # Plot direction vector R
            if R is not None:
                ax.arrow(P[0], P[1], R[0], R[1],
                        head_width=12, head_length=12, fc='purple', ec='purple',
                        linewidth=2, label='Direction Vector (R)', zorder=8, alpha=0.7)

            ax.set_xlabel('X Position (m)', fontsize=12)
            ax.set_ylabel('Y Position (m)', fontsize=12)
            ax.set_title(f'Iteration {iteration_count} (Window Q={Q})\n'
                        f'Dataset: {self.colmap_path}\n'
                        f'Window D: {D_cam_ids}, Remaining: {len(A_cam_ids)} cameras',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Save figure
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"iteration_{iteration_count:03d}_Q_{Q:03d}.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✅ Iteration {iteration_count} visualization saved to: {output_file}")

            plt.close()

        except Exception as e:
            print(f"⚠️  Could not create iteration visualization: {e}")
            import traceback
            traceback.print_exc()

    def run(self, iterations_per_window: int = 60):
        """
        Main execution pipeline - Dynamic Window Approach

        Args:
            iterations_per_window: Training iterations per window
        """
        print("="*60)
        print("Starting Progressive Training with Dynamic Window")
        print(f"Initial cameras: {self.initial_cameras}, Iterations per window: {iterations_per_window}")
        print("="*60)

        # Validate and adjust max_window_size if needed
        max_window_size = getattr(self, 'max_window_size', None)

        # Treat negative values as unlimited
        if max_window_size is not None and max_window_size < 0:
            print(f"\n   max_window_size is negative ({max_window_size}), treating as unlimited")
            self.max_window_size = None
            max_window_size = None
        elif max_window_size is not None and max_window_size < self.initial_cameras:
            print(f"\n⚠️  Warning: max_window_size ({max_window_size}) is less than initial_cameras ({self.initial_cameras})")
            print(f"   Adjusting max_window_size to {self.initial_cameras} to allow initial window creation")
            self.max_window_size = self.initial_cameras
            max_window_size = self.initial_cameras

        if max_window_size is not None:
            print(f"   Max window size: {max_window_size} cameras")
        else:
            print(f"   Max window size: Unlimited")

        # Step 1: Compute camera footprints
        self.compute_camera_footprints()

        # Initialize WindowSelector for new algorithm
        from progressive_learning.window_selector import WindowSelector

        # Extract camera positions as dict {cam_id: [x, y]}
        camera_positions = {}
        for img_id in sorted(self.images.keys()):
            pos = self.images[img_id]['position']  # [tx, ty, tz]
            camera_positions[img_id] = np.array([pos[0], pos[1]])  # Take only (x, y)

        # Create WindowSelector
        f_mode = getattr(self, 'f_mode', 'global')
        self.window_selector = WindowSelector(
            camera_positions=camera_positions,
            camera_footprints=self.camera_footprints,
            f_mode=f_mode
        )

        print(f"\n✅ WindowSelector initialized:")
        print(f"   Total cameras: {len(camera_positions)}")
        print(f"   Initial cameras: {self.initial_cameras}")
        print(f"   Footprints available: {len(self.camera_footprints)}")

        # Initialize sliding window global state
        all_camera_ids = sorted(self.images.keys())
        all_point_ids = set(self.points3D.keys())

        # Steps 1-5: Select initial window using WindowSelector
        print("\n" + "="*80)
        print("ALGORITHM STEPS 1-5: Initial Window Selection")
        print("="*80)
        window_cameras = self.window_selector.select_initial_window(n_cameras=self.initial_cameras)

        print(f"\n✅ Initial window selected:")
        print(f"   Camera IDs: {window_cameras}")
        print("="*80)

        # Steps 6-7: Compute window mean P and footprint union Z
        print("\n" + "="*80)
        print("ALGORITHM STEPS 6-7: Window Mean and Footprint Union")
        print("="*80)

        # Step 6: Compute window mean P
        P = self.window_selector.compute_window_mean(window_cameras)

        # Step 7: Compute footprint union Z
        Z = self.window_selector.compute_window_footprint_union(window_cameras)

        print(f"\n✅ Steps 6-7 completed:")
        print(f"   Window mean P: ({P[0]:.3f}, {P[1]:.3f})")
        if Z is not None:
            if Z.geom_type == 'Polygon':
                print(f"   Union Z: Single polygon, area = {Z.area:.2f} m²")
            elif Z.geom_type == 'MultiPolygon':
                total_area = sum(poly.area for poly in Z.geoms)
                print(f"   Union Z: {len(Z.geoms)} polygons, total area = {total_area:.2f} m²")
        else:
            print(f"   Union Z: None (error)")
        print("="*80)

        # Visualize Steps 1-7
        if not self.skip_4_fast_debug:
            self._visualize_steps_1_7(camera_positions, window_cameras, P, Z)
        else:
            print("⚠️  SKIP_4_FAST_DEBUG: Skipping Steps 1-7 visualization")

        '''
        print("\n" + "="*80)
        print("✅ Steps 1-7 verification completed!")
        print("="*80)
        sys.exit(0)  # Exit for testing Steps 1-7
        '''
        # Initialize sliding window state sets
        self.unprocessed_cameras = set(all_camera_ids) - set(window_cameras)  # Cameras remaining to be processed
        self.processed_cameras = set()  # Processed cameras (initially empty)
        self.all_processed_cameras = set(window_cameras)  # All cameras ever in any window (starts with initial window)
        self.unprocessed_points = all_point_ids.copy()  # All points start as unprocessed
        self.trained_gaussians = set()  # Trained Gaussians (initially empty)
        # Note: We only track unprocessed_points and processed_gaussians

        print(f"📊 Initial sliding window state:")
        print(f"   Total cameras: {len(all_camera_ids)}")
        print(f"   Initial window cameras: {len(window_cameras)}")
        print(f"   Unprocessed cameras: {len(self.unprocessed_cameras)}")
        print(f"   Total 3D points: {len(all_point_ids)}")
        print(f"   Unprocessed points: {len(self.unprocessed_points)}")

        # Store initial window camera IDs for FIFO removal strategy
        self.initial_window_cam_ids = set(window_cameras)
        print(f"   Initial window camera IDs stored: {self.initial_window_cam_ids}")

        # Create remaining cameras list (excluding selected initial cameras)
        remaining_cameras = [cam_id for cam_id in all_camera_ids if cam_id not in window_cameras]
        camera_index = 0  # Index for remaining_cameras list

        # Initial training
        print("\n" + "="*80)
        print(f"🚀 STARTING INITIAL WINDOW")
        print(f"📷 Cameras: {window_cameras}")
        print(f"🔄 Iterations: {iterations_per_window}")
        print("="*80)
        dataset = self.create_sliding_dataset(window_cameras)

        # Calculate and store window mean for initial window
        self.current_window_mean = self._calculate_window_mean(window_cameras)

        # Create visualization for initial window if debug mode
        if self.debug and not self.skip_4_fast_debug:
            print("   Creating visualization for initial window...")
            self._visualize_sliding_window_coverage(dataset, window_cameras, 0)  # window_num=0 for initial

        # Store current window cameras for next iteration
        self.prev_window_cameras = list(window_cameras)

        if not self.skip_4_fast_debug:
            self.train_grendel_gs(dataset, -1, -1, iterations_per_window, "initial",sliding_window=True)
        else:
            print("   ⚠️  SKIP_4_FAST_DEBUG: Skipping training for initial window")

        print("="*80)
        print(f"✅ INITIAL WINDOW COMPLETED")
        print("="*80)

        # Initialize trajectory history for balanced_smooth_trajectory strategy
        e_strategy = getattr(self, 'e_selection_strategy', 'default')
        if e_strategy == 'balanced_smooth_trajectory':
            # Initialize history with initial window (P is already 2D from window_selector)
            self.balanced_window_center_history = [P.copy()]  # Initial window center
            self.balanced_camera_history = []  # No cameras "added" yet (initial cameras don't count)
            print(f"\n📊 Initialized trajectory history for {e_strategy} strategy")
            print(f"   Initial window center: ({P[0]:.3f}, {P[1]:.3f})")

        # ========== ALGORITHM STEP 9: Find next camera E (for Window 1) ==========
        print("\n  " + "="*80)
        print("  ALGORITHM STEP 9: Find Next Camera E (for Window 1)")
        print("  " + "="*80)

        # For Window 1, use projection-based selection (unified with Window 2+)
        # yy = P (initial window mean), R = [0, 0] (no direction yet)

        # Initialize F (global mean)
        f_mode = getattr(self, 'f_mode', 'global')
        if f_mode == 'remaining':
            # Dynamic F: Use only remaining cameras in A
            A_positions = np.array([camera_positions[cam_id] for cam_id in self.window_selector.A])
            F = np.mean(A_positions, axis=0)
            self.current_F_remaining = F  # Store for visualization
            print(f"\n     [9] Using projection-based E selection (unified with Window 2+)")
            print(f"         F mode: 'remaining' (mean of {len(self.window_selector.A)} cameras in A): ({F[0]:.3f}, {F[1]:.3f})")
        else:
            # Static F: Use all cameras (original behavior)
            F = self.window_selector.F
            print(f"\n     [9] Using projection-based E selection (unified with Window 2+)")
            print(f"         F mode: 'global' (mean of all cameras): ({F[0]:.3f}, {F[1]:.3f})")
        print(f"         P (initial window mean): ({P[0]:.3f}, {P[1]:.3f})")

        # Step 9.1: Find cameras in A that intersect with Z
        intersection_threshold = getattr(self, 'footprint_intersection_threshold', 0.0)
        print(f"\n     [9.1] Finding cameras in A that intersect with Z...")
        print(f"         Intersection threshold: {intersection_threshold:.2f}")
        S = set()
        if Z is not None:
            Z_area = Z.area
            for cam_id in self.window_selector.A:
                if cam_id in self.camera_footprints:
                    footprint_coords = self.camera_footprints[cam_id]
                    from shapely.geometry import Polygon
                    footprint_polygon = Polygon(footprint_coords)

                    # Check if footprints intersect
                    if footprint_polygon.intersects(Z):
                        # If threshold > 0, check intersection area ratio
                        if intersection_threshold > 0.0:
                            intersection = footprint_polygon.intersection(Z)
                            intersection_area = intersection.area
                            candidate_area = footprint_polygon.area
                            ratio = intersection_area / candidate_area if candidate_area > 0 else 0.0

                            if ratio >= intersection_threshold:
                                S.add(cam_id)
                                if self.debug:
                                    print(f"         Camera {cam_id}: intersection ratio = {ratio:.3f} >= {intersection_threshold:.2f} ✓")
                            else:
                                if self.debug:
                                    print(f"         Camera {cam_id}: intersection ratio = {ratio:.3f} < {intersection_threshold:.2f} ✗")
                        else:
                            # No threshold, just check intersection
                            S.add(cam_id)

        print(f"         Cameras intersecting with Z: {len(S)}")
        print(f"         S = {sorted(S)}")

        # DEBUG: Exit after showing intersection filtering results
        if intersection_threshold > 0.0:
            print("\n" + "="*80)
            print("✅ INTERSECTION FILTERING TEST COMPLETED (Window 1)")
            print(f"   Current window D: {window_cameras}")
            print(f"   Threshold: {intersection_threshold:.2f}")
            print(f"   Total cameras in A: {len(self.window_selector.A)}")
            print(f"   Cameras passing filter: {len(S)}")
            print(f"   Filtered cameras: {sorted(S)}")
            print("="*80)
            #import sys
            #sys.exit(0)

        if len(S) == 0:
            print("  ❌ Error: No cameras intersecting with Z. Stopping.")
            return

        # Step 9.2: For Window 1, skip forward hemisphere filtering (R = [0,0])
        # All cameras in S are candidates
        X = S
        print(f"\n     [9.2] Window 1: Using all intersecting cameras (no direction filtering)")
        print(f"         X = {sorted(X)}")

        # Step 9.3: Find E using selected strategy
        print(f"\n     [9.3] Finding camera E using '{e_strategy}' strategy...")

        # Check if we should use balanced_smooth_trajectory
        if e_strategy == 'balanced_smooth_trajectory':
            # Use balanced_smooth_trajectory for Window 1
            # Note: smooth_window and smooth_camera scores will be 0 (no history yet)

            # Get weights
            outward_weight = getattr(self, 'e_outward_weight', 0.04)
            compact_weight = getattr(self, 'e_compact_weight', 2.5)
            smooth_window_weight = getattr(self, 'e_smooth_window_weight', 2.8)
            smooth_camera_weight = getattr(self, 'e_smooth_camera_weight', 0.7)
            distance_weight = getattr(self, 'e_distance_weight', 0.5)
            directional_weight = getattr(self, 'e_directional_weight', 0.0)

            print(f"            Strategy: balanced_smooth_trajectory")
            print(f"            Weights: outward={outward_weight:.2f}, compact={compact_weight:.2f}, "
                  f"smooth_win={smooth_window_weight:.2f}, smooth_cam={smooth_camera_weight:.2f}, "
                  f"distance={distance_weight:.2f}, directional={directional_weight:.2f}")
            print(f"            F (dynamic): ({F[0]:.3f}, {F[1]:.3f})")
            print(f"            Note: Window 1 has no history, so smooth scores will be 0")

            # Build arrays for vectorized calculation
            all_cam_ids = sorted(camera_positions.keys())
            positions_array = np.array([camera_positions[cam_id] for cam_id in all_cam_ids])
            cam_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(all_cam_ids)}

            # Current window D indices (Window 1 uses window_cameras)
            D_array_indices = [cam_id_to_idx[cam_id] for cam_id in window_cameras]
            window_size = max_window_size  # Use max_window_size, not current size

            # Convert initial_window_cam_ids to array indices
            initial_window_array_indices = [cam_id_to_idx[cam_id] for cam_id in self.initial_window_cam_ids] if hasattr(self, 'initial_window_cam_ids') else None

            # For Window 1: no previous history
            prev_window_center = None
            prev_movement = None
            last_added_array_idx = None
            second_last_added_array_idx = None

            # Calculate scores for all candidates
            scores = []
            max_score = -float('inf')
            E_cam_id = None

            # Create reverse mapping: array_idx -> cam_id
            idx_to_cam_id = {idx: cam_id for cam_id, idx in cam_id_to_idx.items()}

            for xx_cam_id in X:
                xx_array_idx = cam_id_to_idx[xx_cam_id]

                # Enable debug for specific cameras (21, 35) to diagnose compact score differences
                enable_debug = (xx_cam_id in [21, 35])

                # Calculate 6-force score
                total_score, outward_score, compact_score, smooth_window_score, smooth_camera_score, distance_score, directional_score, D_prime_center, variance = \
                    calculate_balanced_smooth_score(
                        xx_array_idx, D_array_indices, positions_array, F,
                        prev_window_center, prev_movement, window_size,
                        outward_weight, compact_weight, smooth_window_weight, smooth_camera_weight, distance_weight, directional_weight,
                        last_added_array_idx, second_last_added_array_idx, initial_window_array_indices,
                        debug=enable_debug, candidate_cam_id=xx_cam_id,
                        D_cam_ids=window_cameras, idx_to_cam_id=idx_to_cam_id
                    )

                scores.append((xx_cam_id, total_score, outward_score, compact_score,
                              smooth_window_score, smooth_camera_score, distance_score, directional_score, variance))

                if total_score > max_score:
                    max_score = total_score
                    E_cam_id = xx_cam_id

            # Print ALL candidates with their scores
            print(f"\n" + "="*80)
            print(f"📊 WINDOW 1 CANDIDATE SCORES (total={len(scores)} candidates)")
            print("="*80)
            print(f"Weights applied: outward={outward_weight}, compact={compact_weight}, "
                  f"smooth_win={smooth_window_weight}, smooth_cam={smooth_camera_weight}, "
                  f"distance={distance_weight}, directional={directional_weight}")
            print("="*80)

            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            for rank, (cam_id, total, outward, compact, smooth_win, smooth_cam, dist, direc, var) in enumerate(sorted_scores, 1):
                cam_pos = camera_positions[cam_id]
                selected_marker = "✓ SELECTED" if cam_id == E_cam_id else ""
                print(f"  [{rank:2d}] Camera {cam_id:3d}: total={total:7.4f} | "
                      f"outward={outward:7.3f} | compact={compact:.3f} | "
                      f"smooth_win={smooth_win:.3f} | smooth_cam={smooth_cam:.3f} | "
                      f"dist={dist:.3f} | direc={direc:.3f} | var={var:6.2f} {selected_marker}")

            print("="*80)
            print(f"🎯 SELECTED: Camera {E_cam_id} with total score {max_score:.4f}")
            print("="*80)

            # Exit immediately after Window 1 if camera 21 or 35 is selected
            if E_cam_id in [21, 35]:
                print(f"\n✅ Window 1 selected Camera {E_cam_id} (21 or 35). Exiting as requested.")
                #import sys
                #sys.exit(0)

            if E_cam_id is None:
                print("  ❌ Error: Could not find camera E. Stopping.")
                return

            E_pos = camera_positions[E_cam_id]
            print(f"\n         Selected E: camera {E_cam_id} at ({E_pos[0]:.3f}, {E_pos[1]:.3f})")

        else:
            # Default: Simple projection-based selection
            print(f"            Using simple projection (default strategy)")
            yy_direction = P - F  # Direction from F to initial window mean
            yy_direction_norm = yy_direction / np.linalg.norm(yy_direction)
            print(f"            yy direction (P - F): ({yy_direction[0]:.3f}, {yy_direction[1]:.3f})")

            max_projection = -float('inf')
            E_cam_id = None
            for xx_cam_id in X:
                xx_pos = camera_positions[xx_cam_id]
                xx_direction = xx_pos - F
                projection = np.dot(xx_direction, yy_direction_norm)

                if projection > max_projection:
                    max_projection = projection
                    E_cam_id = xx_cam_id

                print(f"            Camera {xx_cam_id}: projection={projection:.3f}")

            if E_cam_id is None:
                print("  ❌ Error: Could not find camera E. Stopping.")
                return

            E_pos = camera_positions[E_cam_id]
            print(f"\n         Selected E: camera {E_cam_id} at ({E_pos[0]:.3f}, {E_pos[1]:.3f})")
            print(f"         Maximum projection from F: {max_projection:.3f}")

        print(f"\n  ✅ Step 9 completed:")
        print(f"     E camera ID: {E_cam_id}")
        print(f"     E position: ({E_pos[0]:.3f}, {E_pos[1]:.3f})")
        print("  " + "="*80)


        # ========== ALGORITHM STEP 10: Compute direction vector R ==========
        print("\n  " + "="*80)
        print("  ALGORITHM STEP 10: Compute Direction Vector R")
        print("  " + "="*80)

        # Step 10: R = E - P
        R = E_pos - P

        print(f"\n  ✅ Step 10 completed:")
        print(f"     P (window mean): ({P[0]:.3f}, {P[1]:.3f})")
        print(f"     E (new camera): ({E_pos[0]:.3f}, {E_pos[1]:.3f})")
        print(f"     R (direction vector): ({R[0]:.3f}, {R[1]:.3f})")
        print(f"     R magnitude: {np.linalg.norm(R):.3f}")
        print("  " + "="*80)

        # ========== ALGORITHM STEP 11-12: Add E to D, Remove E from A ==========
        print("\n  " + "="*80)
        print("  ALGORITHM STEPS 11-12: Update Window D and Set A")
        print("  " + "="*80)

        # Step 11: D = D + E
        D_cam_ids = list(window_cameras)  # Current window as list
        D_cam_ids.append(E_cam_id)

        # Track all cameras that have ever been in any window
        self.all_processed_cameras.add(E_cam_id)

        # Step 12: A = A - E
        self.window_selector.A.discard(E_cam_id)

        print(f"\n  ✅ Steps 11-12 completed:")
        print(f"     D before: {window_cameras} (size: {len(window_cameras)})")
        print(f"     Added E: {E_cam_id}")
        print(f"     D after: {D_cam_ids} (size: {len(D_cam_ids)})")
        print(f"     A size: {len(self.window_selector.A)} cameras remaining")
        print("  " + "="*80)

        # Update trajectory history for balanced_smooth_trajectory strategy
        if e_strategy == 'balanced_smooth_trajectory':
            # Calculate new window center after adding E (2D)
            new_window_center_2d = self._calculate_window_mean_2d(D_cam_ids)
            self.balanced_window_center_history.append(new_window_center_2d.copy())
            self.balanced_camera_history.append(E_cam_id)
            print(f"\n📊 Updated trajectory history:")
            print(f"   New window center: ({new_window_center_2d[0]:.3f}, {new_window_center_2d[1]:.3f})")
            print(f"   Added camera: {E_cam_id}")
            print(f"   Total window centers: {len(self.balanced_window_center_history)}")
            print(f"   Total added cameras: {len(self.balanced_camera_history)}")

        # ========== REMOVAL DECISION (Memory + Window Size) ==========
        print("\n  " + "="*80)
        print("  REMOVAL DECISION (Memory + Window Size)")
        print("  " + "="*80)

        # Load GPU metrics from initial window's state.json
        initial_state_file = self.output_path / "model_initial" / "state.json"
        peak_usage_ratio = 0.0
        if initial_state_file.exists():
            import json
            with open(initial_state_file, 'r') as f:
                state = json.load(f)
                gpu_metrics = state.get('gpu_metrics', {})
                peak_usage_ratio = gpu_metrics.get('peak_usage_ratio', 0.0)
                print(f"\n  📊 GPU Memory Metrics from Initial Window:")
                print(f"     Peak: {gpu_metrics.get('peak_memory_gb', 0):.2f} GB / {gpu_metrics.get('total_memory_gb', 0):.2f} GB ({peak_usage_ratio*100:.1f}%)")
        else:
            print(f"\n  ⚠️  Warning: Initial window state.json not found at {initial_state_file}")
            print(f"     Assuming memory sufficient, will not remove camera")

        # Check memory condition
        camera_removal_threshold = self.densify_memory_limit_percentage - self.camera_removal_margin
        print(f"\n  📊 Memory Threshold Calculation:")
        print(f"     Densify limit: {self.densify_memory_limit_percentage*100:.1f}%")
        print(f"     Removal margin: {self.camera_removal_margin*100:.1f}%")
        print(f"     Removal threshold: {camera_removal_threshold*100:.1f}%")
        print(f"     Current peak usage: {peak_usage_ratio*100:.1f}%")

        memory_exceeded = peak_usage_ratio >= camera_removal_threshold

        # Check window size condition
        max_window_size = getattr(self, 'max_window_size', None)
        window_size_exceeded = False
        if max_window_size is not None:
            current_window_size = len(D_cam_ids)
            print(f"\n  📊 Window Size Check:")
            print(f"     Max window size: {max_window_size}")
            print(f"     Current window size: {current_window_size}")
            window_size_exceeded = current_window_size >= max_window_size
            if window_size_exceeded:
                print(f"     ⚠️  Window size limit reached!")
        else:
            print(f"\n  📊 Window Size Check:")
            print(f"     Max window size: Not set (unlimited)")

        # Make removal decision (remove if EITHER condition is met)
        skip_removal = not (memory_exceeded or window_size_exceeded)

        if skip_removal:
            print(f"\n  💡 Removal not needed:")
            if not memory_exceeded:
                print(f"     ✓ Memory OK: {peak_usage_ratio*100:.1f}% < {camera_removal_threshold*100:.1f}%")
            if max_window_size is None or not window_size_exceeded:
                print(f"     ✓ Window size OK: {len(D_cam_ids)} < {max_window_size if max_window_size else 'unlimited'}")
            print(f"     → Window will grow to {len(D_cam_ids)} cameras: {D_cam_ids}")
            G_cam_id = None
        else:
            print(f"\n  ⚠️  Camera removal required:")
            if memory_exceeded:
                print(f"     ✗ Memory exceeded: {peak_usage_ratio*100:.1f}% >= {camera_removal_threshold*100:.1f}%")
            if window_size_exceeded:
                print(f"     ✗ Window size exceeded: {len(D_cam_ids)} >= {max_window_size}")
            print(f"     → Will remove one camera from window")

        print(f"\n  ✅ Removal decision completed")
        print("  " + "="*80)

        # ========== ALGORITHM STEP 13: Find camera G (only if removal needed) ==========
        if not skip_removal:
            removal_strategy = getattr(self, 'removal_strategy', 'farthest')

            print("\n  " + "="*80)
            if removal_strategy == 'fifo':
                print("  ALGORITHM STEP 13: Find Camera G (FIFO - Oldest Camera)")
            else:
                print("  ALGORITHM STEP 13: Find Camera G (Farthest from E)")
            print("  " + "="*80)

            # Step 13: Find G based on strategy
            print(f"\n     Current D: {D_cam_ids}")
            print(f"     Removal strategy: {removal_strategy}")

            if len(D_cam_ids) > 0:
                if removal_strategy == 'fifo':
                    # FIFO with initial camera priority
                    # Check if any initial window cameras remain in D
                    remaining_initial_cams = [cam_id for cam_id in D_cam_ids if cam_id in self.initial_window_cam_ids]

                    if len(remaining_initial_cams) >= 2:
                        # 2개 이상 남아있으면 → E와 가장 먼 것 제거 (farthest 적용)
                        G_cam_id = self.window_selector.find_farthest_from_camera(E_cam_id, remaining_initial_cams)
                        print(f"     Initial window cameras remaining in D: {remaining_initial_cams} (>= 2)")
                        print(f"     G camera ID: {G_cam_id} (initial camera, farthest from E={E_cam_id})")
                    else:
                        # 1개 이하면 → 진짜 FIFO (D_cam_ids[0] 제거)
                        G_cam_id = D_cam_ids[0]
                        if len(remaining_initial_cams) == 1:
                            print(f"     Only 1 initial window camera remaining: {remaining_initial_cams}")
                        else:
                            print(f"     All initial window cameras removed")
                        print(f"     G camera ID: {G_cam_id} (position 0 in D - oldest, FIFO)")
                else:
                    # Farthest: Remove camera farthest from newly added E
                    G_cam_id = self.window_selector.find_farthest_from_camera(E_cam_id, D_cam_ids)
                    print(f"     G camera ID: {G_cam_id} (farthest from E={E_cam_id})")
            else:
                print("  ❌ Error: D is empty, cannot find camera G.")
                G_cam_id = None
        else:
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 13: SKIPPED (No removal needed)")
            print("  " + "="*80)
            G_cam_id = None

        if not skip_removal:
            if G_cam_id is None:
                print("  ❌ Error: Could not find camera G. Stopping.")
                return

            G_pos = camera_positions[G_cam_id]
            removal_strategy = getattr(self, 'removal_strategy', 'farthest')

            print(f"\n  ✅ Step 13 completed:")
            print(f"     E camera ID: {E_cam_id}, position: ({E_pos[0]:.3f}, {E_pos[1]:.3f})")
            print(f"     G camera ID: {G_cam_id}, position: ({G_pos[0]:.3f}, {G_pos[1]:.3f}) ({removal_strategy})")
            print("  " + "="*80)

        # ========== ALGORITHM STEP 14: Remove G from D (only if G was found) ==========
        if not skip_removal and G_cam_id is not None:
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 14: Remove G from D (Create window_1)")
            print("  " + "="*80)

            # Step 14: D = D - G
            D_cam_ids.remove(G_cam_id)

            print(f"\n  ✅ Step 14 completed:")
            print(f"     Removed G: {G_cam_id}")
            print(f"     D (window_1): {D_cam_ids} (size: {len(D_cam_ids)})")
            print("  " + "="*80)
        else:
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 14: SKIPPED (No G to remove)")
            print("  " + "="*80)
            print(f"\n     D (window_1): {D_cam_ids} (size: {len(D_cam_ids)})")
            print("  " + "="*80)

        # ========== ALGORITHM STEP 15: Set Q = 1 ==========
        print("\n  " + "="*80)
        print("  ALGORITHM STEP 15: Initialize Window Counter")
        print("  " + "="*80)

        # Step 15: Q = 1
        Q = 1

        print(f"\n  ✅ Step 15 completed:")
        print(f"     Q = {Q} (window counter initialized)")
        print("  " + "="*80)

        # Visualize Window 1 state before training
        if self.debug:
            print(f"\n  📊 Creating visualization for Window {Q}...")
            # Calculate P and Z for visualization (2D)
            P_vis = self._calculate_window_mean_2d(D_cam_ids)
            Z_vis = self.window_selector.compute_window_footprint_union(D_cam_ids)
            # Calculate R for visualization (2D)
            prev_window_mean_2d = self._calculate_window_mean_2d(window_cameras)
            R_vis = E_pos - prev_window_mean_2d
            # Call visualization with iteration_count=1 for Window 1
            self._visualize_iteration(camera_positions, D_cam_ids, P_vis, Z_vis, R_vis,
                                     iteration_count=1, Q=Q, G_cam_id=G_cam_id,
                                     current_removed_region=None, E_cam_id=E_cam_id,
                                     current_added_region=None)

        # ========== ALGORITHM STEP 16: Iterative Expansion Loop ==========
        print("\n" + "="*80)
        print("ALGORITHM STEP 16: Iterative Expansion Loop")
        print("="*80)

        # Initialize F (global mean)
        f_mode = getattr(self, 'f_mode', 'global')
        if f_mode == 'remaining':
            # Dynamic F: Use only remaining cameras in A
            A_positions = np.array([camera_positions[cam_id] for cam_id in self.window_selector.A])
            F = np.mean(A_positions, axis=0)
            self.current_F_remaining = F  # Store for visualization
            print(f"\n  F mode: 'remaining' (dynamic, based on cameras in A)")
            print(f"  Current F (mean of {len(self.window_selector.A)} cameras in A): ({F[0]:.3f}, {F[1]:.3f})")
        else:
            # Static F: Use all cameras (original behavior)
            F = self.window_selector.F
            print(f"\n  F mode: 'global' (static, based on all cameras)")
            print(f"  Global mean F: ({F[0]:.3f}, {F[1]:.3f})")

        # Initialize camera history for momentum strategy
        self.camera_history = []

        # Initialize variables for first iteration
        # Store the initial window cameras before E was added and G was removed
        prev_D_cam_ids = list(window_cameras)  # This is [15, 63, 38]
        # Store E and G from Steps 9-14 for first iteration
        current_E_cam_id = E_cam_id  # Camera to add
        current_G_cam_id = G_cam_id  # Camera to remove

        # Step 16: Loop until all cameras processed
        iteration_count = 0
        while True:
            iteration_count += 1
            print(f"\n  {'='*80}")
            print(f"  ITERATION {iteration_count} (Window Q={Q})")
            print(f"     Remaining cameras in A: {len(self.window_selector.A)}")
            print(f"     Current window D: {D_cam_ids} (size: {len(D_cam_ids)})")
            print(f"  {'='*80}")

            # Step 16.1: Compute window mean P
            print(f"\n     [16.1] Computing window mean P...")

            # If FIFO removal will happen, compute mean of future window E (D - {G})
            if current_G_cam_id is not None:
                future_window_E_for_mean = [c for c in D_cam_ids if c != current_G_cam_id]
                print(f"           FIFO will remove Camera {current_G_cam_id}")
                print(f"           Using future window E for mean: {future_window_E_for_mean}")
                P = self.window_selector.compute_window_mean(future_window_E_for_mean)
            else:
                print(f"           No FIFO removal, using current window D: {D_cam_ids}")
                P = self.window_selector.compute_window_mean(D_cam_ids)

            print(f"           P = ({P[0]:.3f}, {P[1]:.3f})")

            # Step 16.2: Compute footprint union Z
            print(f"\n     [16.2] Computing footprint union Z...")

            # If FIFO removal will happen, use future window E (D - {G})
            # Otherwise, use current window D
            if current_G_cam_id is not None:
                future_window_E = [c for c in D_cam_ids if c != current_G_cam_id]
                print(f"           FIFO will remove Camera {current_G_cam_id}")
                print(f"           Using future window E: {future_window_E}")
                Z = self.window_selector.compute_window_footprint_union(future_window_E)
            else:
                print(f"           No FIFO removal, using current window D: {D_cam_ids}")
                Z = self.window_selector.compute_window_footprint_union(D_cam_ids)

            if Z is not None:
                if Z.geom_type == 'Polygon':
                    print(f"           Z: Single polygon, area = {Z.area:.2f} m²")
                elif Z.geom_type == 'MultiPolygon':
                    total_area = sum(poly.area for poly in Z.geoms)
                    print(f"           Z: {len(Z.geoms)} polygons, total area = {total_area:.2f} m²")
            else:
                print(f"           Z: None (error)")

            # Calculate removed region (G footprint - current window union Z)
            current_removed_region = None
            if current_G_cam_id is not None and current_G_cam_id in self.camera_footprints and Z is not None:
                from shapely.geometry import Polygon
                G_footprint_coords = self.camera_footprints[current_G_cam_id]
                G_footprint = Polygon(G_footprint_coords)
                current_removed_region = G_footprint.difference(Z)

                # Update cumulative removed regions
                if self.cumulative_removed_regions is None:
                    self.cumulative_removed_regions = current_removed_region
                else:
                    from shapely.ops import unary_union
                    self.cumulative_removed_regions = unary_union([self.cumulative_removed_regions, current_removed_region])

                print(f"\n     [Removed Region] Camera {current_G_cam_id} removed region area: {current_removed_region.area:.2f} m²")
                print(f"     [Cumulative] Total removed regions area: {self.cumulative_removed_regions.area:.2f} m²")

            # Calculate added region (E footprint - previous window union)
            current_added_region = None
            if current_E_cam_id is not None and current_E_cam_id in self.camera_footprints:
                from shapely.geometry import Polygon
                from shapely.ops import unary_union

                E_footprint_coords = self.camera_footprints[current_E_cam_id]
                E_footprint = Polygon(E_footprint_coords)

                # Calculate previous window's footprint union (before adding E)
                # Use prev_D_cam_ids which are the cameras used for training (before E was added)
                prev_window_polys = []
                for cam_id in prev_D_cam_ids:
                    if cam_id in self.camera_footprints:
                        fp_coords = self.camera_footprints[cam_id]
                        prev_window_polys.append(Polygon(fp_coords))

                if prev_window_polys:
                    prev_window_union = unary_union(prev_window_polys)
                    current_added_region = E_footprint.difference(prev_window_union)
                else:
                    # If no previous window, entire E footprint is added
                    current_added_region = E_footprint

                print(f"\n     [Added Region] Camera {current_E_cam_id} added region area: {current_added_region.area:.2f} m²")

            # Visualize iteration state
            self._visualize_iteration(camera_positions, D_cam_ids, P, Z, R, iteration_count, Q, current_G_cam_id, current_removed_region, current_E_cam_id, current_added_region)

            print(f"\n  ✅ Steps 16.1-16.2 completed for iteration {iteration_count}")
            print("  " + "="*80)

            # Step 16.3: Train Grendel-GS on window_{Q}
            print("\n" + "="*80)
            print(f"🚀 STARTING WINDOW_{Q:03d}")
            print(f"📷 Cameras: {D_cam_ids}")
            print(f"🔄 Iterations: {iterations_per_window}")
            print("="*80)

            # Create dataset for current window
            dataset = self.create_sliding_dataset(D_cam_ids)

            # Calculate and store window mean
            self.current_window_mean = self._calculate_window_mean(D_cam_ids)

            # Create visualization if debug mode
            if self.debug and not self.skip_4_fast_debug:
                print(f"   Creating visualization for window_{Q:03d}...")
                self._visualize_sliding_window_coverage(dataset, D_cam_ids, Q)

            # Save current D before training (this will be prev_D for next iteration)
            training_D_cam_ids = list(D_cam_ids)

            # Set prev_window_cameras for train_grendel_gs
            self.prev_window_cameras = list(prev_D_cam_ids)

            print(f"   📊 Training with:")
            print(f"      Previous window cameras: {self.prev_window_cameras}")
            print(f"      Camera to remove: {current_G_cam_id}")
            print(f"      Camera to add: {current_E_cam_id}")
            print(f"      Current window cameras: {D_cam_ids}")

            # Train Grendel-GS with correct E and G
            # Note: train_internal.py will exit after verifying camera IDs
            if not self.skip_4_fast_debug:
                self.train_grendel_gs(dataset, current_G_cam_id, current_E_cam_id, iterations_per_window, f"window_{Q:03d}", sliding_window=True)
            else:
                print(f"   ⚠️  SKIP_4_FAST_DEBUG: Skipping training for window_{Q:03d}")

            # Load GPU metrics from state.json (saved by _save_progressive_state)
            state_file = self.output_path / f"model_window_{Q:03d}" / "state.json"
            peak_usage_ratio = 0.0
            if state_file.exists():
                import json
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    gpu_metrics = state.get('gpu_metrics', {})
                    peak_usage_ratio = gpu_metrics.get('peak_usage_ratio', 0.0)
                    print(f"\n📊 GPU Memory Metrics for Window {Q}:")
                    print(f"   Peak: {gpu_metrics.get('peak_memory_gb', 0):.2f} GB / {gpu_metrics.get('total_memory_gb', 0):.2f} GB ({peak_usage_ratio*100:.1f}%)")
            else:
                print(f"\n⚠️  Warning: State file not found: {state_file}")

            print("="*80)
            print(f"✅ WINDOW_{Q:03d} COMPLETED")
            print("="*80)

            # Check if A is empty (all cameras processed)
            if len(self.window_selector.A) == 0:
                print("\n" + "="*80)
                print("✅ ALL CAMERAS PROCESSED - TRAINING COMPLETED")
                print("="*80)
                break

            # Step 16.4: Find cameras in A that intersect with Z
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 16.4: Find Cameras in A that Intersect with Z")
            print("  " + "="*80)

            intersection_threshold = getattr(self, 'footprint_intersection_threshold', 0.0)
            print(f"\n     [16.4] Finding cameras in A that intersect with footprint union Z...")
            print(f"            Z type: {Z.geom_type if Z else 'None'}")
            print(f"            A size: {len(self.window_selector.A)}")
            print(f"            Intersection threshold: {intersection_threshold:.2f}")

            S = set()  # Cameras in A that intersect with Z

            if Z is None:
                print("  ⚠️  Warning: Z is None. Cannot find intersecting cameras.")
            else:
                Z_area = Z.area
                for cam_id in self.window_selector.A:
                    if cam_id in self.camera_footprints:
                        footprint_coords = self.camera_footprints[cam_id]

                        # Convert footprint to Shapely Polygon
                        from shapely.geometry import Polygon
                        footprint_polygon = Polygon(footprint_coords)

                        # Check if footprints intersect
                        if footprint_polygon.intersects(Z):
                            # If threshold > 0, check intersection area ratio
                            if intersection_threshold > 0.0:
                                intersection = footprint_polygon.intersection(Z)
                                intersection_area = intersection.area
                                candidate_area = footprint_polygon.area
                                ratio = intersection_area / candidate_area if candidate_area > 0 else 0.0

                                if ratio >= intersection_threshold:
                                    S.add(cam_id)
                                    if self.debug:
                                        print(f"            Camera {cam_id}: intersection ratio = {ratio:.3f} >= {intersection_threshold:.2f} ✓")
                                else:
                                    if self.debug:
                                        print(f"            Camera {cam_id}: intersection ratio = {ratio:.3f} < {intersection_threshold:.2f} ✗")
                            else:
                                # No threshold, just check intersection
                                S.add(cam_id)

            print(f"\n  ✅ Step 16.4 completed:")
            print(f"     Current window D: {D_cam_ids}")
            print(f"     Total cameras in A: {len(self.window_selector.A)}")
            print(f"     Cameras intersecting with Z: {len(S)}")
            print(f"     S = {sorted(S)}")
            print("  " + "="*80)

            # DEBUG: Exit after showing intersection filtering results for Window 2+
            if intersection_threshold > 0.0 and iteration_count == 1:
                print("\n" + "="*80)
                print(f"✅ INTERSECTION FILTERING TEST COMPLETED (Window {Q})")
                print(f"   Current window D: {D_cam_ids}")
                if current_G_cam_id is not None:
                    future_window_E = [c for c in D_cam_ids if c != current_G_cam_id]
                    print(f"   FIFO removal: Camera {current_G_cam_id} will be removed")
                    print(f"   Future window E (after FIFO): {future_window_E}")
                    print(f"   ✅ Using E's footprint union for intersection check")
                else:
                    print(f"   No FIFO removal (window size < max)")
                    print(f"   Using D's footprint union for intersection check")
                print(f"   Threshold: {intersection_threshold:.2f}")
                print(f"   Total cameras in A: {len(self.window_selector.A)}")
                print(f"   Cameras passing filter: {len(S)}")
                print(f"   Filtered cameras: {sorted(S)}")
                print("="*80)
                #import sys
                #sys.exit(0)

            if len(S) == 0:
                print("  ⚠️  No cameras intersecting with Z. Stopping iteration.")
                break

            # Step 16.5: Initialize X = {}
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 16.5: Initialize Set X")
            print("  " + "="*80)
            X = set()
            print(f"\n  ✅ Step 16.5 completed: X = {X}")
            print("  " + "="*80)

            # Step 16.6: Filter S by angle with R (optional)
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 16.6: Filter Cameras by Direction")
            print("  " + "="*80)

            enable_direction_filtering = getattr(self, 'enable_direction_filtering', True)

            if enable_direction_filtering:
                print(f"\n     [16.6] Filtering cameras in S by angle with R...")
                print(f"            Current P: ({P[0]:.3f}, {P[1]:.3f})")
                print(f"            Direction R: ({R[0]:.3f}, {R[1]:.3f})")

                for T_cam_id in S:
                    T_pos = camera_positions[T_cam_id]

                    # Step 16.6.1: V = T - P
                    V = T_pos - P

                    # Step 16.6.2: Check angle between V and R
                    # angle <= 90 degrees means forward hemisphere (same direction as R)
                    # This is equivalent to dot(V, R) >= 0
                    dot_product = np.dot(V, R)
                    V_norm = np.linalg.norm(V)
                    R_norm = np.linalg.norm(R)

                    if V_norm > 0 and R_norm > 0:
                        cos_angle = np.clip(dot_product / (V_norm * R_norm), -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_angle))

                        if angle_deg <= 90:  # Forward hemisphere only
                            X.add(T_cam_id)
                            print(f"            Camera {T_cam_id}: angle={angle_deg:.1f}° → Added to X")
                        else:
                            print(f"            Camera {T_cam_id}: angle={angle_deg:.1f}° → Rejected (backward)")

                print(f"\n  ✅ Step 16.6 completed:")
                print(f"     Cameras in X (forward direction): {len(X)}")
                print(f"     X = {sorted(X)}")
                print("  " + "="*80)

                if len(X) == 0:
                    print("  ⚠️  No cameras in forward direction. Stopping iteration.")
                    break
            else:
                # Direction filtering disabled: use all cameras in S
                X = S.copy()
                print(f"\n     [16.6] Direction filtering disabled")
                print(f"            Using all cameras in S (footprint intersection only)")
                print(f"            X = S = {sorted(X)}")
                print("  " + "="*80)

            # Step 16.7: Find E (maximum projection from F along computed direction)
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 16.7: Find Camera E (Maximum Projection from F)")
            print("  " + "="*80)

            # Get E selection strategy
            e_strategy = getattr(self, 'e_selection_strategy', 'default')

            # yy = most recently added camera (current_E_cam_id)
            yy_cam_id = current_E_cam_id
            yy_pos = camera_positions[yy_cam_id]

            print(f"\n     [16.7] Finding camera E using '{e_strategy}' strategy...")
            print(f"            F (global mean): ({F[0]:.3f}, {F[1]:.3f})")
            print(f"            yy (last added camera): {yy_cam_id} at ({yy_pos[0]:.3f}, {yy_pos[1]:.3f})")
            print(f"            R (current direction): ({R[0]:.3f}, {R[1]:.3f})")

            # Compute projection direction based on strategy
            if e_strategy == 'default':
                # Original: Use (yy - F) only
                projection_direction = yy_pos - F
                print(f"            Strategy: default - using (yy - F)")

            elif e_strategy == 'momentum':
                # Momentum: Use recent camera movement
                # Need at least 2 previous cameras for momentum
                if not hasattr(self, 'camera_history'):
                    self.camera_history = []
                self.camera_history.append(yy_cam_id)

                if len(self.camera_history) >= 3:
                    yy_prev1_id = self.camera_history[-2]
                    yy_prev2_id = self.camera_history[-3]
                    yy_prev1_pos = camera_positions[yy_prev1_id]
                    yy_prev2_pos = camera_positions[yy_prev2_id]

                    # Momentum = average of recent movements
                    momentum = (yy_pos - yy_prev1_pos) + (yy_prev1_pos - yy_prev2_pos)
                    projection_direction = momentum + (yy_pos - F)
                    print(f"            Strategy: momentum - using momentum + (yy - F)")
                    print(f"            Momentum from cameras: {yy_prev2_id} -> {yy_prev1_id} -> {yy_cam_id}")
                else:
                    # Not enough history, fall back to default
                    projection_direction = yy_pos - F
                    print(f"            Strategy: momentum (insufficient history, using default)")

            elif e_strategy == 'weighted':
                # Weighted: alpha * R + beta * (yy - F)
                alpha = getattr(self, 'e_weighted_alpha', 0.7)
                beta = getattr(self, 'e_weighted_beta', 0.3)
                projection_direction = alpha * R + beta * (yy_pos - F)
                print(f"            Strategy: weighted - {alpha:.2f}*R + {beta:.2f}*(yy - F)")

            elif e_strategy == 'tangential':
                # Tangential: R + c * R_perpendicular
                c = getattr(self, 'e_tangential_coeff', 0.5)
                R_perpendicular = np.array([-R[1], R[0]])  # 90 degree rotation
                projection_direction = R + c * R_perpendicular
                print(f"            Strategy: tangential - R + {c:.2f}*R_perp")
                print(f"            R_perpendicular: ({R_perpendicular[0]:.3f}, {R_perpendicular[1]:.3f})")

            elif e_strategy == 'polar':
                # Polar: Fixed angle/radius steps from F
                angle_step = getattr(self, 'e_polar_angle_step', 30.0)  # degrees
                radius_step = getattr(self, 'e_polar_radius_step', 1.2)

                # Track current angle and radius from F
                if not hasattr(self, 'polar_angle'):
                    # Initialize with yy's angle
                    yy_from_F = yy_pos - F
                    self.polar_angle = np.arctan2(yy_from_F[1], yy_from_F[0])
                    self.polar_radius = np.linalg.norm(yy_from_F)

                # Increment angle and radius
                self.polar_angle += np.radians(angle_step)
                self.polar_radius *= radius_step

                # Convert to Cartesian direction
                projection_direction = np.array([
                    self.polar_radius * np.cos(self.polar_angle),
                    self.polar_radius * np.sin(self.polar_angle)
                ])
                print(f"            Strategy: polar - angle={np.degrees(self.polar_angle):.1f}°, radius={self.polar_radius:.1f}")

            elif e_strategy == 'outward_spiral_compact':
                # Outward Spiral Compact: Outward + Spiral + Compact window
                # Score-based selection with 3 components:
                # 1. Distance: prefer closer to window center
                # 2. Diversity: prefer perpendicular to previous direction
                # 3. Variance: minimize window variance

                alpha = getattr(self, 'e_spiral_alpha', 1.0)
                beta = getattr(self, 'e_spiral_beta', 0.3)
                gamma = getattr(self, 'e_spiral_gamma', 0.5)

                print(f"            Strategy: outward_spiral_compact")
                print(f"            Weights: alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")

                # Step 1: Outward filtering (F 기준으로 멀어지는 것만)
                Y = set()
                dist_yy_to_F = np.linalg.norm(yy_pos - F)
                for xx_cam_id in X:
                    xx_pos = camera_positions[xx_cam_id]
                    dist_to_F = np.linalg.norm(xx_pos - F)
                    if dist_to_F > dist_yy_to_F:  # Outward
                        Y.add(xx_cam_id)

                print(f"            Outward filtering: {len(X)} → {len(Y)} candidates (dist > {dist_yy_to_F:.3f})")

                if len(Y) == 0:
                    print("            ⚠️  No outward candidates, using all X")
                    Y = X

                # Get current window D (use D_cam_ids from Step 16, not window_selector.D)
                D_indices = list(D_cam_ids)
                D_positions = np.array([camera_positions[cam_id] for cam_id in D_indices])

                # Handle case when D has only 1 camera
                if len(D_indices) == 1:
                    D_positions = D_positions.reshape(1, -1)

                D_center = np.mean(D_positions, axis=0)
                D_variance = np.var(np.linalg.norm(D_positions - D_center, axis=1))
                D_std = np.std(np.linalg.norm(D_positions - D_center, axis=1)) + 1e-6

                print(f"            Current window: center=({D_center[0]:.3f}, {D_center[1]:.3f}), std={D_std:.3f}")

                # Previous direction (for diversity)
                prev_direction = None
                if hasattr(self, 'prev_spiral_direction'):
                    prev_direction = self.prev_spiral_direction

                # Step 2: Score calculation for each candidate
                max_score = -float('inf')
                next_E_cam_id = None
                scores = []

                for xx_cam_id in Y:
                    xx_pos = camera_positions[xx_cam_id]

                    # 1. Distance score: closer to window center is better
                    dist_to_window = np.linalg.norm(xx_pos - D_center)
                    distance_score = np.exp(-dist_to_window / (2 * D_std))

                    # 2. Diversity score: perpendicular to previous direction
                    diversity_score = 0
                    if prev_direction is not None:
                        current_direction = xx_pos - D_center
                        current_norm = np.linalg.norm(current_direction)
                        if current_norm > 1e-6:
                            current_direction = current_direction / current_norm
                            cos_angle = np.dot(prev_direction, current_direction)
                            diversity_score = 1 - abs(cos_angle)  # Max at perpendicular

                    # 3. Variance penalty: minimize window variance increase
                    temp_positions = np.vstack([D_positions, xx_pos])
                    new_center = np.mean(temp_positions, axis=0)
                    new_variance = np.var(np.linalg.norm(temp_positions - new_center, axis=1))
                    variance_penalty = new_variance - D_variance

                    # Final score
                    score = alpha * distance_score + beta * diversity_score - gamma * variance_penalty
                    scores.append((xx_cam_id, score, distance_score, diversity_score, variance_penalty))

                    if score > max_score:
                        max_score = score
                        next_E_cam_id = xx_cam_id

                # Print scores
                print(f"\n            Candidate scores:")
                for cam_id, score, d_score, div_score, var_pen in sorted(scores, key=lambda x: x[1], reverse=True)[:5]:
                    cam_pos = camera_positions[cam_id]
                    print(f"              Camera {cam_id}: score={score:.4f} (dist={d_score:.3f}, div={div_score:.3f}, var_pen={var_pen:.3f})")

                # Update previous direction
                if next_E_cam_id is not None:
                    next_E_pos = camera_positions[next_E_cam_id]
                    direction = next_E_pos - D_center
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-6:
                        self.prev_spiral_direction = direction / direction_norm

                # Skip projection-based selection
                use_score_selection = True

            elif e_strategy == 'balanced_smooth_trajectory':
                # Balanced Smooth Trajectory: 5-force camera selection
                # 1. Outward: Window center moves away from F
                # 2. Compact: Window radius minimization
                # 3. Smooth Window: Window center trajectory continuity
                # 4. Smooth Camera: Camera addition trajectory continuity
                # 5. Distance: Candidate close to current window center

                # Get weights (with defaults matching visualization)
                outward_weight = getattr(self, 'e_outward_weight', 0.04)
                compact_weight = getattr(self, 'e_compact_weight', 2.5)
                smooth_window_weight = getattr(self, 'e_smooth_window_weight', 2.8)
                smooth_camera_weight = getattr(self, 'e_smooth_camera_weight', 0.7)
                distance_weight = getattr(self, 'e_distance_weight', 0.5)
                directional_weight = getattr(self, 'e_directional_weight', 0.0)

                print(f"            Strategy: balanced_smooth_trajectory")
                print(f"            Weights: outward={outward_weight:.2f}, compact={compact_weight:.2f}, "
                      f"smooth_win={smooth_window_weight:.2f}, smooth_cam={smooth_camera_weight:.2f}, "
                      f"distance={distance_weight:.2f}, directional={directional_weight:.2f}")

                # Initialize history tracking if needed
                if not hasattr(self, 'balanced_window_center_history'):
                    self.balanced_window_center_history = []
                if not hasattr(self, 'balanced_camera_history'):
                    self.balanced_camera_history = []

                # Get current window D
                D_indices = list(D_cam_ids)
                window_size = max_window_size  # Use max_window_size, not current size

                # Previous window center and movement
                prev_window_center = None
                prev_movement = None
                if len(self.balanced_window_center_history) >= 1:
                    prev_window_center = self.balanced_window_center_history[-1]
                if len(self.balanced_window_center_history) >= 2:
                    prev_movement = self.balanced_window_center_history[-1] - self.balanced_window_center_history[-2]

                # Last added cameras
                last_added_idx = None
                second_last_added_idx = None
                if len(self.balanced_camera_history) >= 1:
                    last_added_idx = self.balanced_camera_history[-1]
                if len(self.balanced_camera_history) >= 2:
                    second_last_added_idx = self.balanced_camera_history[-2]

                # Create positions array mapping camera_id to [x, y]
                # Need to create a consistent indexing scheme
                all_cam_ids = sorted(camera_positions.keys())
                cam_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(all_cam_ids)}
                positions_array = np.array([camera_positions[cam_id] for cam_id in all_cam_ids])

                # Convert D_indices and X to array indices
                D_array_indices = [cam_id_to_idx[cam_id] for cam_id in D_indices]

                # Convert initial_window_cam_ids to array indices
                initial_window_array_indices = [cam_id_to_idx[cam_id] for cam_id in self.initial_window_cam_ids] if hasattr(self, 'initial_window_cam_ids') else None

                # Update F based on f_mode (for 'remaining' mode, recalculate based on current A)
                if f_mode == 'remaining':
                    A_positions = np.array([camera_positions[cam_id] for cam_id in self.window_selector.A])
                    F = np.mean(A_positions, axis=0)
                    self.current_F_remaining = F  # Store for visualization
                    print(f"            F (dynamic): ({F[0]:.3f}, {F[1]:.3f}) based on {len(self.window_selector.A)} cameras in A")

                # Score calculation for each candidate (NO hard filtering - outward is just a weighted component)
                print(f"            Evaluating all {len(X)} candidates in X (no hard filtering)")
                max_score = -float('inf')
                next_E_cam_id = None
                scores = []

                # Convert history indices to array indices
                last_added_array_idx = cam_id_to_idx[last_added_idx] if last_added_idx is not None else None
                second_last_added_array_idx = cam_id_to_idx[second_last_added_idx] if second_last_added_idx is not None else None

                # Create reverse mapping: array_idx -> cam_id
                idx_to_cam_id = {idx: cam_id for cam_id, idx in cam_id_to_idx.items()}

                for xx_cam_id in X:
                    xx_array_idx = cam_id_to_idx[xx_cam_id]

                    # Enable debug for specific cameras (21, 35) to diagnose compact score differences
                    enable_debug = (xx_cam_id in [21, 35])

                    # Calculate 6-force score
                    total_score, outward_score, compact_score, smooth_window_score, smooth_camera_score, distance_score, directional_score, D_prime_center, variance = \
                        calculate_balanced_smooth_score(
                            xx_array_idx, D_array_indices, positions_array, F,
                            prev_window_center, prev_movement, window_size,
                            outward_weight, compact_weight, smooth_window_weight, smooth_camera_weight, distance_weight, directional_weight,
                            last_added_array_idx, second_last_added_array_idx, initial_window_array_indices,
                            debug=enable_debug, candidate_cam_id=xx_cam_id,
                            D_cam_ids=window_cameras, idx_to_cam_id=idx_to_cam_id
                        )

                    scores.append((xx_cam_id, total_score, outward_score, compact_score,
                                  smooth_window_score, smooth_camera_score, distance_score, directional_score, variance))

                    if total_score > max_score:
                        max_score = total_score
                        next_E_cam_id = xx_cam_id
                        next_E_window_center = D_prime_center

                # Print ALL candidates with their scores
                print(f"\n" + "="*80)
                print(f"📊 ALL CANDIDATE SCORES (total={len(scores)} candidates)")
                print("="*80)
                print(f"Weights applied: outward={outward_weight}, compact={compact_weight}, "
                      f"smooth_win={smooth_window_weight}, smooth_cam={smooth_camera_weight}, "
                      f"distance={distance_weight}, directional={directional_weight}")
                print("="*80)

                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
                for rank, (cam_id, total, outward, compact, smooth_win, smooth_cam, dist, direc, var) in enumerate(sorted_scores, 1):
                    cam_pos = camera_positions[cam_id]
                    selected_marker = "✓ SELECTED" if cam_id == next_E_cam_id else ""
                    print(f"  [{rank:2d}] Camera {cam_id:3d}: total={total:7.4f} | "
                          f"outward={outward:7.3f} | compact={compact:.3f} | "
                          f"smooth_win={smooth_win:.3f} | smooth_cam={smooth_cam:.3f} | "
                          f"dist={dist:.3f} | direc={direc:.3f} | var={var:6.2f} {selected_marker}")

                print("="*80)
                print(f"🎯 SELECTED: Camera {next_E_cam_id} with total score {max_score:.4f}")
                print("="*80)

                # VERIFICATION EXIT
                print("\n" + "="*80)
                print("🛑 WEIGHT VERIFICATION COMPLETE - EXITING")
                print("="*80)
                print("This confirms all 5 weights are applied:")
                print(f"  - Outward weight:       {outward_weight}")
                print(f"  - Compact weight:       {compact_weight}")
                print(f"  - Smooth window weight: {smooth_window_weight}")
                print(f"  - Smooth camera weight: {smooth_camera_weight}")
                print(f"  - Distance weight:      {distance_weight}")
                print("="*80)
                print("Remove this sys.exit(0) to continue training")
                print("="*80)
                #import sys
                #sys.exit(0)

                # Update history
                if next_E_cam_id is not None:
                    self.balanced_window_center_history.append(next_E_window_center)
                    self.balanced_camera_history.append(next_E_cam_id)

                    # Limit history size to avoid memory issues
                    if len(self.balanced_window_center_history) > 100:
                        self.balanced_window_center_history = self.balanced_window_center_history[-100:]
                    if len(self.balanced_camera_history) > 100:
                        self.balanced_camera_history = self.balanced_camera_history[-100:]

                # Skip projection-based selection
                use_score_selection = True

            else:
                # Fallback to default
                projection_direction = yy_pos - F
                print(f"            Strategy: unknown '{e_strategy}', using default")
                use_score_selection = False

            # Projection-based selection (only if not score-based)
            if not locals().get('use_score_selection', False):
                # Normalize direction
                yy_direction_norm = projection_direction / np.linalg.norm(projection_direction)
                print(f"            Projection direction: ({projection_direction[0]:.3f}, {projection_direction[1]:.3f})")

                max_projection = -float('inf')
                next_E_cam_id = None
                for xx_cam_id in X:
                    xx_pos = camera_positions[xx_cam_id]
                    # (xx - F) projected onto (yy - F)
                    xx_direction = xx_pos - F
                    projection = np.dot(xx_direction, yy_direction_norm)

                    if projection > max_projection:
                        max_projection = projection
                        next_E_cam_id = xx_cam_id

                    print(f"            Camera {xx_cam_id}: projection={projection:.3f}")

            if next_E_cam_id is None:
                print("  ⚠️  Could not find camera E. Stopping.")
                break

            next_E_pos = camera_positions[next_E_cam_id]
            print(f"\n            Selected E: camera {next_E_cam_id} at ({next_E_pos[0]:.3f}, {next_E_pos[1]:.3f})")

            # Only print max_projection if it was calculated (projection-based selection)
            if 'max_projection' in locals():
                print(f"            Maximum projection from F: {max_projection:.3f}")

            print(f"\n  ✅ Step 16.7 completed:")
            print(f"     E camera ID: {next_E_cam_id}")
            print("  " + "="*80)

            # TEMPORARY: Exit for Window 2 E verification
            print("\n" + "="*80)
            print("🛑 VERIFICATION COMPLETE - Exiting after Step 16.7 (Window 2 E selection)")
            print("="*80)
            #import sys
            #sys.exit(0)

            # Step 16.8: Update R = E - P
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 16.8: Update Direction Vector R")
            print("  " + "="*80)

            print(f"\n     [16.8] Updating R = E - P...")
            print(f"            P: ({P[0]:.3f}, {P[1]:.3f})")
            print(f"            E: ({next_E_pos[0]:.3f}, {next_E_pos[1]:.3f})")
            R = next_E_pos - P
            print(f"            New R: ({R[0]:.3f}, {R[1]:.3f})")
            print(f"            R magnitude: {np.linalg.norm(R):.3f}")

            print(f"\n  ✅ Step 16.8 completed")
            print("  " + "="*80)

            # Step 16.9: D = D + E
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 16.9: Add E to D")
            print("  " + "="*80)

            print(f"\n     [16.9] Adding E to D...")
            print(f"            D before: {D_cam_ids} (size: {len(D_cam_ids)})")
            D_cam_ids.append(next_E_cam_id)

            # Track all cameras that have ever been in any window
            self.all_processed_cameras.add(next_E_cam_id)

            print(f"            D after: {D_cam_ids} (size: {len(D_cam_ids)})")

            print(f"\n  ✅ Step 16.9 completed")
            print("  " + "="*80)

            # Step 16.10: A = A - E
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 16.10: Remove E from A")
            print("  " + "="*80)

            print(f"\n     [16.10] Removing E from A...")
            print(f"            A size before: {len(self.window_selector.A)}")
            self.window_selector.A.discard(next_E_cam_id)
            print(f"            A size after: {len(self.window_selector.A)}")

            print(f"\n  ✅ Step 16.10 completed")
            print("  " + "="*80)

            # Step 16.10.5: Check if camera removal is needed based on memory + window size
            print("\n  " + "="*80)
            print("  REMOVAL DECISION (Memory + Window Size)")
            print("  " + "="*80)

            # Check memory condition
            camera_removal_threshold = self.densify_memory_limit_percentage - self.camera_removal_margin
            print(f"\n     Densify limit: {self.densify_memory_limit_percentage*100:.1f}%")
            print(f"     Removal margin: {self.camera_removal_margin*100:.1f}%")
            print(f"     Removal threshold: {camera_removal_threshold*100:.1f}%")
            print(f"     Current peak usage: {peak_usage_ratio*100:.1f}%")

            memory_exceeded = peak_usage_ratio >= camera_removal_threshold

            # Check window size condition
            max_window_size = getattr(self, 'max_window_size', None)
            window_size_exceeded = False
            if max_window_size is not None:
                current_window_size = len(D_cam_ids)  # Current window D in Step 16 loop
                print(f"\n     Max window size: {max_window_size}")
                print(f"     Current window size: {current_window_size}")
                window_size_exceeded = current_window_size > max_window_size  # > not >=, allow up to max_window_size
                if window_size_exceeded:
                    print(f"     ⚠️  Window size limit reached!")
            else:
                print(f"\n     Max window size: Not set (unlimited)")

            # Make removal decision (remove if EITHER condition is met)
            skip_removal = not (memory_exceeded or window_size_exceeded)

            if skip_removal:
                print(f"\n     💡 Removal not needed:")
                if not memory_exceeded:
                    print(f"        ✓ Memory OK: {peak_usage_ratio*100:.1f}% < {camera_removal_threshold*100:.1f}%")
                if max_window_size is None or not window_size_exceeded:
                    print(f"        ✓ Window size OK")
                print(f"        → Window will grow")
                next_G_cam_id = None
            else:
                print(f"\n     ⚠️  Camera removal required:")
                if memory_exceeded:
                    print(f"        ✗ Memory exceeded: {peak_usage_ratio*100:.1f}% >= {camera_removal_threshold*100:.1f}%")
                if window_size_exceeded:
                    print(f"        ✗ Window size exceeded: {len(D_cam_ids)} > {max_window_size}")  # Current window D
                print(f"        → Will remove one camera from window")

            print(f"\n  ✅ Removal decision completed")
            print("  " + "="*80)

            # Step 16.11: Find G - only if removal needed
            if not skip_removal:
                removal_strategy = getattr(self, 'removal_strategy', 'farthest')

                print("\n  " + "="*80)
                if removal_strategy == 'fifo':
                    print("  ALGORITHM STEP 16.11: Find Camera G (FIFO - Oldest Camera)")
                else:
                    print("  ALGORITHM STEP 16.11: Find Camera G (Farthest from E)")
                print("  " + "="*80)

                print(f"\n     [16.11] Finding camera G using {removal_strategy} strategy...")
                print(f"            Current D: {D_cam_ids}")

                if len(D_cam_ids) > 0:
                    if removal_strategy == 'fifo':
                        # FIFO with initial camera priority
                        # Check if any initial window cameras remain in D
                        remaining_initial_cams = [cam_id for cam_id in D_cam_ids if cam_id in self.initial_window_cam_ids]

                        if len(remaining_initial_cams) >= 2:
                            # 2개 이상 남아있으면 → E와 가장 먼 것 제거 (farthest 적용)
                            next_G_cam_id = self.window_selector.find_farthest_from_camera(next_E_cam_id, remaining_initial_cams)
                            next_G_pos = camera_positions[next_G_cam_id]
                            print(f"            Initial window cameras remaining in D: {remaining_initial_cams} (>= 2)")
                            print(f"            G camera ID: {next_G_cam_id} (initial camera, farthest from E={next_E_cam_id})")
                        else:
                            # 1개 이하면 → 진짜 FIFO (D_cam_ids[0] 제거)
                            next_G_cam_id = D_cam_ids[0]
                            next_G_pos = camera_positions[next_G_cam_id]
                            if len(remaining_initial_cams) == 1:
                                print(f"            Only 1 initial window camera remaining: {remaining_initial_cams}")
                            else:
                                print(f"            All initial window cameras removed")
                            print(f"            G camera ID: {next_G_cam_id} (position 0 in D - oldest, FIFO)")
                    else:
                        # Farthest: Remove camera farthest from newly added E
                        next_G_cam_id = self.window_selector.find_farthest_from_camera(next_E_cam_id, D_cam_ids)
                        next_G_pos = camera_positions[next_G_cam_id]
                        print(f"            G camera ID: {next_G_cam_id} (farthest from E={next_E_cam_id})")

                    print(f"            G position: ({next_G_pos[0]:.3f}, {next_G_pos[1]:.3f})")
                    print(f"            E camera ID: {next_E_cam_id}, position: ({next_E_pos[0]:.3f}, {next_E_pos[1]:.3f})")
                else:
                    print("  ⚠️  D is empty, cannot find camera G. Stopping.")
                    next_G_cam_id = None
                    break

                print(f"\n  ✅ Step 16.11 completed:")
                print(f"     G camera ID: {next_G_cam_id} ({removal_strategy})")
                print("  " + "="*80)
            else:
                # Skip camera removal - window grows
                print("\n  " + "="*80)
                print("  ALGORITHM STEP 16.11: SKIPPED (No removal needed)")
                print("  " + "="*80)
                print(f"\n     Window will continue growing")
                print(f"     Current D size: {len(D_cam_ids)}")
                next_G_cam_id = None
                print(f"\n  ✅ Step 16.11 skipped")
                print("  " + "="*80)

            # Check if this is the first camera removal (for testing)
            if self.exit_after_first_removal and next_G_cam_id is not None and current_G_cam_id is None:
                print("\n" + "="*80)
                print("🛑 FIRST CAMERA REMOVAL DETECTED - EXITING FOR TESTING")
                print("="*80)
                print(f"   Camera to be removed: {next_G_cam_id}")
                print(f"   Current window size: {len(D_cam_ids)}")
                print(f"   This means memory threshold was reached and sliding window mode started")
                print("="*80)
                #import sys
                #sys.exit(0)

            # Step 16.12: Q = Q + 1
            print("\n  " + "="*80)
            print("  ALGORITHM STEP 16.12: Increment Window Counter")
            print("  " + "="*80)

            print(f"\n     [16.12] Incrementing Q...")
            print(f"            Q before: {Q}")
            Q = Q + 1
            print(f"            Q after: {Q}")

            print(f"\n  ✅ Step 16.12 completed")
            print("  " + "="*80)

            # Step 16.13: D = D - G (only if G was selected)
            if next_G_cam_id is not None:
                print("\n  " + "="*80)
                print("  ALGORITHM STEP 16.13: Remove G from D")
                print("  " + "="*80)

                print(f"\n     [16.13] Removing G from D...")
                print(f"            D before: {D_cam_ids} (size: {len(D_cam_ids)})")
                D_cam_ids.remove(next_G_cam_id)
                print(f"            D after: {D_cam_ids} (size: {len(D_cam_ids)})")

                print(f"\n  ✅ Step 16.13 completed")
                print("  " + "="*80)
            else:
                print("\n  " + "="*80)
                print("  ALGORITHM STEP 16.13: SKIPPED (No G to remove)")
                print("  " + "="*80)
                print(f"\n     D remains: {D_cam_ids} (size: {len(D_cam_ids)})")
                print(f"\n  ✅ Step 16.13 skipped")
                print("  " + "="*80)

            # Update variables for next iteration
            # prev_D_cam_ids = D used for training in this iteration (saved before Steps 16.9-16.13)
            prev_D_cam_ids = training_D_cam_ids
            current_E_cam_id = next_E_cam_id
            current_G_cam_id = next_G_cam_id

            print(f"\n  📊 Updated state for next iteration:")
            print(f"     Previous D (for next training): {prev_D_cam_ids}")
            print(f"     Current D (for next iteration): {D_cam_ids}")
            print(f"     Next E: {current_E_cam_id}")
            print(f"     Next G: {current_G_cam_id}")
            print(f"     Next Q: {Q}")
            print(f"     Cameras remaining in A: {len(self.window_selector.A)}")
            print("  " + "="*80)

        # All cameras processed - Algorithm complete
        print("\n" + "="*80)
        print("🎉 ALGORITHM STEP 16 COMPLETED - ALL CAMERAS PROCESSED!")
        print("="*80)
        print(f"\n  📊 Final Statistics:")
        print(f"     Total windows trained: {Q}")
        print(f"     Total iterations: {iteration_count}")
        print(f"     All cameras from set A have been processed")
        print(f"     Final window D: {D_cam_ids} (size: {len(D_cam_ids)})")
        print(f"     Output saved to: {self.output_path}")
        print("\n" + "="*80)
        print("✅ PROGRESSIVE TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)

        return  # Exit progressive training method

        #sys.exit(1)
        window_num = 1
        self.current_window_cameras = set(window_cameras)  # Track current window cameras

        # Slide through remaining cameras
        while camera_index < len(remaining_cameras):
            # STEP 1: Add new camera first (mean-based selection with gaussian cleanup)
            added, window_cameras = self._add_camera_to_window(window_cameras, window_num)
            if added is None:
                break  # No more unprocessed cameras
            # STEP 2: Remove camera farthest from newly added camera
            removed = self._remove_camera_from_window(window_cameras, added)
            '''
            # Update points: move points covered by new camera from unprocessed to processed
            if added in self.points_in_view:
                new_points = set(self.points_in_view[added]) & self.unprocessed_points
                self.unprocessed_points -= new_points  # Remove from unprocessed
                # Note: processed_points doesn't exist, only processed_gaussians
            print(f"   New points processed: {len(new_points) if added in self.points_in_view else 0}")
            print(f"   Unprocessed points: {len(self.unprocessed_points)}")
            '''
            camera_index += 1

            print(f"📊 Updated sliding window state:")
            print(f"   Removed camera {removed} -> processed_cameras: {len(self.processed_cameras)} cameras")
            print(f"   Added camera {added} -> unprocessed_cameras: {len(self.unprocessed_cameras)} cameras remaining")
            print(f"   Total processed gaussians: {len(self.processed_gaussians) if hasattr(self, 'processed_gaussians') else 0}")

            print("\n" + "="*80)
            print(f"🔄 STARTING WINDOW {window_num}")
            print(f"➖ Removed camera: {removed}")
            print(f"➕ Added camera: {added}")
            print(f"📷 Current window: {window_cameras}")
            print(f"📊 Progress: {len(self.current_window_cameras)} / {len(all_camera_ids)} cameras processed")
            print(f"⏳ Remaining: {len(remaining_cameras) - camera_index} cameras to add")
            print(f"🔄 Iterations: {iterations_per_window}")
            print("="*80)

            # Create dataset for current window
            dataset = self.create_sliding_dataset(window_cameras)

            # Calculate and store updated window mean
            self.current_window_mean = self._calculate_window_mean(window_cameras)
            #exit(1)
            # Create visualization for current window coverage
            if self.debug and not self.skip_4_fast_debug:
                print(f"Creating visualization for window {window_num}...")
                self._visualize_sliding_window_coverage(dataset, window_cameras, window_num, removed, added)

            # Train with checkpoint from previous window
            if not self.skip_4_fast_debug:
                self.train_grendel_gs(dataset, removed, added, iterations_per_window, f"window_{window_num}", sliding_window=True)
            else:
                print(f"   ⚠️  SKIP_4_FAST_DEBUG: Skipping training for window_{window_num}")

            print("="*80)
            print(f"✅ WINDOW {window_num} COMPLETED")
            print("="*80)

            # Update previous window cameras for next iteration
            self.prev_window_cameras = list(window_cameras)

            window_num += 1

        print("\n" + "="*80)
        print("🎉 PROGRESSIVE TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print(f"📊 Total cameras processed: {len(self.current_window_cameras)} / {len(all_camera_ids)}")
        print(f"🪟 Total windows completed: {window_num}")
        print(f"💾 Output saved to: {self.output_path}")
        print("="*80)



# Example usage
if __name__ == "__main__":
    print('TEST: Progressive trainer main called')
    print('TEST: About to create trainer')
    trainer = ProgressiveTrainer(
        colmap_path="/data/colmap_reconstruction",
        output_path="/output/progressive_results",
        initial_cameras=4,
        camera_removal_margin=0.1,
        debug=True
    )
    print('TEST: Trainer created, about to run')
    trainer.run()
