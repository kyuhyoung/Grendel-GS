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
        initial_cameras: int = 4,
        gpu_memory_threshold: float = 0.9,
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
            gpu_memory_threshold: GPU memory usage threshold (0-1)
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
        self.gpu_memory_threshold = gpu_memory_threshold
        self.debug = debug
        self.only_actually_visible = only_actually_visible
        
        # Data containers
        self.cameras = {}
        self.images = {}
        self.points3D = {}
        
        # Computed data
        self.W = None  # (x,y) coordinates of all 3D points
        self.camera_footprints = {}  # F_c for each camera
        self.points_in_view = {}  # Camera ID -> List of 3D point IDs visible through projection
        
        # Training state
        self.current_window_cameras = set()  # Current sliding window cameras
        self.H = None  # Union of selected footprints
        self.V = None  # Current Gaussian set
        self.I = []  # Removed Gaussians (saved to PLY)

        # Sliding window global state
        self.unprocessed_cameras = set()  # Cameras remaining to be processed
        self.processed_cameras = set()  # Cameras that have been fully processed
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
                        self.images[img_id] = {
                            'id': img_id,
                            'qw': float(parts[1]),
                            'qx': float(parts[2]),
                            'qy': float(parts[3]),
                            'qz': float(parts[4]),
                            'tx': float(parts[5]),
                            'ty': float(parts[6]),
                            'tz': float(parts[7]),
                            'camera_id': int(parts[8]),
                            'name': parts[9],
                            'position': np.array([float(parts[5]), float(parts[6]), float(parts[7])])
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
                    
                    if len(corners_3d) >= 3:
                        # Valid footprint
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
                        # Fallback to camera position
                        pos = img_data['position']
                        S_c = pos[:2]
                        radius = 50.0  # meters
                        F_c = np.array([
                            S_c + [-radius, -radius],
                            S_c + [radius, -radius],
                            S_c + [radius, radius],
                            S_c + [-radius, radius]
                        ])
                        # Filter points within radius - store point IDs
                        point_ids = []
                        all_point_ids = list(self.points3D.keys())
                        for i, point_xy in enumerate(self.W):
                            if np.linalg.norm(point_xy - S_c) <= radius:
                                point_ids.append(all_point_ids[i])
                        W_c = point_ids
                        
                except Exception as e:
                    if self.debug:
                        print(f"Error processing camera {img_id}: {e}")
                    # Fallback to simplified computation
                    pos = img_data['position']
                    S_c = pos[:2]
                    radius = 50.0
                    F_c = np.array([
                        S_c + [-radius, -radius],
                        S_c + [radius, -radius],
                        S_c + [radius, radius],
                        S_c + [-radius, radius]
                    ])
                    # Filter points within radius - store point IDs
                    point_ids = []
                    all_point_ids = list(self.points3D.keys())
                    for i, point_xy in enumerate(self.W):
                        if np.linalg.norm(point_xy - S_c) <= radius:
                            point_ids.append(all_point_ids[i])
                    W_c = point_ids
            else:
                # Simplified computation without DTM
                # Use camera position as approximate center
                S_c = img_data['position'][:2]  # (x,y) only

                # Simple radius-based footprint (placeholder)
                radius = 50.0  # meters, adjust based on camera height
                F_c = {'center': S_c, 'radius': radius, 'type': 'circle'}

                # Filter points within radius - store point IDs
                point_ids = []
                all_point_ids = list(self.points3D.keys())
                for i, point_xy in enumerate(self.W):
                    if np.linalg.norm(point_xy - S_c) <= radius:
                        point_ids.append(all_point_ids[i])
                W_c = point_ids
            
            #self.camera_footprints[img_id] = F_c
            
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

            print(f"Creating sliding window visualization at: {window_path}")

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

            print(f"Current cameras in window: {current_cameras}")
            if removed_camera is not None:
                print(f"Removed camera: {removed_camera}")
            if added_camera is not None:
                print(f"Added camera: {added_camera}")

            # Calculate median position for visualization
            #positions = np.array([img['position'] for img in self.images.values()])
            #median_position = np.median(positions, axis=0)

            # Create visualization - show only selected cameras in sliding window
            self.dtm_module.create_nadir_view_multi(subsets_info, save_path=str(window_path),
                                                  only_selected=True, median_point=self.cam_pos_med)
            print(f"✓ Sliding window coverage saved to: {window_path}")

        except Exception as e:
            print(f"Warning: Could not create sliding window visualization: {e}")

    def get_previous_iteration_name(self, current_name: str) -> str:
        """Get the previous iteration name for checkpoint loading"""
        if current_name == "initial":
            return None
        # Parse iteration number
        if current_name.startswith("iter_"):
            current_num = int(current_name.split("_")[1])
            if current_num == 1:
                return "initial"
            return f"iter_{current_num - 1}"
        elif current_name.startswith("window_"):
            current_num = int(current_name.split("_")[1])
            if current_num == 1:
                return "initial"
            return f"window_{current_num - 1}"
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
        print(f"Training Grendel-GS ({iteration_name})...")

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

        # Add show_memory_debug_info if set
        if hasattr(self, 'show_memory_debug_info') and self.show_memory_debug_info:
            cmd.append("--show_memory_debug_info")

        # Progressive training should not auto-save final iteration by default
        if not hasattr(self, 'auto_save_final_iteration') or not self.auto_save_final_iteration:
            cmd.append("--no_auto_save_final_iteration")

        # Add track_by_projection flag if set
        if hasattr(self, 'track_by_projection') and self.track_by_projection:
            cmd.append("--track_by_projection")

        # Force checkpoint save at final iteration for progressive training continuity
        final_iter = total_iterations if total_iterations else self.iterations
        cmd.extend(["--checkpoint_iterations", str(final_iter)])
        print(f"DEBUG: Forcing checkpoint save at iteration {final_iter}")

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
                prev_model_name = f"window_{prev_window_num}"

            prev_state_file = self.output_path / f"model_{prev_model_name}" / "state.json"
            if prev_state_file.exists():
                cmd.extend(["--previous_state", str(prev_state_file)])
            else:
                print(f"⚠️  Warning: Previous state file not found: {prev_state_file}")
                cmd.extend(["--previous_state", ""])  # Fallback to empty

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

                # Add cameras to delete and add
                if cam_id_2_delete is not None:
                    cmd.extend(["--cams_2_delete", str(cam_id_2_delete)])
                if new_camera_id is not None:
                    cmd.extend(["--cams_2_add", str(new_camera_id)])


        if self.debug:
            print(f"Training command: {' '.join(cmd)}")
        
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

    def _calculate_window_mean(self, camera_ids):
        """Calculate mean position of camera window"""
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
            print(f"📍 Calculated window mean: {mean_pos}")
            #exit(1)
            return mean_pos
        return None

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

    def _remove_gaussians_outside_camera_view(self, new_camera_id, margin_pixels = 5, consider_distortion = True, use_opencv = True):
        """
        Remove gaussians that are not visible from the new camera and add them to processed_gaussians

        NOTE: This function should operate on actual trained gaussians, not initial COLMAP points.
        However, current implementation is limited because we need access to the trained gaussian model.

        Args:
            new_camera_id: ID of the new camera to check visibility against
            margin_pixels: Margin in pixels to consider gaussian still in view
            consider_distortion: Whether to apply camera distortion correction (ignored if use_opencv=True)
            use_opencv: If True, use cv2.projectPoints (always considers distortion)
        """
        if new_camera_id not in self.images:
            print(f"⚠️  Camera {new_camera_id} not found in images")
            return

        if not hasattr(self.dtm_module, 'project_point_to_image'):
            print(f"⚠️  DTM module does not have project_point_to_image function")
            return

        print(f"🔍 Checking gaussian visibility for camera {new_camera_id} with {margin_pixels}px margin")

        # TODO: Load actual gaussian positions from trained model
        # For now, we fallback to using COLMAP points as a proxy
        # This is not ideal but better than nothing until we implement proper gaussian loading

        print("⚠️  WARNING: Currently using COLMAP points as proxy for gaussians.")
        print("⚠️  This should be replaced with actual trained gaussian positions.")

        # Try to load gaussians from previous training iteration
        gaussian_positions = None
        if hasattr(self, 'current_iteration_name'):
            prev_model_path = self.output_path / f"model_{self.current_iteration_name}"
            if prev_model_path.exists():
                gaussian_positions = self._load_gaussian_positions_from_model(prev_model_path)

        if gaussian_positions is not None:
            # Use actual trained gaussian positions
            print(f"📊 Processing {len(gaussian_positions)} actual gaussians")

            visible_gaussians = []
            invisible_gaussians = []

            for i, gaussian_pos in enumerate(gaussian_positions):
                is_visible, u, v = self.dtm_module.project_point_to_image(
                    gaussian_pos, new_camera_id, margin_pixels, consider_distortion, use_opencv)

                if is_visible:
                    visible_gaussians.append(i)
                else:
                    invisible_gaussians.append(i)

            print(f"📊 Actual gaussian projection results for camera {new_camera_id}:")
            print(f"   Total gaussians checked: {len(gaussian_positions)}")
            print(f"   Visible gaussians: {len(visible_gaussians)} ({len(visible_gaussians)/len(gaussian_positions)*100:.1f}%)")
            print(f"   Invisible gaussians: {len(invisible_gaussians)} ({len(invisible_gaussians)/len(gaussian_positions)*100:.1f}%)")

            # Note: Currently we cannot actually remove gaussians from the model
            # This would require modifying the gaussian model file or implementing
            # selective gaussian masking in the training process
            print(f"⚠️  NOTE: Gaussian removal not implemented yet. This is for analysis only.")

        else:
            # Fallback to COLMAP points (temporary solution)
            print("📊 Fallback: Using COLMAP points as gaussian proxy")

            # Get current gaussians (using COLMAP points as proxy)
            if hasattr(self, 'current_gaussians'):
                current_gaussians = self.current_gaussians.copy()
            else:
                # If no current_gaussians set, use all unprocessed points as current
                current_gaussians = self.unprocessed_points.copy()
                self.current_gaussians = current_gaussians

            if not current_gaussians:
                print("📊 No current gaussians to check")
                return

            # Project COLMAP points to the new camera using DTM module's projection function
            visible_gaussians = set()
            invisible_gaussians = set()

            total_checked = 0

            for point_id in current_gaussians:
                if point_id in self.points3D:
                    point_3d = self.points3D[point_id]['xyz']
                    is_visible, u, v = self.dtm_module.project_point_to_image(point_3d, new_camera_id, margin_pixels, consider_distortion, use_opencv)

                    total_checked += 1
                    if is_visible:
                        visible_gaussians.add(point_id)
                    else:
                        invisible_gaussians.add(point_id)

            print(f"📊 COLMAP proxy projection results for camera {new_camera_id}:")
            print(f"   Total points checked: {total_checked}")
            print(f"   Visible points: {len(visible_gaussians)} ({len(visible_gaussians)/total_checked*100:.1f}%)")
            print(f"   Invisible points: {len(invisible_gaussians)} ({len(invisible_gaussians)/total_checked*100:.1f}%)")

            # Remove invisible gaussians from current state (proxy-based)
            if invisible_gaussians:
                # Update current_gaussians
                self.current_gaussians -= invisible_gaussians

                # Remove from unprocessed_points
                self.unprocessed_points -= invisible_gaussians

                # Add to processed gaussians (create set if not exists)
                if not hasattr(self, 'processed_gaussians'):
                    self.processed_gaussians = set()
                self.processed_gaussians.update(invisible_gaussians)

                print(f"🗑️  Removed {len(invisible_gaussians)} proxy points not visible from camera {new_camera_id}")
                print(f"📊 Updated state:")
                print(f"   Current gaussians: {len(self.current_gaussians)}")
                print(f"   Unprocessed points: {len(self.unprocessed_points)}")
                print(f"   Processed gaussians: {len(self.processed_gaussians)}")
            else:
                print(f"✅ All {len(current_gaussians)} proxy points remain visible from camera {new_camera_id}")

    def _add_camera_to_window(self, window_cameras, window_num):
        """Select and add new camera to window after cleaning up invisible gaussians"""
        # STEP 1: Select next camera based on previous window mean
        previous_mean = None
        if window_num >= 1:
            # Load previous window's progressive state to get mean
            prev_iteration_name = self.get_previous_iteration_name(f"window_{window_num}")
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

        # Get current window's checkpoint paths after training completion
        all_checkpoint_paths = {}

        # Try to load checkpoint paths from temporary file created by train_internal.py
        temp_checkpoint_file = model_output / f"window_{window_number}_checkpoint_paths.json"
        if temp_checkpoint_file.exists():
            try:
                with open(temp_checkpoint_file, 'r') as f:
                    all_checkpoint_paths = json.load(f)
                print(f"📂 Collected checkpoint paths: {all_checkpoint_paths}")
                # Remove temporary file since we'll store in state.json
                temp_checkpoint_file.unlink()
                print(f"🗑️  Removed temporary checkpoint file: {temp_checkpoint_file}")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint paths: {e}")

        state = {
            "iteration_name": iteration_name,
            "window_number": window_number,
            "unprocessed_cameras": list(self.unprocessed_cameras),
            "processed_cameras": list(self.processed_cameras),
            "unprocessed_points": list(self.unprocessed_points),
            # "processed_points": removed - only track processed_gaussians
            "trained_gaussians": list(self.trained_gaussians),
            "current_window_cameras": list(self.current_window_cameras),
            "current_window_mean": current_window_mean,
            "all_checkpoint_paths": all_checkpoint_paths,
            "total_cameras": len(self.images),
            "total_points": len(self.points3D),
            "gpu_memory_threshold": self.gpu_memory_threshold,
            "debug": self.debug
        }

        state_file = model_output / "state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"💾 Progressive state saved to: {state_file}")
        print(f"   iteration_name: {state['iteration_name']}")
        print(f"   window_number: {state['window_number']}")
        print(f"   unprocessed_cameras: {state['unprocessed_cameras']}")
        print(f"   processed_cameras: {state['processed_cameras']}")
        print(f"   unprocessed_points: {len(state['unprocessed_points'])} points")
        # print(f"   processed_points: removed - only track processed_gaussians")
        print(f"   trained_gaussians: {len(state['trained_gaussians'])} files")
        print(f"   current_window_cameras: {state['current_window_cameras']}")
        print(f"   current_window_mean: {state['current_window_mean']}")
        print(f"   total_cameras: {state['total_cameras']}")
        print(f"   total_points: {state['total_points']}")
        print(f"   gpu_memory_threshold: {state['gpu_memory_threshold']}")
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
        
        while self.get_gpu_memory_usage() < self.gpu_memory_threshold:
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
            if self.get_gpu_memory_usage() >= self.gpu_memory_threshold:
                print(f"GPU memory threshold reached ({self.gpu_memory_threshold*100}%)")
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
        print(f"📊 Create dataset for sliding window cameras: {camera_ids}")
        #print(f"📊 Total 3D points in memory: {len(self.points3D)}")
        #print(f'📊 Camera visible points available: {list(self.points_in_view.keys())}')

        # Include points visible from selected cameras
        included_points = {}
        total_footprint_points = 0

        for cam_id in camera_ids:
            if cam_id in self.points_in_view:
                visible_points = self.points_in_view[cam_id]
                print(f"📊 Camera {cam_id} visible points: {len(visible_points)}")
                total_footprint_points += len(visible_points)

                valid_points_in_view = 0
                for pt_idx in visible_points:
                    if pt_idx in self.points3D:
                        included_points[pt_idx] = self.points3D[pt_idx]
                        valid_points_in_view += 1

                print(f"📊 Camera {cam_id}: {valid_points_in_view} valid points added to dataset")
            else:
                print(f"⚠️  Camera {cam_id} has no footprint data")

        print(f"📊 Total footprint points from all cameras: {total_footprint_points}")
        print(f"📊 Unique points included in dataset: {len(included_points)}")

        final_points = included_points if included_points else self.points3D
        print(f"📊 Final dataset points count: {len(final_points)}")

        return {
            'cameras': self.cameras,
            'images': {img_id: self.images[img_id] for img_id in camera_ids},
            'points3D': final_points
        }


    def run(self, sliding_window_size: int = 3, iterations_per_window: int = 60):
        """
        Main execution pipeline - Sliding Window Approach

        Args:
            sliding_window_size: Number of cameras in sliding window
            iterations_per_window: Training iterations per window
        """
        print("="*60)
        print("Starting Progressive Training with Sliding Window")
        print(f"Window size: {sliding_window_size}, Iterations per window: {iterations_per_window}")
        print("="*60)

        # Step 1: Compute camera footprints
        self.compute_camera_footprints()

        # Initialize sliding window global state
        all_camera_ids = sorted(self.images.keys())
        all_point_ids = set(self.points3D.keys())

        # Initialize sliding window state sets
        window_cameras = self.select_initial_cameras()
        self.unprocessed_cameras = set(all_camera_ids) - set(window_cameras)  # Cameras remaining to be processed
        self.processed_cameras = set()  # Processed cameras (initially empty)
        self.unprocessed_points = all_point_ids.copy()  # All points start as unprocessed
        self.trained_gaussians = set()  # Trained Gaussians (initially empty)
        # Note: We only track unprocessed_points and processed_gaussians

        print(f"📊 Initial sliding window state:")
        print(f"   Total cameras: {len(all_camera_ids)}")
        print(f"   Initial window cameras: {len(window_cameras)}")
        print(f"   Unprocessed cameras: {len(self.unprocessed_cameras)}")
        print(f"   Total 3D points: {len(all_point_ids)}")
        print(f"   Unprocessed points: {len(self.unprocessed_points)}")

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
        if self.debug:
            print("Creating visualization for initial window...")
            self._visualize_sliding_window_coverage(dataset, window_cameras, 0)  # window_num=0 for initial

        # Store current window cameras for next iteration
        self.prev_window_cameras = list(window_cameras)

        self.train_grendel_gs(dataset, -1, -1, iterations_per_window, "initial",sliding_window=True)

        print("="*80)
        print(f"✅ INITIAL WINDOW COMPLETED")
        print("="*80)

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
            if self.debug:
                print(f"Creating visualization for window {window_num}...")
                self._visualize_sliding_window_coverage(dataset, window_cameras, window_num, removed, added)

            # Train with checkpoint from previous window
            self.train_grendel_gs(dataset, removed, added, iterations_per_window, f"window_{window_num}", sliding_window=True)

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
        gpu_memory_threshold=0.9,
        debug=True
    )
    print('TEST: Trainer created, about to run')
    trainer.run()
