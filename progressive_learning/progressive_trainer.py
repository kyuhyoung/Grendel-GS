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
        self.points_in_footprint = {}  # W_c for each camera
        
        # Training state
        self.P = set()  # Selected camera indices
        self.H = None  # Union of selected footprints
        self.V = None  # Current Gaussian set
        self.I = []  # Removed Gaussians (saved to PLY)
        
        # Find COLMAP files and image directory
        self._find_colmap_files()
        
        # Load COLMAP data
        self._load_colmap_data()
        
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
            with open(points_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        pt_id = int(parts[0])
                        xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

                        # only_positive_z 플래그 체크
                        if self.only_positive_z and xyz[2] < 0:
                            continue  # 음수 Z값 포인트는 제외

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

                        self.points3D[pt_id] = {
                            'id': pt_id,
                            'xyz': xyz,
                            'rgb': rgb,
                            'error': error,
                            'track': track
                        }
        
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
                self.dtm_module.read_points3d_txt()
                self.dtm_module.create_dtm(resolution=2.0)
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
            self.points_in_footprint[img_id] = W_c if W_c else []
            
        #print(f"Computed {len(self.camera_footprints)} footprints")
        avg_points = np.mean([len(pts) for pts in self.points_in_footprint.values()])
        print(f"Average points per footprint: {avg_points:.1f}")
        #print(f'self.debug : {self.debug}');    exit(1)    
        if self.debug:
            # Use existing verified visualization from colmap_visualizer
            try:
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                scene_path = os.path.join(self.output_path, f"progressive_3d_scene_{timestamp}.png")
                ortho_path = os.path.join(self.output_path, f"progressive_ortho_view_{timestamp}.png")
                
                scene_center = self.dtm_module.visualize_3d_scene(save_path=scene_path)
                self.dtm_module.render_orthographic_view(scene_center, save_path=ortho_path)
                
                print(f"Visualizations saved:")
                print(f"  - 3D scene: {scene_path}")
                print(f"  - Orthographic: {ortho_path}")
            except Exception as e:
                print(f"Visualization failed: {e}")

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
            positions = np.array([img['position'] for img in self.images.values()])
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

    def train_grendel_gs(self, dataset: Dict, iteration_name: str = "initial",
                        total_iterations: int = None, sliding_window: bool = False) -> None:
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

        # For sliding window testing, use shorter iterations
        if sliding_window and total_iterations is None:
            total_iterations = 60  # Quick test iterations

        # Save dataset to temporary COLMAP format
        # NOTE: This I/O could be avoided if train.py supported in-memory datasets
        temp_path = self.output_path / f"temp_{iteration_name}"
        temp_path.mkdir(exist_ok=True)

        # Save to COLMAP format
        self._save_colmap_format(dataset, temp_path)
        
        # Prepare model output path
        model_output = self.output_path / f"model_{iteration_name}"
        model_output.mkdir(exist_ok=True)
        
        # Build training command
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

        if gpu_count > 1:
            # Multi-GPU training with torchrun
            cmd = [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={gpu_count}",
                str(train_script),
                "-s", str(temp_path),
                "-m", str(model_output),
                "--iterations", str(total_iterations if total_iterations else self.iterations),
                "--sh_degree", str(self.sh_degree),
                "--bsz", "1"  # Batch size for multi-GPU
            ]
        else:
            # Single GPU or CPU training
            cmd = [
                sys.executable, str(train_script),
                "-s", str(temp_path),
                "-m", str(model_output),
                "--iterations", str(total_iterations if total_iterations else self.iterations),
                "--sh_degree", str(self.sh_degree)
            ]

        # Add checkpoint support for sliding window
        if iteration_name != "initial":
            # Check if previous model exists
            prev_model = self.output_path / f"model_{self.get_previous_iteration_name(iteration_name)}"
            if prev_model.exists():
                cmd.extend(["--start_checkpoint", str(prev_model)])
                print(f"Starting from checkpoint: {prev_model}")
        
        if self.backend == "gsplat":
            cmd.extend(["--backend", "gsplat"])
        
        if self.deterministic:
            cmd.append("--deterministic")
        
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
            else:
                error_msg = f"Training failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nError: {result.stderr}"
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Training timed out for {iteration_name}")
        except Exception as e:
            raise RuntimeError(f"Training failed for {iteration_name}: {e}")
        
        # Load resulting Gaussian set V
        # self.V = load_gaussians(...)
        
    def _save_colmap_format(self, dataset: Dict, path: Path):
        """Save dataset in COLMAP format"""
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

        print(f"✓ COLMAP format saved to: {sparse_path}")
        print(f"  - cameras.txt: {len(dataset['cameras'])} cameras")
        print(f"  - images.txt: {len(dataset['images'])} images")
        print(f"  - points3D.txt: {len(dataset['points3D'])} points")
        exit(1)

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
        available = set(self.images.keys()) - self.P
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
            
            self.P.add(j)
            print(f"Adding camera {j} (total: {len(self.P)})")
            
            # Step 10.2: Find new points R_j not in H
            W_j = self.points_in_footprint[j]  # List of point IDs
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
        while len(self.P) < total_cameras:
            # Step 12.1: Find next closest camera
            j = self.find_next_closest_camera()
            if j is None:
                break
            
            self.P.add(j)
            print(f"Processing camera {j} ({len(self.P)}/{total_cameras})")
            
            # Step 12.2: Find new points R_j
            W_j = self.points_in_footprint[j]  # List of point IDs
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
        print(f"Create dataset for sliding window cameras");
        print(f'self.points_in_footprint.keys() : {self.points_in_footprint.keys()}')
        #print(f'self.points3D.keys() : {self.points3D.keys()}')
        # Include points visible from selected cameras
        included_points = {}
        for cam_id in camera_ids:
            if cam_id in self.points_in_footprint:
                #t0 = self.points_in_footprint[cam_id]
                #print(f't0 length : {len(t0)}')
                for pt_idx in self.points_in_footprint[cam_id]:
                    #print(f'pt_idx : {pt_idx}')
                    if pt_idx in self.points3D:
                        included_points[pt_idx] = self.points3D[pt_idx]

        return {
            'cameras': self.cameras,
            'images': {img_id: self.images[img_id] for img_id in camera_ids},
            'points3D': included_points if included_points else self.points3D
        }

    def test_sliding_window(self, iterations_per_window: int = 60, num_windows: int = 5):
        """
        Test sliding window approach with small iterations

        Args:
            iterations_per_window: Number of training iterations per window
            num_windows: Number of sliding windows to test
        """
        print("="*60)
        print(f"Testing Sliding Window: {iterations_per_window} iterations x {num_windows} windows")
        print("="*60)

        # Get initial camera set
        initial_cameras = self.select_initial_cameras()
        window_cameras = list(initial_cameras)
        remaining_cameras = [cam for cam in self.images.keys() if cam not in window_cameras]

        # Initial training
        print(f"Initial window: cameras {window_cameras}")
        dataset = self.create_initial_dataset(window_cameras)
        self.train_grendel_gs(dataset, "initial",
                            total_iterations=iterations_per_window,
                            sliding_window=True)

        # Sliding windows
        for window_idx in range(min(num_windows, len(remaining_cameras))):
            if not remaining_cameras:
                break

            # Remove oldest camera
            removed_camera = window_cameras.pop(0)
            # Add new camera
            new_camera = remaining_cameras.pop(0)
            window_cameras.append(new_camera)

            print(f"\nWindow {window_idx + 1}: removed camera {removed_camera}, added camera {new_camera}")
            print(f"Current cameras: {window_cameras}")

            # Create new dataset with sliding window
            dataset = {
                'cameras': self.cameras,
                'images': {img_id: self.images[img_id] for img_id in window_cameras},
                'points3D': self.points3D  # For simplicity, use all points
            }

            # Train with checkpoint from previous iteration
            iteration_name = f"window_{window_idx + 1}"
            self.train_grendel_gs(dataset, iteration_name,
                                total_iterations=iterations_per_window,
                                sliding_window=True)

        print("="*60)
        print("Sliding Window Test Complete!")
        print("="*60)

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

        # Initialize with sliding window approach using select_initial_cameras
        all_camera_ids = sorted(self.images.keys())

        # Use select_initial_cameras to get optimal starting cameras
        self.initial_cameras = sliding_window_size  # Set window size as initial_cameras
        window_cameras = self.select_initial_cameras()

        # Create remaining cameras list (excluding selected initial cameras)
        remaining_cameras = [cam_id for cam_id in all_camera_ids if cam_id not in window_cameras]
        camera_index = 0  # Index for remaining_cameras list

        # Initial training
        print(f"\nInitial window: cameras {window_cameras}")
        dataset = self.create_sliding_dataset(window_cameras)

        # Create visualization for initial window if debug mode
        if self.debug:
            print("Creating visualization for initial window...")
            self._visualize_sliding_window_coverage(dataset, window_cameras, 0)  # window_num=0 for initial

        self.train_grendel_gs(dataset, "initial",
                            total_iterations=iterations_per_window,
                            sliding_window=True)

        window_num = 1
        self.P = set(window_cameras)  # Track processed cameras

        # Slide through remaining cameras
        while camera_index < len(remaining_cameras):
            # Remove oldest, add newest
            removed = window_cameras.pop(0)
            added = remaining_cameras[camera_index]
            window_cameras.append(added)
            self.P.add(added)
            camera_index += 1

            print(f"\n--- Window {window_num} ---")
            print(f"Removed camera {removed}, added camera {added}")
            print(f"Current window: {window_cameras}")
            print(f"Total processed: {len(self.P)} / {len(all_camera_ids)}")
            print(f"Remaining cameras to add: {len(remaining_cameras) - camera_index}")

            # Create dataset for current window
            dataset = self.create_sliding_dataset(window_cameras)

            # Create visualization for current window coverage
            if self.debug:
                print(f"Creating visualization for window {window_num}...")
                self._visualize_sliding_window_coverage(dataset, window_cameras, window_num, removed, added)

            # Train with checkpoint from previous window
            self.train_grendel_gs(dataset, f"window_{window_num}",
                                total_iterations=iterations_per_window,
                                sliding_window=True)

            window_num += 1

        print("="*60)
        print("Sliding Window Training Complete!")
        print(f"Processed {len(self.P)} cameras in {window_num} windows")
        print(f"Output saved to {self.output_path}")
        print("="*60)


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
