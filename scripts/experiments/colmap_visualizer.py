#!/usr/bin/env python3
"""
COLMAP Output 3D Visualization Tool
카메라 위치, 이미지 raycast, DTM 생성 및 상공 렌더링
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용 (서버 환경)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import os
import struct
import sys
sys.path.insert(0, '/workspace/Grendel-GS')
from utils.camera_param_parser import parse_camera_parameters_heuristic

class COLMAPVisualizer:
    def __init__(self, colmap_path=None, only_actually_visible=False):
        self.colmap_path = colmap_path
        self.cameras = {}
        self.images = {}
        self.points3d = {}
        self.only_actually_visible = only_actually_visible
        self.color_cam = [
            'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
            'lime', 'pink', 'brown', 'olive', 'navy', 'maroon', 'teal', 'gold', 'indigo', 'coral',
            'tomato', 'orangered', 'deeppink', 'hotpink', 'springgreen', 'mediumseagreen',
            'royalblue', 'steelblue', 'mediumorchid', 'crimson', 'forestgreen',
            'dodgerblue', 'sienna', 'orchid', 'turquoise', 'limegreen', 'goldenrod',
            'mediumblue', 'mediumvioletred', 'peru', 'chocolate', 'saddlebrown',
            'midnightblue', 'firebrick', 'mediumaquamarine', 'cadetblue',
            'cornflowerblue', 'mediumturquoise', 'lawngreen', 'aqua', 'fuchsia',
            'deepskyblue', 'chartreuse', 'yellowgreen', 'palegreen', 'violet',
            'rosybrown', 'mediumpurple', 'blueviolet', 'tan', 'khaki', 'skyblue',
            'plum', 'salmon', 'peachpuff', 'palevioletred', 'sandybrown', 'powderblue',
            'aquamarine', 'wheat', 'moccasin', 'bisque'
        ]
    
    def set_external_data(self, cameras, images, points3d):
        """외부에서 이미 로드된 데이터를 설정"""
        print("Setting external data...")
        self.cameras = cameras

        # Convert images to the same format as load_images()
        converted_images = {}
        for img_id, img_data in images.items():
            # Reconstruct R from quaternion
            R = self.quaternion_to_rotation_matrix([img_data['qw'], img_data['qx'],
                                                   img_data['qy'], img_data['qz']])
            t = img_data['position'] if 'position' in img_data else np.array([img_data['tx'],
                                                                              img_data['ty'],
                                                                              img_data['tz']])
            # Compute camera center in world coordinates
            camera_center = -R.T @ t

            converted_images[img_id] = {
                'id': img_id,
                'R': R,
                't': t,
                'camera_center': camera_center,
                'camera_id': img_data['camera_id'],
                'name': img_data['name']
            }
        self.images = converted_images

        # points3d 형식을 colmap_visualizer 형식으로 변환
        converted_points3d = {}
        for pt_id, pt_data in points3d.items():
            converted_points3d[pt_id] = {
                'id': pt_id,
                'xyz': pt_data['xyz'],
                'rgb': pt_data['rgb'],
                'error': pt_data['error']
            }
        self.points3d = converted_points3d
        print(f"Set external data: {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3d)} points")
        
    def read_cameras_txt(self):
        """cameras.txt 파싱"""
        cameras_file = os.path.join(self.colmap_path, 'cameras.txt')
        cameras = {}
        
        with open(cameras_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
                
            # print(f"DEBUG: Parsing camera line: '{line.strip()}'")
            parts = line.strip().split()
            # print(f"DEBUG: Parsed parts: {parts}")
            
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            raw_params = [float(x) for x in parts[4:]]
            
            # print(f"DEBUG: Raw params: {raw_params}")
            # print(f"DEBUG: Num params: {len(raw_params)}")
            # print(f"DEBUG: Width/2={width/2:.1f}, Height/2={height/2:.1f}")
            
            # 파라미터 자동 파싱 (utility 함수 사용)
            params = parse_camera_parameters_heuristic(raw_params, width, height, model)
            
            # print(f"DEBUG: Parsed params: fx={params['fx']:.1f}, fy={params['fy']:.1f}, cx={params['cx']:.1f}, cy={params['cy']:.1f}")
            
            cameras[camera_id] = {
                'id': camera_id,
                'model': model, 
                'width': width,
                'height': height,
                'params': params,
                'raw_params': raw_params  # 원본 저장
            }
            
        self.cameras = cameras
        print(f"Loaded {len(cameras)} cameras")
        #exit(1)
        return cameras
    
    def read_images_txt(self):
        """images.txt 파싱"""
        images_file = os.path.join(self.colmap_path, 'images.txt')
        images = {}
        
        with open(images_file, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
                
            # Image info line
            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            
            # Convert quaternion to rotation matrix
            R = self.quaternion_to_rotation_matrix([qw, qx, qy, qz])
            t = np.array([tx, ty, tz])
            
            # Camera center in world coordinates
            camera_center = -R.T @ t
            
            images[image_id] = {
                'id': image_id,
                'R': R,
                't': t,
                'camera_center': camera_center,
                'camera_id': camera_id,
                'name': name
            }
            
            i += 2  # Skip points2d line
            
        self.images = images
        print(f"Loaded {len(images)} images")
        #exit(1)
        return images
    
    def read_points3d_txt(self, point_cloud_format="auto"):
        """points3D 파싱 - 다양한 형식 지원"""
        points_file = os.path.join(self.colmap_path, 'points3D.txt')

        # Define all possible paths
        ply_path = os.path.join(self.colmap_path, "points3D.ply")
        bin_path = os.path.join(self.colmap_path, "points3D.bin")
        txt_path = points_file

        points3d = {}

        print(f"🔍 Loading points3D with format={point_cloud_format}")

        if point_cloud_format == "ply":
            # Load PLY directly
            self._load_points_from_ply(ply_path, points3d)
        elif point_cloud_format == "txt":
            # Load TXT directly
            self._load_points_from_txt(txt_path, points3d)
        elif point_cloud_format == "bin":
            # Load BIN directly
            self._load_points_from_bin(bin_path, points3d)
        else:  # auto mode - try bin -> txt -> ply (same as scene/dataset_readers.py)
            loaded = False

            # Try BIN first
            if os.path.exists(bin_path):
                print(f"Trying to load BIN file: {bin_path}")
                self._load_points_from_bin(bin_path, points3d)
                if len(points3d) > 0:
                    loaded = True
                    print(f"Successfully loaded {len(points3d)} points from BIN file")
            else:
                print(f"BIN file not found: {bin_path}")

            # Try TXT if BIN failed or not found
            if not loaded and os.path.exists(txt_path):
                print(f"Trying to load TXT file: {txt_path}")
                self._load_points_from_txt(txt_path, points3d)
                if len(points3d) > 0:
                    loaded = True
                    print(f"Successfully loaded {len(points3d)} points from TXT file")
            elif not loaded:
                print(f"TXT file not found: {txt_path}")

            # Try PLY as last resort
            if not loaded and os.path.exists(ply_path):
                print(f"Trying to load PLY file: {ply_path}")
                self._load_points_from_ply(ply_path, points3d)
                if len(points3d) > 0:
                    loaded = True
                    print(f"Successfully loaded {len(points3d)} points from PLY file")
            elif not loaded:
                print(f"PLY file not found: {ply_path}")

            if not loaded:
                print(f"No point cloud file found at {bin_path}, {txt_path}, or {ply_path}")
                print("No point cloud data available")
            
        self.points3d = points3d
        print(f"Loaded {len(points3d)} 3D points")
        
        # 관찰되지 않는 포인트 분석
        self.analyze_unobserved_points()

    def _load_points_from_txt(self, txt_path, points3d):
        """Load points from TXT file"""
        if not os.path.exists(txt_path):
            print(f"TXT file not found: {txt_path}")
            return

        print(f"Loading points from TXT: {txt_path}")
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split()
            point_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

            # Z값이 음수인 포인트 제외 (항공 사진의 경우)
            if xyz[2] < 0:
                continue

            rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
            error = float(parts[7])

            # Track 정보 파싱 (이미지와 2D 포인트 ID 쌍)
            track = []
            for i in range(8, len(parts), 2):
                if i + 1 < len(parts):
                    image_id = int(parts[i])
                    point2d_id = int(parts[i + 1])
                    track.append((image_id, point2d_id))

            points3d[point_id] = {
                'id': point_id,
                'xyz': xyz,
                'rgb': rgb,
                'error': error,
                'track': track  # 이 포인트를 관찰한 이미지들
            }

    def _load_points_from_ply(self, ply_path, points3d):
        """Load points from PLY file using fetchPly function"""
        import sys
        import os

        if not os.path.exists(ply_path):
            print(f"PLY file not found: {ply_path}")
            return

        print(f"✅ PLY 파일에서 포인트 로딩: {ply_path}")

        # Import fetchPly function
        scene_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scene')
        if scene_path not in sys.path:
            sys.path.append(scene_path)
        from dataset_readers import fetchPly

        try:
            point_cloud = fetchPly(ply_path)
            print(f"🎯 PLY 파일에서 {len(point_cloud.points)} 개의 포인트를 성공적으로 로딩했습니다!")

            # Convert BasicPointCloud to points3d format
            for i, (xyz, rgb) in enumerate(zip(point_cloud.points, point_cloud.colors)):
                point_id = i + 1  # PLY doesn't have point IDs, so we generate them

                # Z값이 음수인 포인트 제외 (항공 사진의 경우)
                if xyz[2] < 0:
                    continue

                # Convert RGB from [0,1] to [0,255] if needed
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(int)
                else:
                    rgb = rgb.astype(int)

                points3d[point_id] = {
                    'id': point_id,
                    'xyz': xyz,
                    'rgb': rgb,
                    'error': 0.0,  # PLY doesn't have error info
                    'track': []     # PLY doesn't have track info
                }

            print(f"🚀 PLY에서 변환된 최종 포인트 수: {len(points3d)}")

        except Exception as e:
            print(f"❌ PLY 파일 로딩 실패: {e}")
            import traceback
            traceback.print_exc()

    def _load_points_from_bin(self, bin_path, points3d):
        """Load points from BIN file"""
        if not os.path.exists(bin_path):
            print(f"BIN file not found: {bin_path}")
            return

        print(f"Loading points from BIN: {bin_path}")

        try:
            # Import COLMAP binary reader
            import sys
            scene_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scene')
            if scene_path not in sys.path:
                sys.path.append(scene_path)
            from colmap_loader import read_points3D_binary

            # Read binary points
            points3D_data = read_points3D_binary(bin_path)

            for point_id, point_data in points3D_data.items():
                xyz = point_data.xyz
                rgb = point_data.rgb
                error = point_data.error
                track = [(image_id, point2d_idx) for image_id, point2d_idx in zip(point_data.image_ids, point_data.point2D_idxs)]

                # Z값이 음수인 포인트 제외 (항공 사진의 경우)
                if xyz[2] < 0:
                    continue

                points3d[point_id] = {
                    'id': point_id,
                    'xyz': xyz,
                    'rgb': rgb,
                    'error': error,
                    'track': track
                }

            print(f"Successfully loaded {len(points3d)} points from BIN file")

        except Exception as e:
            print(f"❌ BIN 파일 로딩 실패: {e}")
            import traceback
            traceback.print_exc()

    def analyze_unobserved_points(self):
        """3D 포인트를 실제로 카메라에 투영하여 이미지 범위 내에 들어오는지 분석"""
        if not self.points3d or not self.images or not self.cameras:
            return
            
        print("\n" + "="*60)
        print("3D POINT PROJECTION ANALYSIS")
        print("="*60)
        print(f"Total 3D points: {len(self.points3d)}")
        print(f"Total images: {len(self.images)}")
        
        # 각 포인트가 실제로 투영 가능한 이미지 수 계산
        points_projectable = {}  # point_id: [list of image_ids where point is in frame]
        points_outside = {}      # point_id: [list of image_ids where point is outside frame]
        
        # 모든 포인트를 모든 카메라에 투영 테스트
        # Batch projection optimization: process all points for each camera at once
        print("🚀 Using batch projection optimization for faster processing...")

        # Prepare all points as numpy array
        all_points = []
        point_ids = []
        for point_id, point in self.points3d.items():
            all_points.append(point['xyz'])
            point_ids.append(point_id)

        if not all_points:
            print("No points to process")
            return

        all_points = np.array(all_points)  # Shape: (N, 3)

        for image_id, image in self.images.items():
            # Batch project all points to this camera
            camera = self.cameras[image['camera_id']]

            try:
                from utils.projection_utils import project_points_to_camera

                # Camera matrix and distortion
                fx = camera['params']['fx']
                fy = camera['params']['fy'] if 'fy' in camera['params'] else fx
                cx = camera['params']['cx']
                cy = camera['params']['cy']

                camera_matrix = np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]], dtype=np.float64)

                dist_coeffs = None
                if 'distortion' in camera['params'] and camera['params']['distortion']:
                    dist_coeffs = np.array(camera['params']['distortion'], dtype=np.float64)

                # Rotation and translation
                R = image['R']
                t = image['t']

                # Use common projection utility
                result = project_points_to_camera(
                    all_points, R, t, camera_matrix, dist_coeffs,
                    check_behind_camera=False,  # Not checking in original code
                    image_width=camera['width'],
                    image_height=camera['height'],
                    margin_pixels=0
                )

                image_points = result['points_2d']
                in_bounds = result['in_bounds_mask']

                # Assign results to points
                for i, point_id in enumerate(point_ids):
                    if point_id not in points_projectable:
                        points_projectable[point_id] = []
                        points_outside[point_id] = []

                    if in_bounds[i]:
                        points_projectable[point_id].append(image_id)
                    else:
                        points_outside[point_id].append(image_id)

            except Exception as e:
                print(f"Error in batch projection for camera {image_id}: {e}")
                import traceback
                traceback.print_exc()
                # Skip this camera on error
                continue
        
        # 통계 계산
        never_visible = []  # 어떤 이미지에서도 보이지 않는 포인트
        always_outside = []  # 투영은 되지만 항상 프레임 밖인 포인트
        sometimes_visible = []  # 최소 1개 이미지에서 보이는 포인트
        
        for point_id, projectable in points_projectable.items():
            if len(projectable) == 0:
                if len(points_outside[point_id]) > 0:
                    always_outside.append(point_id)
                else:
                    never_visible.append(point_id)
            else:
                sometimes_visible.append(point_id)
        
        print(f"\nProjection results:")
        print(f"  Points visible in at least 1 image: {len(sometimes_visible)}")
        print(f"  Points NEVER visible (always outside frame): {len(always_outside)}")
        print(f"  Points behind all cameras: {len(never_visible)}")
        
        
        # 문제가 있는 포인트들의 위치 분석
        if len(always_outside) > 0:
            outside_xyz = np.array([self.points3d[pid]['xyz'] for pid in always_outside])
            print(f"\nPoints always outside frame - bounding box:")
            print(f"  X: [{outside_xyz[:, 0].min():.2f}, {outside_xyz[:, 0].max():.2f}]")
            print(f"  Y: [{outside_xyz[:, 1].min():.2f}, {outside_xyz[:, 1].max():.2f}]")
            print(f"  Z: [{outside_xyz[:, 2].min():.2f}, {outside_xyz[:, 2].max():.2f}]")
        
        # Track 정보와 실제 투영 가능성 비교
        mismatch_points = set()
        for point_id, point in self.points3d.items():
            track_images = set([img_id for img_id, _ in point['track']])
            projectable = set(points_projectable.get(point_id, []))

            # Track에는 있지만 실제로 프레임 밖인 경우
            if len(track_images - projectable) > 0:
                mismatch_points.add(point_id)

        if len(mismatch_points) > 0:
            print(f"\n⚠️  WARNING: {len(mismatch_points)} points have track info but project outside frame!")

        # only_actually_visible 플래그가 켜져있으면 프레임 밖 포인트 제거
        #print(f'self.only_actually_visible : {self.only_actually_visible}, len(always_outside) : {len(always_outside)}');  exit(1)
        if self.only_actually_visible:
            points_to_remove = set(always_outside) | mismatch_points  # Union of both sets

            if len(points_to_remove) > 0:
                print(f"\n🔧 Removing {len(points_to_remove)} points:")
                print(f"   - Always outside frame: {len(always_outside)}")
                print(f"   - Track/projection mismatch: {len(mismatch_points)}")

                # 제거 전 포인트 수
                original_count = len(self.points3d)

                # 프레임 밖 포인트 제거
                for point_id in points_to_remove:
                    if point_id in self.points3d:
                        del self.points3d[point_id]

                # 제거 후 포인트 수
                remaining_count = len(self.points3d)
                print(f"   Points reduced from {original_count} to {remaining_count}")

                # 제거 후 bounding box 다시 계산
                if remaining_count > 0:
                    remaining_xyz = np.array([p['xyz'] for p in self.points3d.values()])
                    print(f"\n   New bounding box after removal:")
                    print(f"   X: [{remaining_xyz[:, 0].min():.2f}, {remaining_xyz[:, 0].max():.2f}]")
                    print(f"   Y: [{remaining_xyz[:, 1].min():.2f}, {remaining_xyz[:, 1].max():.2f}]")
                    print(f"   Z: [{remaining_xyz[:, 2].min():.2f}, {remaining_xyz[:, 2].max():.2f}]")
            
        # 카메라별 visible points 딕셔너리 생성 (points_projectable을 역변환)
        camera_visible_points = {}
        for camera_id in self.images.keys():
            camera_visible_points[camera_id] = []

        for point_id, visible_cameras in points_projectable.items():
            for camera_id in visible_cameras:
                if camera_id in camera_visible_points:
                    camera_visible_points[camera_id].append(point_id)

        # 클래스 변수로 저장하여 progressive_trainer에서 접근 가능하게 함
        self.camera_visible_points = camera_visible_points

        print(f"\n📷 Camera visible points summary:")
        for camera_id, visible_points in camera_visible_points.items():
            print(f"  Camera {camera_id}: {len(visible_points)} visible points")

        print("="*60)
        #exit(1)

    # DELETED: project_point_to_image
    # This function has been replaced by _check_points_visibility_batch in train_internal.py
    # which provides batch processing for better performance

    def quaternion_to_rotation_matrix(self, q):
        """쿼터니언을 회전 행렬로 변환"""
        w, x, y, z = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
        return R
    
    def create_dtm(self, resolution=1.0):
        """points3D로부터 DTM(Digital Terrain Model) 생성"""
        if not self.points3d:
            raise ValueError("points3D not loaded")
            
        # 모든 3D 점들 추출
        xyz_points = np.array([p['xyz'] for p in self.points3d.values()])
        
        # X, Y, Z 분리
        x_coords = xyz_points[:, 0]
        y_coords = xyz_points[:, 1] 
        z_coords = xyz_points[:, 2]
        
        # Grid 생성
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        xi = np.arange(x_min, x_max, resolution)
        yi = np.arange(y_min, y_max, resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # 3D 점들을 2D Grid로 보간
        zi_grid = griddata((x_coords, y_coords), z_coords, 
                          (xi_grid, yi_grid), method='linear')
        
        # NaN 값들을 nearest neighbor로 채우기
        mask = ~np.isnan(zi_grid)
        if mask.sum() > 0:
            zi_grid[~mask] = griddata((x_coords, y_coords), z_coords,
                                     (xi_grid[~mask], yi_grid[~mask]), 
                                     method='nearest')
        
        self.dtm = {
            'x_grid': xi_grid,
            'y_grid': yi_grid, 
            'z_grid': zi_grid,
            'resolution': resolution,
            'bounds': (x_min, x_max, y_min, y_max)
        }
        
        print(f"Created DTM with resolution {resolution}m, size {zi_grid.shape}")
        
        # DTM과 원래 point cloud의 bounding box 비교
        #self.compare_dtm_vs_pointcloud_bounds()
        
        return self.dtm
    
    def compare_dtm_vs_pointcloud_bounds(self):
        """DTM과 원래 point cloud의 3D bounding box 비교"""
        if not hasattr(self, 'dtm'):
            print("DTM not created yet")
            return
            
        #''' 
        # 원래 point cloud bounding box
        xyz_points = np.array([p['xyz'] for p in self.points3d.values()])
        pc_x_min, pc_x_max = xyz_points[:, 0].min(), xyz_points[:, 0].max()
        pc_y_min, pc_y_max = xyz_points[:, 1].min(), xyz_points[:, 1].max()
        pc_z_min, pc_z_max = xyz_points[:, 2].min(), xyz_points[:, 2].max()
        
        # DTM bounding box
        dtm_x_min, dtm_x_max = self.dtm['x_grid'].min(), self.dtm['x_grid'].max()
        dtm_y_min, dtm_y_max = self.dtm['y_grid'].min(), self.dtm['y_grid'].max()
        dtm_z_min, dtm_z_max = self.dtm['z_grid'].min(), self.dtm['z_grid'].max()
        print("\n" + "="*60)
        print("DTM vs POINT CLOUD BOUNDING BOX COMPARISON")
        print("="*60)
        
        print(f"ORIGINAL POINT CLOUD:")
        print(f"  X: [{pc_x_min:.2f}, {pc_x_max:.2f}] (range: {pc_x_max-pc_x_min:.2f}m)")
        print(f"  Y: [{pc_y_min:.2f}, {pc_y_max:.2f}] (range: {pc_y_max-pc_y_min:.2f}m)")
        print(f"  Z: [{pc_z_min:.2f}, {pc_z_max:.2f}] (range: {pc_z_max-pc_z_min:.2f}m)")
        
        print(f"\nDTM INTERPOLATED:")
        print(f"  X: [{dtm_x_min:.2f}, {dtm_x_max:.2f}] (range: {dtm_x_max-dtm_x_min:.2f}m)")
        print(f"  Y: [{dtm_y_min:.2f}, {dtm_y_max:.2f}] (range: {dtm_y_max-dtm_y_min:.2f}m)")
        print(f"  Z: [{dtm_z_min:.2f}, {dtm_z_max:.2f}] (range: {dtm_z_max-dtm_z_min:.2f}m)")
        
        print(f"\nDIFFERENCES:")
        x_diff = (dtm_x_max - dtm_x_min) - (pc_x_max - pc_x_min)
        y_diff = (dtm_y_max - dtm_y_min) - (pc_y_max - pc_y_min)
        z_diff = (dtm_z_max - dtm_z_min) - (pc_z_max - pc_z_min)
        
        print(f"  X range difference: {x_diff:.2f}m ({'expanded' if x_diff > 0 else 'contracted' if x_diff < 0 else 'same'})")
        print(f"  Y range difference: {y_diff:.2f}m ({'expanded' if y_diff > 0 else 'contracted' if y_diff < 0 else 'same'})")
        print(f"  Z range difference: {z_diff:.2f}m ({'expanded' if z_diff > 0 else 'contracted' if z_diff < 0 else 'same'})")
        # DTM의 Z값이 원래 포인트 범위를 벗어나는지 확인
        z_below_min = (dtm_z_min < pc_z_min)
        z_above_max = (dtm_z_max > pc_z_max)
        
        if z_below_min or z_above_max:
            print(f"\n⚠️  DTM EXTRAPOLATION DETECTED:")
            if z_below_min:
                print(f"   - DTM has Z values below original min: {dtm_z_min:.2f} < {pc_z_min:.2f}")
            if z_above_max:
                print(f"   - DTM has Z values above original max: {dtm_z_max:.2f} > {pc_z_max:.2f}")
        else:
            print(f"\n✅ DTM Z values are within original point cloud range")
            
        print("="*60)
        #'''
        #exit(1)

    def get_camera_corners(self, image_id):
        """이미지의 4개 꼭지점을 3D 공간에서 구하기"""
        if image_id not in self.images:
            raise ValueError(f"Image {image_id} not found")
            
        image = self.images[image_id]
        #print(f'image.keys() : {image.keys()}')
        camera = self.cameras[image['camera_id']]
        
        # 이미지 꼭지점들 (pixel coordinates)
        w, h = camera['width'], camera['height']
        corners_2d = np.array([
            [0, 0, 1],      # 왼쪽 상단
            [w, 0, 1],      # 오른쪽 상단  
            [w, h, 1],      # 오른쪽 하단
            [0, h, 1]       # 왼쪽 하단
        ]).T
        #print(f'corners_2d : {corners_2d}');    print(f'camera : {camera}')

        # Camera intrinsics (자동 파싱된 파라미터 사용)
        fx = camera['params']['fx']
        fy = camera['params']['fy']
        cx = camera['params']['cx']
        cy = camera['params']['cy']
            
        #print('555')
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy], 
            [0, 0, 1]
        ])
        
        #print(f'K : {K}')
        # Normalize coordinates
        '''
        t0 = np.linalg.inv(K);  print(f't0 : {t0}')
        t1 = t0 @ corners_2d;   print(f't1 : {t1}')
        '''
        corners_normalized = np.linalg.inv(K) @ corners_2d
        
        #print(f"corners_normalized : {corners_normalized}")
        #print(f"image['R'] : {image['R']}")
        # Camera center and rotation

        R = image['R']
        #print('888')
        camera_center = image['camera_center']
        #print('999')
        
        # Ray directions in world coordinates
        ray_dirs = R.T @ corners_normalized
        
        # Normalize ray directions to unit vectors
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=0, keepdims=True)
        
        return camera_center, ray_dirs
    
    def get_scene_bounds(self):
        """Point cloud의 확장된 bounding box 계산 (extrapolation 고려)"""
        if not hasattr(self, '_scene_bounds'):
            xyz_points = np.array([p['xyz'] for p in self.points3d.values()])
            
            # 90% percentile bounds 계산
            x_min, x_max = np.percentile(xyz_points[:, 0], [5, 95])
            y_min, y_max = np.percentile(xyz_points[:, 1], [5, 95])
            z_min, z_max = np.percentile(xyz_points[:, 2], [5, 95])
            
            # 카메라 ray가 더 넓은 영역을 커버할 수 있도록 100% 확장
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            x_min_expanded = x_min - x_range * 1.0
            x_max_expanded = x_max + x_range * 1.0
            y_min_expanded = y_min - y_range * 1.0
            y_max_expanded = y_max + y_range * 1.0
            
            self._scene_bounds = {
                'x_range': [x_min_expanded, x_max_expanded],
                'y_range': [y_min_expanded, y_max_expanded], 
                'z_range': [z_min, z_max]
            }
            # print(f"DEBUG: Scene bounds (expanded): X:[{x_min_expanded:.1f},{x_max_expanded:.1f}] Y:[{y_min_expanded:.1f},{y_max_expanded:.1f}] Z:[{z_min:.1f},{z_max:.1f}]")
        
        return self._scene_bounds

    def raycast_to_dtm(self, ray_origin, ray_dir):
        """Ray와 DTM surface의 실제 교차점 찾기 (iterative method)"""
        if not hasattr(self, 'dtm'):
            raise ValueError("DTM not created. Call create_dtm() first")
            
        # Ray가 아래로 향하는지 확인
        # # print(f"DEBUG:       Ray origin: {ray_origin}")
        # # print(f"DEBUG:       Ray direction: {ray_dir}")
        # # print(f"DEBUG:       Ray dir Z: {ray_dir[2]}")
        
        # # print(f"DEBUG:       Ray dir Z: {ray_dir[2]:.3f}")
        if ray_dir[2] >= 0:  # 위로 향하면 아래 지면과 교차하지 않음 (항공 사진의 경우)
            # # print(f"DEBUG:       FAIL: Ray pointing upward (dir_z={ray_dir[2]:.3f})")
            return None, 'upward_ray'
        else:
            # # print(f"DEBUG:       Ray pointing downward (dir_z={ray_dir[2]:.3f}) - continuing...")
            pass
            
        # Scene bounds 가져오기
        bounds = self.get_scene_bounds()
        x_min, x_max = bounds['x_range']
        y_min, y_max = bounds['y_range']
        z_min, z_max = bounds['z_range']
        
        # 간단한 접근: Z축에서만 교차점 계산 (항공 카메라의 경우)
        # 카메라가 scene 위에 있고 아래로 향하므로, Z 축 교차만 고려
        
        # Z축에서 scene bounds와의 교차점들
        t_z_top = (z_max - ray_origin[2]) / ray_dir[2]  # 상단면과 교차
        t_z_bottom = (z_min - ray_origin[2]) / ray_dir[2]  # 하단면과 교차
        
        # # print(f"DEBUG:       t_z_top: {t_z_top}, t_z_bottom: {t_z_bottom}")
        
        # Ray가 아래로 향하므로 t_z_top이 더 작은 값 (먼저 만나는 면)
        t_start = max(0, t_z_top)  # Scene 상단부터 시작
        t_end = t_z_bottom  # Scene 하단까지
        
        # # print(f"DEBUG:       Final t_start: {t_start}, t_end: {t_end}")
        
        # 유효한 범위인지 확인
        if t_start >= t_end or t_end <= 0:
            # # print("DEBUG:       Invalid t range for Z intersection")
            return None, 'out_of_bounds'
            
        # 이 범위에서 XY가 scene bounds 내에 있는지 확인
        start_point = ray_origin + t_start * ray_dir
        end_point = ray_origin + t_end * ray_dir
        
        # # print(f"DEBUG:       Start point: {start_point}")
        # # print(f"DEBUG:       End point: {end_point}")
        
        # Ray가 XY bounds를 지나가는지 확인 (더 관대한 체크)
        # 적어도 ray의 일부가 scene XY 영역을 지나가면 OK
        
        # X, Y 축에서도 교차점 계산
        t_ranges = []
        
        # X축 체크
        if abs(ray_dir[0]) > 1e-6:
            t_x1 = (x_min - ray_origin[0]) / ray_dir[0]
            t_x2 = (x_max - ray_origin[0]) / ray_dir[0]
            t_x_enter = min(t_x1, t_x2)
            t_x_exit = max(t_x1, t_x2)
            # # print(f"DEBUG:       X axis: t_enter={t_x_enter}, t_exit={t_x_exit}")
            t_ranges.append((t_x_enter, t_x_exit))
        # else:
            # # print("DEBUG:       X axis: Ray parallel to X bounds")
            
        # Y축 체크  
        if abs(ray_dir[1]) > 1e-6:
            t_y1 = (y_min - ray_origin[1]) / ray_dir[1]
            t_y2 = (y_max - ray_origin[1]) / ray_dir[1]
            t_y_enter = min(t_y1, t_y2)
            t_y_exit = max(t_y1, t_y2)
            # # print(f"DEBUG:       Y axis: t_enter={t_y_enter}, t_exit={t_y_exit}")
            t_ranges.append((t_y_enter, t_y_exit))
        # else:
            # # print("DEBUG:       Y axis: Ray parallel to Y bounds")
            
        # Z축 범위
        # # print(f"DEBUG:       Z axis: t_enter={t_start}, t_exit={t_end}")
        t_ranges.append((t_start, t_end))
        
        # # print(f"DEBUG:       All t_ranges: {t_ranges}")
        
        # 모든 축의 교집합 계산
        final_t_start = max([r[0] for r in t_ranges] + [0])  # 0보다 큰 값만
        final_t_end = min([r[1] for r in t_ranges])
        
        # # print(f"DEBUG:       XY+Z intersection: t_start={final_t_start}, t_end={final_t_end}")
        
        if final_t_start >= final_t_end:
            # # print(f"DEBUG:       FAIL: No intersection with scene bounds (t={final_t_start:.1f} >= {final_t_end:.1f})")
            return None, 'out_of_bounds'
            
        # 교집합 범위 사용
        t_start = final_t_start
        t_end = final_t_end
            
        # 무한대면 적당한 값으로 제한
        if t_end == float('inf'):
            t_end = 1000
        
        # Binary search로 교차점 찾기
        # # print(f"DEBUG:       Starting binary search with t_start={t_start}, t_end={t_end}")
        
        for i in range(20):  # 최대 20회 반복
            t_mid = (t_start + t_end) / 2
            # # print(f"DEBUG:       Binary search iteration {i+1}: t_mid={t_mid}")
            
            # Ray 위의 점 계산
            ray_point = ray_origin + t_mid * ray_dir
            # # print(f"DEBUG:       Ray point: {ray_point}")
            
            # DTM에서 해당 XY 위치의 Z 값 보간 (최적화된 방법)
            try:
                # # print(f"DEBUG:       Getting DTM Z for XY: [{ray_point[0]:.2f}, {ray_point[1]:.2f}]")
                
                # DTM grid에서 가장 가까운 점들 찾기 (griddata 대신)
                x_grid_flat = self.dtm['x_grid'].flatten()
                y_grid_flat = self.dtm['y_grid'].flatten()
                z_grid_flat = self.dtm['z_grid'].flatten()
                
                # 가장 가까운 점 찾기 (nearest neighbor with extrapolation)
                distances = np.sqrt((x_grid_flat - ray_point[0])**2 + (y_grid_flat - ray_point[1])**2)
                nearest_idx = np.argmin(distances)
                dtm_z = z_grid_flat[nearest_idx]
                
                # 거리 체크 - 너무 멀면 extrapolation 경고
                min_distance = distances[nearest_idx]
                if min_distance > 100:  # 100m 이상 떨어져 있으면 extrapolation
                    # # print(f"DEBUG:       Using extrapolation (distance: {min_distance:.1f}m)")
                    pass
                
                # # print(f"DEBUG:       DTM Z (nearest): {dtm_z}, Ray Z: {ray_point[2]}, distance: {distances[nearest_idx]:.2f}")
                
                if np.isnan(dtm_z):
                    pass
                    # print("DEBUG:       FAIL: DTM Z is NaN at this location")
                    return None, 'dtm_nan'
                    
                # Ray Z와 DTM Z 비교
                z_diff = ray_point[2] - dtm_z
                # # print(f"DEBUG:       Z difference: {z_diff}")
                
                if abs(z_diff) < 50.0:  # 50m 이내 정밀도로 극도로 완화
                    # # print(f"DEBUG:     Found intersection at [{ray_point[0]:.1f}, {ray_point[1]:.1f}, {dtm_z:.1f}]")
                    return np.array([ray_point[0], ray_point[1], dtm_z]), 'success'
                elif z_diff > 0:  # Ray가 DTM보다 위에 있음
                    # # print("DEBUG:       Ray above DTM, moving t_start forward")
                    t_start = t_mid
                else:  # Ray가 DTM보다 아래에 있음
                    # # print("DEBUG:       Ray below DTM, moving t_end backward")
                    t_end = t_mid
                    
            except Exception as e:
                # # print(f"DEBUG:       Exception in griddata: {e}")
                return None, 'dtm_nan'
                
        # 최종 근사값 반환 (binary search 완료 후)
        # print(f"DEBUG:       FAIL: Binary search couldn't converge to 50m precision")
        final_point = ray_origin + t_mid * ray_dir
        try:
            # Nearest neighbor 방식 사용 (griddata 대신)
            x_grid_flat = self.dtm['x_grid'].flatten()
            y_grid_flat = self.dtm['y_grid'].flatten()
            z_grid_flat = self.dtm['z_grid'].flatten()
            
            distances = np.sqrt((x_grid_flat - final_point[0])**2 + (y_grid_flat - final_point[1])**2)
            nearest_idx = np.argmin(distances)
            dtm_z = z_grid_flat[nearest_idx]
            
            # Extrapolation 거리 체크
            if distances[nearest_idx] > 100:
                pass
                # print(f"DEBUG:       Using extrapolation for final point (distance: {distances[nearest_idx]:.1f}m)")
                pass
            
            # print(f"DEBUG:       Final result: [{final_point[0]:.2f}, {final_point[1]:.2f}, {dtm_z:.2f}]")
            return np.array([final_point[0], final_point[1], dtm_z]), 'no_convergence'
        except Exception as e:
            pass
            # print(f"DEBUG:       Error in final approximation: {e}")
            return None, 'dtm_nan'
    
    def visualize_3d_scene(self, save_path='colmap_3d_scene.png', create_nadir_view=False):
        """3D 장면 시각화"""
        # print("DEBUG: visualize_3d_scene() - ENTRY POINT")
        # print(f"DEBUG: save_path = {save_path}")
        
        # print("DEBUG: Checking data availability...")
        # print(f"DEBUG: self.cameras exists: {bool(self.cameras)}")
        # print(f"DEBUG: self.images exists: {bool(self.images)}")
        # print(f"DEBUG: self.points3d exists: {bool(self.points3d)}")
        
        if not all([self.cameras, self.images, self.points3d]):
            pass
            # print("DEBUG: ERROR - Missing required data!")
            raise ValueError("Load all COLMAP data first")
        
        # print("DEBUG: All data available, proceeding...")
        # print(f"DEBUG: Number of cameras: {len(self.cameras)}")
        # print(f"DEBUG: Number of images: {len(self.images)}")
        # print(f"DEBUG: Number of 3D points: {len(self.points3d)}")
        
        # print("DEBUG: Creating matplotlib figure...")
        try:
            fig = plt.figure(figsize=(15, 12))
            # print("DEBUG: Figure created successfully")
        except Exception as e:
            pass
            # print(f"DEBUG: ERROR creating figure: {e}")
            raise
            
        # print("DEBUG: Adding 3D subplot...")
        try:
            ax = fig.add_subplot(111, projection='3d')
            # print("DEBUG: 3D subplot added successfully")
        except Exception as e:
            pass
            # print(f"DEBUG: ERROR adding 3D subplot: {e}")
            raise
        
        # 1. DTM 표시
        # print("DEBUG: Checking DTM availability...")
        if hasattr(self, 'dtm'):
            pass
            # print("DEBUG: DTM exists, adding surface plot...")
            try:
                ax.plot_wireframe(self.dtm['x_grid'], self.dtm['y_grid'], self.dtm['z_grid'],
                                alpha=0.6, color='darkgray', linewidth=0.8)
                # print("DEBUG: DTM wireframe plot added successfully")
            except Exception as e:
                pass
                # print(f"DEBUG: ERROR adding DTM surface: {e}")
                pass
        else:
            pass
            # print("DEBUG: No DTM available")
            pass
        
        # 2. 3D 점들 표시 (생략 - DTM만 사용)
        # print("DEBUG: Skipping 3D points display - using DTM wireframe only")
        
        # 3. 카메라들과 ray casting
        # print("DEBUG: Processing cameras and ray casting...")
        camera_centers = []
        
        # 실패 원인 통계 수집
        failure_stats = {
            'upward_ray': 0,
            'downward_ray': 0,
            'out_of_bounds': 0, 
            'dtm_nan': 0,
            'no_convergence': 0,
            'success': 0
        }
        
        image_count = 0
        # 모든 이미지 처리
        for image_id, image in self.images.items():
            image_count += 1
            # 카메라별 색상 선택 (순환)
            #camera_color = colors[(image_count - 1) % len(colors)]
            camera_color = self.color_cam[image_id % len(self.color_cam)]
            
            # print(f"DEBUG: Processing image {image_count}/{len(self.images)} (ID: {image_id}) - Color: {camera_color}")
            
            try:
                camera_center = image['camera_center']
                # # print(f"DEBUG:   Camera center: {camera_center}")
                camera_centers.append(camera_center)
                
                # 카메라 0의 바라보는 방향 계산 및 출력
                '''
                if image_count == 1:  # 첫 번째 카메라 (카메라 0)
                    R = image['R']
                    # COLMAP에서 카메라의 Z축(viewing direction)은 [0, 0, 1]
                    viewing_direction = R.T @ np.array([0, 0, 1])
                    print(f"\n=== CAMERA 0 VIEWING DIRECTION ===")
                    print(f"Viewing direction (world coords): [{viewing_direction[0]:.6f}, {viewing_direction[1]:.6f}, {viewing_direction[2]:.6f}]")
                    print(f"Z component: {viewing_direction[2]:.6f}")
                    if viewing_direction[2] > 0:
                        print("Camera looking UP (positive Z)")
                    else:
                        print("Camera looking DOWN (negative Z)")
                    print("=== END VIEWING DIRECTION ===\n")
                '''
                # 카메라 위치 표시 (각각 다른 색상)
                ax.scatter(*camera_center, c=camera_color, s=100, marker='^')
                # # print(f"DEBUG:   Camera position plotted successfully")
                
                # 이미지 꼭지점들과 지표면 교점 구하기
                # # print("DEBUG:   Getting camera corners...")
                _, ray_dirs = self.get_camera_corners(image_id)
                # # print(f"DEBUG:   Ray directions shape: {ray_dirs.shape}")
                
                # 각 카메라의 corner ray direction 출력  
                if image_count <= 10:  # 처음 10대 카메라만 출력 (너무 많으면 제한)
                    # corners_normalized도 출력하기 위해 다시 계산
                    camera = self.cameras[image['camera_id']]
                    w, h = camera['width'], camera['height']
                    corners_2d = np.array([
                        [0, 0, 1],      # 왼쪽 상단
                        [w, 0, 1],      # 오른쪽 상단  
                        [w, h, 1],      # 오른쪽 하단
                        [0, h, 1]       # 왼쪽 하단
                    ]).T
                    # Camera intrinsics (자동 파싱된 파라미터 사용)
                    fx = camera['params']['fx']
                    fy = camera['params']['fy']
                    cx = camera['params']['cx']
                    cy = camera['params']['cy']
                        
                    K = np.array([
                        [fx, 0, cx],
                        [0, fy, cy], 
                        [0, 0, 1]
                    ])
                    corners_normalized = np.linalg.inv(K) @ corners_2d
                   
                    '''
                    print(f"\n=== CAMERA {image_count} (ID: {image_id}) CORNER ANALYSIS ===")
                    print(f"Image size: width={w}, height={h}")
                    print(f"Camera params: fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")
                    print(f"Model: {camera['model']}, Raw params: {camera['raw_params']}")
                    print(f"Principal point: ({cx}, {cy})")
                    print(f"Focal length: fx={fx:.1f}, fy={fy:.1f}")
                    
                    print(f"\nPixel corners:")
                    pixel_corners = [[0, 0], [w, 0], [w, h], [0, h]]
                    corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                    for i in range(4):
                        px, py = pixel_corners[i]
                        print(f"Corner {i+1} ({corner_names[i]}): pixel=({px}, {py})")
                    
                    print(f"\nCorners normalized (camera coordinates):")
                    corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                    for i in range(4):
                        norm_coord = corners_normalized[:, i]
                        px, py = pixel_corners[i]
                        # 수동 계산으로 확인
                        manual_nx = (px - cx) / fx
                        manual_ny = (py - cy) / fy
                        print(f"Corner {i+1} ({corner_names[i]}): [{norm_coord[0]:.6f}, {norm_coord[1]:.6f}, {norm_coord[2]:.6f}]")
                        print(f"  -> Manual calc: [({px}-{cx:.1f})/{fx:.1f}, ({py}-{cy:.1f})/{fy:.1f}] = [{manual_nx:.6f}, {manual_ny:.6f}]")
                    print(f"\nRay directions (world coordinates):")
                    corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                    nadir_vector = np.array([0, 0, -1])  # 수직 아래 방향
                    
                    for i in range(4):
                        ray_dir = ray_dirs[:, i]
                        
                        # Off nadir angle 계산 (도 단위)
                        dot_product = np.dot(ray_dir, nadir_vector)
                        off_nadir_angle = np.arccos(np.abs(dot_product)) * 180.0 / np.pi
                        
                        print(f"Corner {i+1} ({corner_names[i]}): [{ray_dir[0]:.6f}, {ray_dir[1]:.6f}, {ray_dir[2]:.6f}]")
                        print(f"  -> Off nadir angle: {off_nadir_angle:.2f}°")
                        if ray_dir[2] > 0:
                            print(f"  -> Ray pointing UP (Z={ray_dir[2]:.6f})")
                        else:
                            print(f"  -> Ray pointing DOWN (Z={ray_dir[2]:.6f})")
                    print(f"=== END CAMERA {image_count} ANALYSIS ===\n")
                    '''
                ground_points = []
                for i in range(4):  # 4개 꼭지점
                    # # print(f"DEBUG:     Raycasting corner {i+1}/4...")
                    result = self.raycast_to_dtm(camera_center, ray_dirs[:, i])
                    
                    if result[0] is not None:  # 성공
                        intersection, reason = result
                        # # print(f"DEBUG:     Intersection found: {intersection}")
                        ground_points.append(intersection)
                        failure_stats[reason] += 1
                        
                        # 카메라에서 지표면으로 ray (카메라별 색상)
                        ax.plot([camera_center[0], intersection[0]],
                               [camera_center[1], intersection[1]],
                               [camera_center[2], intersection[2]], 
                               color=camera_color, alpha=1.0, linewidth=1.0, linestyle='-')
                        # # print(f"DEBUG:     Ray line plotted successfully")
                    else:  # 실패
                        _, reason = result
                        failure_stats[reason] += 1
                        # print(f"DEBUG:     No intersection found for corner {i+1} - Reason: {reason}")
                
                # print(f"DEBUG:   Found {len(ground_points)}/4 ground intersections for camera {image_count}")
               
                '''
                # 카메라 0의 교점들을 콘솔에 출력 (몇 개든 상관없이)
                print(f"\n=== CAMERA {image_count} INTERSECTIONS ({len(ground_points)}/4 found) ===")
                for i, point in enumerate(ground_points):
                    print(f"Intersection {i+1}: [{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}]")
                print("=== END INTERSECTIONS ===\n")
                '''

                # 지표면 사각형 그리기 (2개 이상의 점이 있으면 연결)
                if len(ground_points) >= 2:
                    # # print(f"DEBUG:   Drawing ground polygon with {len(ground_points)} points...")
                    
                    # 모든 점을 연결하여 polygon 형성
                    ground_points_array = np.array(ground_points)
                    
                    # 점들을 순서대로 연결 (카메라별 색상)
                    for i in range(len(ground_points)):
                        j = (i + 1) % len(ground_points)
                        ax.plot([ground_points[i][0], ground_points[j][0]],
                               [ground_points[i][1], ground_points[j][1]], 
                               [ground_points[i][2], ground_points[j][2]],
                               color=camera_color, alpha=0.9, linewidth=0.8)
                    
                    # 교차점 표시 제거 - 4각형만 그리기
                    
                    # # print(f"DEBUG:   Ground polygon drawn with {len(ground_points)} vertices")
                elif len(ground_points) == 1:
                    # 하나의 점만 있을 때는 점으로 표시 (카메라별 색상)
                    ax.scatter(ground_points[0][0], ground_points[0][1], ground_points[0][2], 
                              c=camera_color, s=50, marker='o', alpha=0.8)
                    # print("DEBUG:   Single ground point drawn")
                else:
                    # footprint가 없는 경우 처리
                    pass
                    # print(f"DEBUG:   No ground intersections found for this image")
                    
            except Exception as e:
                pass
                # print(f"DEBUG:   ERROR processing image {image_id}: {e}")
                continue
        
        # print(f"DEBUG: Processed {len(camera_centers)} camera centers")
        
        # 4. 카메라들의 중심 구하기
        # print("DEBUG: Computing scene center...")
        try:
            camera_centers = np.array(camera_centers)
            # print(f"DEBUG: Camera centers array shape: {camera_centers.shape}")
            scene_center = np.mean(camera_centers, axis=0)
            # print(f"DEBUG: Scene center: {scene_center}")
            
            # Scene center 표시는 뒤에서 한 번만 하기
        except Exception as e:
            pass
            # print(f"DEBUG: ERROR computing scene center: {e}")
            raise
        
        # 상공 카메라 제거됨
        
        # 설정
        # print("DEBUG: Setting plot labels and legend...")
        try:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)') 
            ax.set_zlabel('Z (m)')
            
            # 축 스케일 동일하게 설정
            # Get current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim() 
            zlim = ax.get_zlim()
            
            # Calculate ranges
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            z_range = zlim[1] - zlim[0]
            
            # Find maximum range
            max_range = max(x_range, y_range, z_range)
            
            # Set equal ranges centered on current centers
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            z_center = (zlim[0] + zlim[1]) / 2
            
            ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
            
            # 시각화 각도 설정 (더 옆에서 보기)
            ax.view_init(elev=10, azim=60)
            
            # Scene center 표시
            ax.scatter(*scene_center, c='yellow', s=200, marker='*', 
                      edgecolors='black', linewidth=2, label='Scene Center')
            ax.legend()
            ax.set_title('COLMAP 3D Scene Visualization')
            # print("DEBUG: Plot configuration completed")
        except Exception as e:
            pass
            # print(f"DEBUG: ERROR setting plot configuration: {e}")
        
        # print("DEBUG: Preparing to save plot...")
        try:
            plt.tight_layout()
            # print("DEBUG: tight_layout() completed")
        except Exception as e:
            pass
            # print(f"DEBUG: WARNING - tight_layout() failed: {e}")
            
        # print(f"DEBUG: About to save plot to: {save_path}")
        # print(f"DEBUG: Current working directory: {os.getcwd()}")
        
        # 간단한 테스트로 빈 파일이라도 생성해보기
        try:
            with open('test_file.txt', 'w') as f:
                f.write('test')
            # print("DEBUG: Test file creation successful")
        except Exception as e:
            pass
            # print(f"DEBUG: Test file creation failed: {e}")
        
        #'''
        # print(f"DEBUG: Saving to {save_path}...")
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # print(f"DEBUG: Plot saved successfully to {save_path}")
            # print(f"DEBUG: PNG file should be created at: {save_path}")
            
            # 파일이 실제로 생성되었는지 확인
            if os.path.exists(save_path):
                size = os.path.getsize(save_path)
                # print(f"DEBUG: File exists with size: {size} bytes")
            else:
                pass
                # print(f"DEBUG: ERROR - File does not exist: {save_path}")
                
        except Exception as e:
            pass
            # print(f"DEBUG: ERROR saving plot: {e}")
            import traceback
            traceback.print_exc()
            raise
        #'''

        # print("DEBUG: Closing figure...")
        try:
            plt.close()  # 메모리 절약을 위해 figure 닫기
            # print("DEBUG: Figure closed successfully")
        except Exception as e:
            pass
            # print(f"DEBUG: WARNING - error closing figure: {e}")
        
        print(f"3D scene saved to {save_path}")
        
        # Nadir view 추가 생성
        if create_nadir_view:
            nadir_save_path = save_path.replace('_3d_scene_', '_nadir_view_')
            self.create_nadir_view(nadir_save_path)
        
        # 실패 원인 통계 테이블 출력
        print("\n" + "="*50)
        print("RAY CASTING STATISTICS")
        print("="*50)
        total_rays = sum(failure_stats.values())
        print(f"{'Reason':<20} {'Count':<8} {'Percentage':<10}")
        print("-"*40)
        
        reason_names = {
            'success': 'Success',
            'upward_ray': 'Upward Ray',
            'downward_ray': 'Downward Ray',
            'out_of_bounds': 'Out of Bounds', 
            'dtm_nan': 'DTM NaN',
            'no_convergence': 'No Convergence'
        }
        
        for reason, count in failure_stats.items():
            percentage = (count / total_rays * 100) if total_rays > 0 else 0
            print(f"{reason_names[reason]:<20} {count:<8} {percentage:>6.1f}%")
        
        print("-"*40)
        print(f"{'TOTAL':<20} {total_rays:<8} {100.0:>6.1f}%")
        print("="*50)
        
        # print(f"DEBUG: visualize_3d_scene() - EXIT POINT")
        # print(f"DEBUG: Returning scene_center={scene_center}")
        return scene_center
    
    def render_orthographic_view(self, scene_center, 
                                save_path='orthographic_view.png'):
        """상공에서 nadir orthographic projection 렌더링"""
        #print('111') 
        # DTM 기반으로 orthographic view 생성
        if not hasattr(self, 'dtm'):
            raise ValueError("DTM not created")
            
        #print('222') 
        fig, ax = plt.subplots(figsize=(12, 12))
        
        #print('333') 
        # DTM contour map
        contour = ax.contourf(self.dtm['x_grid'], self.dtm['y_grid'], self.dtm['z_grid'],
                             levels=50, cmap='terrain', alpha=0.8)
        
        #print('444') 
        # 카메라 위치들과 footprint 표시
        image_count = 0

        # Prepare points data for footprint computation
        if self.points3d:
            all_points_3d = np.array([pt['xyz'] for pt in self.points3d.values()])
            all_point_ids = list(self.points3d.keys())
        else:
            all_points_3d = np.array([])
            all_point_ids = []

        for image_id, image in self.images.items():
            image_count += 1
            camera_color = self.color_cam[image_id % len(self.color_cam)]

            # Get camera parameters
            camera_center = image['camera_center']
            camera_rotation = image['R']
            camera = self.cameras[image['camera_id']]

            # Camera position marker
            ax.plot(camera_center[0], camera_center[1], '^',
                   color=camera_color, markersize=10, alpha=1.0,
                   label=f'Cam {image_id}' if image_count <= 5 else "")

            # Prepare camera intrinsics
            camera_intrinsics = {
                'fx': camera['params']['fx'],
                'fy': camera['params']['fy'],
                'cx': camera['params']['cx'],
                'cy': camera['params']['cy'],
                'width': camera['width'],
                'height': camera['height']
            }

            # Compute footprint using the new method
            try:
                footprint, points_in_footprint = self.compute_camera_footprint(
                    camera_center,
                    camera_rotation,
                    camera_intrinsics,
                    all_points_3d if len(all_points_3d) > 0 else np.zeros((0, 3)),
                    all_point_ids
                )

                # Draw footprint polygon with filled area
                polygon = plt.Polygon(footprint,
                                     fill=True,               # Fill the polygon
                                     facecolor=camera_color,   # Fill color
                                     alpha=0.2,                # Transparency for fill
                                     edgecolor=camera_color,   # Border color
                                     linewidth=2,              # Border width
                                     linestyle='-')            # Solid line
                ax.add_patch(polygon)

                # Add text label near camera position
                if image_count <= 10:  # Limit labels to avoid clutter
                    ax.text(camera_center[0], camera_center[1], f'{image_id}',
                           fontsize=8, ha='center', va='bottom')

            except Exception as e:
                print(f"Warning: Could not compute footprint for camera {image_id}: {e}")
        
        #print('555') 
        # Scene center 표시
        ax.plot(scene_center[0], scene_center[1], 'y*', 
               markersize=15, label='Scene Center')
        
        #print('666') 
        # 원래 point cloud 점들 표시 (주석처리 - 대량 데이터로 인한 성능 문제)
        # if self.points3d:
        #     xyz_points = np.array([p['xyz'] for p in self.points3d.values()])
        #     ax.scatter(xyz_points[:, 0], xyz_points[:, 1], 
        #               c=xyz_points[:, 2], cmap='viridis', s=0.1, alpha=0.6, 
        #               label='Original Points')
        
        # Aerial camera 제거됨
        
        # 컬러바
        plt.colorbar(contour, ax=ax, label='Elevation (m)')
        
        #print('777')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Orthographic View with Camera Footprints ({image_count} cameras)')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        #print('888') 
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 메모리 절약을 위해 figure 닫기
        
        print(f"Orthographic view saved to {save_path}")

    def create_nadir_view(self, save_path='nadir_view.png'):
        """3D 장면을 위에서 아래로 보는 nadir view로 렌더링"""
        if not all([self.cameras, self.images, self.points3d]):
            raise ValueError("Load all COLMAP data first")
        
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. DTM 표시 (진하게)
        if hasattr(self, 'dtm'):
            ax.plot_wireframe(self.dtm['x_grid'], self.dtm['y_grid'], self.dtm['z_grid'],
                            alpha=0.9, color='black', linewidth=1.5)
            
            '''
            # Nadir view에서 실제 그려지는 DTM 범위 출력
            print("\n" + "="*60)
            print("NADIR VIEW - ACTUAL DTM RENDERING BOUNDS")
            print("="*60)
            print(f"DTM Grid shape: {self.dtm['x_grid'].shape}")
            print(f"X range: [{self.dtm['x_grid'].min():.2f}, {self.dtm['x_grid'].max():.2f}]")
            print(f"Y range: [{self.dtm['y_grid'].min():.2f}, {self.dtm['y_grid'].max():.2f}]")
            print(f"Z range: [{self.dtm['z_grid'].min():.2f}, {self.dtm['z_grid'].max():.2f}]")
            
            # NaN 값 체크
            nan_count = np.isnan(self.dtm['z_grid']).sum()
            total_points = self.dtm['z_grid'].size
            print(f"NaN values in DTM: {nan_count}/{total_points} ({100*nan_count/total_points:.1f}%)")
            
            # 실제 유효한 Z값 범위 (NaN 제외)
            valid_z = self.dtm['z_grid'][~np.isnan(self.dtm['z_grid'])]
            if len(valid_z) > 0:
                print(f"Valid Z range (excluding NaN): [{valid_z.min():.2f}, {valid_z.max():.2f}]")
            print("="*60)
            '''
        # 2. Point cloud 표시
        if self.points3d:
            xyz_points = np.array([pt['xyz'] for pt in self.points3d.values()])
            rgb_points = np.array([pt['rgb'] for pt in self.points3d.values()]) / 255.0  # Normalize to 0-1

            # Point cloud scatter plot
            ax.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2],
                      c=rgb_points, s=0.1, alpha=0.6)
            print(f"Displayed {len(xyz_points)} 3D points")

        # 3. 카메라들과 ray casting (동일한 로직)
        
        image_count = 0
        for image_id, image in self.images.items():
            #print(f'type(image_id) : {type(image_id)}');    exit(1)
            #type(image_id) : <class 'int'>
            image_count += 1
            #camera_color = colors[(image_count - 1) % len(colors)]
            camera_color = self.color_cam[image_id % len(self.color_cam)]
            
            camera_center = image['camera_center']
            ax.scatter(*camera_center, c=camera_color, s=100, marker='^')
            
            # Ray casting 및 footprint 표시 (기존과 동일)
            try:
                _, ray_dirs = self.get_camera_corners(image_id)
                ground_points = []
                
                for i in range(4):
                    result = self.raycast_to_dtm(camera_center, ray_dirs[:, i])
                    if result[0] is not None:
                        intersection, _ = result
                        ground_points.append(intersection)
                        
                if len(ground_points) >= 3:
                    ground_points = np.array(ground_points)
                    
                    # 3D 다각형 그리기
                    for i in range(len(ground_points)):
                        next_i = (i + 1) % len(ground_points)
                        ax.plot([ground_points[i][0], ground_points[next_i][0]],
                               [ground_points[i][1], ground_points[next_i][1]],
                               [ground_points[i][2], ground_points[next_i][2]],
                               color=camera_color, linewidth=3, alpha=0.7)
                        
                    # 카메라에서 ground로 연결선
                    for point in ground_points:
                        ax.plot([camera_center[0], point[0]],
                               [camera_center[1], point[1]], 
                               [camera_center[2], point[2]],
                               color=camera_color, linewidth=1, alpha=0.3)
                        
            except Exception as e:
                pass
                # print(f"DEBUG: Error processing camera {image_id}: {e}")
        
        # 4. Nadir view 설정 (위에서 아래로)
        if hasattr(self, 'dtm'):
            # DTM 중심점 계산
            x_center = (self.dtm['x_grid'].min() + self.dtm['x_grid'].max()) / 2
            y_center = (self.dtm['y_grid'].min() + self.dtm['y_grid'].max()) / 2
            z_max = self.dtm['z_grid'].max()
        else:
            # Point cloud 중심점 사용
            xyz_points = np.array([p['xyz'] for p in self.points3d.values()])
            x_center = xyz_points[:, 0].mean()
            y_center = xyz_points[:, 1].mean()
            z_max = xyz_points[:, 2].max()
        
        # 카메라를 장면 위에 배치하고 아래를 바라보도록 설정
        ax.view_init(elev=90, azim=0)  # 90도 위에서 내려다보기
        
        # 축 범위 설정
        if hasattr(self, 'dtm'):
            x_lim = [self.dtm['x_grid'].min(), self.dtm['x_grid'].max()]
            y_lim = [self.dtm['y_grid'].min(), self.dtm['y_grid'].max()]
            z_lim = [self.dtm['z_grid'].min(), self.dtm['z_grid'].max() + 200]
            
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ''' 
            print(f"\nNADIR VIEW - AXIS LIMITS SET:")
            print(f"X axis: {x_lim}")
            print(f"Y axis: {y_lim}")
            print(f"Z axis: {z_lim}")
            '''
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Scene - Nadir View (Top-Down)')
        
        # Grid 선 제거
        ax.grid(False)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Nadir view saved to {save_path}")

    def create_nadir_view_multi(self, subsets_info, save_path='nadir_view_multi.png', only_selected=False, median_point=None):
        """
        create_nadir_view와 동일하지만 subset 카메라들을 색상으로 구분하여 표시

        Args:
            subsets_info: List of dictionaries, each containing:
                {
                    'name': str,  # Subset name (e.g., 'initial', 'iteration_1')
                    'camera_ids': List[int],  # List of camera IDs in this subset
                    'color': str (optional),  # Color for visualization
                }
            save_path: Output path for the visualization
            only_selected: If True, only show cameras in subsets (hide other cameras)
            median_point: If provided, display median position as a special marker
        """
        if not all([self.cameras, self.images, self.points3d]):
            raise ValueError("Load all COLMAP data first")

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 1. DTM 표시 (진하게) - create_nadir_view와 동일
        if hasattr(self, 'dtm'):
            ax.plot_wireframe(self.dtm['x_grid'], self.dtm['y_grid'], self.dtm['z_grid'],
                            alpha=0.9, color='black', linewidth=1.5)

        # 2. Point cloud 표시 - create_nadir_view와 동일
        if self.points3d:
            xyz_points = np.array([pt['xyz'] for pt in self.points3d.values()])
            rgb_points = np.array([pt['rgb'] for pt in self.points3d.values()]) / 255.0  # Normalize to 0-1

            # Point cloud scatter plot
            ax.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2],
                      c=rgb_points, s=0.1, alpha=0.6)
            print(f"Displayed {len(xyz_points)} 3D ppoints")

        # 3. 카메라 색상 매핑 준비
        camera_color_map_set = {}  # camera_id -> color
        camera_subset_map = {}  # camera_id -> subset_name
        default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

        # Assign colors to cameras based on subsets
        for idx, subset in enumerate(subsets_info):
            subset_name = subset.get('name', f'Subset_{idx}')
            camera_ids = subset.get('camera_ids', [])
            color_set = subset.get('color', default_colors[idx % len(default_colors)])
            print(f'subset_name : {subset_name}, camera_ids : {camera_ids}, color_set : {color_set}');   #exit(1);
            for cam_id in camera_ids:
                #kolor = self.color_cam[cam_id % len(self.color_cam)]
                #camera_color_map[cam_id] = kolor
                camera_color_map_set[cam_id] = color_set
                camera_subset_map[cam_id] = subset_name

        # Default color for cameras not in any subset
        default_camera_color = 'gray'

        # 4. 카메라들과 ray casting - create_nadir_view와 동일한 로직이지만 색상만 다름
        #print(f'self.images.keys() : {self.images.keys()}');  exit(1)
        #self.images.keys() : dict_keys([7, 65, 41, 36, 46])
        for image_id, image in self.images.items():
            #print(f'image_id : {image_id}, camera_color_map : {camera_color_map}');  exit(1)
            #image_id : 7, camera_color_map : {41: 'red', 46: 'red', 36: 'red'}
            # Get color for this camera
            #camera_color = camera_color_map.get(image_id, default_camera_color)
            camera_color = self.color_cam[image_id % len(self.color_cam)]
            camera_color_set = camera_color_map_set.get(image_id, default_camera_color)
            is_in_subset = image_id in camera_color_map_set

            # Skip non-selected cameras if only_selected is True
            if only_selected and not is_in_subset:
                continue

            # Camera marker size and alpha based on whether it's in a subset
            marker_size = 150 if is_in_subset else 50
            marker_alpha = 1.0 if is_in_subset else 0.3
            #print(f'image.keys() : {image.keys()}');   #exit(1)
            #image.keys() : dict_keys(['id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'camera_id', 'name', 'position'])

            # Handle both data structures: camera_center (from load_images) vs position (from progressive_trainer)
            if 'camera_center' in image:
                camera_center = image['camera_center']
                #print('111');
            elif 'position' in image:
                # position is just t vector, need to compute actual camera center
                # Reconstruct rotation matrix from quaternion
                R = self.quaternion_to_rotation_matrix([image['qw'], image['qx'], image['qy'], image['qz']])
                #print('222');
                t = image['position']
                camera_center = -R.T @ t
            else:
                #print('333');
                raise KeyError(f"Image {image_id} has neither 'camera_center' nor 'position' key")

            #print('4444');
            ax.scatter(*camera_center, c=camera_color, s=marker_size, marker='^', alpha=marker_alpha)

            # Annotate camera ID for subset cameras
            #print('5555');
            if is_in_subset:
                #print('555');
                ax.text(camera_center[0], camera_center[1], camera_center[2] + 10,
                       str(image_id), fontsize=10, color=camera_color, weight='bold')

            #print('6666');
            # Ray casting 및 footprint 표시 (기존과 동일)
            try:
                #print('zzz');
                _, ray_dirs = self.get_camera_corners(image_id)
                #print('aaa');
                ground_points = []
                
                for i in range(4):
                    result = self.raycast_to_dtm(camera_center, ray_dirs[:, i])
                    if result[0] is not None:
                        intersection, _ = result
                        ground_points.append(intersection)

                #print('bbb');
                if len(ground_points) >= 3:
                    ground_points = np.array(ground_points)

                    # Footprint line width and alpha based on subset membership
                    footprint_linewidth = 3.0 if is_in_subset else 1.0  # Increased thickness
                    footprint_alpha = 0.9 if is_in_subset else 0.4

                    # 3D 다각형 그리기
                    for i in range(len(ground_points)):
                        next_i = (i + 1) % len(ground_points)
                        ax.plot([ground_points[i][0], ground_points[next_i][0]],
                               [ground_points[i][1], ground_points[next_i][1]],
                               [ground_points[i][2], ground_points[next_i][2]],
                               color=camera_color, linewidth = footprint_linewidth * 2,alpha=footprint_alpha)
                        ax.plot([ground_points[i][0], ground_points[next_i][0]],
                               [ground_points[i][1], ground_points[next_i][1]],
                               [ground_points[i][2], ground_points[next_i][2]],
                               color=camera_color_set, linewidth = footprint_linewidth * 0.6,alpha=footprint_alpha)

                    # 카메라에서 ground로 연결선
                    for point in ground_points:
                        ax.plot([camera_center[0], point[0]],
                               [camera_center[1], point[1]],
                               [camera_center[2], point[2]],
                               color=camera_color, linewidth=footprint_linewidth, alpha=footprint_alpha)
                        ax.plot([camera_center[0], point[0]],
                               [camera_center[1], point[1]],
                               [camera_center[2], point[2]],
                               color=camera_color_set, linewidth=footprint_linewidth * 0.3, alpha=footprint_alpha)

                #print('ccc');
            except Exception as e:
                if hasattr(self, 'debug') and self.debug:
                    pass
                    # print(f"DEBUG: Error processing camera {image_id}: {e}")

        #print('aaaa');  #exit(1)
        # 5. Nadir view 설정 (위에서 아래로) - create_nadir_view와 동일
        ax.view_init(elev=90, azim=0)

        # 6. 축 범위 설정 - create_nadir_view와 동일
        if hasattr(self, 'dtm'):
            x_lim = [self.dtm['x_grid'].min(), self.dtm['x_grid'].max()]
            y_lim = [self.dtm['y_grid'].min(), self.dtm['y_grid'].max()]
            z_lim = [self.dtm['z_grid'].min(), self.dtm['z_grid'].max() + 200]
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
        else:
            # Point cloud 중심점 사용
            xyz_points = np.array([p['xyz'] for p in self.points3d.values()])
            x_center = xyz_points[:, 0].mean()
            y_center = xyz_points[:, 1].mean()
            z_max = xyz_points[:, 2].max()

        # 7. Labels and formatting - create_nadir_view와 유사
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Build title with subset information
        title = '3D Scene - Nadir View (Multi-Subset)\n'
        title += f'Total: {len(self.images)} cameras, {len(self.points3d)} points\n'

        # Add subset statistics to title
        subset_stats = []
        for subset in subsets_info:
            name = subset.get('name', 'Unknown')
            n_cams = len(subset.get('camera_ids', []))
            subset_stats.append(f'{name}: {n_cams} cameras')

        if subset_stats:
            title += 'Subsets: ' + ', '.join(subset_stats)

        ax.set_title(title)

        # Grid 선 제거 - create_nadir_view와 동일
        ax.grid(False)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        # Legend for subsets
        from matplotlib.patches import Patch
        legend_elements = []
        for idx, subset in enumerate(subsets_info):
            color = subset.get('color', default_colors[idx % len(default_colors)])
            name = subset.get('name', f'Subset_{idx}')
            legend_elements.append(Patch(facecolor=color, label=name))

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')

        # Add median point if provided
        if median_point is not None:
            ax.scatter(median_point[0], median_point[1], median_point[2],
                      c='gold', s=200, marker='*', alpha=1.0,
                      edgecolors='black', linewidth=2,
                      label='Median Position')
            print(f"Added median point at: ({median_point[0]:.2f}, {median_point[1]:.2f}, {median_point[2]:.2f})")

        # Save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        #print(f"Nadir multi view saved to {save_path}")
        #exit(1)

    def compute_camera_footprint(self,
                                camera_center: np.ndarray,
                                camera_rotation: np.ndarray,
                                camera_intrinsics: dict,
                                points_3d: np.ndarray,
                                point_ids: list) -> tuple:
        """
        Compute camera footprint on DTM surface from camera parameters

        Args:
            camera_center: [3] - Camera position in world coordinates (x, y, z)
            camera_rotation: [3, 3] - Rotation matrix R (world -> camera)
            camera_intrinsics: dict - {'fx': float, 'fy': float, 'cx': float, 'cy': float, 'width': int, 'height': int}
            points_3d: [N, 3] - 3D point coordinates (x, y, z)
            point_ids: [N] - Point IDs corresponding to points_3d

        Returns:
            footprint: [4, 2] - Footprint rectangle corners on DTM (x, y)
            points_in_footprint: List[int] - Point IDs inside the footprint
        """
        if not hasattr(self, 'dtm'):
            raise ValueError("DTM not created. Call create_dtm() first")

        # Extract camera parameters
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        width = camera_intrinsics['width']
        height = camera_intrinsics['height']

        # Camera intrinsic matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # Image corner coordinates (pixel coordinates)
        corners_2d = np.array([
            [0, 0, 1],        # Top-left
            [width, 0, 1],    # Top-right
            [width, height, 1], # Bottom-right
            [0, height, 1]    # Bottom-left
        ]).T

        # Normalize coordinates to camera space
        corners_normalized = np.linalg.inv(K) @ corners_2d

        # Transform ray directions to world coordinates
        ray_dirs = camera_rotation.T @ corners_normalized

        # Normalize ray directions to unit vectors
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=0, keepdims=True)

        # Compute footprint from ray-DTM intersections
        corners_3d = []
        for i in range(4):  # 4 corners
            ray_dir = ray_dirs[:, i].copy()

            # For aerial photos: if ray points upward, flip Z direction
            if ray_dir[2] > 0:
                ray_dir[2] = -ray_dir[2]  # Flip Z component to point downward

            # Ray-cast to DTM surface
            intersection, status = self.raycast_to_dtm(camera_center, ray_dir)
            if intersection is not None:
                corners_3d.append(intersection[:2])  # Only (x, y) coordinates

        if len(corners_3d) != 4:
            raise RuntimeError(
                f"Failed to compute footprint from DTM ray-casting. "
                f"Got {len(corners_3d)} corners but need exactly 4 for rectangular footprint."
            )

        # Create footprint polygon
        footprint = np.array(corners_3d)

        # Filter points within footprint using polygon containment
        from matplotlib.path import Path
        poly_path = Path(footprint)

        # Extract (x, y) coordinates from points_3d
        points_2d = points_3d[:, :2]  # Only x, y coordinates
        mask = poly_path.contains_points(points_2d)

        # Get point IDs that are within the footprint
        points_in_footprint = [point_ids[i] for i, is_inside in enumerate(mask) if is_inside]

        return footprint, points_in_footprint

def main():
    """메인 실행 함수"""
    # COLMAP 데이터 경로 설정
    colmap_path = "/path/to/colmap/sparse/0"  # 실제 경로로 변경
    
    # 시각화 객체 생성
    viz = COLMAPVisualizer(colmap_path)
    
    try:
        # 1. COLMAP 데이터 로드
        print("Loading COLMAP data...")
        viz.read_cameras_txt()
        viz.read_images_txt() 
        viz.read_points3d_txt()
        
        # 2. DTM 생성
        print("Creating DTM...")
        viz.create_dtm(resolution=1.0)
        
        # 3. 3D 시각화
        print("Creating 3D visualization...")
        scene_center, aerial_camera = viz.visualize_3d_scene()
        
        # 4. Orthographic 렌더링
        print("Rendering orthographic view...")
        viz.render_orthographic_view(aerial_camera, scene_center)
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
