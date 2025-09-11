#!/usr/bin/env python3
"""
COLMAP Output 3D Visualization Tool
카메라 위치, 이미지 raycast, DTM 생성 및 상공 렌더링
"""

print("DEBUG: colmap_visualizer.py starting...")

import numpy as np
print("DEBUG: numpy imported")

import matplotlib
print("DEBUG: matplotlib imported")
matplotlib.use('Agg')  # GUI 없는 백엔드 사용 (서버 환경)
print("DEBUG: matplotlib backend set to Agg")

import matplotlib.pyplot as plt
print("DEBUG: matplotlib.pyplot imported")

from mpl_toolkits.mplot3d import Axes3D
print("DEBUG: Axes3D imported")

import matplotlib.patches as patches
print("DEBUG: matplotlib.patches imported")

print("DEBUG: About to import scipy modules...")
from scipy.spatial import cKDTree
print("DEBUG: cKDTree imported")

from scipy.interpolate import griddata
print("DEBUG: griddata imported")

from scipy.spatial.distance import cdist
print("DEBUG: cdist imported")

import os
print("DEBUG: os imported")

import struct
print("DEBUG: struct imported")

print("DEBUG: All imports completed in colmap_visualizer.py")

class COLMAPVisualizer:
    def __init__(self, colmap_path):
        print(f"DEBUG: COLMAPVisualizer.__init__ called with path: {colmap_path}")
        self.colmap_path = colmap_path
        self.cameras = {}
        self.images = {}
        self.points3d = {}
        print("DEBUG: COLMAPVisualizer.__init__ completed")
        
    def read_cameras_txt(self):
        """cameras.txt 파싱"""
        cameras_file = os.path.join(self.colmap_path, 'cameras.txt')
        cameras = {}
        
        with open(cameras_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
                
            print(f"DEBUG: Parsing camera line: '{line.strip()}'")
            parts = line.strip().split()
            print(f"DEBUG: Parsed parts: {parts}")
            
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            raw_params = [float(x) for x in parts[4:]]
            
            print(f"DEBUG: Raw params: {raw_params}")
            print(f"DEBUG: Num params: {len(raw_params)}")
            print(f"DEBUG: Width/2={width/2:.1f}, Height/2={height/2:.1f}")
            
            # 파라미터 자동 파싱
            params = {}
            if len(raw_params) >= 4:
                # 4개 이상: fx, fy, cx, cy 순서 가능성 (PINHOLE 등)
                # cx, cy가 width/2, height/2에 가까운지 체크
                potential_cx_cy_pairs = [
                    (raw_params[2], raw_params[3]),  # 일반적인 fx,fy,cx,cy 순서
                    (raw_params[1], raw_params[2]),  # f,cx,cy,... 순서  
                ]
                
                best_match = None
                best_score = float('inf')
                
                for i, (cx, cy) in enumerate(potential_cx_cy_pairs):
                    cx_error = abs(cx - width/2)
                    cy_error = abs(cy - height/2)
                    score = cx_error + cy_error
                    print(f"DEBUG: Pattern {i}: cx={cx:.1f}, cy={cy:.1f}, error={score:.1f}")
                    
                    if score < best_score:
                        best_score = score
                        best_match = i
                
                print(f"DEBUG: Best match: pattern {best_match}")
                
                if best_match == 0:  # fx,fy,cx,cy 순서
                    params = {
                        'fx': raw_params[0],
                        'fy': raw_params[1], 
                        'cx': raw_params[2],
                        'cy': raw_params[3],
                        'distortion': raw_params[4:] if len(raw_params) > 4 else []
                    }
                elif best_match == 1:  # f,cx,cy 순서
                    params = {
                        'fx': raw_params[0],
                        'fy': raw_params[0],  # 단일 focal length
                        'cx': raw_params[1],
                        'cy': raw_params[2],
                        'distortion': raw_params[3:] if len(raw_params) > 3 else []
                    }
            else:
                # 3개: f, cx, cy 순서 가능성
                if len(raw_params) == 3:
                    params = {
                        'fx': raw_params[0],
                        'fy': raw_params[0],
                        'cx': raw_params[1], 
                        'cy': raw_params[2],
                        'distortion': []
                    }
            
            print(f"DEBUG: Parsed params: fx={params['fx']:.1f}, fy={params['fy']:.1f}, cx={params['cx']:.1f}, cy={params['cy']:.1f}")
            
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
        return images
    
    def read_points3d_txt(self):
        """points3D.txt 파싱"""
        points_file = os.path.join(self.colmap_path, 'points3D.txt')
        points3d = {}
        
        with open(points_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
                
            parts = line.strip().split()
            point_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
            error = float(parts[7])
            
            points3d[point_id] = {
                'id': point_id,
                'xyz': xyz,
                'rgb': rgb,
                'error': error
            }
            
        self.points3d = points3d
        print(f"Loaded {len(points3d)} 3D points")
        return points3d
        
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
        return self.dtm
    
    def get_camera_corners(self, image_id):
        """이미지의 4개 꼭지점을 3D 공간에서 구하기"""
        if image_id not in self.images:
            raise ValueError(f"Image {image_id} not found")
            
        image = self.images[image_id]
        camera = self.cameras[image['camera_id']]
        
        # 이미지 꼭지점들 (pixel coordinates)
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
        
        # Normalize coordinates
        corners_normalized = np.linalg.inv(K) @ corners_2d
        
        # Camera center and rotation
        R = image['R']
        camera_center = image['camera_center']
        
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
            
            # 카메라 ray가 더 넓은 영역을 커버할 수 있도록 500% 확장
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            x_min_expanded = x_min - x_range * 5.0
            x_max_expanded = x_max + x_range * 5.0
            y_min_expanded = y_min - y_range * 5.0
            y_max_expanded = y_max + y_range * 5.0
            
            self._scene_bounds = {
                'x_range': [x_min_expanded, x_max_expanded],
                'y_range': [y_min_expanded, y_max_expanded], 
                'z_range': [z_min, z_max]
            }
            print(f"DEBUG: Scene bounds (expanded): X:[{x_min_expanded:.1f},{x_max_expanded:.1f}] Y:[{y_min_expanded:.1f},{y_max_expanded:.1f}] Z:[{z_min:.1f},{z_max:.1f}]")
        
        return self._scene_bounds

    def raycast_to_dtm(self, ray_origin, ray_dir):
        """Ray와 DTM surface의 실제 교차점 찾기 (iterative method)"""
        if not hasattr(self, 'dtm'):
            raise ValueError("DTM not created. Call create_dtm() first")
            
        # Ray가 아래로 향하는지 확인
        # print(f"DEBUG:       Ray origin: {ray_origin}")
        # print(f"DEBUG:       Ray direction: {ray_dir}")
        # print(f"DEBUG:       Ray dir Z: {ray_dir[2]}")
        
        print(f"DEBUG:       Ray dir Z: {ray_dir[2]:.3f}")
        if ray_dir[2] >= 0:  # 위로 향하면 아래 지면과 교차하지 않음 (항공 사진의 경우)
            print(f"DEBUG:       FAIL: Ray pointing upward (dir_z={ray_dir[2]:.3f})")
            return None, 'upward_ray'
        else:
            print(f"DEBUG:       Ray pointing downward (dir_z={ray_dir[2]:.3f}) - continuing...")
            
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
        
        print(f"DEBUG:       t_z_top: {t_z_top}, t_z_bottom: {t_z_bottom}")
        
        # Ray가 아래로 향하므로 t_z_top이 더 작은 값 (먼저 만나는 면)
        t_start = max(0, t_z_top)  # Scene 상단부터 시작
        t_end = t_z_bottom  # Scene 하단까지
        
        # print(f"DEBUG:       Final t_start: {t_start}, t_end: {t_end}")
        
        # 유효한 범위인지 확인
        if t_start >= t_end or t_end <= 0:
            # print("DEBUG:       Invalid t range for Z intersection")
            return None, 'out_of_bounds'
            
        # 이 범위에서 XY가 scene bounds 내에 있는지 확인
        start_point = ray_origin + t_start * ray_dir
        end_point = ray_origin + t_end * ray_dir
        
        # print(f"DEBUG:       Start point: {start_point}")
        # print(f"DEBUG:       End point: {end_point}")
        
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
            # print(f"DEBUG:       X axis: t_enter={t_x_enter}, t_exit={t_x_exit}")
            t_ranges.append((t_x_enter, t_x_exit))
        # else:
            # print("DEBUG:       X axis: Ray parallel to X bounds")
            
        # Y축 체크  
        if abs(ray_dir[1]) > 1e-6:
            t_y1 = (y_min - ray_origin[1]) / ray_dir[1]
            t_y2 = (y_max - ray_origin[1]) / ray_dir[1]
            t_y_enter = min(t_y1, t_y2)
            t_y_exit = max(t_y1, t_y2)
            # print(f"DEBUG:       Y axis: t_enter={t_y_enter}, t_exit={t_y_exit}")
            t_ranges.append((t_y_enter, t_y_exit))
        # else:
            # print("DEBUG:       Y axis: Ray parallel to Y bounds")
            
        # Z축 범위
        # print(f"DEBUG:       Z axis: t_enter={t_start}, t_exit={t_end}")
        t_ranges.append((t_start, t_end))
        
        # print(f"DEBUG:       All t_ranges: {t_ranges}")
        
        # 모든 축의 교집합 계산
        final_t_start = max([r[0] for r in t_ranges] + [0])  # 0보다 큰 값만
        final_t_end = min([r[1] for r in t_ranges])
        
        # print(f"DEBUG:       XY+Z intersection: t_start={final_t_start}, t_end={final_t_end}")
        
        if final_t_start >= final_t_end:
            # print(f"DEBUG:       FAIL: No intersection with scene bounds (t={final_t_start:.1f} >= {final_t_end:.1f})")
            return None, 'out_of_bounds'
            
        # 교집합 범위 사용
        t_start = final_t_start
        t_end = final_t_end
            
        # 무한대면 적당한 값으로 제한
        if t_end == float('inf'):
            t_end = 1000
        
        # Binary search로 교차점 찾기
        # print(f"DEBUG:       Starting binary search with t_start={t_start}, t_end={t_end}")
        
        for i in range(20):  # 최대 20회 반복
            t_mid = (t_start + t_end) / 2
            # print(f"DEBUG:       Binary search iteration {i+1}: t_mid={t_mid}")
            
            # Ray 위의 점 계산
            ray_point = ray_origin + t_mid * ray_dir
            # print(f"DEBUG:       Ray point: {ray_point}")
            
            # DTM에서 해당 XY 위치의 Z 값 보간 (최적화된 방법)
            try:
                # print(f"DEBUG:       Getting DTM Z for XY: [{ray_point[0]:.2f}, {ray_point[1]:.2f}]")
                
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
                    # print(f"DEBUG:       Using extrapolation (distance: {min_distance:.1f}m)")
                    pass
                
                # print(f"DEBUG:       DTM Z (nearest): {dtm_z}, Ray Z: {ray_point[2]}, distance: {distances[nearest_idx]:.2f}")
                
                if np.isnan(dtm_z):
                    print("DEBUG:       FAIL: DTM Z is NaN at this location")
                    return None, 'dtm_nan'
                    
                # Ray Z와 DTM Z 비교
                z_diff = ray_point[2] - dtm_z
                # print(f"DEBUG:       Z difference: {z_diff}")
                
                if abs(z_diff) < 50.0:  # 50m 이내 정밀도로 극도로 완화
                    print(f"DEBUG:     Found intersection at [{ray_point[0]:.1f}, {ray_point[1]:.1f}, {dtm_z:.1f}]")
                    return np.array([ray_point[0], ray_point[1], dtm_z]), 'success'
                elif z_diff > 0:  # Ray가 DTM보다 위에 있음
                    # print("DEBUG:       Ray above DTM, moving t_start forward")
                    t_start = t_mid
                else:  # Ray가 DTM보다 아래에 있음
                    # print("DEBUG:       Ray below DTM, moving t_end backward")
                    t_end = t_mid
                    
            except Exception as e:
                # print(f"DEBUG:       Exception in griddata: {e}")
                return None, 'dtm_nan'
                
        # 최종 근사값 반환 (binary search 완료 후)
        print(f"DEBUG:       FAIL: Binary search couldn't converge to 50m precision")
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
                print(f"DEBUG:       Using extrapolation for final point (distance: {distances[nearest_idx]:.1f}m)")
            
            print(f"DEBUG:       Final result: [{final_point[0]:.2f}, {final_point[1]:.2f}, {dtm_z:.2f}]")
            return np.array([final_point[0], final_point[1], dtm_z]), 'no_convergence'
        except Exception as e:
            print(f"DEBUG:       Error in final approximation: {e}")
            return None, 'dtm_nan'
    
    def visualize_3d_scene(self, save_path='colmap_3d_scene.png'):
        """3D 장면 시각화"""
        print("DEBUG: visualize_3d_scene() - ENTRY POINT")
        print(f"DEBUG: save_path = {save_path}")
        
        print("DEBUG: Checking data availability...")
        print(f"DEBUG: self.cameras exists: {bool(self.cameras)}")
        print(f"DEBUG: self.images exists: {bool(self.images)}")
        print(f"DEBUG: self.points3d exists: {bool(self.points3d)}")
        
        if not all([self.cameras, self.images, self.points3d]):
            print("DEBUG: ERROR - Missing required data!")
            raise ValueError("Load all COLMAP data first")
        
        print("DEBUG: All data available, proceeding...")
        print(f"DEBUG: Number of cameras: {len(self.cameras)}")
        print(f"DEBUG: Number of images: {len(self.images)}")
        print(f"DEBUG: Number of 3D points: {len(self.points3d)}")
        
        print("DEBUG: Creating matplotlib figure...")
        try:
            fig = plt.figure(figsize=(15, 12))
            print("DEBUG: Figure created successfully")
        except Exception as e:
            print(f"DEBUG: ERROR creating figure: {e}")
            raise
            
        print("DEBUG: Adding 3D subplot...")
        try:
            ax = fig.add_subplot(111, projection='3d')
            print("DEBUG: 3D subplot added successfully")
        except Exception as e:
            print(f"DEBUG: ERROR adding 3D subplot: {e}")
            raise
        
        # 1. DTM 표시
        print("DEBUG: Checking DTM availability...")
        if hasattr(self, 'dtm'):
            print("DEBUG: DTM exists, adding surface plot...")
            try:
                ax.plot_wireframe(self.dtm['x_grid'], self.dtm['y_grid'], self.dtm['z_grid'],
                                alpha=0.6, color='darkgray', linewidth=0.8)
                print("DEBUG: DTM wireframe plot added successfully")
            except Exception as e:
                print(f"DEBUG: ERROR adding DTM surface: {e}")
        else:
            print("DEBUG: No DTM available")
        
        # 2. 3D 점들 표시 (생략 - DTM만 사용)
        print("DEBUG: Skipping 3D points display - using DTM wireframe only")
        
        # 3. 카메라들과 ray casting
        print("DEBUG: Processing cameras and ray casting...")
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
        
        # 카메라별 색상 생성 (색상 팔레트)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 
                 'lime', 'pink', 'brown', 'gray', 'olive', 'navy', 'maroon', 'teal',
                 'silver', 'gold', 'indigo', 'coral']
        
        image_count = 0
        # 모든 이미지 처리
        for image_id, image in self.images.items():
            image_count += 1
            # 카메라별 색상 선택 (순환)
            camera_color = colors[(image_count - 1) % len(colors)]
            
            print(f"DEBUG: Processing image {image_count}/{len(self.images)} (ID: {image_id}) - Color: {camera_color}")
            
            try:
                camera_center = image['camera_center']
                print(f"DEBUG:   Camera center: {camera_center}")
                camera_centers.append(camera_center)
                
                # 카메라 0의 바라보는 방향 계산 및 출력
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
                
                # 카메라 위치 표시 (각각 다른 색상)
                ax.scatter(*camera_center, c=camera_color, s=100, marker='^')
                print(f"DEBUG:   Camera position plotted successfully")
                
                # 이미지 꼭지점들과 지표면 교점 구하기
                print("DEBUG:   Getting camera corners...")
                _, ray_dirs = self.get_camera_corners(image_id)
                print(f"DEBUG:   Ray directions shape: {ray_dirs.shape}")
                
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
                    
                    print(f"\n=== CAMERA {image_count-1} (ID: {image_id}) CORNER ANALYSIS ===")
                    print(f"Image size: width={w}, height={h}")
                    print(f"Camera params: fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")
                    print(f"Model: {camera['model']}, Raw params: {camera['raw_params']}")
                    print(f"Principal point: ({cx}, {cy})")
                    print(f"Focal length: fx={fx:.1f}, fy={fy:.1f}")
                    
                    print(f"\nPixel corners:")
                    pixel_corners = [[0, 0], [w, 0], [w, h], [0, h]]
                    for i in range(4):
                        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                        px, py = pixel_corners[i]
                        print(f"Corner {i+1} ({corner_names[i]}): pixel=({px}, {py})")
                    
                    print(f"\nCorners normalized (camera coordinates):")
                    for i in range(4):
                        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                        norm_coord = corners_normalized[:, i]
                        px, py = pixel_corners[i]
                        # 수동 계산으로 확인
                        manual_nx = (px - cx) / fx
                        manual_ny = (py - cy) / fy
                        print(f"Corner {i+1} ({corner_names[i]}): [{norm_coord[0]:.6f}, {norm_coord[1]:.6f}, {norm_coord[2]:.6f}]")
                        print(f"  -> Manual calc: [({px}-{cx:.1f})/{fx:.1f}, ({py}-{cy:.1f})/{fy:.1f}] = [{manual_nx:.6f}, {manual_ny:.6f}]")
                    
                    print(f"\nRay directions (world coordinates):")
                    nadir_vector = np.array([0, 0, -1])  # 수직 아래 방향
                    
                    for i in range(4):
                        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
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
                    print(f"=== END CAMERA {image_count-1} ANALYSIS ===\n")
                
                ground_points = []
                for i in range(4):  # 4개 꼭지점
                    print(f"DEBUG:     Raycasting corner {i+1}/4...")
                    result = self.raycast_to_dtm(camera_center, ray_dirs[:, i])
                    
                    if result[0] is not None:  # 성공
                        intersection, reason = result
                        print(f"DEBUG:     Intersection found: {intersection}")
                        ground_points.append(intersection)
                        failure_stats[reason] += 1
                        
                        # 카메라에서 지표면으로 ray (카메라별 색상)
                        ax.plot([camera_center[0], intersection[0]],
                               [camera_center[1], intersection[1]],
                               [camera_center[2], intersection[2]], 
                               color=camera_color, alpha=1.0, linewidth=1.0, linestyle='-')
                        print(f"DEBUG:     Ray line plotted successfully")
                    else:  # 실패
                        _, reason = result
                        failure_stats[reason] += 1
                        print(f"DEBUG:     No intersection found for corner {i+1} - Reason: {reason}")
                
                print(f"DEBUG:   Found {len(ground_points)}/4 ground intersections for camera {image_count}")
                
                # 카메라 0의 교점들을 콘솔에 출력 (몇 개든 상관없이)
                print(f"\n=== CAMERA 0 INTERSECTIONS ({len(ground_points)}/4 found) ===")
                for i, point in enumerate(ground_points):
                    print(f"Intersection {i+1}: [{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}]")
                print("=== END INTERSECTIONS ===\n")
                
                # 지표면 사각형 그리기 (2개 이상의 점이 있으면 연결)
                if len(ground_points) >= 2:
                    print(f"DEBUG:   Drawing ground polygon with {len(ground_points)} points...")
                    
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
                    
                    print(f"DEBUG:   Ground polygon drawn with {len(ground_points)} vertices")
                elif len(ground_points) == 1:
                    # 하나의 점만 있을 때는 점으로 표시 (카메라별 색상)
                    ax.scatter(ground_points[0][0], ground_points[0][1], ground_points[0][2], 
                              c=camera_color, s=50, marker='o', alpha=0.8)
                    print("DEBUG:   Single ground point drawn")
                else:
                    # footprint가 없는 경우 처리
                    pass
                    print(f"DEBUG:   No ground intersections found for this image")
                    
            except Exception as e:
                print(f"DEBUG:   ERROR processing image {image_id}: {e}")
                continue
        
        print(f"DEBUG: Processed {len(camera_centers)} camera centers")
        
        # 4. 카메라들의 중심 구하기
        print("DEBUG: Computing scene center...")
        try:
            camera_centers = np.array(camera_centers)
            print(f"DEBUG: Camera centers array shape: {camera_centers.shape}")
            scene_center = np.mean(camera_centers, axis=0)
            print(f"DEBUG: Scene center: {scene_center}")
            
            # Scene center 표시는 뒤에서 한 번만 하기
        except Exception as e:
            print(f"DEBUG: ERROR computing scene center: {e}")
            raise
        
        # 상공 카메라 제거됨
        
        # 설정
        print("DEBUG: Setting plot labels and legend...")
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
            print("DEBUG: Plot configuration completed")
        except Exception as e:
            print(f"DEBUG: ERROR setting plot configuration: {e}")
        
        print("DEBUG: Preparing to save plot...")
        try:
            plt.tight_layout()
            print("DEBUG: tight_layout() completed")
        except Exception as e:
            print(f"DEBUG: WARNING - tight_layout() failed: {e}")
            
        print(f"DEBUG: About to save plot to: {save_path}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        
        # 간단한 테스트로 빈 파일이라도 생성해보기
        try:
            with open('test_file.txt', 'w') as f:
                f.write('test')
            print("DEBUG: Test file creation successful")
        except Exception as e:
            print(f"DEBUG: Test file creation failed: {e}")
        
        print(f"DEBUG: Saving to {save_path}...")
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"DEBUG: Plot saved successfully to {save_path}")
            print(f"DEBUG: PNG file should be created at: {save_path}")
            
            # 파일이 실제로 생성되었는지 확인
            if os.path.exists(save_path):
                size = os.path.getsize(save_path)
                print(f"DEBUG: File exists with size: {size} bytes")
            else:
                print(f"DEBUG: ERROR - File does not exist: {save_path}")
                
        except Exception as e:
            print(f"DEBUG: ERROR saving plot: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("DEBUG: Closing figure...")
        try:
            plt.close()  # 메모리 절약을 위해 figure 닫기
            print("DEBUG: Figure closed successfully")
        except Exception as e:
            print(f"DEBUG: WARNING - error closing figure: {e}")
        
        print(f"3D scene saved to {save_path}")
        
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
        
        print(f"DEBUG: visualize_3d_scene() - EXIT POINT")
        print(f"DEBUG: Returning scene_center={scene_center}")
        return scene_center
    
    def render_orthographic_view(self, scene_center, 
                                save_path='orthographic_view.png'):
        """상공에서 nadir orthographic projection 렌더링"""
        
        # DTM 기반으로 orthographic view 생성
        if not hasattr(self, 'dtm'):
            raise ValueError("DTM not created")
            
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # DTM contour map
        contour = ax.contourf(self.dtm['x_grid'], self.dtm['y_grid'], self.dtm['z_grid'],
                             levels=50, cmap='terrain', alpha=0.8)
        
        # 카메라별 색상 생성 (3D 시각화와 동일한 색상 팔레트)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 
                 'lime', 'pink', 'brown', 'gray', 'olive', 'navy', 'maroon', 'teal',
                 'silver', 'gold', 'indigo', 'coral']
        
        # 카메라 위치들 표시
        image_count = 0
        for image_id, image in self.images.items():
            image_count += 1
            camera_color = colors[(image_count - 1) % len(colors)]
            
            camera_center = image['camera_center']
            ax.plot(camera_center[0], camera_center[1], '^', 
                   color=camera_color, markersize=8, alpha=0.8)
            
            # 이미지 footprint
            _, ray_dirs = self.get_camera_corners(image_id)
            ground_points = []
            
            for i in range(4):
                result = self.raycast_to_dtm(camera_center, ray_dirs[:, i])
                if result[0] is not None:  # 성공한 경우
                    intersection, _ = result
                    ground_points.append(intersection[:2])  # X, Y만
                    
            if len(ground_points) >= 3:  # 최소 3개 점이 있으면 다각형 그리기
                try:
                    ground_points = np.array(ground_points)
                    # 다각형 그리기 (카메라별 색상 사용)
                    polygon = plt.Polygon(ground_points, fill=False, 
                                         edgecolor=camera_color, linewidth=1, alpha=0.7)
                    ax.add_patch(polygon)
                except Exception as e:
                    print(f"DEBUG: Error creating polygon: {e}")
        
        # Scene center 표시
        ax.plot(scene_center[0], scene_center[1], 'y*', 
               markersize=15, label='Scene Center')
        
        # Aerial camera 제거됨
        
        # 컬러바
        plt.colorbar(contour, ax=ax, label='Elevation (m)')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Orthographic View from Aerial Camera')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 메모리 절약을 위해 figure 닫기
        
        print(f"Orthographic view saved to {save_path}")

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