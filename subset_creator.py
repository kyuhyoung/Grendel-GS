#!/usr/bin/env python3
"""
Subset Creator for Divide and Conquer 3D Gaussian Splatting

This module implements the subset creation algorithm based on footprint constraints:
1. 하나의 subset의 footprint의 union에 해당하는 이미지를 B라고 하자
2. B의 pixel 개수 C는 A를 넘을 수 없다.
3. B의 width와 height의 비율은 1에 가까워야 한다.
4. subset 끼리는 이미지를 공유하지 않는다.
5. 모든 이미지는 하나의 subset에 속한다.
6. min(C)와 max(C)의 비율이 D보다 커야 한다.
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import json

# PuLP import - required dependency
from pulp import *

logger = logging.getLogger(__name__)

class FootprintCalculator:
    """Calculate camera footprints from COLMAP data"""
    
    @staticmethod
    def load_colmap_data(colmap_path: Path) -> Tuple[Dict, Dict]:
        """Load COLMAP cameras and images data"""
        
        # Find sparse directory
        sparse_dirs = [
            colmap_path,
            colmap_path / "sparse",
            colmap_path / "sparse" / "0"
        ]
        
        sparse_dir = None
        for sdir in sparse_dirs:
            if (sdir / "cameras.txt").exists() and (sdir / "images.txt").exists():
                sparse_dir = sdir
                break
        
        if sparse_dir is None:
            raise FileNotFoundError(f"Could not find COLMAP files in {colmap_path}")
        
        logger.info(f"Loading COLMAP data from: {sparse_dir}")
        
        try:
            # Load cameras
            cameras = {}
            with open(sparse_dir / "cameras.txt", 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cam_id = int(parts[0])
                            model = parts[1]
                            width = int(parts[2])
                            height = int(parts[3])
                            params = [float(x) for x in parts[4:]]
                            cameras[cam_id] = {
                                'model': model,
                                'width': width,
                                'height': height,
                                'params': params
                            }
                        except ValueError as e:
                            logger.error(f"Error parsing camera line {line_num}: {line.strip()}")
                            # logger.error(f"Parts: {parts}")
                            logger.error(f"ValueError: {e}")
                            continue
            
            # Load images
            images = {}
            with open(sparse_dir / "images.txt", 'r') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('#') or not line:
                        i += 1
                        continue
                    
                    # Parse image metadata line (first line)
                    parts = line.split()
                    if len(parts) >= 10:
                        try:
                            img_id = int(parts[0])
                            qw, qx, qy, qz = map(float, parts[1:5])
                            tx, ty, tz = map(float, parts[5:8])
                            cam_id = int(parts[8])
                            name = parts[9]
                            images[img_id] = {
                                'quat': [qw, qx, qy, qz],
                                'trans': [tx, ty, tz],
                                'camera_id': cam_id,
                                'name': name
                            }
                            # Skip the next line (2D points data)
                            i += 2
                        except ValueError as e:
                            logger.error(f"Error parsing image line {i+1}: {line}")
                            logger.error(f"ValueError: {e}")
                            i += 1
                    else:
                        i += 1
            
            logger.info(f"Loaded {len(cameras)} cameras, {len(images)} images")
            return cameras, images
            
        except Exception as e:
            import traceback
            logger.error(f"Fatal error loading COLMAP data: {e}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise
    
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    @staticmethod
    def calculate_footprint(camera: Dict, image: Dict, ground_height: float = 0.0) -> Polygon:
        """Calculate camera footprint polygon"""
        
        # Camera intrinsics
        width = camera['width']
        height = camera['height']
        
        # Get focal lengths
        if len(camera['params']) >= 2:
            fx, fy = camera['params'][0], camera['params'][1]
        else:
            fx = fy = camera['params'][0] if camera['params'] else width
        
        # Camera pose
        quat = image['quat']
        trans = np.array(image['trans'])
        
        # Convert quaternion to rotation matrix
        R = FootprintCalculator.quaternion_to_rotation_matrix(quat)
        
        # Camera center in world coordinates
        C = -R.T @ trans
        
        # Estimate ground intersection
        # For aerial photos, assume camera is looking downward
        camera_height = C[2] - ground_height
        
        if camera_height <= 0:
            # Camera is at or below ground, use a small default footprint
            camera_height = 1.0
        
        # Calculate ground coverage based on camera parameters
        ground_width = width * camera_height / fx
        ground_height_actual = height * camera_height / fy
        
        # Project camera frustum corners to ground plane
        # Image corners in normalized coordinates
        corners_norm = np.array([
            [-width/2, -height/2, fx],  # Top-left
            [width/2, -height/2, fx],   # Top-right
            [width/2, height/2, fx],    # Bottom-right
            [-width/2, height/2, fx]    # Bottom-left
        ])
        
        # Transform to world coordinates and project to ground
        ground_points = []
        for corner in corners_norm:
            # Ray direction in camera coordinates
            ray_dir_cam = corner / np.linalg.norm(corner)
            
            # Transform to world coordinates
            ray_dir_world = R.T @ ray_dir_cam
            
            # Intersect with ground plane (z = ground_height)
            if ray_dir_world[2] != 0:
                t = (ground_height - C[2]) / ray_dir_world[2]
                if t > 0:  # Ray goes toward ground
                    ground_point = C + t * ray_dir_world
                    ground_points.append([ground_point[0], ground_point[1]])
        
        if len(ground_points) < 3:
            # Fallback: create rectangular footprint
            center_x, center_y = C[0], C[1]
            half_width = ground_width / 2
            half_height = ground_height_actual / 2
            
            ground_points = [
                [center_x - half_width, center_y - half_height],
                [center_x + half_width, center_y - half_height],
                [center_x + half_width, center_y + half_height],
                [center_x - half_width, center_y + half_height]
            ]
        
        try:
            # Create polygon from ground points
            footprint = Polygon(ground_points)
            
            # Ensure valid polygon
            if not footprint.is_valid:
                footprint = footprint.buffer(0)  # Fix self-intersections
            
            return footprint
            
        except Exception as e:
            logger.warning(f"Failed to create footprint polygon: {e}, using fallback")
            # Fallback rectangular footprint
            center_x, center_y = C[0], C[1]
            half_width = ground_width / 2
            half_height = ground_height_actual / 2
            return box(center_x - half_width, center_y - half_height,
                      center_x + half_width, center_y + half_height)


class SubsetCreator:
    """Create image subsets based on footprint constraints"""
    
    def __init__(self, cameras: Dict, images: Dict):
        self.cameras = cameras
        self.images = images
        self.image_footprints = {}
        
        # Calculate footprints for all images
        logger.info("Calculating footprints for all images...")
        for img_id, image in images.items():
            camera = cameras[image['camera_id']]
            footprint = FootprintCalculator.calculate_footprint(camera, image)
            self.image_footprints[img_id] = footprint
            
        logger.info(f"Calculated {len(self.image_footprints)} footprints")
    
    def calculate_union_metrics(self, image_ids: List[int]) -> Dict:
        """Calculate metrics for union of footprints"""
        if not image_ids:
            return {'area': 0, 'pixel_count': 0, 'aspect_ratio': 1.0, 'width': 0, 'height': 0}
        
        # Get footprints for subset
        footprints = [self.image_footprints[img_id] for img_id in image_ids]
        
        # Calculate union
        union_polygon = unary_union(footprints)
        
        if union_polygon.is_empty:
            return {'area': 0, 'pixel_count': 0, 'aspect_ratio': 1.0, 'width': 0, 'height': 0}
        
        # Get bounding box
        minx, miny, maxx, maxy = union_polygon.bounds
        width = maxx - minx
        height = maxy - miny
        
        # Calculate area and estimate pixel count
        area = union_polygon.area
        
        # Estimate pixel count based on image resolutions
        total_pixels = 0
        for img_id in image_ids:
            image = self.images[img_id]
            camera = self.cameras[image['camera_id']]
            total_pixels += camera['width'] * camera['height']
        
        # Aspect ratio (should be close to 1.0)
        aspect_ratio = width / height if height > 0 else 1.0
        
        return {
            'area': area,
            'pixel_count': total_pixels,
            'aspect_ratio': aspect_ratio,
            'width': width,
            'height': height,
            'union_polygon': union_polygon
        }
    
    def two_stage_subset_creation(self, target_a: int, max_subsets: int) -> List[List[int]]:
        """
        Two-Stage 서브셋 생성:
        Stage 1: ILP로 픽셀 합 기반 초기 할당
        Stage 2: 실제 union 계산으로 local optimization
        """
        logger.info("Starting Two-Stage subset creation...")
        logger.info(f"Target A: {target_a:,}, Max subsets: {max_subsets}")
        
        # Stage 1: ILP 기반 초기 할당
        stage1_subsets = self.stage1_ilp_approximation(target_a, max_subsets)
        
        # Stage 2: 실제 union으로 최적화
        final_subsets = self.stage2_union_optimization(stage1_subsets, target_a)
        
        return final_subsets
    
    def stage1_ilp_approximation(self, target_a: int, max_subsets: int) -> List[List[int]]:
        """
        Stage 1: ILP를 사용한 초기 서브셋 할당 (픽셀 합 근사)
        """
            
        logger.info("Stage 1: ILP 기반 초기 할당 시작...")
        
        image_ids = list(self.images.keys())
        N = len(image_ids)
        K = min(max_subsets, max(1, N // 2))  # 최대 subset 수 결정
        
        logger.info(f"Images: {N}, Max subsets: {K}")
        
        # 변수 정의
        x = {}  # x[i,j] = 1 if image i assigned to subset j
        y = {}  # y[j] = 1 if subset j is used
        
        for i in range(N):
            for j in range(K):
                x[i,j] = LpVariable(f"x_{i}_{j}", cat='Binary')
        
        for j in range(K):
            y[j] = LpVariable(f"y_{j}", cat='Binary')
        
        # 각 서브셋의 픽셀 수 (근사치)
        subset_pixels = {}
        for j in range(K):
            subset_pixels[j] = lpSum([
                self.cameras[self.images[image_ids[i]]['camera_id']]['width'] * 
                self.cameras[self.images[image_ids[i]]['camera_id']]['height'] / 
                self.image_footprints[image_ids[i]].area * x[i,j] 
                for i in range(N)
            ])
        
        # 목적함수: target_a로부터의 편차 최소화 + 분산 최소화
        prob = LpProblem("Stage1_Subset_Assignment", LpMinimize)
        
        # A로부터의 편차 계산
        deviations = {}
        for j in range(K):
            dev_pos = LpVariable(f"dev_pos_{j}", lowBound=0)
            dev_neg = LpVariable(f"dev_neg_{j}", lowBound=0)
            prob += subset_pixels[j] - target_a == dev_pos - dev_neg
            deviations[j] = dev_pos + dev_neg
        
        # 분산 최소화를 위한 max-min 계산
        max_pixels = LpVariable("max_pixels", lowBound=0)
        min_pixels = LpVariable("min_pixels", lowBound=0)
        
        for j in range(K):
            prob += max_pixels >= subset_pixels[j] - (1 - y[j]) * target_a * 2
            prob += min_pixels <= subset_pixels[j] + (1 - y[j]) * target_a * 2
        
        # 가중 목적함수
        total_deviation = lpSum(deviations.values())
        pixel_range = max_pixels - min_pixels
        prob += 1.0 * total_deviation + 2.0 * pixel_range
        
        # 제약조건들
        # 1. 각 이미지는 정확히 하나의 서브셋에만
        for i in range(N):
            prob += lpSum([x[i, j] for j in range(K)]) == 1
        
        # 2. 각 서브셋 최소 2개 이미지
        for j in range(K):
            prob += lpSum([x[i, j] for i in range(N)]) >= 2 * y[j]
            prob += lpSum([x[i, j] for i in range(N)]) <= N * y[j]
        
        # 3. 픽셀 수 상한선 (target_a의 150%)
        for j in range(K):
            prob += subset_pixels[j] <= target_a * 1.5
        
        # 문제 해결
        logger.info("ILP 문제 해결 중...")
        prob.solve(PULP_CBC_CMD(msg=0))
        
        if prob.status != LpStatusOptimal:
            logger.warning(f"ILP 최적해를 찾지 못함: {LpStatus[prob.status]}")
            # Fallback to greedy method
            return self.greedy_subset_creation_fallback(target_a, max_subsets)
        
        # 결과 추출
        subsets = []
        for j in range(K):
            if y[j].value() > 0.5:
                subset = []
                for i in range(N):
                    if x[i, j].value() > 0.5:
                        subset.append(image_ids[i])
                if subset:
                    subsets.append(subset)
        
        logger.info(f"Stage 1 완료: {len(subsets)}개 서브셋 생성")
        return subsets
    
    def stage2_union_optimization(self, initial_subsets: List[List[int]], target_a: int, max_iterations: int = 100) -> List[List[int]]:
        """
        Stage 2: 실제 footprint union 계산으로 local optimization
        """
        logger.info("Stage 2: 실제 union 계산 기반 최적화 시작...")
        
        current_subsets = [subset[:] for subset in initial_subsets]  # 깊은 복사
        best_score = float('inf')
        best_subsets = [subset[:] for subset in current_subsets]
        
        # 초기 점수 계산
        current_score = self.calculate_union_based_score(current_subsets, target_a)
        best_score = current_score
        
        logger.info(f"초기 점수: {current_score:.0f}")
        
        for iteration in range(max_iterations):
            improved = False
            
            # 모든 이미지에 대해 다른 서브셋으로 이동 시도
            for subset_i in range(len(current_subsets)):
                for img_id in current_subsets[subset_i][:]:  # 복사본으로 반복
                    if len(current_subsets[subset_i]) <= 2:  # 최소 2개 유지
                        continue
                    
                    for subset_j in range(len(current_subsets)):
                        if subset_i == subset_j:
                            continue
                        
                        # 임시로 이미지 이동
                        current_subsets[subset_i].remove(img_id)
                        current_subsets[subset_j].append(img_id)
                        
                        # 새로운 점수 계산
                        new_score = self.calculate_union_based_score(current_subsets, target_a)
                        
                        if new_score < best_score:
                            best_score = new_score
                            best_subsets = [subset[:] for subset in current_subsets]
                            improved = True
                            logger.info(f"Iteration {iteration}: 개선된 점수 {new_score:.0f} "
                                      f"(이미지 {img_id}: 서브셋 {subset_i+1} → {subset_j+1})")
                        
                        # 원복 (이동하지 않기로 결정)
                        current_subsets[subset_j].remove(img_id)
                        current_subsets[subset_i].append(img_id)
            
            # 개선된 해가 있으면 적용
            if improved:
                current_subsets = [subset[:] for subset in best_subsets]
                current_score = best_score
            else:
                break  # 더 이상 개선되지 않음
        
        logger.info(f"Stage 2 완료: 최종 점수 {best_score:.0f} ({iteration+1}번 반복)")
        return best_subsets
    
    def calculate_union_based_score(self, subsets: List[List[int]], target_a: int) -> float:
        """
        실제 footprint union을 기반으로 한 점수 계산
        """
        if not subsets or any(len(subset) < 2 for subset in subsets):
            return float('inf')  # 무효한 해
        
        pixel_counts = []
        aspect_penalties = 0
        deviation_from_a = 0
        
        for subset in subsets:
            # 실제 union 계산
            union_polygon = self.calculate_real_footprint_union(subset)
            if union_polygon is None or union_polygon.is_empty:
                return float('inf')
            
            # Union 영역의 픽셀 수 계산 (근사)
            union_pixels = self.estimate_union_pixels(union_polygon, subset)
            pixel_counts.append(union_pixels)
            
            # A로부터 편차
            deviation_from_a += abs(union_pixels - target_a)
            
            # Aspect ratio 페널티
            bounds = union_polygon.bounds
            width = bounds[2] - bounds[0] if bounds[2] > bounds[0] else 1
            height = bounds[3] - bounds[1] if bounds[3] > bounds[1] else 1
            aspect_ratio = width / height
            aspect_penalties += abs(aspect_ratio - 1.0)
        
        # 분산 계산
        variance = np.var(pixel_counts) if len(pixel_counts) > 1 else 0
        
        # 가중 점수
        total_score = (
            1.0 * deviation_from_a +      # A에 가까워야 함 (조건 2)
            2.0 * variance +               # 분산 최소화 (조건 6)  
            0.5 * aspect_penalties         # 정사각형에 가까워야 함 (조건 3)
        )
        
        return total_score
    
    def calculate_real_footprint_union(self, image_list: List[int]):
        """실제 footprint들의 union 계산"""
        if not image_list:
            return None
        
        polygons = []
        for img_id in image_list:
            if img_id in self.image_footprints:
                footprint = self.image_footprints[img_id]
                polygons.append(footprint)
        
        if not polygons:
            return None
        
        from shapely.ops import unary_union
        union_polygon = unary_union(polygons)
        return union_polygon
    
    def estimate_union_pixels(self, union_polygon, image_list: List[int]) -> int:
        """Union 다각형의 픽셀 수 추정"""
        # 방법 1: 면적 기반 추정
        area_m2 = union_polygon.area
        
        # 해당 서브셋 이미지들의 평균 픽셀 밀도 계산
        total_pixels = 0
        total_area = 0
        
        for img_id in image_list:
            if img_id in self.image_footprints:
                camera = self.cameras[self.images[img_id]['camera_id']]
                img_pixels = camera['width'] * camera['height']
                img_area = self.image_footprints[img_id].area
                
                total_pixels += img_pixels
                total_area += img_area
        
        if total_area > 0:
            avg_pixel_density = total_pixels / total_area
            estimated_pixels = int(area_m2 * avg_pixel_density * 0.7)  # overlap 고려 계수
        else:
            estimated_pixels = int(area_m2 * 1000)  # fallback
        
        return estimated_pixels
    
    def greedy_subset_creation_fallback(self, target_a: int, max_subsets: int) -> List[List[int]]:
        """ILP 실패 시 fallback greedy 방법"""
        return self.greedy_subset_creation(target_a, 0.7, max_subsets)  # 기존 greedy 메서드 호출

    def greedy_subset_creation(self, pixel_threshold_a: int, min_max_ratio_d: float, max_subsets: int) -> List[List[int]]:
        """Create subsets using greedy algorithm"""
        
        image_ids = list(self.images.keys())
        unassigned = set(image_ids)
        subsets = []
        
        logger.info(f"Creating subsets for {len(image_ids)} images")
        logger.info(f"Pixel threshold A: {pixel_threshold_a:,}")
        logger.info(f"Min/Max ratio D: {min_max_ratio_d}")
        logger.info(f"Max subsets: {max_subsets}")
        logger.info(f"Minimum images per subset: 2")
        
        iteration = 0
        while unassigned and len(subsets) < max_subsets:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration}: {len(unassigned)} images remaining ---")
            
            # Check if we have at least 2 images left to form a valid subset
            if len(unassigned) < 2:
                logger.warning(f"Only {len(unassigned)} image(s) remaining, cannot form valid subset (min 2 required)")
                break
            
            # Start new subset
            current_subset = []
            
            # Find best starting image (central or with good coverage)
            best_start = self.find_best_starting_image(unassigned)
            current_subset.append(best_start)
            unassigned.remove(best_start)
            
            logger.info(f"Started subset {len(subsets) + 1} with image {best_start}")
            
            # Ensure minimum 2 images per subset - try to add at least one more image
            if len(current_subset) == 1 and unassigned:
                # Force add closest image to ensure minimum subset size
                closest_img = None
                closest_dist = float('inf')
                
                for candidate_id in unassigned:
                    proximity_score = self.calculate_proximity_score(current_subset, candidate_id)
                    if proximity_score < closest_dist:
                        closest_dist = proximity_score
                        closest_img = candidate_id
                
                if closest_img is not None:
                    current_subset.append(closest_img)
                    unassigned.remove(closest_img)
                    logger.info(f"  Force added image {closest_img} to meet minimum size requirement")
            
            # Greedily add more images to current subset
            improved = True
            while improved and unassigned:
                improved = False
                best_candidate = None
                best_metrics = None
                best_score = float('inf')
                
                for candidate_id in list(unassigned):
                    test_subset = current_subset + [candidate_id]
                    metrics = self.calculate_union_metrics(test_subset)
                    
                    # Check pixel threshold constraint
                    if metrics['pixel_count'] > pixel_threshold_a:
                        continue
                    
                    # Score based on aspect ratio (prefer close to 1.0) and compactness
                    aspect_penalty = abs(1.0 - metrics['aspect_ratio'])
                    
                    # Also consider spatial proximity (prefer clustered images)
                    proximity_score = self.calculate_proximity_score(current_subset, candidate_id)
                    
                    # Combined score (lower is better)
                    score = aspect_penalty + 0.1 * proximity_score
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = candidate_id
                        best_metrics = metrics
                
                if best_candidate is not None:
                    current_subset.append(best_candidate)
                    unassigned.remove(best_candidate)
                    improved = True
                    
                    logger.info(f"  Added image {best_candidate}: "
                              f"pixels={best_metrics['pixel_count']:,}, "
                              f"aspect={best_metrics['aspect_ratio']:.3f}, "
                              f"size={len(current_subset)}")
            
            # Only add subset if it has at least 2 images
            if len(current_subset) >= 2:
                subsets.append(current_subset)
            final_metrics = self.calculate_union_metrics(current_subset)
            logger.info(f"Completed subset {len(subsets)}: {len(current_subset)} images, "
                       f"{final_metrics['pixel_count']:,} pixels, "
                       f"aspect ratio {final_metrics['aspect_ratio']:.3f}")
            logger.info(f"  Image IDs: {current_subset[:10]}{'...' if len(current_subset) > 10 else ''}")
        
        # Handle remaining images (add to smallest subset or create new one)
        if unassigned:
            remaining_count = len(unassigned)
            
            if remaining_count >= 2 and len(subsets) < max_subsets:
                # Create final subset with remaining images if we have at least 2
                remaining_list = list(unassigned)
                subsets.append(remaining_list)
                logger.info(f"Created final subset {len(subsets)} with {len(remaining_list)} remaining images")
            elif remaining_count == 1:
                # Only 1 image left - add to smallest existing subset
                logger.warning(f"1 image remaining (ID: {list(unassigned)[0]}), adding to smallest subset")
                if subsets:
                    smallest_idx = min(range(len(subsets)), key=lambda i: len(subsets[i]))
                    subsets[smallest_idx].extend(list(unassigned))
                    logger.info(f"  Added to subset {smallest_idx + 1}")
                else:
                    logger.error("No subsets created and only 1 image remaining - cannot satisfy minimum size constraint")
            else:
                # Distribute remaining images to existing subsets
                self.distribute_remaining_images(subsets, list(unassigned), pixel_threshold_a)
        
        return subsets
    
    def find_best_starting_image(self, available_images: Set[int]) -> int:
        """Find best starting image for a new subset"""
        # For simplicity, pick image closest to centroid of remaining images
        if len(available_images) == 1:
            return list(available_images)[0]
        
        # Calculate centroid of available images
        positions = []
        for img_id in available_images:
            footprint = self.image_footprints[img_id]
            centroid = footprint.centroid
            positions.append([centroid.x, centroid.y])
        
        centroid = np.mean(positions, axis=0)
        
        # Find image closest to centroid
        best_img = None
        best_dist = float('inf')
        
        for img_id in available_images:
            footprint = self.image_footprints[img_id]
            img_centroid = footprint.centroid
            dist = np.sqrt((img_centroid.x - centroid[0])**2 + (img_centroid.y - centroid[1])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_img = img_id
        
        return best_img
    
    def calculate_proximity_score(self, current_subset: List[int], candidate_id: int) -> float:
        """Calculate proximity score for adding candidate to current subset"""
        if not current_subset:
            return 0.0
        
        candidate_footprint = self.image_footprints[candidate_id]
        candidate_centroid = candidate_footprint.centroid
        
        # Calculate average distance to current subset images
        distances = []
        for img_id in current_subset:
            footprint = self.image_footprints[img_id]
            centroid = footprint.centroid
            dist = np.sqrt((candidate_centroid.x - centroid.x)**2 + (candidate_centroid.y - centroid.y)**2)
            distances.append(dist)
        
        return np.mean(distances)
    
    def distribute_remaining_images(self, subsets: List[List[int]], remaining: List[int], pixel_threshold_a: int):
        """Distribute remaining images to existing subsets"""
        logger.info(f"Distributing {len(remaining)} remaining images to existing subsets")
        
        for img_id in remaining:
            best_subset_idx = None
            best_score = float('inf')
            
            for i, subset in enumerate(subsets):
                test_subset = subset + [img_id]
                metrics = self.calculate_union_metrics(test_subset)
                
                # Check if adding this image violates pixel threshold
                if metrics['pixel_count'] <= pixel_threshold_a:
                    # Score based on aspect ratio improvement
                    score = abs(1.0 - metrics['aspect_ratio'])
                    
                    if score < best_score:
                        best_score = score
                        best_subset_idx = i
            
            if best_subset_idx is not None:
                subsets[best_subset_idx].append(img_id)
                logger.info(f"  Added image {img_id} to subset {best_subset_idx + 1}")
            else:
                # Force add to smallest subset (may violate constraints)
                smallest_idx = min(range(len(subsets)), key=lambda i: len(subsets[i]))
                subsets[smallest_idx].append(img_id)
                logger.warning(f"  Force added image {img_id} to subset {smallest_idx + 1} (may violate constraints)")
    
    def validate_subsets(self, subsets: List[List[int]], pixel_threshold_a: int, min_max_ratio_d: float) -> bool:
        """Validate subset constraints"""
        logger.info(f"\nValidating {len(subsets)} subsets...")
        
        # Check minimum size constraint (at least 2 images per subset)
        for i, subset in enumerate(subsets):
            if len(subset) < 2:
                logger.error(f"Minimum size constraint violated: Subset {i+1} has only {len(subset)} image(s), minimum 2 required")
                return False
        
        # Check constraint 4: no image sharing
        all_images = set()
        for i, subset in enumerate(subsets):
            subset_set = set(subset)
            if all_images & subset_set:
                logger.error("Constraint 4 violated: Subsets share images!")
                return False
            all_images.update(subset_set)
        
        # Check constraint 5: all images assigned
        expected_images = set(self.images.keys())
        if all_images != expected_images:
            missing = expected_images - all_images
            logger.error(f"Constraint 5 violated: {len(missing)} images not assigned")
            return False
        
        # Check constraints 2, 3, 6
        pixel_counts = []
        valid = True
        
        for i, subset in enumerate(subsets):
            metrics = self.calculate_union_metrics(subset)
            pixel_counts.append(metrics['pixel_count'])
            
            logger.info(f"  Subset {i+1}: {len(subset)} images, "
                       f"{metrics['pixel_count']:,} pixels, "
                       f"aspect ratio {metrics['aspect_ratio']:.3f}")
            logger.info(f"    Image IDs: {subset[:10]}{'...' if len(subset) > 10 else ''}")
            
            # Constraint 2: pixel count <= A
            if metrics['pixel_count'] > pixel_threshold_a:
                logger.warning(f"Constraint 2 violated: Subset {i+1} exceeds pixel threshold "
                             f"({metrics['pixel_count']:,} > {pixel_threshold_a:,})")
                valid = False
            
            # Constraint 3: aspect ratio close to 1.0 (warning only)
            if abs(1.0 - metrics['aspect_ratio']) > 0.5:
                logger.warning(f"Constraint 3 concern: Subset {i+1} has poor aspect ratio "
                             f"({metrics['aspect_ratio']:.3f})")
        
        # Constraint 6: min/max ratio >= D
        if pixel_counts:
            min_pixels = min(pixel_counts)
            max_pixels = max(pixel_counts)
            ratio = min_pixels / max_pixels if max_pixels > 0 else 0
            
            logger.info(f"\nPixel count ratio: {ratio:.3f} (min: {min_pixels:,}, max: {max_pixels:,})")
            
            if ratio < min_max_ratio_d:
                logger.warning(f"Constraint 6 violated: Ratio {ratio:.3f} < {min_max_ratio_d}")
                valid = False
        
        if valid:
            logger.info("✓ All constraints satisfied!")
        else:
            logger.warning("⚠ Some constraints violated, but proceeding...")
        
        return valid


def create_subsets_with_footprints(source_path: Path, output_path: Path, 
                                  pixel_threshold_a: int, min_max_ratio_d: float, 
                                  max_subsets: int) -> List[List[int]]:
    """Main function to create subsets with footprint constraints"""
    
    logger.info("Starting subset creation with footprint constraints")
    
    # Load COLMAP data
    cameras, images = FootprintCalculator.load_colmap_data(source_path)
    
    # Create subset creator
    creator = SubsetCreator(cameras, images)
    
    # Create subsets using Two-Stage approach
    subsets = creator.two_stage_subset_creation(pixel_threshold_a, max_subsets)
    
    # Validate subsets
    valid = creator.validate_subsets(subsets, pixel_threshold_a, min_max_ratio_d)
    
    # Save detailed results
    results = {
        'subsets': subsets,
        'metadata': {
            'total_images': len(images),
            'num_subsets': len(subsets),
            'pixel_threshold_a': pixel_threshold_a,
            'min_max_ratio_d': min_max_ratio_d,
            'max_subsets': max_subsets,
            'validation_passed': valid
        },
        'subset_details': []
    }
    
    # Add detailed metrics for each subset
    for i, subset in enumerate(subsets):
        metrics = creator.calculate_union_metrics(subset)
        results['subset_details'].append({
            'subset_id': i,
            'image_count': len(subset),
            'image_ids': subset,
            'pixel_count': metrics['pixel_count'],
            'aspect_ratio': metrics['aspect_ratio'],
            'width': metrics['width'],
            'height': metrics['height'],
            'area': metrics['area']
        })
    
    # Save results
    results_file = output_path / "subset_creation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Subset creation completed. Results saved to: {results_file}")
    logger.info(f"Created {len(subsets)} subsets for {len(images)} images")
    
    # Print final subset summary
    logger.info("\n=== SUBSET SUMMARY ===")
    for i, subset in enumerate(subsets):
        logger.info(f"Subset {i+1}: {len(subset)} images")
        # Show first 20 image IDs for each subset
        if len(subset) <= 20:
            logger.info(f"  IDs: {subset}")
        else:
            logger.info(f"  IDs: {subset[:20]}... (and {len(subset)-20} more)")
    logger.info("===================")
    
    return subsets