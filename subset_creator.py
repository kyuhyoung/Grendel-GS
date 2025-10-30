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
        
        # Load cameras
        cameras = {}
        with open(sparse_dir / "cameras.txt", 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
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
        
        # Load images
        images = {}
        with open(sparse_dir / "images.txt", 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 10:
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
        
        logger.info(f"Loaded {len(cameras)} cameras, {len(images)} images")
        return cameras, images
    
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
    
    # Create subsets
    subsets = creator.greedy_subset_creation(pixel_threshold_a, min_max_ratio_d, max_subsets)
    
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
    
    return subsets