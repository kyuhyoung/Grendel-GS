"""
공간 분할 (Uniform Grid + Overlap)
"""

import numpy as np
from typing import Dict, Tuple, List
from .tile_storage import BBox


class UniformGrid:
    """균등 Grid 공간 분할 (Overlap 지원)"""
    
    def __init__(self, points: np.ndarray, grid_size: Tuple[int, int, int], 
                 overlap_meters: float = 10.0):
        """
        Args:
            points: [N, 3] Gaussian 중심점
            grid_size: (nx, ny, nz) 타일 개수
            overlap_meters: Overlap 크기 (미터)
        """
        self.grid_size = np.array(grid_size)
        self.overlap = overlap_meters
        
        # BBox 계산
        self.scene_min = points.min(axis=0)
        self.scene_max = points.max(axis=0)
        self.scene_extent = self.scene_max - self.scene_min
        
        # 타일 크기 (미터)
        self.tile_size = self.scene_extent / self.grid_size
        
        print(f"[UniformGrid] Scene: {self.scene_min} ~ {self.scene_max}")
        print(f"[UniformGrid] Extent: {self.scene_extent}")
        print(f"[UniformGrid] Grid: {self.grid_size}")
        print(f"[UniformGrid] Tile size: {self.tile_size}")
        print(f"[UniformGrid] Overlap: {self.overlap}m")
    
    def point_to_tile(self, point: np.ndarray) -> Tuple[int, int, int]:
        """점 → 타일 좌표"""
        normalized = (point - self.scene_min) / self.scene_extent
        coords = (normalized * self.grid_size).astype(int)
        
        # Clamp
        coords = np.clip(coords, 0, self.grid_size - 1)
        
        return tuple(coords)
    
    def get_tile_bbox(self, tile_coord: Tuple[int, int, int], 
                      with_overlap: bool = False) -> BBox:
        """타일의 BBox (Overlap 포함 옵션)"""
        
        tile_min = self.scene_min + np.array(tile_coord) * self.tile_size
        tile_max = tile_min + self.tile_size
        
        if with_overlap:
            tile_min -= self.overlap
            tile_max += self.overlap
        
        return BBox(min_xyz=tile_min, max_xyz=tile_max)
    
    def assign_points_to_tiles(self, points: np.ndarray, 
                               scales: np.ndarray) -> Dict[Tuple[int, int, int], List[int]]:
        """
        점들을 타일에 할당 (Overlap 고려) - Vectorized Version
        
        Args:
            points: [N, 3] 중심점
            scales: [N, 3] scale (선형 공간, overlap 계산용)
            
        Returns:
            {tile_coord: [point_indices]}
        """
        
        print(f"[UniformGrid] Assigning {len(points):,} points to tiles (Vectorized)...")
        
        # 1. Calculate influence radius for all points
        # Clamp max scale to prevent explosion (e.g. 50.0)
        max_scales = scales.max(axis=1)
        max_scales = np.minimum(max_scales, 50.0) 
        influence_radii = 3 * max_scales + self.overlap
        
        # 2. Calculate BBox for all points
        # [N, 3]
        bbox_min = points - influence_radii[:, None]
        bbox_max = points + influence_radii[:, None]
        
        # 3. Convert to Tile Indices
        # (point - scene_min) / tile_size
        # [N, 3]
        tile_min_idx = np.floor((bbox_min - self.scene_min) / self.tile_size).astype(int)
        tile_max_idx = np.floor((bbox_max - self.scene_min) / self.tile_size).astype(int)
        
        # Clamp to grid bounds
        tile_min_idx = np.clip(tile_min_idx, 0, self.grid_size - 1)
        tile_max_idx = np.clip(tile_max_idx, 0, self.grid_size - 1)
        
        # 4. Iterate over possible offsets to find assignments
        # Most points span 1-2 tiles. We iterate over the max span.
        # Calculate spans
        spans = tile_max_idx - tile_min_idx
        max_span = spans.max(axis=0)
        
        print(f"[UniformGrid] Max span: {max_span}")
        
        tile_assignments = {}
        
        # Iterate over all offsets within the max span
        for dx in range(max_span[0] + 1):
            for dy in range(max_span[1] + 1):
                for dz in range(max_span[2] + 1):
                    # Check which points cover this offset
                    # condition: tile_min + offset <= tile_max
                    # i.e. offset <= span
                    mask = (dx <= spans[:, 0]) & (dy <= spans[:, 1]) & (dz <= spans[:, 2])
                    
                    if not np.any(mask):
                        continue
                        
                    # Get valid points
                    valid_indices = np.where(mask)[0]
                    
                    # Calculate target tile coordinates for these points
                    target_tiles = tile_min_idx[valid_indices] + np.array([dx, dy, dz])
                    
                    # Add to dictionary
                    # This part is still a loop but over chunks of points, or we can group by tile
                    # Grouping by tile is faster
                    
                    # Create unique tile keys
                    # We can use a structured array or simple loop over unique tiles
                    unique_tiles, inverse_indices = np.unique(target_tiles, axis=0, return_inverse=True)
                    
                    for i, tile in enumerate(unique_tiles):
                        # Convert numpy int64 to python int for JSON serialization
                        tile_coord = tuple(map(int, tile))
                        
                        # Points belonging to this tile in this offset group
                        points_in_tile = valid_indices[inverse_indices == i]
                        
                        if tile_coord not in tile_assignments:
                            tile_assignments[tile_coord] = []
                        
                        tile_assignments[tile_coord].extend(points_in_tile.tolist())

        # 통계
        total_assigned = sum(len(ids) for ids in tile_assignments.values())
        duplication_ratio = total_assigned / len(points) if len(points) > 0 else 0
        
        print(f"[UniformGrid] Assignment complete:")
        print(f"  Tiles with data: {len(tile_assignments):,}")
        print(f"  Total assignments: {total_assigned:,}")
        print(f"  Duplication ratio: {duplication_ratio:.2f}x")
        
        return tile_assignments
    
    def assign_points_to_tiles_strict(self, points: np.ndarray) -> Dict[Tuple[int, int, int], List[int]]:
        """
        점들을 타일에 할당 (Overlap 없이 중심점 기준) - Vectorized Version
        
        Args:
            points: [N, 3] 중심점
            
        Returns:
            {tile_coord: [point_indices]}
        """
        # (point - scene_min) / tile_size
        # [N, 3]
        tile_idx = np.floor((points - self.scene_min) / self.tile_size).astype(int)
        
        # Clamp to grid bounds
        tile_idx = np.clip(tile_idx, 0, self.grid_size - 1)
        
        # Group by tile index
        # We can use a dictionary or sort
        assignments = {}
        
        # Convert to tuple keys
        # This loop might be slow in Python for millions of points?
        # But assign_points_to_tiles also iterates.
        # Let's try to be efficient.
        
        # Unique tiles
        unique_tiles, inverse_indices = np.unique(tile_idx, axis=0, return_inverse=True)
        
        for i, tile in enumerate(unique_tiles):
            coord = tuple(tile)
            # Find points belonging to this tile
            indices = np.where(inverse_indices == i)[0]
            assignments[coord] = indices.tolist()
            
        return assignments
