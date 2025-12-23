"""
타일 기반 장면 관리 + 3DGS Rasterization
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys

# 3DGS imports
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tile_storage import TileStorage, BBox


class LRUCache:
    """간단한 LRU 캐시"""
    
    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.cache = {}
        self.timestamp = 0
        
        self.hits = 0
        self.misses = 0
    
    def get(self, tile_coord: Tuple[int, int, int]) -> Optional[Dict]:
        """캐시에서 가져오기"""
        if tile_coord in self.cache:
            tile_data, _ = self.cache[tile_coord]
            self.timestamp += 1
            self.cache[tile_coord] = (tile_data, self.timestamp)
            self.hits += 1
            return tile_data
        else:
            self.misses += 1
            return None
    
    def put(self, tile_coord: Tuple[int, int, int], tile_data: Dict):
        """캐시에 추가"""
        if len(self.cache) >= self.capacity and tile_coord not in self.cache:
            oldest_coord = min(self.cache.keys(), 
                             key=lambda k: self.cache[k][1])
            del self.cache[oldest_coord]
        
        self.timestamp += 1
        self.cache[tile_coord] = (tile_data, self.timestamp)
    
    def get_hit_rate(self) -> float:
        """캐시 히트율"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class TiledScene:
    """타일 기반 3D Gaussian Splatting 장면"""
    
    def __init__(self, scene_dir: str, device: str = "cuda", cache_size: int = 200):
        self.storage = TileStorage(scene_dir)
        self.device = device
        self.cache = LRUCache(capacity=cache_size)
        
        self.active_gaussians = None
        self.active_tile_coords = []
        
        print(f"[TiledScene] Initialized")
        print(f"  Device: {device}")
        print(f"  Cache capacity: {cache_size} tiles")
    
    def prepare_for_rendering(self, visible_tile_coords: List[Tuple[int, int, int]]):
        """렌더링 준비: 필요한 타일만 GPU로 로드"""
        
        tiles_data = []
        
        for coord in visible_tile_coords:
            tile_data = self.cache.get(coord)
            
            if tile_data is None:
                tile_data = self.storage.load_tile(coord)
                if tile_data is not None:
                    self.cache.put(coord, tile_data)
            
            if tile_data is not None:
                tiles_data.append(tile_data)
        
        if tiles_data:
            self.active_gaussians = self._merge_to_gpu(tiles_data)
            self.active_tile_coords = visible_tile_coords
        else:
            self.active_gaussians = None
            self.active_tile_coords = []
    
    def _merge_to_gpu(self, tiles: List[Dict]) -> Dict:
        """타일들을 GPU 텐서로 병합 + activation 적용"""
        
        # 타일 데이터는 log/logit space로 저장되어 있음
        # rasterizer는 activated 값을 기대하므로 여기서 적용
        
        merged = {
            'xyz': torch.cat([torch.from_numpy(t['means']) for t in tiles]).to(self.device),
            'rotation': torch.cat([torch.from_numpy(t['quats']) for t in tiles]).to(self.device),
            'scaling': torch.cat([torch.from_numpy(t['scales']) for t in tiles]).to(self.device),  # log space
            'opacity': torch.cat([torch.from_numpy(t['opacities']) for t in tiles]).to(self.device),  # logit space
            'features_dc': torch.cat([torch.from_numpy(t['sh0']) for t in tiles]).to(self.device),
            'features_rest': torch.cat([torch.from_numpy(t['shN']) for t in tiles]).to(self.device),
        }
        
        # ⭐ Activation 적용 (3DGS와 동일)
        merged['scaling'] = torch.exp(merged['scaling'])  # log → linear
        merged['opacity'] = torch.sigmoid(merged['opacity'])  # logit → [0,1]
        merged['rotation'] = torch.nn.functional.normalize(merged['rotation'], dim=1)  # quaternion 정규화
        
        # ⭐ Clamp extreme scales to prevent artifacts (optional quality improvement)
        # GS Viewer might do similar clamping internally
        merged['scaling'] = torch.clamp(merged['scaling'], min=0.0001, max=50.0)
        
        for key in merged:
            merged[key] = merged[key].requires_grad_(True)
        
        return merged
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계"""
        return {
            'size': len(self.cache.cache),
            'capacity': self.cache.capacity,
            'hits': self.cache.hits,
            'misses': self.cache.misses,
            'hit_rate': self.cache.get_hit_rate()
        }


def create_simple_camera(width: int = 800, height: int = 600, device: str = "cuda"):
    """테스트용 간단한 카메라 생성"""
    
    fov_x = np.radians(60)
    fov_y = np.radians(45)
    
    world_view_transform = torch.eye(4, device=device)
    
    znear, zfar = 0.01, 10000.0
    
    tanfovx = np.tan(fov_x * 0.5)
    tanfovy = np.tan(fov_y * 0.5)
    
    top = tanfovy * znear
    bottom = -top
    right = tanfovx * znear
    left = -right
    
    P = torch.zeros(4, 4, device=device)
    
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = -1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    
    projection_matrix = P.transpose(0, 1)
    full_proj = world_view_transform @ projection_matrix
    
    camera_center = torch.inverse(world_view_transform)[3, :3]
    
    return {
        'image_height': height,
        'image_width': width,
        'tanfovx': tanfovx,
        'tanfovy': tanfovy,
        'world_view_transform': world_view_transform,
        'projection_matrix': projection_matrix,
        'full_proj_transform': full_proj,
        'camera_center': camera_center
    }


def render_tiled(camera: Dict, scene: TiledScene, bg_color: torch.Tensor, scaling_modifier: float = 1.0):
    """타일 기반 렌더링 (안전한 버전)"""
    
    gaussians = scene.active_gaussians
    
    if gaussians is None or len(gaussians['xyz']) == 0:
        return torch.zeros((3, camera['image_height'], camera['image_width']), 
                          device=scene.device), None
    
    tile_x = (int(camera['image_width']) + 15) // 16
    tile_y = (int(camera['image_height']) + 15) // 16

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera['image_height']),
        image_width=int(camera['image_width']),
        tanfovx=float(camera['tanfovx']),
        tanfovy=float(camera['tanfovy']),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera['world_view_transform'],
        projmatrix=camera['full_proj_transform'],
        sh_degree=3,
        campos=camera['camera_center'],
        prefiltered=False,
        debug=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    shs = torch.cat([gaussians['features_dc'], gaussians['features_rest']], dim=1)

    cuda_args = {
        'mode': 'test',
        'world_size': '1',
        'global_rank': '0',
        'local_rank': '0',
        'mp_world_size': '1',
        'mp_rank': '0',
        'log_folder': '.',
        'log_interval': '1',
        'iteration': '0',
        'zhx_debug': 'False',
        'zhx_time': 'False',
        'avoid_pixel_all2all': False,
        'stats_collector': {}
    }

    means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
        means3D=gaussians['xyz'],
        scales=gaussians['scaling'],
        rotations=gaussians['rotation'],
        shs=shs,
        opacities=gaussians['opacity'],
        cuda_args=cuda_args
    )

    compute_locally = torch.ones((tile_y, tile_x), dtype=torch.bool, device=scene.device)
    extended_compute_locally = compute_locally

    results = rasterizer.render_gaussians(
        means2D=means2D,
        conic_opacity=conic_opacity,
        rgb=rgb,
        depths=depths,
        radii=radii,
        compute_locally=compute_locally,
        extended_compute_locally=extended_compute_locally,
        cuda_args=cuda_args
    )

    if isinstance(results, tuple):
        rendered_image = results[0]
    else:
        rendered_image = results
    
    return rendered_image, radii
