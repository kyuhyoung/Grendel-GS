"""
타일 저장/로딩 시스템 (JSON 기반)
"""

import json
import numpy as np
import uuid
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import fcntl
import os


class BBox:
    """Bounding Box"""
    def __init__(self, min_xyz, max_xyz):
        self.min = np.array(min_xyz, dtype=np.float32)
        self.max = np.array(max_xyz, dtype=np.float32)
    
    def to_dict(self):
        return {
            'min': self.min.tolist(),
            'max': self.max.tolist()
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(min_xyz=d['min'], max_xyz=d['max'])


class TileStorage:
    """타일 기반 저장 시스템 (JSON 메타데이터)"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.tiles_dir = self.root_dir / "tiles"
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 메타데이터
        self.metadata_path = self.root_dir / "tiles_metadata.json"
        self.metadata = self._load_metadata()
        
        print(f"[TileStorage] Initialized at {self.root_dir}")
    
    def _load_metadata(self) -> Dict:
        """메타데이터 로드"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    data = json.load(f)
                    print(f"[TileStorage] Loaded {len(data['tiles'])} tiles from metadata")
                    return data
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            return {'tiles': {}, 'version': '1.0'}
    
    def _save_metadata(self):
        """메타데이터 저장 (Thread-safe & Process-safe)"""
        # 1. Lock & Read latest
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r+') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        disk_data = json.load(f)
                    except json.JSONDecodeError:
                        disk_data = {'tiles': {}, 'version': '1.0'}
                    
                    # 2. Merge local changes into disk data
                    # We assume local changes are newer/authoritative for the tiles this rank owns
                    for tid, tdata in self.metadata['tiles'].items():
                        disk_data['tiles'][tid] = tdata
                    
                    # 3. Write back
                    f.seek(0)
                    f.truncate()
                    json.dump(disk_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                    
                    # 4. Update local to match (optional, but good for consistency)
                    self.metadata = disk_data
                    
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            # Create new file
            with open(self.metadata_path, 'w') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    json.dump(self.metadata, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
    
    def _tile_coord_to_id(self, tile_coord: Tuple[int, int, int]) -> str:
        """타일 좌표 → ID"""
        return f"{tile_coord[0]}_{tile_coord[1]}_{tile_coord[2]}"
    
    def _tile_id_to_coord(self, tile_id: str) -> Tuple[int, int, int]:
        """타일 ID → 좌표"""
        parts = tile_id.split('_')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    def save_tile(self, tile_coord: Tuple[int, int, int], tile_data: Dict):
        """
        타일 저장
        
        Args:
            tile_coord: (x, y, z) 타일 좌표
            tile_data: Gaussian 데이터 + bbox
        """
        tile_id = self._tile_coord_to_id(tile_coord)
        file_path = self.tiles_dir / f"tile_{tile_id}.npz"
        
        # NumPy 압축 저장
        save_dict = {
            'means': tile_data['means'],
            'quats': tile_data['quats'],
            'scales': tile_data['scales'],
            'opacities': tile_data['opacities'],
            'sh0': tile_data['sh0'],
            'shN': tile_data['shN']
        }
        
        # Densification stats (Optional)
        if 'accum_grad' in tile_data:
            save_dict['accum_grad'] = tile_data['accum_grad']
        if 'accum_count' in tile_data:
            save_dict['accum_count'] = tile_data['accum_count']
        if 'max_radii2D' in tile_data:
            save_dict['max_radii2D'] = tile_data['max_radii2D']
            
        # Atomic Save: Write to .tmp first, then rename
        # Use UUID to prevent race conditions between processes
        unique_suffix = str(uuid.uuid4())[:8]
        tmp_path = file_path.with_suffix(f'.tmp.{unique_suffix}.npz')
        try:
            np.savez_compressed(tmp_path, **save_dict)
            tmp_path.replace(file_path)  # Atomic rename
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise e
        
        # 메타데이터에 기록
        bbox = tile_data['bbox']
        self.metadata['tiles'][tile_id] = {
            'coord': list(tile_coord),
            'num_gaussians': len(tile_data['means']),
            'bbox': bbox.to_dict(),
            'file_path': str(file_path.relative_to(self.root_dir)),
            'file_size': file_path.stat().st_size
        }
        
        return tile_id
    
    def load_tile(self, tile_coord: Tuple[int, int, int]) -> Optional[Dict]:
        """타일 로드"""
        tile_id = self._tile_coord_to_id(tile_coord)
        
        if tile_id not in self.metadata['tiles']:
            return None
        
        tile_info = self.metadata['tiles'][tile_id]
        file_path = self.root_dir / tile_info['file_path']
        
        if not file_path.exists():
            print(f"[Warning] Tile file not found: {file_path}")
            return None
        
        # NPZ 로드
        data = np.load(file_path)
        
        result = {
            'means': data['means'],
            'quats': data['quats'],
            'scales': data['scales'],
            'opacities': data['opacities'],
            'sh0': data['sh0'],
            'shN': data['shN'],
            'bbox': BBox.from_dict(tile_info['bbox'])
        }
        
        if 'accum_grad' in data:
            result['accum_grad'] = data['accum_grad']
        if 'accum_count' in data:
            result['accum_count'] = data['accum_count']
        if 'max_radii2D' in data:
            result['max_radii2D'] = data['max_radii2D']
            
        return result
    
    def get_tile_info(self, tile_coord: Tuple[int, int, int]) -> Optional[Dict]:
        """타일 메타데이터만 가져오기"""
        tile_id = self._tile_coord_to_id(tile_coord)
        
        if tile_id not in self.metadata['tiles']:
            return None
        
        info = self.metadata['tiles'][tile_id].copy()
        info['bbox'] = BBox.from_dict(info['bbox'])
        info['file_size_mb'] = info['file_size'] / 1024**2
        
        return info
    
    def list_all_tiles(self) -> List[Tuple[int, int, int]]:
        """모든 타일 좌표 리스트"""
        return [tuple(info['coord']) for info in self.metadata['tiles'].values()]
    
    def get_statistics(self) -> Dict:
        """전체 통계"""
        tiles = self.metadata['tiles'].values()
        
        if not tiles:
            return {
                'num_tiles': 0,
                'total_gaussians': 0,
                'avg_gaussians_per_tile': 0,
                'min_gaussians': 0,
                'max_gaussians': 0,
                'total_size_gb': 0
            }
        
        num_gaussians = [t['num_gaussians'] for t in tiles]
        file_sizes = [t['file_size'] for t in tiles]
        
        return {
            'num_tiles': len(tiles),
            'total_gaussians': sum(num_gaussians),
            'avg_gaussians_per_tile': sum(num_gaussians) / len(num_gaussians),
            'min_gaussians': min(num_gaussians),
            'max_gaussians': max(num_gaussians),
            'total_size_gb': sum(file_sizes) / 1024**3
        }
    
    def close(self):
        """메타데이터 저장 (명시적)"""
        self._save_metadata()
        print(f"[TileStorage] Metadata saved to {self.metadata_path}")