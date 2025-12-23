#!/usr/bin/env python3
"""
Create Gaussians from COLMAP PLY using Grendel-GS logic and tile them for training.
This script replaces the two-step process (create_gaussians -> initialize_tiled_scene)
with a single pipeline that ensures consistent initialization parameters.
"""

import argparse
import sys
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tqdm import tqdm
import json

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Grendel-GS"))

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
import utils.general_utils as utils
from plyfile import PlyData
from src.tile_storage import TileStorage, BBox
from src.spatial_index import UniformGrid

def load_colmap_ply(path):
    print(f"Loading PLY from {path}...")
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def main():
    parser = argparse.ArgumentParser(description="Create Gaussians and Tiles from COLMAP PLY")
    parser.add_argument("--input_ply", required=True, help="Input COLMAP points3D.ply")
    parser.add_argument("--output_dir", required=True, help="Output directory for tiles")
    parser.add_argument("--grid_size", nargs=3, type=int, default=[32, 32, 1], help="Grid size (X Y Z)")
    parser.add_argument("--overlap", type=float, default=10.0, help="Overlap in meters")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--opaque", action="store_true", help="Force opacity to 1.0 (logit 10.0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory {output_dir} already exists. Use --overwrite to replace.")
            return
    output_dir.mkdir(parents=True)

    if not torch.cuda.is_available():
        print("Error: CUDA is required")
        return

    # Setup Grendel args
    runtime_args = SimpleNamespace(
        gaussians_distribution=False,
        distributed_save=False,
        drop_initial_3dgs_p=0.0,
        bsz=1,
        lr_scale_mode="linear",
        lr_scale_pos_and_scale=1.0,
        check_gpu_memory=False,
        check_cpu_memory=False,
        random_background=False,
        white_background=False,
        image_distribution=False,
        image_distribution_mode="single",
        log_folder=str(output_dir),
    )
    
    utils.set_args(runtime_args)
    utils.GLOBAL_RANK = 0
    utils.WORLD_SIZE = 1
    utils.DEFAULT_GROUP = utils.SingleGPUGroup()
    utils.set_log_file(sys.stdout)
    
    # 1. Load PLY
    pcd = load_colmap_ply(args.input_ply)
    
    # 2. Create Gaussian Model
    print("Creating Gaussian Model...")
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    
    # 3. Initialize from PCD (This uses simple_knn internally)
    print("Initializing from PCD (running create_from_pcd)...")
    gaussians.create_from_pcd(pcd, spatial_lr_scale=1.0)
    
    # 4. Optional: Boost opacity
    if args.opaque:
        print("Boosting opacity to 1.0 (logit 10.0)...")
        new_opacity = torch.ones_like(gaussians._opacity) * 10.0
        gaussians._opacity.data = new_opacity
    
    # 5. Extract parameters for tiling
    print("Extracting parameters...")
    with torch.no_grad():
        means = gaussians._xyz.detach().cpu().numpy()
        quats = gaussians._rotation.detach().cpu().numpy()
        scales_log = gaussians._scaling.detach().cpu().numpy()
        # Flatten opacity to [N] for consistency with TileStorage
        opacities_logit = gaussians._opacity.detach().cpu().numpy().flatten()
        sh0 = gaussians._features_dc.detach().cpu().numpy() # [N, 1, 3]
        shN = gaussians._features_rest.detach().cpu().numpy() # [N, 15, 3]
        
        # Calculate linear scales for spatial indexing
        scales_linear = torch.exp(gaussians._scaling).detach().cpu().numpy()

    # 6. Build Spatial Index
    print(f"Building spatial index (Grid: {args.grid_size})...")
    spatial_index = UniformGrid(
        means, 
        grid_size=tuple(args.grid_size),
        overlap_meters=args.overlap
    )
    
    # 7. Assign to tiles
    print("Assigning Gaussians to tiles...")
    tile_assignments = spatial_index.assign_points_to_tiles(
        means,
        scales_linear
    )
    
    # 8. Save tiles
    print("Saving tiles...")
    storage = TileStorage(output_dir)
    tile_count = 0
    
    for tile_coord, gaussian_ids in tqdm(tile_assignments.items(), desc="Saving"):
        if len(gaussian_ids) == 0:
            continue
        
        tile_data = {
            'means': means[gaussian_ids],
            'quats': quats[gaussian_ids],
            'scales': scales_log[gaussian_ids],
            'opacities': opacities_logit[gaussian_ids],
            'sh0': sh0[gaussian_ids],
            'shN': shN[gaussian_ids],
        }
        
        bbox_min = tile_data['means'].min(axis=0)
        bbox_max = tile_data['means'].max(axis=0)
        tile_data['bbox'] = BBox(min_xyz=bbox_min, max_xyz=bbox_max)
        
        storage.save_tile(tile_coord, tile_data)
        tile_count += 1
        
    # 9. Save Metadata
    stats = storage.get_statistics()
    metadata = {
        'source_ply': str(args.input_ply),
        'total_gaussians': len(means),
        'grid_size': list(args.grid_size),
        'overlap_meters': args.overlap,
        'num_tiles': stats['num_tiles'],
        'scene_bbox': {
            'min': spatial_index.scene_min.tolist(),
            'max': spatial_index.scene_max.tolist(),
        },
        'tile_size_meters': spatial_index.tile_size.tolist(),
        'duplication_ratio': stats['total_gaussians'] / len(means),
        'activations_applied': False, # Stored in log/logit space
        'scale_stats': {
            'log_space': {
                'min': float(scales_log.min()),
                'max': float(scales_log.max()),
                'mean': float(scales_log.mean())
            },
            'linear_space': {
                'min': float(scales_linear.min()),
                'max': float(scales_linear.max()),
                'mean': float(scales_linear.mean())
            }
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    storage.close()
    
    print("\n" + "="*70)
    print("Tiling Complete!")
    print(f"Output: {output_dir}")
    print(f"Total Gaussians: {len(means):,}")
    print(f"Tiles Created: {tile_count:,}")
    print("="*70)

if __name__ == "__main__":
    main()
