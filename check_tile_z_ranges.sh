#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG_FILE="./check_tile_z_ranges.log"
> "${LOG_FILE}"
exec > >(tee "${LOG_FILE}") 2>&1

python -c "
import numpy as np
import os
import sys
sys.path.insert(0, 'Grendel-GS')

ply_path = '/data/dabeeo/samsung_dong_mini_30/sparse/0/points3D.ply'
bin_path = '/data/dabeeo/samsung_dong_mini_30/sparse/0/points3D.bin'

print(f'PLY exists: {os.path.exists(ply_path)}')
print(f'BIN exists: {os.path.exists(bin_path)}')

if os.path.exists(ply_path):
    print(f'Loading PLY: {ply_path}')
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    xyz = np.stack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']], axis=1)
elif os.path.exists(bin_path):
    print(f'Loading BIN: {bin_path}')
    from scene.colmap_loader import read_points3D_binary
    xyz, _, _ = read_points3D_binary(bin_path)
else:
    print('ERROR: No point cloud file found!')
    sys.exit(1)

print(f'Loaded from: {ply_path if os.path.exists(ply_path) else bin_path}')

print(f'Total points: {len(xyz)}')
print(f'Full Z range: {xyz[:,2].min():.2f} ~ {xyz[:,2].max():.2f}')
print()

tiles = {
    'tile_0003 (1/4, SUCCESS)': (-1290.24, -59.16, -881.00, 19.66, 15.69, 176.58),
    'tile_0009 (1/8)': (-59.16, 556.37, -881.00, -430.67, 15.69, 176.58),
    'tile_0014 (1/32, FAIL)': (-59.16, 248.61, -205.50, 19.66, 15.69, 176.58),
    'tile_0015 (1/64, FAIL)': (-59.16, 94.72, -205.50, 19.66, 15.69, 176.58),
}

for name, (x0,x1,y0,y1,z0,z1) in tiles.items():
    mask = (xyz[:,0]>=x0)&(xyz[:,0]<=x1)&(xyz[:,1]>=y0)&(xyz[:,1]<=y1)&(xyz[:,2]>=z0)&(xyz[:,2]<=z1)
    pts = xyz[mask]
    print('='*60)
    print(name)
    print('='*60)
    print(f'  Bbox: X[{x0:.0f}~{x1:.0f}] Y[{y0:.0f}~{y1:.0f}]')
    print(f'  Points: {len(pts)}')
    if len(pts) > 0:
        print(f'  Z range: {pts[:,2].min():.1f} ~ {pts[:,2].max():.1f} (span: {pts[:,2].max()-pts[:,2].min():.1f})')
        pct = np.percentile(pts[:,2], [0,25,50,75,100])
        print(f'  Z pct [0,25,50,75,100]: [{pct[0]:.1f}, {pct[1]:.1f}, {pct[2]:.1f}, {pct[3]:.1f}, {pct[4]:.1f}]')
    print()
"
