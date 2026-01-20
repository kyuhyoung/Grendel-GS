#!/usr/bin/env python3
"""
Compare Z value ranges for different tiles in the point cloud.
"""

import numpy as np
import sys
sys.path.insert(0, '/data/kevin/work/etc/ooc-gaussian-splatting/Grendel-GS')

from plyfile import PlyData

PLY_PATH = '/data/dabeeo/samsung_dong_mini_30/sparse/0/points3D.ply'

# Tile bboxes from log
TILES = {
    'tile_0003 (1/4, SUCCESS)': {
        'x_min': -1290.24, 'x_max': -59.16,
        'y_min': -881.00, 'y_max': 19.66,
        'z_min': 15.69, 'z_max': 176.58
    },
    'tile_0009 (1/8)': {
        'x_min': -59.16, 'x_max': 556.37,
        'y_min': -881.00, 'y_max': -430.67,
        'z_min': 15.69, 'z_max': 176.58
    },
    'tile_0014 (1/32, FAIL)': {
        'x_min': -59.16, 'x_max': 248.61,
        'y_min': -205.50, 'y_max': 19.66,
        'z_min': 15.69, 'z_max': 176.58
    },
    'tile_0015 (1/64, FAIL)': {
        'x_min': -59.16, 'x_max': 94.72,
        'y_min': -205.50, 'y_max': 19.66,
        'z_min': 15.69, 'z_max': 176.58
    },
}

def main():
    print(f"Loading PLY: {PLY_PATH}")
    ply = PlyData.read(PLY_PATH)
    vertex = ply['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    print(f"Total points: {len(xyz)}")
    print(f"Full X range: {xyz[:,0].min():.2f} ~ {xyz[:,0].max():.2f}")
    print(f"Full Y range: {xyz[:,1].min():.2f} ~ {xyz[:,1].max():.2f}")
    print(f"Full Z range: {xyz[:,2].min():.2f} ~ {xyz[:,2].max():.2f}")
    print()

    for name, bbox in TILES.items():
        mask = (
            (xyz[:, 0] >= bbox['x_min']) & (xyz[:, 0] <= bbox['x_max']) &
            (xyz[:, 1] >= bbox['y_min']) & (xyz[:, 1] <= bbox['y_max']) &
            (xyz[:, 2] >= bbox['z_min']) & (xyz[:, 2] <= bbox['z_max'])
        )
        pts = xyz[mask]

        print(f"{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        print(f"  Bbox X: {bbox['x_min']:.1f} ~ {bbox['x_max']:.1f} (size: {bbox['x_max']-bbox['x_min']:.1f})")
        print(f"  Bbox Y: {bbox['y_min']:.1f} ~ {bbox['y_max']:.1f} (size: {bbox['y_max']-bbox['y_min']:.1f})")
        print(f"  Points in tile: {len(pts)}")

        if len(pts) > 0:
            print(f"  Actual X: {pts[:,0].min():.1f} ~ {pts[:,0].max():.1f} (span: {pts[:,0].max()-pts[:,0].min():.1f})")
            print(f"  Actual Y: {pts[:,1].min():.1f} ~ {pts[:,1].max():.1f} (span: {pts[:,1].max()-pts[:,1].min():.1f})")
            print(f"  Actual Z: {pts[:,2].min():.1f} ~ {pts[:,2].max():.1f} (span: {pts[:,2].max()-pts[:,2].min():.1f})")

            z_pcts = np.percentile(pts[:,2], [0, 10, 25, 50, 75, 90, 100])
            print(f"  Z percentiles:")
            print(f"    0%:   {z_pcts[0]:.1f}")
            print(f"    10%:  {z_pcts[1]:.1f}")
            print(f"    25%:  {z_pcts[2]:.1f}")
            print(f"    50%:  {z_pcts[3]:.1f}")
            print(f"    75%:  {z_pcts[4]:.1f}")
            print(f"    90%:  {z_pcts[5]:.1f}")
            print(f"    100%: {z_pcts[6]:.1f}")
        print()

if __name__ == "__main__":
    main()
