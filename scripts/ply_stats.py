#!/usr/bin/env python3
"""
PLY 파일의 x, y, z 좌표 통계 및 분포 확인
Usage: python scripts/ply_stats.py /path/to/points3D.ply
"""
import sys
import struct
import numpy as np


def read_ply_stats(ply_path: str):
    """Read PLY file and compute xyz statistics."""
    with open(ply_path, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        num_vertices = 0
        is_binary = False
        is_little_endian = True

        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif 'binary_little_endian' in line:
                is_binary = True
                is_little_endian = True
            elif 'binary_big_endian' in line:
                is_binary = True
                is_little_endian = False

        print(f"File: {ply_path}")
        print(f"Vertices: {num_vertices}")

        if is_binary:
            # Binary format: x,y,z (float), nx,ny,nz (float), r,g,b (uchar)
            vertex_size = 27
            endian = '<' if is_little_endian else '>'

            xyz = np.zeros((num_vertices, 3), dtype=np.float32)
            for i in range(num_vertices):
                data = f.read(vertex_size)
                xyz[i] = struct.unpack(f'{endian}fff', data[:12])
        else:
            xyz = np.zeros((num_vertices, 3), dtype=np.float32)
            for i in range(num_vertices):
                line = f.readline().decode('utf-8').strip()
                parts = line.split()
                xyz[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    print(f"\n{'='*60}")
    print("Basic Statistics")
    print('='*60)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vals = xyz[:, i]
        print(f"\n{axis}:")
        print(f"  Min:    {vals.min():.3f}")
        print(f"  Max:    {vals.max():.3f}")
        print(f"  Range:  {vals.max() - vals.min():.3f}")
        print(f"  Mean:   {vals.mean():.3f}")
        print(f"  Median: {np.median(vals):.3f}")
        print(f"  Std:    {vals.std():.3f}")

    print(f"\n{'='*60}")
    print("Percentiles (outlier detection)")
    print('='*60)
    percentiles = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vals = xyz[:, i]
        print(f"\n{axis}:")
        for p in percentiles:
            print(f"  {p:5.1f}%: {np.percentile(vals, p):.3f}")

    # Check for outliers (beyond 3 std)
    print(f"\n{'='*60}")
    print("Outlier Check (beyond 3 std)")
    print('='*60)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vals = xyz[:, i]
        mean, std = vals.mean(), vals.std()
        lower, upper = mean - 3*std, mean + 3*std
        mask = (vals >= lower) & (vals <= upper)
        outliers = np.sum(~mask)
        clean_vals = vals[mask]
        print(f"{axis}: {outliers} outliers ({outliers/len(vals)*100:.2f}%)")
        print(f"   3-std threshold: {lower:.3f} ~ {upper:.3f}")
        print(f"   Actual range (without outliers): {clean_vals.min():.3f} ~ {clean_vals.max():.3f}")

    # Summary: recommended bbox
    print(f"\n{'='*60}")
    print("Recommended BBox (99.8% of points)")
    print('='*60)
    pct_low, pct_high = 0.1, 99.9
    x_min, y_min, z_min = np.percentile(xyz, pct_low, axis=0)
    x_max, y_max, z_max = np.percentile(xyz, pct_high, axis=0)
    print(f"X: {x_min:.3f} ~ {x_max:.3f} (size: {x_max-x_min:.3f})")
    print(f"Y: {y_min:.3f} ~ {y_max:.3f} (size: {y_max-y_min:.3f})")
    print(f"Z: {z_min:.3f} ~ {z_max:.3f} (size: {z_max-z_min:.3f})")
    print(f"\nBBox string: {x_min},{y_min},{z_min},{x_max},{y_max},{z_max}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ply_stats.py /path/to/points3D.ply")
        sys.exit(1)

    read_ply_stats(sys.argv[1])
