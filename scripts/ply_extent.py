#!/usr/bin/env python3
"""
PLY 파일의 x, y, z 좌표 범위 계산
Usage: python scripts/ply_extent.py /path/to/points3D.ply
"""
import sys
import struct
from pathlib import Path


def read_ply_extent(ply_path: str):
    """Read PLY file and compute xyz extent."""
    with open(ply_path, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse header info
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
        print(f"Format: {'binary' if is_binary else 'ascii'}")

        if not is_binary:
            # ASCII format
            x_vals, y_vals, z_vals = [], [], []
            for _ in range(num_vertices):
                line = f.readline().decode('utf-8').strip()
                parts = line.split()
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
                z_vals.append(float(parts[2]))
        else:
            # Binary format: x,y,z (float), nx,ny,nz (float), r,g,b (uchar)
            # = 3*4 + 3*4 + 3*1 = 27 bytes per vertex
            vertex_size = 27
            endian = '<' if is_little_endian else '>'

            x_vals, y_vals, z_vals = [], [], []
            for _ in range(num_vertices):
                data = f.read(vertex_size)
                x, y, z = struct.unpack(f'{endian}fff', data[:12])
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)

        print(f"\nExtent:")
        print(f"  X: {x_min:.6f} ~ {x_max:.6f} (size: {x_max - x_min:.6f})")
        print(f"  Y: {y_min:.6f} ~ {y_max:.6f} (size: {y_max - y_min:.6f})")
        print(f"  Z: {z_min:.6f} ~ {z_max:.6f} (size: {z_max - z_min:.6f})")

        print(f"\nBBox string:")
        print(f"  {x_min},{y_min},{z_min},{x_max},{y_max},{z_max}")

        return (x_min, y_min, z_min, x_max, y_max, z_max)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ply_extent.py /path/to/points3D.ply")
        sys.exit(1)

    read_ply_extent(sys.argv[1])
