#!/usr/bin/env python3
"""
Script to visualize how Gaussians are distributed across GPUs.
Creates color-coded PLY files showing which GPU owns which Gaussians.
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path


def read_ply_file(filepath):
    """Read PLY file and return Gaussian parameters."""
    properties = []
    with open(filepath, 'rb') as f:
        lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            lines.append(line)
            if line.startswith('property'):
                properties.append(line.split()[2])
            if line == 'end_header':
                header_end = f.tell()
                break
        
        # Parse vertex count
        for line in lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                break
        
        # Read binary data
        f.seek(header_end)
        data = np.fromfile(f, dtype=np.float32, count=vertex_count * len(properties))
        data = data.reshape(vertex_count, len(properties))
    
    return data, properties, vertex_count


def write_colored_ply(filepath, positions, colors, properties=None):
    """Write PLY file with custom colors."""
    if properties is None:
        properties = ['x', 'y', 'z', 'red', 'green', 'blue']
    
    vertex_count = len(positions)
    
    with open(filepath, 'wb') as f:
        # Write header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {vertex_count}\n'.encode('ascii'))
        
        # Position properties
        f.write(b'property float x\n')
        f.write(b'property float y\n') 
        f.write(b'property float z\n')
        
        # Color properties
        f.write(b'property uchar red\n')
        f.write(b'property uchar green\n')
        f.write(b'property uchar blue\n')
        
        f.write(b'end_header\n')
        
        # Write data
        for i in range(vertex_count):
            # Position (3 floats)
            f.write(struct.pack('<fff', positions[i][0], positions[i][1], positions[i][2]))
            # Color (3 uchar)
            f.write(struct.pack('<BBB', 
                int(colors[i][0] * 255), 
                int(colors[i][1] * 255), 
                int(colors[i][2] * 255)))


def simulate_gpu_distribution(total_gaussians, world_size=8):
    """Simulate how Gaussians would be distributed across GPUs."""
    gaussians_per_gpu = total_gaussians // world_size
    remainder = total_gaussians % world_size
    
    distribution = []
    start_idx = 0
    
    for rank in range(world_size):
        # Some GPUs get one extra Gaussian if there's remainder
        count = gaussians_per_gpu + (1 if rank < remainder else 0)
        end_idx = start_idx + count
        distribution.append((start_idx, end_idx, count))
        start_idx = end_idx
    
    return distribution


def create_gpu_colors(world_size=8):
    """Create distinct colors for each GPU."""
    colors = [
        [1.0, 0.0, 0.0],  # Red - GPU 0
        [0.0, 1.0, 0.0],  # Green - GPU 1  
        [0.0, 0.0, 1.0],  # Blue - GPU 2
        [1.0, 1.0, 0.0],  # Yellow - GPU 3
        [1.0, 0.0, 1.0],  # Magenta - GPU 4
        [0.0, 1.0, 1.0],  # Cyan - GPU 5
        [1.0, 0.5, 0.0],  # Orange - GPU 6
        [0.5, 0.0, 1.0],  # Purple - GPU 7
    ]
    return colors[:world_size]


def visualize_gpu_distribution(ply_file, output_dir="./gpu_vis", world_size=8):
    """Create visualizations showing GPU distribution."""
    print(f"Analyzing PLY file: {ply_file}")
    
    # Read PLY file
    data, properties, vertex_count = read_ply_file(ply_file)
    positions = data[:, 0:3]  # x, y, z
    
    print(f"Total Gaussians: {vertex_count}")
    
    # Simulate distribution
    distribution = simulate_gpu_distribution(vertex_count, world_size)
    gpu_colors = create_gpu_colors(world_size)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create colors array
    colors = np.zeros((vertex_count, 3))
    
    print("\nGPU Distribution:")
    for rank, (start_idx, end_idx, count) in enumerate(distribution):
        print(f"GPU {rank}: Gaussians {start_idx:6d}-{end_idx-1:6d} (count: {count:6d}) - {gpu_colors[rank]}")
        colors[start_idx:end_idx] = gpu_colors[rank]
    
    # Write combined visualization
    output_file = os.path.join(output_dir, f"gpu_distribution_combined.ply")
    write_colored_ply(output_file, positions, colors)
    print(f"\nCombined visualization saved to: {output_file}")
    
    # Write individual GPU files
    for rank, (start_idx, end_idx, count) in enumerate(distribution):
        if count > 0:
            gpu_positions = positions[start_idx:end_idx]
            gpu_colors = np.full((count, 3), gpu_colors[rank])
            
            output_file = os.path.join(output_dir, f"gpu_{rank}_gaussians.ply")
            write_colored_ply(output_file, gpu_positions, gpu_colors)
            print(f"GPU {rank} file saved to: {output_file}")
    
    # Create summary file
    summary_file = os.path.join(output_dir, "distribution_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Gaussian Distribution Analysis\n")
        f.write(f"==============================\n\n")
        f.write(f"Source PLY: {ply_file}\n")
        f.write(f"Total Gaussians: {vertex_count:,}\n")
        f.write(f"World Size (GPUs): {world_size}\n\n")
        
        f.write(f"Distribution:\n")
        for rank, (start_idx, end_idx, count) in enumerate(distribution):
            percentage = 100.0 * count / vertex_count
            f.write(f"GPU {rank}: {start_idx:8d} - {end_idx-1:8d} ({count:8d} gaussians, {percentage:5.1f}%)\n")
    
    print(f"\nSummary saved to: {summary_file}")
    return output_dir


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_gpu_distribution.py <ply_file> [world_size]")
        print("\nExample:")
        print("python visualize_gpu_distribution.py output/sillim_ew_mini_30/point_cloud/iteration_006000/point_cloud_i_06000_g_01717592_l_0.104.ply")
        return
    
    ply_file = sys.argv[1]
    world_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    if not os.path.exists(ply_file):
        print(f"Error: PLY file not found: {ply_file}")
        return
    
    # Create visualization
    output_dir = visualize_gpu_distribution(ply_file, world_size=world_size)
    
    print(f"\nVisualization complete!")
    print(f"Load the files in SuperSplat or other 3D viewer:")
    print(f"- Combined view: {output_dir}/gpu_distribution_combined.ply")
    print(f"- Individual GPUs: {output_dir}/gpu_*.ply")


if __name__ == "__main__":
    main()