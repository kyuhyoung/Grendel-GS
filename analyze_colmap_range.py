#!/usr/bin/env python3
"""
Analyze COLMAP reconstruction to calculate bounding boxes and camera positions.
"""

import sys
import numpy as np
import os

def read_points3d(points3d_path):
    """Read points3D.txt and extract 3D coordinates."""
    points = []
    
    if not os.path.exists(points3d_path):
        print(f"Error: {points3d_path} not found")
        return None
    
    with open(points3d_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 6:  # ID X Y Z R G B ERROR TRACK_LIST
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
                except ValueError:
                    continue
    
    return np.array(points) if points else None

def read_cameras(cameras_path):
    """Read camera intrinsics from cameras.txt."""
    cameras = {}
    
    if not os.path.exists(cameras_path):
        print(f"Error: {cameras_path} not found")
        return None
    
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 4:  # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
                try:
                    camera_id = int(parts[0])
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = [float(p) for p in parts[4:]]
                    
                    cameras[camera_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'params': params
                    }
                except (ValueError, IndexError):
                    continue
    
    return cameras

def read_images(images_path):
    """Read camera positions and camera IDs from images.txt."""
    camera_positions = []
    camera_ids = []
    
    if not os.path.exists(images_path):
        print(f"Error: {images_path} not found")
        return None, None
    
    with open(images_path, 'r') as f:
        lines = f.readlines()
    
    # Parse images.txt - each image has 2 lines: image info and points2D
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 10:  # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            try:
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                camera_id = int(parts[8])
                camera_positions.append([tx, ty, tz])
                camera_ids.append(camera_id)
                # Skip the next line (POINTS2D)
                i += 2
            except (ValueError, IndexError):
                i += 1
        else:
            i += 1
    
    return np.array(camera_positions) if camera_positions else None, camera_ids

def calculate_bounding_box(points):
    """Calculate bounding box from 3D points."""
    if points is None or len(points) == 0:
        return None
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (min_coords + max_coords) / 2
    
    return {
        'min': min_coords,
        'max': max_coords,
        'center': center,
        'size': max_coords - min_coords
    }

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points."""
    return np.linalg.norm(point1 - point2)

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_colmap_range.py <colmap_output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    
    # File paths
    points3d_path = os.path.join(sparse_dir, "points3D.txt")
    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    images_path = os.path.join(sparse_dir, "images.txt")
    
    print("="*60)
    print("COLMAP RECONSTRUCTION ANALYSIS")
    print("="*60)
    
    # Read 3D points
    print("Reading 3D points...")
    points = read_points3d(points3d_path)
    
    if points is not None and len(points) > 0:
        print(f"Found {len(points)} 3D points")
        
        # Calculate point cloud bounding box
        bbox = calculate_bounding_box(points)
        
        print("\nPOINT CLOUD BOUNDING BOX:")
        print("-" * 40)
        print(f"Min coordinates: ({bbox['min'][0]:.6f}, {bbox['min'][1]:.6f}, {bbox['min'][2]:.6f})")
        print(f"Max coordinates: ({bbox['max'][0]:.6f}, {bbox['max'][1]:.6f}, {bbox['max'][2]:.6f})")
        print(f"Center:          ({bbox['center'][0]:.6f}, {bbox['center'][1]:.6f}, {bbox['center'][2]:.6f})")
        print(f"Size (WxHxD):    ({bbox['size'][0]:.6f}, {bbox['size'][1]:.6f}, {bbox['size'][2]:.6f})")
        
        # Corner coordinates
        print(f"\nCorner coordinates:")
        print(f"Front-Top-Left:     ({bbox['min'][0]:.6f}, {bbox['max'][1]:.6f}, {bbox['min'][2]:.6f})")
        print(f"Back-Bottom-Right:  ({bbox['max'][0]:.6f}, {bbox['min'][1]:.6f}, {bbox['max'][2]:.6f})")
    else:
        print("No valid 3D points found")
        bbox = None
    
    # Read camera intrinsics
    print("\nReading camera intrinsics...")
    cameras = read_cameras(cameras_path)
    
    if cameras:
        print(f"Found {len(cameras)} unique camera(s)")
        
        print("\nCAMERA INTRINSICS:")
        print("-" * 40)
        for camera_id, camera_info in cameras.items():
            print(f"Camera ID {camera_id}:")
            print(f"  Model: {camera_info['model']}")
            print(f"  Resolution: {camera_info['width']} x {camera_info['height']}")
            print(f"  Parameters: {camera_info['params']}")
            
            # Interpret parameters based on camera model
            if camera_info['model'] == 'PINHOLE':
                if len(camera_info['params']) >= 4:
                    fx, fy, cx, cy = camera_info['params'][:4]
                    print(f"    Focal length: fx={fx:.2f}, fy={fy:.2f}")
                    print(f"    Principal point: cx={cx:.2f}, cy={cy:.2f}")
            elif camera_info['model'] == 'SIMPLE_RADIAL':
                if len(camera_info['params']) >= 3:
                    f, cx, cy, k = camera_info['params'][0], camera_info['params'][1], camera_info['params'][2], camera_info['params'][3] if len(camera_info['params']) > 3 else 0
                    print(f"    Focal length: f={f:.2f}")
                    print(f"    Principal point: cx={cx:.2f}, cy={cy:.2f}")
                    if len(camera_info['params']) > 3:
                        print(f"    Radial distortion: k={k:.6f}")
            elif camera_info['model'] == 'OPENCV':
                if len(camera_info['params']) >= 8:
                    fx, fy, cx, cy, k1, k2, p1, p2 = camera_info['params'][:8]
                    print(f"    Focal length: fx={fx:.2f}, fy={fy:.2f}")
                    print(f"    Principal point: cx={cx:.2f}, cy={cy:.2f}")
                    print(f"    Distortion: k1={k1:.6f}, k2={k2:.6f}, p1={p1:.6f}, p2={p2:.6f}")
            print()
    
    # Read camera positions
    print("Reading camera positions...")
    camera_positions, camera_ids = read_images(images_path)
    
    if camera_positions is not None and len(camera_positions) > 0:
        print(f"Found {len(camera_positions)} camera positions")
        
        # Calculate camera center
        camera_center = np.mean(camera_positions, axis=0)
        
        print("\nCAMERA POSITIONS:")
        print("-" * 40)
        print(f"Camera center: ({camera_center[0]:.6f}, {camera_center[1]:.6f}, {camera_center[2]:.6f})")
        
        # Camera bounding box
        camera_bbox = calculate_bounding_box(camera_positions)
        print(f"Camera min:    ({camera_bbox['min'][0]:.6f}, {camera_bbox['min'][1]:.6f}, {camera_bbox['min'][2]:.6f})")
        print(f"Camera max:    ({camera_bbox['max'][0]:.6f}, {camera_bbox['max'][1]:.6f}, {camera_bbox['max'][2]:.6f})")
        print(f"Camera spread: ({camera_bbox['size'][0]:.6f}, {camera_bbox['size'][1]:.6f}, {camera_bbox['size'][2]:.6f})")
        
        # Distance from camera center to point cloud center
        if bbox is not None:
            distance = calculate_distance(camera_center, bbox['center'])
            print(f"\nDISTANCE ANALYSIS:")
            print("-" * 40)
            print(f"Distance from camera center to point cloud center: {distance:.6f}")
            
            # Scale analysis
            point_cloud_diagonal = np.linalg.norm(bbox['size'])
            camera_spread_diagonal = np.linalg.norm(camera_bbox['size'])
            print(f"Point cloud diagonal: {point_cloud_diagonal:.6f}")
            print(f"Camera spread diagonal: {camera_spread_diagonal:.6f}")
            print(f"Scale ratio (PC/Camera): {point_cloud_diagonal/camera_spread_diagonal:.2f}")
        
    else:
        print("No valid camera positions found")
    
    print("="*60)

if __name__ == "__main__":
    main()