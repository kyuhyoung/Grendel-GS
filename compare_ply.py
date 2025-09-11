#!/usr/bin/env python3
"""
PLY File Comparison Tool for 3D Gaussian Splatting

Compares two PLY files containing 3D Gaussian parameters and reports differences.
Minimal dependencies: plyfile, numpy, os

Usage:
    python compare_ply.py file1.ply file2.ply
"""

import sys
import os
import numpy as np
from plyfile import PlyData

def load_ply_data(ply_path):
    """Load PLY file and extract gaussian parameters."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    try:
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        
        # Get field names safely
        if hasattr(vertex, 'dtype') and hasattr(vertex.dtype, 'names'):
            field_names = vertex.dtype.names
        elif hasattr(vertex, 'data') and hasattr(vertex.data, 'dtype'):
            field_names = vertex.data.dtype.names
        else:
            # Try to access the vertex element directly
            vertex_data = vertex.data if hasattr(vertex, 'data') else vertex
            field_names = vertex_data.dtype.names if hasattr(vertex_data.dtype, 'names') else []
        
        if not field_names:
            raise ValueError("Cannot access field names from PLY vertex data")
        
        # Extract common gaussian parameters
        data = {}
        
        # Position (xyz)
        if 'x' in field_names and 'y' in field_names and 'z' in field_names:
            data['xyz'] = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        
        # Features (colors/spherical harmonics)
        feature_keys = [name for name in field_names if name.startswith('f_dc_') or name.startswith('f_rest_')]
        if feature_keys:
            features = np.column_stack([vertex[key] for key in sorted(feature_keys)])
            data['features'] = features
        
        # Opacity
        if 'opacity' in field_names:
            data['opacity'] = vertex['opacity']
        
        # Scaling
        scale_keys = [name for name in field_names if name.startswith('scale_')]
        if scale_keys:
            scaling = np.column_stack([vertex[key] for key in sorted(scale_keys)])
            data['scaling'] = scaling
        
        # Rotation
        rot_keys = [name for name in field_names if name.startswith('rot_')]
        if rot_keys:
            rotation = np.column_stack([vertex[key] for key in sorted(rot_keys)])
            data['rotation'] = rotation
        
        return data, len(vertex)
        
    except Exception as e:
        raise ValueError(f"Error loading PLY file {ply_path}: {str(e)}")

def compare_arrays(arr1, arr2, name, tolerance=1e-6):
    """Compare two numpy arrays and report differences."""
    if arr1 is None and arr2 is None:
        return True
    
    if arr1 is None or arr2 is None:
        print(f"❌ {name}: One file missing this parameter")
        return False
    
    if arr1.shape != arr2.shape:
        print(f"❌ {name}: Shape mismatch - {arr1.shape} vs {arr2.shape}")
        return False
    
    # Check for exact equality first
    if np.array_equal(arr1, arr2):
        print(f"✅ {name}: Identical ({arr1.shape})")
        return True
    
    # Check with tolerance
    if np.allclose(arr1, arr2, atol=tolerance, rtol=tolerance):
        max_diff = np.max(np.abs(arr1 - arr2))
        print(f"⚠️  {name}: Nearly identical (max diff: {max_diff:.2e}, shape: {arr1.shape})")
        return True
    
    # Calculate statistics for differences
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    print(f"❌ {name}: Different (shape: {arr1.shape})")
    print(f"   Max diff: {max_diff:.6e}")
    print(f"   Mean diff: {mean_diff:.6e}")
    print(f"   Std diff: {std_diff:.6e}")
    
    return False

def compare_ply_files(file1, file2, tolerance=1e-6):
    """Compare two PLY files and report differences."""
    print(f"Comparing PLY files:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print(f"  Tolerance: {tolerance}")
    print("-" * 60)
    
    try:
        # Load both files
        data1, count1 = load_ply_data(file1)
        data2, count2 = load_ply_data(file2)
        
        print(f"Gaussian count: {count1} vs {count2}")
        
        if count1 != count2:
            print("❌ Different number of gaussians!")
            return False
        
        # Compare each parameter type
        all_keys = set(data1.keys()) | set(data2.keys())
        all_identical = True
        
        for key in sorted(all_keys):
            arr1 = data1.get(key)
            arr2 = data2.get(key)
            
            is_identical = compare_arrays(arr1, arr2, key, tolerance)
            if not is_identical:
                all_identical = False
        
        print("-" * 60)
        if all_identical:
            print("✅ Files are identical (within tolerance)")
        else:
            print("❌ Files have differences")
        
        return all_identical
        
    except Exception as e:
        print(f"❌ Error comparing files: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_ply.py file1.ply file2.ply")
        print("       python compare_ply.py file1.ply file2.ply")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    # Default tolerance for floating point comparison
    tolerance = 1e-6
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"❌ File not found: {file1}")
        sys.exit(1)
    
    if not os.path.exists(file2):
        print(f"❌ File not found: {file2}")
        sys.exit(1)
    
    # Compare files
    are_identical = compare_ply_files(file1, file2, tolerance)
    
    # Exit with appropriate code
    sys.exit(0 if are_identical else 1)

if __name__ == "__main__":
    main()