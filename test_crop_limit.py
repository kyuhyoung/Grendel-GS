#!/usr/bin/env python3
"""
Test script to verify crop size limiting logic.
"""

import sys
import os
import numpy as np

# Add the scene path to sys.path
sys.path.append('Grendel-GS/scene')

from adaptive_tile_utils import TileBBox, compute_tile_crop_for_camera, build_full_proj_transform

def test_crop_limiting():
    """Test that crop size limiting works correctly."""
    
    print("Testing crop size limiting logic...")
    
    # Create a very large tile bbox (simulating initial scene-wide tile)
    tile_bbox = TileBBox(
        x_min=-1000, x_max=1000,
        y_min=-800, y_max=800, 
        z_min=0, z_max=100
    )
    
    # Create a projection transform that would make the tile very large on screen
    # Simulate a camera looking at the large scene from close distance
    R = np.eye(3)  # Identity rotation (camera looking down -Z axis)
    T = np.array([0, 0, -50])  # Camera 50 units back
    fov_x = np.pi/2  # 90 degrees (wide FOV)
    fov_y = np.pi/2
    width = 11310  # Large image size (as seen in logs)
    height = 17310
    
    full_proj = build_full_proj_transform(R, T, fov_x, fov_y, width, height)
    
    # Camera position for inside check
    camera_pos = np.array([0, 0, -50])
    
    print("Testing with max_crop_size=None (unlimited)...")
    crop_unlimited = compute_tile_crop_for_camera(
        tile_bbox, full_proj, width, height, margin=100,
        max_crop_size=None, camera_position=camera_pos
    )
    
    print("Testing with max_crop_size=3072 (limited)...")
    crop_limited = compute_tile_crop_for_camera(
        tile_bbox, full_proj, width, height, margin=100,
        max_crop_size=3072, camera_position=camera_pos
    )
    
    if crop_unlimited:
        print(f"Unlimited crop: {crop_unlimited.width}x{crop_unlimited.height} "
              f"({crop_unlimited.x_min},{crop_unlimited.y_min})-({crop_unlimited.x_max},{crop_unlimited.y_max})")
    else:
        print("Unlimited crop: None")
        
    if crop_limited:
        print(f"Limited crop: {crop_limited.width}x{crop_limited.height} "
              f"({crop_limited.x_min},{crop_limited.y_min})-({crop_limited.x_max},{crop_limited.y_max})")
    else:
        print("Limited crop: None")
    
    # Verify that the limited crop is indeed limited
    if crop_unlimited and crop_limited:
        if (crop_unlimited.width > 3072 or crop_unlimited.height > 3072) and \
           (crop_limited.width <= 3072 and crop_limited.height <= 3072):
            print("✅ SUCCESS: Crop size limiting is working correctly!")
            return True
        else:
            print("❌ FAILED: Crop size limiting is not working")
            return False
    else:
        print("❌ FAILED: One or both crops are None")
        return False

if __name__ == "__main__":
    success = test_crop_limiting()
    sys.exit(0 if success else 1)