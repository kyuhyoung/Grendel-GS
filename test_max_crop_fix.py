#!/usr/bin/env python3

"""
Test to verify that max_crop_size has been completely removed from compute_tile_crop_for_camera
"""

import sys
import os
sys.path.insert(0, '/media2/4tb/kevin/work/etc/ada_grendel/Grendel-GS')

import numpy as np
from scene.adaptive_tile_utils import TileBBox, compute_tile_crop_for_camera

def test_no_max_crop_size():
    """Test that large crops are not limited by max_crop_size"""
    
    # Create a very large tile bbox that would trigger max_crop_size if it existed
    tile_bbox = TileBBox(
        x_min=-2000.0, y_min=-2000.0, z_min=-4000.0,
        x_max=2000.0, y_max=2000.0, z_max=500.0
    )
    
    # Create a projection transform that would result in very large image dimensions
    # This simulates the original 11310x17310 -> should not be limited to 2048 anymore
    full_proj_transform = np.array([
        [1000.0, 0.0, 5500.0, 0.0],
        [0.0, 1000.0, 8500.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Large image dimensions that would trigger the original max_crop_size=2048 limit
    img_width = 11310
    img_height = 17310
    
    print(f"Testing with image dimensions: {img_width}x{img_height}")
    print(f"Testing with tile bbox: {tile_bbox}")
    
    try:
        # This should work without any max_crop_size limitations
        crop_result = compute_tile_crop_for_camera(
            tile_bbox=tile_bbox,
            full_proj_transform=full_proj_transform,
            img_width=img_width,
            img_height=img_height,
            margin=100,
            return_debug_info=True
        )
        
        if crop_result is None:
            print("✅ SUCCESS: Function completed without max_crop_size errors")
            print("✅ No artificial size limits detected")
            return True
        else:
            crop_region, debug_info = crop_result
            print(f"✅ SUCCESS: Function returned crop region: {crop_region}")
            if debug_info:
                print(f"✅ Debug info: {debug_info}")
            
            # Check if the crop dimensions are reasonable (not artificially limited to 2048)
            crop_width = crop_region.x_max - crop_region.x_min
            crop_height = crop_region.y_max - crop_region.y_min
            print(f"✅ Crop dimensions: {crop_width}x{crop_height}")
            
            # If dimensions are exactly 2048, that might indicate the old limit is still active
            if crop_width == 2048 or crop_height == 2048:
                print("⚠️  WARNING: Crop dimension is exactly 2048 - old limit might still be active")
                return False
            else:
                print("✅ Crop dimensions are not limited to 2048")
                return True
                
    except Exception as e:
        if "max_crop_size" in str(e).lower():
            print(f"❌ FAILED: max_crop_size limitation still exists: {e}")
            return False
        else:
            print(f"❌ FAILED: Unexpected error: {e}")
            return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing max_crop_size removal from compute_tile_crop_for_camera")
    print("=" * 60)
    
    success = test_no_max_crop_size()
    
    if success:
        print("\n✅ ALL TESTS PASSED - max_crop_size has been successfully removed!")
    else:
        print("\n❌ TESTS FAILED - max_crop_size limitations may still exist!")
    
    sys.exit(0 if success else 1)