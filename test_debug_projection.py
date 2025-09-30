#!/usr/bin/env python3
"""
Debug script to compare projection between colmap_visualizer and colmap_loader
"""

import sys
import os
sys.path.insert(0, '/workspace/Grendel-GS')
sys.path.insert(0, '/workspace/Grendel-GS/scripts/experiments')

import numpy as np
from colmap_visualizer import COLMAPVisualizer

# Initialize colmap_visualizer
viz = COLMAPVisualizer('/data/samsung_dong_mini_5/sparse/0')

print("Loading COLMAP data...")
try:
    viz.read_cameras_txt()
    viz.read_images_txt()
    viz.read_points3d_txt()

    print(f"Loaded {len(viz.cameras)} cameras")
    print(f"Loaded {len(viz.images)} images")
    print(f"Loaded {len(viz.points3d)} 3D points")

    # Get first 3D point
    first_point_id = list(viz.points3d.keys())[0]
    first_point = viz.points3d[first_point_id]['xyz']

    print(f"\nTesting projection for:")
    print(f"Point ID: {first_point_id}")
    print(f"Point coords: {first_point}")
    print(f"Camera ID: 36")

    # Test projection for camera 36
    if 36 in viz.images:
        print("\n" + "="*60)
        print("COLMAP_VISUALIZER PROJECTION TEST")
        print("="*60)

        is_in_view, u, v = viz.project_point_to_image(first_point, 36)

        print(f"\nResult:")
        print(f"  Is in view: {is_in_view}")
        print(f"  Projected coords: ({u:.2f}, {v:.2f})")

    else:
        print("Camera 36 not found in images")
        print(f"Available image IDs: {list(viz.images.keys())[:10]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()