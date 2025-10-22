#!/usr/bin/env python3
"""
Test script for footprint visualization in orthographic view
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/workspace/Grendel-GS')

from scripts.experiments.colmap_visualizer import COLMAPVisualizer

def test_footprint_visualization():
    """Test footprint visualization with sample COLMAP data"""

    # Find COLMAP data path
    possible_paths = [
        Path("/workspace/Grendel-GS/data/colmap"),
        Path("/workspace/Grendel-GS/data/sillim_outdoor/colmap"),
        Path("/workspace/Grendel-GS/data/outdoor_sillim2/colmap"),
        Path("/workspace/Grendel-GS/colmap")
    ]

    colmap_path = None
    for path in possible_paths:
        if path.exists() and (path / "cameras.txt").exists():
            colmap_path = path
            break
        elif path.exists() and (path / "sparse" / "0" / "cameras.txt").exists():
            colmap_path = path / "sparse" / "0"
            break

    if colmap_path is None:
        print("Error: Could not find COLMAP data")
        print("Searched paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return

    print(f"Using COLMAP data from: {colmap_path}")

    # Create visualizer
    viz = COLMAPVisualizer(str(colmap_path))

    try:
        # Load COLMAP data
        print("Loading COLMAP data...")
        viz.read_cameras_txt()
        viz.read_images_txt()
        viz.read_points3d_txt()

        print(f"Loaded {len(viz.cameras)} cameras, {len(viz.images)} images, {len(viz.points3d)} points")

        # Create DTM
        print("Creating DTM...")
        viz.create_dtm(resolution=2.0)

        # Get scene center
        if viz.points3d:
            xyz_points = [pt['xyz'] for pt in viz.points3d.values()]
            import numpy as np
            scene_center = np.mean(xyz_points, axis=0)
        else:
            scene_center = [0, 0, 0]

        # Render orthographic view with footprints
        print("Rendering orthographic view with footprints...")
        output_path = '/workspace/Grendel-GS/orthographic_with_footprints.png'
        viz.render_orthographic_view(scene_center, save_path=output_path)

        print(f"✓ Visualization saved to: {output_path}")

        # Print footprint statistics
        print("\nFootprint Statistics:")
        print(f"  Total cameras: {len(viz.images)}")
        print(f"  DTM resolution: 2.0 m")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_footprint_visualization()