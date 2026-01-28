#!/usr/bin/env python3
"""
Visual debugging tool for projection and crop calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import os
import sys

# Add paths
sys.path.append('Grendel-GS/scene')
sys.path.append('src')

def visualize_tile_projection(tile_bbox, camera_params, image_size, crop_region=None, 
                             principal_point=None, output_path="projection_debug.png"):
    """
    Visualize how a tile projects onto an image with crop region.
    
    Args:
        tile_bbox: Dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        camera_params: Dict with 'R', 'T', 'fov_x', 'fov_y'
        image_size: Tuple (width, height)
        crop_region: Optional dict with 'x_min', 'x_max', 'y_min', 'y_max'
        principal_point: Optional tuple (cx, cy)
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ============ 1. 3D Scene View (Top-down) ============
    ax = axes[0, 0]
    ax.set_title("3D Scene (Top View)", fontsize=14, fontweight='bold')
    
    # Draw tile bounding box
    tile_rect = Rectangle((tile_bbox['x_min'], tile_bbox['y_min']), 
                          tile_bbox['x_max'] - tile_bbox['x_min'],
                          tile_bbox['y_max'] - tile_bbox['y_min'],
                          fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(tile_rect)
    
    # Draw camera position
    cam_pos = -camera_params['R'].T @ camera_params['T']
    ax.plot(cam_pos[0], cam_pos[1], 'ro', markersize=10, label='Camera')
    
    # Draw camera view frustum (simplified)
    fov_x = camera_params['fov_x']
    view_distance = 500  # Arbitrary for visualization
    left_angle = -fov_x / 2
    right_angle = fov_x / 2
    
    # Camera forward direction (assuming looking along -Z in camera space)
    forward = camera_params['R'].T @ np.array([0, 0, -1])
    forward_2d = forward[:2] / np.linalg.norm(forward[:2])
    
    # Draw frustum lines
    left_dir = np.array([np.cos(left_angle) * forward_2d[0] - np.sin(left_angle) * forward_2d[1],
                         np.cos(left_angle) * forward_2d[1] + np.sin(left_angle) * forward_2d[0]])
    right_dir = np.array([np.cos(right_angle) * forward_2d[0] - np.sin(right_angle) * forward_2d[1],
                          np.cos(right_angle) * forward_2d[1] + np.sin(right_angle) * forward_2d[0]])
    
    ax.plot([cam_pos[0], cam_pos[0] + left_dir[0] * view_distance],
            [cam_pos[1], cam_pos[1] + left_dir[1] * view_distance], 'g--', alpha=0.5)
    ax.plot([cam_pos[0], cam_pos[0] + right_dir[0] * view_distance],
            [cam_pos[1], cam_pos[1] + right_dir[1] * view_distance], 'g--', alpha=0.5)
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # ============ 2. Full Image with Projected Tile ============
    ax = axes[0, 1]
    ax.set_title("Full Image with Tile Projection", fontsize=14, fontweight='bold')
    
    # Draw image boundary
    img_rect = Rectangle((0, 0), image_size[0], image_size[1],
                         fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(img_rect)
    
    # Project tile corners to image
    from adaptive_tile_utils import TileBBox, build_full_proj_transform, project_points_to_camera
    
    tile = TileBBox(tile_bbox['x_min'], tile_bbox['x_max'],
                    tile_bbox['y_min'], tile_bbox['y_max'],
                    tile_bbox['z_min'], tile_bbox['z_max'])
    corners_3d = tile.get_corners()
    
    full_proj = build_full_proj_transform(
        camera_params['R'], camera_params['T'],
        camera_params['fov_x'], camera_params['fov_y'],
        image_size[0], image_size[1]
    )
    
    corners_2d, valid = project_points_to_camera(
        corners_3d, full_proj, image_size[0], image_size[1]
    )
    
    if np.any(valid):
        valid_corners = corners_2d[valid]
        # Draw convex hull of projected corners
        from scipy.spatial import ConvexHull
        if len(valid_corners) >= 3:
            hull = ConvexHull(valid_corners)
            for simplex in hull.simplices:
                ax.plot(valid_corners[simplex, 0], valid_corners[simplex, 1], 'b-', alpha=0.5)
        
        # Fill the projected area
        if len(valid_corners) >= 3:
            hull_points = valid_corners[hull.vertices]
            from matplotlib.patches import Polygon
            poly = Polygon(hull_points, alpha=0.3, facecolor='blue', edgecolor='blue')
            ax.add_patch(poly)
    
    # Draw principal point if provided
    if principal_point:
        ax.plot(principal_point[0], principal_point[1], 'r+', markersize=15, 
                markeredgewidth=2, label=f'Principal Point ({principal_point[0]:.0f}, {principal_point[1]:.0f})')
    
    # Draw crop region if provided
    if crop_region:
        crop_rect = Rectangle((crop_region['x_min'], crop_region['y_min']),
                              crop_region['x_max'] - crop_region['x_min'],
                              crop_region['y_max'] - crop_region['y_min'],
                              fill=False, edgecolor='green', linewidth=3,
                              linestyle='--', label='Crop Region')
        ax.add_patch(crop_rect)
    
    ax.set_xlim(-500, image_size[0] + 500)
    ax.set_ylim(-500, image_size[1] + 500)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.invert_yaxis()  # Image coordinate system
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # ============ 3. Cropped View ============
    ax = axes[0, 2]
    ax.set_title("Cropped Image Region", fontsize=14, fontweight='bold')
    
    if crop_region:
        # Draw crop boundary
        crop_w = crop_region['x_max'] - crop_region['x_min']
        crop_h = crop_region['y_max'] - crop_region['y_min']
        crop_rect = Rectangle((0, 0), crop_w, crop_h,
                              fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(crop_rect)
        
        # Transform projected corners to crop coordinates
        if np.any(valid):
            cropped_corners = valid_corners - np.array([crop_region['x_min'], crop_region['y_min']])
            
            # Draw in crop space
            for corner in cropped_corners:
                if 0 <= corner[0] <= crop_w and 0 <= corner[1] <= crop_h:
                    ax.plot(corner[0], corner[1], 'bo', markersize=5)
            
            # Draw principal point in crop space
            if principal_point:
                pp_in_crop = np.array([principal_point[0] - crop_region['x_min'],
                                       principal_point[1] - crop_region['y_min']])
                
                # Principal point might be outside crop!
                color = 'red' if (0 <= pp_in_crop[0] <= crop_w and 0 <= pp_in_crop[1] <= crop_h) else 'orange'
                ax.plot(pp_in_crop[0], pp_in_crop[1], 'x', color=color, markersize=15,
                       markeredgewidth=2, label=f'Principal Point ({"inside" if color == "red" else "OUTSIDE"})')
                
                # Draw line from crop center to principal point
                crop_center = np.array([crop_w/2, crop_h/2])
                ax.plot([crop_center[0], pp_in_crop[0]], 
                       [crop_center[1], pp_in_crop[1]], 'r--', alpha=0.5)
                
                # Add offset text
                offset = pp_in_crop - crop_center
                ax.text(crop_w/2, crop_h - 20, 
                       f'PP Offset: ({offset[0]:.0f}, {offset[1]:.0f})',
                       ha='center', fontsize=10, color='red')
        
        ax.set_xlim(-crop_w * 0.2, crop_w * 1.2)
        ax.set_ylim(-crop_h * 0.2, crop_h * 1.2)
    
    ax.set_xlabel("X (pixels in crop)")
    ax.set_ylabel("Y (pixels in crop)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # ============ 4. NDC Space Visualization ============
    ax = axes[1, 0]
    ax.set_title("NDC Space (Normalized Device Coordinates)", fontsize=14, fontweight='bold')
    
    # Draw NDC boundary (-1 to 1)
    ndc_rect = Rectangle((-1, -1), 2, 2, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(ndc_rect)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    if np.any(valid):
        # Convert to NDC
        ndc_corners = (valid_corners - np.array([image_size[0]/2, image_size[1]/2])) / np.array([image_size[0]/2, image_size[1]/2])
        
        for corner in ndc_corners:
            ax.plot(corner[0], corner[1], 'bo', markersize=5)
        
        # Show principal point in NDC
        if principal_point:
            pp_ndc = (np.array(principal_point) - np.array([image_size[0]/2, image_size[1]/2])) / np.array([image_size[0]/2, image_size[1]/2])
            ax.plot(pp_ndc[0], pp_ndc[1], 'r+', markersize=15, markeredgewidth=2)
            ax.text(pp_ndc[0], pp_ndc[1] - 0.1, f'({pp_ndc[0]:.2f}, {pp_ndc[1]:.2f})', 
                   ha='center', fontsize=9, color='red')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("NDC X")
    ax.set_ylabel("NDC Y")
    ax.grid(True, alpha=0.3)
    
    # ============ 5. Projection Matrix Values ============
    ax = axes[1, 1]
    ax.set_title("Projection Matrix Analysis", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Calculate projection offsets
    if crop_region and principal_point:
        # For cropped view
        crop_center_x = (crop_region['x_min'] + crop_region['x_max']) / 2
        crop_center_y = (crop_region['y_min'] + crop_region['y_max']) / 2
        
        # Calculate what P[0,2] and P[1,2] should be
        crop_w = crop_region['x_max'] - crop_region['x_min']
        crop_h = crop_region['y_max'] - crop_region['y_min']
        
        # In crop space
        pp_in_crop_x = principal_point[0] - crop_region['x_min']
        pp_in_crop_y = principal_point[1] - crop_region['y_min']
        
        # NDC offset (how far principal point is from crop center in NDC)
        ndc_offset_x = (pp_in_crop_x - crop_w/2) / (crop_w/2)
        ndc_offset_y = (pp_in_crop_y - crop_h/2) / (crop_h/2)
        
        info_text = f"""Projection Analysis:
        
Original Image: {image_size[0]} x {image_size[1]}
Principal Point: ({principal_point[0]:.0f}, {principal_point[1]:.0f})

Crop Region: ({crop_region['x_min']:.0f}, {crop_region['y_min']:.0f}) - ({crop_region['x_max']:.0f}, {crop_region['y_max']:.0f})
Crop Size: {crop_w:.0f} x {crop_h:.0f}
Crop Center: ({crop_center_x:.0f}, {crop_center_y:.0f})

PP in Crop Space: ({pp_in_crop_x:.0f}, {pp_in_crop_y:.0f})
PP Offset from Crop Center: ({pp_in_crop_x - crop_w/2:.0f}, {pp_in_crop_y - crop_h/2:.0f})

Expected P[0,2] (NDC): {ndc_offset_x:.3f}
Expected P[1,2] (NDC): {ndc_offset_y:.3f}

Asymmetric Frustum: {'YES' if abs(ndc_offset_x) > 0.1 or abs(ndc_offset_y) > 0.1 else 'NO'}
PP Outside Crop: {'YES' if pp_in_crop_x < 0 or pp_in_crop_x > crop_w or pp_in_crop_y < 0 or pp_in_crop_y > crop_h else 'NO'}
"""
    else:
        info_text = "No crop region specified"
    
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', family='monospace')
    
    # ============ 6. Visual Legend ============
    ax = axes[1, 2]
    ax.set_title("Legend & Status", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    legend_text = """
Legend:
━━━ Black: Image/Crop boundaries
━━━ Blue: Tile bounding box projection
━━━ Green: Crop region
  +  Red: Principal point
  ×  Orange: Principal point (outside crop)
- - - Dashed: View frustum

Status Indicators:
✓ Tile visible in camera
✓ Crop region computed
"""
    
    if crop_region and principal_point:
        pp_in_crop_x = principal_point[0] - crop_region['x_min']
        pp_in_crop_y = principal_point[1] - crop_region['y_min']
        crop_w = crop_region['x_max'] - crop_region['x_min']
        crop_h = crop_region['y_max'] - crop_region['y_min']
        
        if pp_in_crop_x < 0 or pp_in_crop_x > crop_w or pp_in_crop_y < 0 or pp_in_crop_y > crop_h:
            legend_text += "⚠️  Principal point OUTSIDE crop\n"
            legend_text += "   → Asymmetric frustum needed!\n"
        else:
            legend_text += "✓ Principal point inside crop\n"
    
    ax.text(0.1, 0.9, legend_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top')
    
    # Save figure
    plt.suptitle(f"Tile Projection & Crop Debug Visualization", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.show()


def test_with_example():
    """Test with an example similar to the actual data."""
    
    # Example from logs
    tile_bbox = {
        'x_min': -1290.24, 'x_max': 1171.91,
        'y_min': -881.00, 'y_max': 920.32,
        'z_min': 15.69, 'z_max': 176.58
    }
    
    # Camera parameters (example)
    camera_params = {
        'R': np.eye(3),  # Identity for simplicity
        'T': np.array([0, 0, -1000]),  # Camera 1000 units back
        'fov_x': np.radians(60),
        'fov_y': np.radians(40)
    }
    
    # Image size from logs
    image_size = (11310, 17310)
    
    # Principal point (center for this example, but could be off-center)
    principal_point = (5655, 8655)  # Center of image
    
    # Test different crop scenarios
    test_cases = [
        {
            'name': 'full_image',
            'crop': {'x_min': 0, 'x_max': 11310, 'y_min': 0, 'y_max': 17310}
        },
        {
            'name': 'crop_with_pp_inside',
            'crop': {'x_min': 4000, 'x_max': 7000, 'y_min': 7000, 'y_max': 10000}
        },
        {
            'name': 'crop_with_pp_outside',
            'crop': {'x_min': 1000, 'x_max': 4000, 'y_min': 2000, 'y_max': 5000}
        },
        {
            'name': 'crop_limited_size',
            'crop': {'x_min': 4152, 'x_max': 7224, 'y_min': 7119, 'y_max': 10191}
        }
    ]
    
    for test_case in test_cases:
        output_path = f"projection_debug_{test_case['name']}.png"
        print(f"\nGenerating visualization for: {test_case['name']}")
        visualize_tile_projection(
            tile_bbox, camera_params, image_size,
            crop_region=test_case['crop'],
            principal_point=principal_point,
            output_path=output_path
        )


if __name__ == "__main__":
    # Check if we have the required modules
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("Please install matplotlib: pip install matplotlib scipy")
        sys.exit(1)
    
    test_with_example()