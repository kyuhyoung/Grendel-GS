import torch
import numpy as np
import matplotlib.pyplot as plt

def debug_gaussian_colors(gaussians, scene, batched_cameras):
    """
    Debug function to check Gaussian color initialization.
    Analyzes the distribution and range of Gaussian colors.
    """
    print("🎨 DEBUG: Checking Gaussian color initialization...")

    try:
        # Get current Gaussian colors (SH coefficients)
        features_dc = gaussians._features_dc.detach().cpu()  # Shape: (N, 1, 3) for DC component
        features_rest = gaussians._features_rest.detach().cpu()  # Shape: (N, sh_coeffs-1, 3) for higher order

        # Convert SH DC component to RGB (DC component represents base color)
        # SH DC coefficient relates to RGB as: RGB = (DC + 0.5)
        dc_rgb = features_dc[:, 0, :]  # Shape: (N, 3)
        rgb_colors = (dc_rgb + 0.5).clamp(0, 1)  # Convert to [0,1] range

        print(f"📊 Gaussian Color Statistics:")
        print(f"  Total Gaussians: {len(rgb_colors)}")
        print(f"  DC features shape: {features_dc.shape}")
        print(f"  Rest features shape: {features_rest.shape}")

        # Analyze RGB color distribution
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_values = rgb_colors[:, i]
            print(f"  {channel} channel - Min: {channel_values.min():.4f}, Max: {channel_values.max():.4f}, Mean: {channel_values.mean():.4f}, Std: {channel_values.std():.4f}")

        # Check for problematic color values
        invalid_colors = (rgb_colors < 0) | (rgb_colors > 1)
        if invalid_colors.any():
            print(f"⚠️  Warning: {invalid_colors.sum().item()} invalid color values found!")

        # Check if colors are too dark or too bright
        brightness = rgb_colors.mean(dim=1)  # Average of RGB
        too_dark = (brightness < 0.01).sum().item()
        too_bright = (brightness > 0.99).sum().item()

        print(f"  Brightness - Min: {brightness.min():.4f}, Max: {brightness.max():.4f}, Mean: {brightness.mean():.4f}")
        print(f"  Too dark (< 0.01): {too_dark} gaussians")
        print(f"  Too bright (> 0.99): {too_bright} gaussians")

        # Check opacity
        opacity = gaussians.get_opacity.detach().cpu()
        print(f"  Opacity - Min: {opacity.min():.4f}, Max: {opacity.max():.4f}, Mean: {opacity.mean():.4f}")

        # Sample some specific Gaussian colors for detailed inspection
        print(f"📝 Sample Gaussian Colors (first 10):")
        for i in range(min(10, len(rgb_colors))):
            r, g, b = rgb_colors[i]
            opacity_val = opacity[i, 0]
            print(f"  Gaussian {i}: RGB=({r:.3f}, {g:.3f}, {b:.3f}), Opacity={opacity_val:.3f}")

        print("🎨 DEBUG: Gaussian color check complete")

    except Exception as e:
        print(f"❌ Error checking colors: {e}")
        import traceback
        traceback.print_exc()


def debug_point_cloud_colors(point_cloud):
    """
    Debug function to check the original point cloud colors before Gaussian initialization.
    """
    print("🌈 DEBUG: Checking original point cloud colors...")

    try:
        colors = point_cloud.colors  # Shape: (N, 3), values in [0, 255] range
        print(f"📊 Point Cloud Color Statistics:")
        print(f"  Total points: {len(colors)}")
        print(f"  Colors shape: {colors.shape}")
        print(f"  Color range: [{colors.min()}, {colors.max()}]")

        # Check color distribution per channel
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_values = colors[:, i]
            print(f"  {channel} channel - Min: {channel_values.min()}, Max: {channel_values.max()}, Mean: {channel_values.mean():.1f}")

        # Check for invalid values
        if colors.min() < 0 or colors.max() > 255:
            print("⚠️  Warning: Point cloud colors outside [0, 255] range!")

        # Sample some colors
        print(f"📝 Sample Point Colors (first 10):")
        for i in range(min(10, len(colors))):
            r, g, b = colors[i]
            print(f"  Point {i}: RGB=({r:.0f}, {g:.0f}, {b:.0f})")

        print("🌈 DEBUG: Point cloud color check complete")

    except Exception as e:
        print(f"❌ Error checking point cloud colors: {e}")
        import traceback
        traceback.print_exc()