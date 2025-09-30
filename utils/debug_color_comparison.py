import torch
import numpy as np
import os

def debug_colmap_vs_gaussian_colors(gaussians, scene, args, num_samples=10):
    """
    Compare COLMAP original point colors with current Gaussian colors.

    Args:
        gaussians: GaussianModel with initialized colors
        scene: Scene object
        args: Arguments containing source_path
        num_samples: Number of points to sample for comparison
    """
    print("🔍 DEBUG: Comparing COLMAP original colors vs Gaussian colors...")

    try:
        from scene.colmap_loader import read_points3D_text, read_points3D_binary

        # Load original COLMAP points3D data
        source_path = args.source_path
        points3d_txt = os.path.join(source_path, "points3D.txt")
        points3d_bin = os.path.join(source_path, "points3D.bin")

        if os.path.exists(points3d_bin):
            print(f"📂 Loading COLMAP points from: {points3d_bin}")
            xyzs, rgbs, errors, tracks = read_points3D_binary(points3d_bin)
        elif os.path.exists(points3d_txt):
            print(f"📂 Loading COLMAP points from: {points3d_txt}")
            xyzs, rgbs, errors, tracks = read_points3D_text(points3d_txt)
        else:
            print("❌ Could not find COLMAP points3D file")
            return

        # Get current Gaussian data
        gaussian_positions = gaussians.get_xyz.detach().cpu().numpy()  # Shape: (N, 3)
        gaussian_features_dc = gaussians._features_dc.detach().cpu()  # Shape: (N, 1, 3)

        # Convert Gaussian SH DC component to RGB
        # SH DC coefficient relates to RGB as: RGB = (DC + 0.5)
        gaussian_rgb = (gaussian_features_dc[:, 0, :] + 0.5).clamp(0, 1)  # Shape: (N, 3)

        print(f"📊 Data loaded:")
        print(f"  COLMAP points: {len(xyzs)}")
        print(f"  Gaussian points: {len(gaussian_positions)}")
        print(f"  COLMAP RGB range: [{rgbs.min()}, {rgbs.max()}]")
        print(f"  Gaussian RGB range: [{gaussian_rgb.min():.3f}, {gaussian_rgb.max():.3f}]")

        # Sample points for comparison
        sample_indices = np.linspace(0, len(gaussian_positions)-1, num_samples, dtype=int)

        print(f"\n🎨 Color Comparison (sampling {num_samples} points):")
        print("Index | COLMAP RGB          | Gaussian RGB        | Difference")
        print("------|--------------------|--------------------|--------------------")

        total_diff = 0.0
        for i, idx in enumerate(sample_indices):
            if idx < len(rgbs):
                # COLMAP color (0-255 range)
                colmap_r, colmap_g, colmap_b = rgbs[idx]
                colmap_rgb_norm = np.array([colmap_r, colmap_g, colmap_b]) / 255.0  # Normalize to [0,1]

                # Gaussian color (already in [0,1] range)
                gauss_r, gauss_g, gauss_b = gaussian_rgb[idx]
                gauss_rgb_array = np.array([gauss_r, gauss_g, gauss_b])

                # Calculate difference
                diff = np.abs(colmap_rgb_norm - gauss_rgb_array)
                avg_diff = np.mean(diff)
                total_diff += avg_diff

                print(f"{idx:5d} | ({colmap_r:3.0f},{colmap_g:3.0f},{colmap_b:3.0f}) -> ({colmap_rgb_norm[0]:.3f},{colmap_rgb_norm[1]:.3f},{colmap_rgb_norm[2]:.3f}) | ({gauss_r:.3f},{gauss_g:.3f},{gauss_b:.3f}) | ({diff[0]:.3f},{diff[1]:.3f},{diff[2]:.3f}) avg={avg_diff:.3f}")

        avg_total_diff = total_diff / num_samples
        print(f"\n📈 Average color difference: {avg_total_diff:.4f}")

        if avg_total_diff > 0.1:
            print("⚠️  Warning: Large color difference detected! Color initialization may be incorrect.")
        elif avg_total_diff > 0.05:
            print("⚠️  Moderate color difference detected.")
        else:
            print("✅ Color differences are small - initialization appears correct.")

        # Check position correspondence
        print(f"\n📍 Position Correspondence Check (first 5 points):")
        for i in range(min(5, len(gaussian_positions), len(xyzs))):
            colmap_pos = xyzs[i]
            gaussian_pos = gaussian_positions[i]
            pos_diff = np.linalg.norm(colmap_pos - gaussian_pos)
            print(f"  Point {i}: COLMAP=({colmap_pos[0]:.3f},{colmap_pos[1]:.3f},{colmap_pos[2]:.3f}) Gaussian=({gaussian_pos[0]:.3f},{gaussian_pos[1]:.3f},{gaussian_pos[2]:.3f}) diff={pos_diff:.6f}")

    except Exception as e:
        print(f"❌ Error in color comparison: {e}")
        import traceback
        traceback.print_exc()

    print("🔍 DEBUG: COLMAP vs Gaussian color comparison complete")