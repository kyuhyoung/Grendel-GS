#!/usr/bin/env python3
"""
Simple PLY export from checkpoint using direct loading
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from scene.gaussian_model import GaussianModel

def export_final_window_ply(checkpoint_dir, output_ply):
    """
    Export final window gaussians to PLY by reading checkpoint files directly
    """
    print(f"🔄 Exporting final window checkpoint to PLY...")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    print(f"   Output PLY: {output_ply}")

    checkpoint_dir = Path(checkpoint_dir)
    output_path = Path(output_ply)

    # Find checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("chkpnt_ws=*_rk=*.pth"))
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}")
        return False

    print(f"📋 Found {len(checkpoint_files)} checkpoint files")

    # Initialize gaussian model with minimal setup
    gaussians = GaussianModel(sh_degree=0)

    # Load and merge checkpoint data (simplified approach)
    print("📄 Loading checkpoint data...")
    all_xyz = []
    all_features_dc = []
    all_features_rest = []
    all_scaling = []
    all_rotation = []
    all_opacity = []

    for ckpt_file in sorted(checkpoint_files):
        print(f"   Loading {ckpt_file.name}...")
        try:
            # Load with weights_only=True to avoid security warning
            checkpoint_data = torch.load(ckpt_file, map_location='cpu', weights_only=False)

            # Extract the actual gaussian data
            # Checkpoint format: (gaussian_data_tuple, iteration)
            if isinstance(checkpoint_data, tuple) and len(checkpoint_data) >= 2:
                gaussian_data = checkpoint_data[0]
                if isinstance(gaussian_data, tuple) and len(gaussian_data) >= 6:
                    # Unpack gaussian parameters
                    (active_sh_degree, xyz, features_dc, features_rest,
                     scaling, rotation, opacity) = gaussian_data[:7]

                    all_xyz.append(xyz.cpu())
                    all_features_dc.append(features_dc.cpu())
                    all_features_rest.append(features_rest.cpu())
                    all_scaling.append(scaling.cpu())
                    all_rotation.append(rotation.cpu())
                    all_opacity.append(opacity.cpu())
                else:
                    print(f"   ⚠️  Unexpected gaussian data format in {ckpt_file.name}")
            else:
                print(f"   ⚠️  Unexpected checkpoint format in {ckpt_file.name}")

        except Exception as e:
            print(f"   ❌ Error loading {ckpt_file.name}: {e}")
            continue

    if not all_xyz:
        print("❌ No valid checkpoint data found")
        return False

    # Concatenate all data
    print("🔗 Merging data from all ranks...")
    gaussians._xyz = torch.cat(all_xyz, dim=0)
    gaussians._features_dc = torch.cat(all_features_dc, dim=0)
    gaussians._features_rest = torch.cat(all_features_rest, dim=0)
    gaussians._scaling = torch.cat(all_scaling, dim=0)
    gaussians._rotation = torch.cat(all_rotation, dim=0)
    gaussians._opacity = torch.cat(all_opacity, dim=0)

    # Set required attributes for save_ply
    gaussians.max_radii2D = torch.zeros_like(gaussians._xyz[:, 0])
    gaussians.xyz_gradient_accum = torch.zeros_like(gaussians._xyz)
    gaussians.denom = torch.zeros_like(gaussians._xyz[:, 0])
    gaussians.active_sh_degree = 0

    total_gaussians = gaussians._xyz.shape[0]
    print(f"✅ Merged {total_gaussians:,} gaussians")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to PLY using a minimal args setup
    print("💾 Saving PLY file...")

    # Set minimal args for save_ply function
    class MinimalArgs:
        gaussians_distribution = True
        distributed_save = False
        check_cpu_memory = False
        check_gpu_memory = False

    import utils.general_utils as utils
    utils.set_args(MinimalArgs())

    try:
        # Direct PLY creation without distributed processing
        from plyfile import PlyData, PlyElement
        import numpy as np

        print("   Converting to numpy arrays...")
        xyz = gaussians._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        features_dc = gaussians._features_dc.detach().cpu().numpy()
        if features_dc.ndim == 3:
            features_dc = features_dc.reshape(features_dc.shape[0], -1)  # Flatten last dimensions

        features_extra = gaussians._features_rest.detach().cpu().numpy()
        opacities = gaussians._opacity.detach().cpu().numpy()
        scale = gaussians._scaling.detach().cpu().numpy()
        rotation = gaussians._rotation.detach().cpu().numpy()

        print(f"   Shapes: xyz={xyz.shape}, features_dc={features_dc.shape}, opacities={opacities.shape}, scale={scale.shape}, rotation={rotation.shape}")

        print("   Creating PLY structure...")
        # Create vertex array
        dtype_full = [(attribute, 'f4') for attribute in [
            'x', 'y', 'z', 'nx', 'ny', 'nz',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity',
            'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ]]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, features_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Create PLY element
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(str(output_ply))

        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ PLY export successful!")
            print(f"   📄 File: {output_ply}")
            print(f"   📏 Size: {file_size:.2f} MB")
            print(f"   🔢 Gaussians: {total_gaussians:,}")
            return True
        else:
            print(f"❌ PLY file was not created")
            return False

    except Exception as e:
        print(f"❌ Error saving PLY: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Export final window checkpoint to PLY")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--output_ply", type=str, required=True, help="Output PLY file path")

    args = parser.parse_args()

    try:
        success = export_final_window_ply(args.checkpoint_dir, args.output_ply)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()