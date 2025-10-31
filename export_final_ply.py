#!/usr/bin/env python3
"""
Export final window gaussians from checkpoint to PLY file
"""

import sys
import torch
import argparse
from pathlib import Path
sys.path.append('.')

from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
import utils

def main():
    parser = argparse.ArgumentParser(description="Export final gaussians from checkpoint to PLY")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--output_ply", type=str, required=True, help="Output PLY file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_ply = Path(args.output_ply)

    print(f"🔄 Loading checkpoint from: {checkpoint_dir}")
    print(f"💾 Output PLY: {output_ply}")

    # Create output directory if needed
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    # Initialize gaussian model
    gaussians = GaussianModel(sh_degree=0)  # Use same SH degree as training

    # Load checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("chkpnt_ws=*_rk=*.pth"))
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}")
        return False

    print(f"📋 Found {len(checkpoint_files)} checkpoint files")

    # Load and merge all checkpoint files
    all_data = {}
    for i, ckpt_file in enumerate(sorted(checkpoint_files)):
        print(f"   📄 Loading {ckpt_file}")
        checkpoint = torch.load(ckpt_file, map_location=args.device)

        # Merge checkpoint data
        for key, value in checkpoint.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(value)

    # Concatenate tensors from all ranks
    merged_data = {}
    for key, tensor_list in all_data.items():
        if isinstance(tensor_list[0], torch.Tensor):
            merged_data[key] = torch.cat(tensor_list, dim=0)
        else:
            merged_data[key] = tensor_list[0]  # Take first value for non-tensors

    print(f"🔄 Restoring gaussians from merged checkpoint data...")

    # Restore gaussian parameters
    gaussians._xyz = merged_data["_xyz"]
    gaussians._features_dc = merged_data["_features_dc"]
    gaussians._features_rest = merged_data["_features_rest"]
    gaussians._scaling = merged_data["_scaling"]
    gaussians._rotation = merged_data["_rotation"]
    gaussians._opacity = merged_data["_opacity"]

    # Set other required attributes
    gaussians.max_radii2D = merged_data.get("max_radii2D", torch.zeros_like(gaussians._xyz[:, 0]))
    gaussians.xyz_gradient_accum = merged_data.get("xyz_gradient_accum", torch.zeros_like(gaussians._xyz))
    gaussians.denom = merged_data.get("denom", torch.zeros_like(gaussians._xyz[:, 0]))

    total_gaussians = gaussians._xyz.shape[0]
    print(f"✅ Loaded {total_gaussians:,} gaussians")

    # Save to PLY
    print(f"💾 Saving PLY file...")
    with torch.no_grad():
        gaussians.save_ply(str(output_ply))

    if output_ply.exists():
        file_size = output_ply.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ PLY file saved successfully!")
        print(f"   📄 File: {output_ply}")
        print(f"   📏 Size: {file_size:.2f} MB")
        print(f"   🔢 Gaussians: {total_gaussians:,}")
        return True
    else:
        print(f"❌ Failed to create PLY file")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)