#!/usr/bin/env python3
"""
Export gaussians from checkpoint to PLY file using existing render.py infrastructure
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from scene.gaussian_model import GaussianModel
from scene import Scene
from arguments import ModelParams, PipelineParams, OptimizationParams
import utils.general_utils as utils

def export_checkpoint_to_ply(model_path, iteration, output_ply):
    """
    Export gaussians from checkpoint to PLY file

    Args:
        model_path: Path to model directory (e.g., ./output/progressive_test/model_window_018)
        iteration: Iteration number (e.g., 12)
        output_ply: Output PLY file path
    """
    print(f"🔄 Exporting checkpoint to PLY...")
    print(f"   Model path: {model_path}")
    print(f"   Iteration: {iteration}")
    print(f"   Output PLY: {output_ply}")

    with torch.no_grad():
        # Load arguments from saved args.json
        import json
        args_file = os.path.join(model_path, "args.json")
        if os.path.exists(args_file):
            with open(args_file, 'r') as f:
                args_dict = json.load(f)

            # Create args object with loaded values
            class Args:
                pass

            args = Args()
            for key, value in args_dict.items():
                setattr(args, key, value)
        else:
            # Fallback to minimal args
            class Args:
                def __init__(self):
                    self.model_path = model_path
                    self.sh_degree = 0
                    self.source_path = ""
                    self.images = "images"
                    self.resolution = -1
                    self.white_background = False
                    self.data_device = "cuda"
                    self.eval = False
                    self.extend_gpu = 0
                    self.check_cpu_memory = False
                    self.check_gpu_memory = False

            args = Args()

        utils.set_args(args)

        # Initialize gaussian model
        gaussians = GaussianModel(args.sh_degree)

        # Load scene from checkpoint
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

        print(f"✅ Loaded gaussians from checkpoint")
        print(f"   Total gaussians: {gaussians._xyz.shape[0]:,}")

        # Create output directory if needed
        output_path = Path(output_ply)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to PLY
        print(f"💾 Saving PLY file...")
        gaussians.save_ply(str(output_ply))

        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ PLY export successful!")
            print(f"   📄 File: {output_ply}")
            print(f"   📏 Size: {file_size:.2f} MB")
            print(f"   🔢 Gaussians: {gaussians._xyz.shape[0]:,}")
            return True
        else:
            print(f"❌ Failed to create PLY file")
            return False

def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to PLY")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--iteration", type=int, default=12, help="Iteration number")
    parser.add_argument("--output_ply", type=str, required=True, help="Output PLY file path")

    args = parser.parse_args()

    try:
        success = export_checkpoint_to_ply(args.model_path, args.iteration, args.output_ply)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()