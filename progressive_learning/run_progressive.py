#!/usr/bin/env python3
"""
Progressive Training Main Script
Fixed version - no longer generated dynamically
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    from progressive_learning.progressive_trainer import ProgressiveTrainer
except ImportError as e:
    print(f"Error importing ProgressiveTrainer: {e}")
    print("Make sure you're running from the Grendel-GS root directory")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Progressive Training for Grendel-GS')
    parser.add_argument('--source_path', required=True, help='Path to COLMAP reconstruction')
    parser.add_argument('--output_path', required=True, help='Output directory')
    parser.add_argument('--initial_cameras', type=int, default=4, help='Number of initial cameras')
    parser.add_argument('--gpu_threshold', type=float, default=0.9, help='GPU memory threshold')
    parser.add_argument('--iterations', type=int, default=30000, help='Training iterations')
    parser.add_argument('--sh_degree', type=int, default=3, help='Spherical harmonics degree')
    parser.add_argument('--resolution', type=int, default=1, help='Resolution downscaling')
    parser.add_argument('--window_size', type=int, default=3, help='Sliding window size')
    parser.add_argument('--iterations_per_window', type=int, default=60, help='Iterations per sliding window')
    parser.add_argument('--densification_interval', type=int, default=100, help='Densification interval')
    parser.add_argument('--densify_from_iter', type=int, default=50, help='Start densification from iteration')
    parser.add_argument('--backend', default='gsplat', help='Rendering backend')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--show_memory_debug_info', action='store_true', help='Show detailed memory and tensor debug info')
    parser.add_argument('--dtm_module', help='Path to external DTM module')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic training')
    parser.add_argument('--only_actually_visible', action='store_true', help='Only keep points visible in camera frames')
    parser.add_argument('--use_chunk', action='store_true', help='Enable chunked SSIM for memory efficiency')
    parser.add_argument('--auto_save_final_iteration', action='store_true', help='Automatically add final iteration to save_iterations')
    parser.add_argument('--track_by_projection', action='store_true', help='Generate tracks by projection instead of using COLMAP tracks')
    parser.add_argument('--prune_by_visibility', action='store_true', help='Prune gaussians outside all camera frustums')
    parser.add_argument('--visibility_prune_margin', type=int, default=20, help='Margin in pixels for visibility-based pruning')

    args = parser.parse_args()

    # Load DTM module if specified
    dtm_module = None
    if args.dtm_module:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("dtm_module", args.dtm_module)
            dtm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dtm_module)
            print(f"Loaded DTM module from {args.dtm_module}")
        except Exception as e:
            print(f"Warning: Could not load DTM module: {e}")
            print("Continuing with simplified footprint computation")

    # Create and run progressive trainer
    trainer = ProgressiveTrainer(
        colmap_path=args.source_path,
        output_path=args.output_path,
        dtm_module=dtm_module,
        initial_cameras=args.initial_cameras,
        gpu_memory_threshold=args.gpu_threshold,
        debug=args.debug,
        only_actually_visible=args.only_actually_visible
    )

    # Store additional training parameters for integration with Grendel-GS
    trainer.iterations = args.iterations
    trainer.sh_degree = args.sh_degree
    trainer.resolution = args.resolution
    trainer.backend = args.backend
    trainer.deterministic = args.deterministic
    trainer.use_chunk = args.use_chunk
    trainer.densification_interval = args.densification_interval
    trainer.densify_from_iter = args.densify_from_iter
    trainer.show_memory_debug_info = args.show_memory_debug_info
    trainer.auto_save_final_iteration = args.auto_save_final_iteration
    trainer.track_by_projection = args.track_by_projection
    trainer.prune_by_visibility = args.prune_by_visibility
    trainer.visibility_prune_margin = args.visibility_prune_margin

    # Run the progressive training pipeline with sliding window
    try:
        # Use sliding window approach with configurable parameters
        window_size = getattr(args, 'window_size', 3)
        iterations_per_window = getattr(args, 'iterations_per_window', 60)
        trainer.run(sliding_window_size=window_size,
                   iterations_per_window=iterations_per_window)
        print("\n" + "="*60)
        print("Progressive Training Completed Successfully!")
        print(f"Results saved to: {args.output_path}")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\nError during progressive training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())