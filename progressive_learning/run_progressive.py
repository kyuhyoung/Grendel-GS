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
    parser.add_argument('--densify_memory_limit_percentage', type=float, default=None, help='GPU memory limit for densification (0.0-1.0)')
    parser.add_argument('--max_window_size', type=int, default=None, help='Maximum window size (number of cameras). If set, removes camera when window reaches this size.')
    parser.add_argument('--removal_strategy', default='farthest', choices=['fifo', 'farthest'], help='Camera removal strategy: fifo or farthest')
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
    parser.add_argument('--opacity_reset_interval', type=int, default=3000, help='Opacity reset interval in iterations')
    parser.add_argument('--opacity_reset_until_iter', type=int, default=14000, help='Continue opacity reset until this iteration')
    parser.add_argument('--resume_from_window', help='Resume from specific window (0=initial, 1+=window_XXX, "auto"=auto-detect)')
    parser.add_argument('--f_mode', type=str, default='global', choices=['global', 'remaining'], help='F (global center) calculation mode: global (all cameras) or remaining (only cameras in A)')
    parser.add_argument('--enable_direction_filtering', action='store_true', help='Enable direction filtering (Step 16.6): only cameras in forward hemisphere (angle <= 90°)')
    parser.add_argument('--e_selection_strategy', type=str, default='default', choices=['default', 'momentum', 'weighted', 'tangential', 'polar', 'outward_spiral_compact', 'balanced_smooth_trajectory'], help='E camera selection strategy')
    parser.add_argument('--e_weighted_alpha', type=float, default=0.7, help='Weight for R in weighted strategy (default: 0.7)')
    parser.add_argument('--e_weighted_beta', type=float, default=0.3, help='Weight for (yy-F) in weighted strategy (default: 0.3)')
    parser.add_argument('--e_tangential_coeff', type=float, default=0.5, help='Tangential coefficient for tangential strategy (default: 0.5)')
    parser.add_argument('--e_polar_angle_step', type=float, default=30.0, help='Angle step in degrees for polar strategy (default: 30)')
    parser.add_argument('--e_polar_radius_step', type=float, default=1.2, help='Radius multiplier for polar strategy (default: 1.2)')
    parser.add_argument('--e_spiral_alpha', type=float, default=1.0, help='Distance weight for outward_spiral_compact strategy (default: 1.0)')
    parser.add_argument('--e_spiral_beta', type=float, default=0.3, help='Diversity weight for outward_spiral_compact strategy (default: 0.3)')
    parser.add_argument('--e_spiral_gamma', type=float, default=0.5, help='Variance penalty weight for outward_spiral_compact strategy (default: 0.5)')
    # Balanced smooth trajectory strategy parameters (6-force)
    parser.add_argument('--e_outward_weight', type=float, default=0.04, help='Weight for outward movement in balanced_smooth_trajectory strategy (default: 0.04)')
    parser.add_argument('--e_compact_weight', type=float, default=2.5, help='Weight for window compactness in balanced_smooth_trajectory strategy (default: 2.5)')
    parser.add_argument('--e_smooth_window_weight', type=float, default=2.8, help='Weight for smooth window trajectory in balanced_smooth_trajectory strategy (default: 2.8)')
    parser.add_argument('--e_smooth_camera_weight', type=float, default=0.7, help='Weight for smooth camera trajectory in balanced_smooth_trajectory strategy (default: 0.7)')
    parser.add_argument('--e_distance_weight', type=float, default=0.5, help='Weight for distance to current window center in balanced_smooth_trajectory strategy (default: 0.5)')
    parser.add_argument('--e_directional_weight', type=float, default=0.0, help='Weight for directional alignment (candidate aligns with window movement) in balanced_smooth_trajectory strategy (default: 0.0)')
    parser.add_argument('--footprint_intersection_threshold', type=float, default=0.0, help='Minimum intersection area ratio (intersection/union) for candidate cameras (0.0-1.0, default: 0.0 = disabled)')
    parser.add_argument('--exit_after_first_removal', action='store_true', help='Exit after first camera removal for testing')
    parser.add_argument('--use_all_processed_cameras', action='store_true', help='Check all processed cameras (not just prev window) when adding gaussians')
    parser.add_argument('--camera_removal_margin', type=float, default=0.15, help='Margin below densify_memory_limit for camera removal (default: 0.15)')
    parser.add_argument('--point_cloud_format', type=str, default='auto', choices=['auto', 'ply', 'txt', 'bin'],
                       help='Point cloud format to load: auto (default), ply, txt, or bin')

    # Adaptive training parameters
    parser.add_argument('--enable_adaptive_training', action='store_true', help='Enable adaptive training with convergence detection')
    parser.add_argument('--min_iterations_per_window', type=int, default=5, help='Minimum iterations per window for adaptive training')
    parser.add_argument('--convergence_start_iter', type=int, default=100, help='Start convergence checking from this iteration')
    parser.add_argument('--convergence_loss_threshold', type=float, default=1e-4, help='Loss threshold for convergence detection')

    # Exponential fitting parameters for dynamic patience
    parser.add_argument('--min_camera_count', type=int, default=2, help='Minimum camera count for exponential fitting')
    parser.add_argument('--max_patience_for_min_cam', type=int, default=50, help='Patience when camera count is min_camera_count')
    parser.add_argument('--max_camera_count', type=int, default=30, help='Maximum camera count for exponential fitting')
    parser.add_argument('--min_patience_for_max_cam', type=int, default=15, help='Patience when camera count is max_camera_count')

    # Visualization control
    parser.add_argument('--skip_heavy_visualization', action='store_true', help='Skip heavy visualization files (3d_scene, ortho, nadir) to save time')

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

    # Parse resume_from_window parameter
    resume_from_window = None
    if args.resume_from_window:
        if args.resume_from_window == "auto":
            resume_from_window = "auto"
        else:
            try:
                resume_from_window = int(args.resume_from_window)
            except ValueError:
                print(f"Error: resume_from_window must be a number or 'auto', got: {args.resume_from_window}")
                sys.exit(1)

    # Create and run progressive trainer
    trainer = ProgressiveTrainer(
        colmap_path=args.source_path,
        output_path=args.output_path,
        dtm_module=dtm_module,
        initial_cameras=args.initial_cameras,
        gpu_memory_threshold=args.gpu_threshold,
        camera_removal_margin=args.camera_removal_margin,
        debug=args.debug,
        only_actually_visible=args.only_actually_visible,
        skip_heavy_visualization=args.skip_heavy_visualization,
        resume_from_window=resume_from_window,
        dataset_path=args.source_path
    )

    # Store additional training parameters for integration with Grendel-GS
    trainer.iterations = args.iterations
    trainer.sh_degree = args.sh_degree
    trainer.resolution = args.resolution
    trainer.backend = args.backend
    trainer.deterministic = args.deterministic
    trainer.point_cloud_format = args.point_cloud_format
    trainer.use_chunk = args.use_chunk
    trainer.densification_interval = args.densification_interval
    trainer.densify_from_iter = args.densify_from_iter
    trainer.densify_memory_limit_percentage = args.densify_memory_limit_percentage
    trainer.max_window_size = args.max_window_size
    trainer.removal_strategy = args.removal_strategy
    trainer.f_mode = args.f_mode
    trainer.enable_direction_filtering = args.enable_direction_filtering
    trainer.e_selection_strategy = args.e_selection_strategy
    trainer.e_weighted_alpha = args.e_weighted_alpha
    trainer.e_weighted_beta = args.e_weighted_beta
    trainer.e_tangential_coeff = args.e_tangential_coeff
    trainer.e_polar_angle_step = args.e_polar_angle_step
    trainer.e_polar_radius_step = args.e_polar_radius_step
    trainer.e_spiral_alpha = args.e_spiral_alpha
    trainer.e_spiral_beta = args.e_spiral_beta
    trainer.e_spiral_gamma = args.e_spiral_gamma
    trainer.e_outward_weight = args.e_outward_weight
    trainer.e_compact_weight = args.e_compact_weight
    trainer.e_smooth_window_weight = args.e_smooth_window_weight
    trainer.e_smooth_camera_weight = args.e_smooth_camera_weight
    trainer.e_distance_weight = args.e_distance_weight
    trainer.e_directional_weight = args.e_directional_weight
    trainer.footprint_intersection_threshold = args.footprint_intersection_threshold
    trainer.exit_after_first_removal = args.exit_after_first_removal
    trainer.use_all_processed_cameras = args.use_all_processed_cameras
    trainer.show_memory_debug_info = args.show_memory_debug_info
    trainer.auto_save_final_iteration = args.auto_save_final_iteration
    trainer.track_by_projection = args.track_by_projection
    trainer.prune_by_visibility = args.prune_by_visibility
    trainer.visibility_prune_margin = args.visibility_prune_margin

    # Opacity reset parameters
    trainer.opacity_reset_interval = args.opacity_reset_interval
    trainer.opacity_reset_until_iter = args.opacity_reset_until_iter

    # Visualization control
    trainer.skip_heavy_visualization = args.skip_heavy_visualization
    print(f"🔍 [DEBUG] Setting trainer.skip_heavy_visualization = {args.skip_heavy_visualization}")

    # Adaptive training parameters
    trainer.enable_adaptive_training = args.enable_adaptive_training
    trainer.min_iterations_per_window = args.min_iterations_per_window
    trainer.convergence_start_iter = args.convergence_start_iter

    # Only set convergence parameters if config file doesn't exist
    # (If config file exists, it was already loaded in __init__)
    import os
    if not os.path.exists(os.path.join(os.getcwd(), "convergence_config.txt")):
        trainer.convergence_loss_threshold = args.convergence_loss_threshold
        # Set exponential fitting parameters
        trainer.min_camera_count = args.min_camera_count
        trainer.max_patience_for_min_cam = args.max_patience_for_min_cam
        trainer.max_camera_count = args.max_camera_count
        trainer.min_patience_for_max_cam = args.min_patience_for_max_cam

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