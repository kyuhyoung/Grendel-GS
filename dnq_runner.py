#!/usr/bin/env python3
"""
Divide and Conquer Runner for 3D Gaussian Splatting

This script implements the main logic for divide and conquer training:
1. Create subsets based on footprint constraints
2. Train each subset in parallel using torchrun
3. Merge all resulting PLY files
"""

import os
import sys
import json
import argparse
import subprocess
import concurrent.futures
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dnq_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DNQRunner:
    """Main class for divide and conquer 3D Gaussian Splatting"""
    
    def __init__(self, args):
        self.args = args
        self.source_path = Path(args.source_path)
        self.output_path = Path(args.output_path)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DNQ Runner initialized")
        logger.info(f"Source: {self.source_path}")
        logger.info(f"Output: {self.output_path}")
        
    def create_subsets(self) -> List[List[int]]:
        """Create subsets using footprint-based algorithm"""
        logger.info("Creating subsets...")
        
        # Import and use existing subset creation logic
        from subset_creator import create_subsets_with_footprints
        
        # Load COLMAP data and create subsets
        subsets = create_subsets_with_footprints(
            source_path=self.source_path,
            output_path=self.output_path,
            pixel_threshold_a=self.args.pixel_threshold_a,
            min_max_ratio_d=self.args.min_max_ratio_d,
            max_subsets=self.args.max_subsets
        )
        
        logger.info(f"Created {len(subsets)} subsets")
        
        # Save subsets metadata
        subsets_data = {
            'subsets': subsets,
            'metadata': {
                'total_subsets': len(subsets),
                'pixel_threshold_a': self.args.pixel_threshold_a,
                'min_max_ratio_d': self.args.min_max_ratio_d,
                'max_subsets': self.args.max_subsets
            }
        }
        
        subsets_file = self.output_path / "subsets.json"
        with open(subsets_file, 'w') as f:
            json.dump(subsets_data, f, indent=2)
        
        logger.info(f"Subsets saved to: {subsets_file}")
        return subsets
    
    def train_subset(self, subset_id: int, image_ids: List[int]) -> bool:
        """Train a single subset using torchrun"""
        logger.info(f"Training subset {subset_id} with {len(image_ids)} images")
        
        subset_output = self.output_path / f"subset_{subset_id:03d}"
        subset_output.mkdir(exist_ok=True)
        
        # Create subset-specific COLMAP data
        subset_colmap_path = self.create_subset_colmap(subset_id, image_ids)
        
        # Build torchrun command for training
        cmd = [
            "torchrun",
            "--nproc_per_node=1",  # Single GPU per subset
            "train.py",
            "-s", str(subset_colmap_path),
            "-m", str(subset_output),
            "--iterations", str(self.args.iterations),
            "--sh_degree", str(self.args.sh_degree),
            "--backend", self.args.backend,
            "--densification_interval", str(self.args.densification_interval),
            "--densify_from_iter", str(self.args.densify_from_iter),
            "--densify_until_iter", str(self.args.densify_until_iter),
            "--opacity_reset_interval", str(self.args.opacity_reset_interval),
        ]
        
        # Add optional flags
        if self.args.deterministic:
            cmd.append("--deterministic")
        if self.args.debug:
            cmd.append("--debug")
        if self.args.white_background:
            cmd.append("--white_background")
        if self.args.convert_SHs_python:
            cmd.append("--convert_SHs_python")
        if self.args.compute_cdf_python:
            cmd.append("--compute_cdf_python")
        
        # Add point cloud format
        if self.args.point_cloud_format != "auto":
            cmd.extend(["--point_cloud_format", self.args.point_cloud_format])
        
        # Add DTM module if specified
        if self.args.dtm_module:
            cmd.extend(["--dtm_module", self.args.dtm_module])
        
        logger.info(f"Training command for subset {subset_id}: {' '.join(cmd)}")
        
        try:
            # Set CUDA device for this subset
            env = os.environ.copy()
            try:
                import torch
                gpu_id = subset_id % torch.cuda.device_count() if torch.cuda.is_available() else 0
            except ImportError:
                gpu_id = subset_id % 4  # Assume 4 GPUs max if torch not available
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Run training
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600 * 4  # 4 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Subset {subset_id} training completed successfully")
                return True
            else:
                logger.error(f"Subset {subset_id} training failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Subset {subset_id} training timed out")
            return False
        except Exception as e:
            logger.error(f"Subset {subset_id} training error: {e}")
            return False
    
    def create_subset_colmap(self, subset_id: int, image_ids: List[int]) -> Path:
        """Create subset-specific COLMAP data"""
        subset_colmap_path = self.output_path / f"subset_{subset_id:03d}_colmap"
        subset_colmap_path.mkdir(exist_ok=True)
        
        # Copy and filter COLMAP files for this subset
        # This is a simplified version - should be implemented properly
        import shutil
        
        # Find original sparse directory
        sparse_dirs = [
            self.source_path,
            self.source_path / "sparse",
            self.source_path / "sparse" / "0"
        ]
        
        sparse_dir = None
        for sdir in sparse_dirs:
            if (sdir / "cameras.txt").exists() and (sdir / "images.txt").exists():
                sparse_dir = sdir
                break
        
        if sparse_dir is None:
            raise FileNotFoundError(f"Could not find COLMAP files in {self.source_path}")
        
        # Create sparse subdirectory
        subset_sparse = subset_colmap_path / "sparse" / "0"
        subset_sparse.mkdir(parents=True, exist_ok=True)
        
        # Copy cameras.txt (usually same for all subsets)
        shutil.copy2(sparse_dir / "cameras.txt", subset_sparse / "cameras.txt")
        
        # Filter images.txt and points3D.txt for this subset
        self.filter_images_txt(sparse_dir / "images.txt", subset_sparse / "images.txt", image_ids)
        
        # Copy points3D.txt (or create empty one)
        if (sparse_dir / "points3D.txt").exists():
            shutil.copy2(sparse_dir / "points3D.txt", subset_sparse / "points3D.txt")
        else:
            # Create empty points3D.txt
            with open(subset_sparse / "points3D.txt", 'w') as f:
                f.write("# 3D point list with one line of data per point:\n")
                f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        # Copy images directory (or create symlinks)
        images_src = self.source_path / "images"
        images_dst = subset_colmap_path / "images"
        
        if images_src.exists() and not images_dst.exists():
            # Create symbolic link to save space
            try:
                images_dst.symlink_to(images_src.absolute())
            except:
                # Fallback to copying
                shutil.copytree(images_src, images_dst)
        
        logger.info(f"Created subset COLMAP data: {subset_colmap_path}")
        return subset_colmap_path
    
    def filter_images_txt(self, input_file: Path, output_file: Path, image_ids: List[int]):
        """Filter images.txt to include only specified image IDs"""
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            skip_next_line = False
            
            for line in f_in:
                if line.startswith('#'):
                    f_out.write(line)
                    continue
                
                # Skip this line if it's a continuation line for a filtered out image
                if skip_next_line:
                    skip_next_line = False
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 10:
                    # This is an image metadata line
                    img_id = int(parts[0])
                    if img_id in image_ids:
                        # Remove folder path from image name (e.g., 'images/800886.tif' -> '800886.tif')
                        if '/' in parts[9]:
                            parts[9] = parts[9].split('/')[-1]
                        modified_line = ' '.join(parts) + '\n'
                        f_out.write(modified_line)
                        # Write the next line (2D points) as well
                        next_line = next(f_in, '')
                        f_out.write(next_line)
                    else:
                        # Skip this image and its 2D points line
                        skip_next_line = True
                elif len(parts) > 0:
                    # This might be a 2D points line that wasn't handled properly
                    # Skip it as it should have been handled with its parent image line
                    continue
    
    def train_subsets_parallel(self, subsets: List[List[int]]) -> bool:
        """Train all subsets in parallel"""
        logger.info(f"Training {len(subsets)} subsets in parallel (max {self.args.parallel_jobs} jobs)")
        
        success_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.parallel_jobs) as executor:
            # Submit all training jobs
            future_to_subset = {
                executor.submit(self.train_subset, i, subset): i 
                for i, subset in enumerate(subsets)
            }
            
            # Wait for completion
            for future in concurrent.futures.as_completed(future_to_subset):
                subset_id = future_to_subset[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        logger.info(f"✓ Subset {subset_id} completed successfully")
                    else:
                        logger.error(f"✗ Subset {subset_id} failed")
                except Exception as e:
                    logger.error(f"✗ Subset {subset_id} exception: {e}")
        
        logger.info(f"Training completed: {success_count}/{len(subsets)} subsets successful")
        return success_count == len(subsets)
    
    def merge_ply_files(self) -> bool:
        """Merge all subset PLY files into final result"""
        logger.info("Merging PLY files...")
        
        # Find all subset PLY files
        ply_files = []
        for subset_dir in sorted(self.output_path.glob("subset_*")):
            if subset_dir.is_dir():
                # Look for final PLY file
                ply_candidates = [
                    subset_dir / "point_cloud.ply",
                    subset_dir / "point_cloud" / f"iteration_{self.args.iterations}" / "point_cloud.ply",
                    subset_dir / "point_cloud" / "iteration_final" / "point_cloud.ply",
                ]
                
                for ply_file in ply_candidates:
                    if ply_file.exists():
                        ply_files.append(ply_file)
                        logger.info(f"Found PLY: {ply_file}")
                        break
        
        if not ply_files:
            logger.error("No PLY files found to merge")
            return False
        
        logger.info(f"Merging {len(ply_files)} PLY files")
        
        # Use existing merge script
        final_ply = self.output_path / "final_merged.ply"
        
        try:
            cmd = [
                "python3", "merge_progressive_ply.py",
                "--output_dir", str(self.output_path),
                "--output_ply", str(final_ply)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ PLY files merged successfully: {final_ply}")
                return True
            else:
                logger.error(f"✗ PLY merging failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"✗ PLY merging exception: {e}")
            return False
    
    def run(self) -> int:
        """Main execution function"""
        logger.info("Starting Divide and Conquer 3D Gaussian Splatting")
        
        try:
            # Step 1: Create subsets
            logger.info("=== Step 1: Creating subsets ===")
            subsets = self.create_subsets()
            
            if not subsets:
                logger.error("No subsets created")
                return 1
            
            # Step 2: Train subsets in parallel
            logger.info("=== Step 2: Training subsets ===")
            if not self.train_subsets_parallel(subsets):
                logger.error("Subset training failed")
                return 1
            
            # Step 3: Merge results
            logger.info("=== Step 3: Merging results ===")
            if not self.merge_ply_files():
                logger.error("PLY merging failed")
                return 1
            
            logger.info("🎉 Divide and Conquer training completed successfully!")
            return 0
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Fatal error: {e}")
            logger.error("Full traceback:")
            logger.error(error_details)
            return 1

def main():
    parser = argparse.ArgumentParser(description='Divide and Conquer 3D Gaussian Splatting')
    
    # Required arguments
    parser.add_argument('--source_path', required=True, help='Path to COLMAP reconstruction')
    parser.add_argument('--output_path', required=True, help='Output directory')
    
    # DNQ specific parameters
    parser.add_argument('--pixel_threshold_a', type=int, default=1000000, 
                       help='Maximum pixel count for subset footprint union')
    parser.add_argument('--min_max_ratio_d', type=float, default=0.7,
                       help='Minimum ratio between min/max subset sizes')
    parser.add_argument('--max_subsets', type=int, default=8,
                       help='Maximum number of subsets')
    parser.add_argument('--parallel_jobs', type=int, default=4,
                       help='Number of parallel training jobs')
    
    # 3DGS training parameters
    parser.add_argument('--iterations', type=int, default=30000,
                       help='Training iterations per subset')
    parser.add_argument('--densification_interval', type=int, default=100,
                       help='Densification interval')
    parser.add_argument('--densify_from_iter', type=int, default=500,
                       help='Start densification from iteration')
    parser.add_argument('--densify_until_iter', type=int, default=15000,
                       help='Stop densification at iteration')
    parser.add_argument('--opacity_reset_interval', type=int, default=3000,
                       help='Opacity reset interval')
    parser.add_argument('--sh_degree', type=int, default=3,
                       help='Spherical harmonics degree')
    parser.add_argument('--backend', default='gsplat', choices=['gsplat', 'diff_gauss'],
                       help='Rendering backend')
    
    # Flags
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic training')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--white_background', action='store_true',
                       help='Use white background')
    parser.add_argument('--convert_SHs_python', action='store_true',
                       help='Convert SH features in Python')
    parser.add_argument('--compute_cdf_python', action='store_true',
                       help='Compute CDF in Python')
    
    # Other options
    parser.add_argument('--point_cloud_format', default='auto',
                       choices=['auto', 'ply', 'txt', 'bin'],
                       help='Point cloud format')
    parser.add_argument('--dtm_module', help='Path to external DTM module')
    
    args = parser.parse_args()
    
    # Create runner and execute
    runner = DNQRunner(args)
    return runner.run()

if __name__ == "__main__":
    sys.exit(main())