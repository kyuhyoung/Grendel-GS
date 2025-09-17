from pathlib import Path
from typing import Set, List, Dict, Tuple
import sys
import shutil
import os
import argparse
import numpy as np
from scipy.spatial.distance import cdist, euclidean
import re

def geometric_median(X, eps=1e-5):
    """Calculate the geometric median of a set of points."""
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def extract_camera_positions(input_colmap_dir: Path) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Extract camera positions from images.txt file.
    Returns a dictionary mapping image_name to (position, image_id).
    """
    images_file = input_colmap_dir / "images.txt"
    if not images_file.exists():
        raise FileNotFoundError(f"images.txt not found in {input_colmap_dir}")
    
    lines = images_file.read_text().splitlines()
    camera_positions = {}
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            i += 1
            continue
        
        # Parse image line
        tokens = line.strip().split()
        if len(tokens) >= 10:  # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            image_id = int(tokens[0])
            qw, qx, qy, qz = map(float, tokens[1:5])
            tx, ty, tz = map(float, tokens[5:8])
            image_name = tokens[9]
            
            # Calculate camera position: C = -R^T * t
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            camera_pos = -R.T @ t
            
            camera_positions[image_name] = (camera_pos, image_id)
        
        i += 2  # Skip the 2D points line
    
    return camera_positions

def compute_camera_distribution(positions: np.ndarray) -> None:
    """
    Compute and print statistics about camera position distribution:
    - 3D bounding box (min/max for each axis)
    - Bounding box center
    - Average position
    - Bounding box dimensions
    """
    if len(positions) == 0:
        print("No camera positions to analyze")
        return
    
    # Compute bounding box
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_dimensions = bbox_max - bbox_min
    
    # Compute average position
    avg_position = np.mean(positions, axis=0)
    
    # Print statistics
    print("\n===== Camera Position Distribution =====")
    print(f"Number of cameras: {len(positions)}")
    print(f"\n3D Bounding Box:")
    print(f"  Min: X={bbox_min[0]:.3f}, Y={bbox_min[1]:.3f}, Z={bbox_min[2]:.3f}")
    print(f"  Max: X={bbox_max[0]:.3f}, Y={bbox_max[1]:.3f}, Z={bbox_max[2]:.3f}")
    print(f"  Dimensions: X={bbox_dimensions[0]:.3f}, Y={bbox_dimensions[1]:.3f}, Z={bbox_dimensions[2]:.3f}")
    print(f"\nBounding Box Center: X={bbox_center[0]:.3f}, Y={bbox_center[1]:.3f}, Z={bbox_center[2]:.3f}")
    print(f"Average Position: X={avg_position[0]:.3f}, Y={avg_position[1]:.3f}, Z={avg_position[2]:.3f}")
    
    # Compute distance between center and average
    center_avg_dist = np.linalg.norm(bbox_center - avg_position)
    print(f"\nDistance between bbox center and average: {center_avg_dist:.3f}")
    print("========================================\n")

def find_closest_images_by_distance(camera_positions: Dict[str, Tuple[np.ndarray, int]], 
                                   threshold: float) -> Tuple[List[str], str]:
    """
    Find closest images to the geometric median based on threshold.
    If threshold > 1: select that many closest images
    If 0 < threshold < 1: select threshold * total_images closest images
    Returns: (list of selected image names, median image name)
    """
    if not camera_positions:
        return [], ""
    
    # Extract positions and names
    names = list(camera_positions.keys())
    positions = np.array([camera_positions[name][0] for name in names])
    
    # Compute and display camera distribution statistics
    compute_camera_distribution(positions)
    
    # Calculate geometric median
    median_pos = geometric_median(positions)
    print(f"Geometric median position: {median_pos}")
    
    # Calculate distances from median
    distances = np.array([np.linalg.norm(pos - median_pos) for pos in positions])
    
    # Sort by distance
    sorted_indices = np.argsort(distances)
    
    # Find the closest image to median (for naming)
    median_image_name = names[sorted_indices[0]]
    median_image_base = os.path.splitext(median_image_name)[0]  # Remove extension
    
    # Determine number of images to select
    total_images = len(names)
    if threshold >= 1:
        n_closest = int(threshold)
    else:
        n_closest = int(threshold * total_images)
    
    n_closest = min(n_closest, total_images)
    print(f"Selecting {n_closest} closest images out of {total_images} total images")
    print(f"Closest image to median: {median_image_name}")
    
    # Select closest images
    selected_indices = sorted_indices[:n_closest]
    selected_names = [names[i] for i in selected_indices]
    
    # Print all images with distances, showing cut line between selected and non-selected
    print("\nAll images sorted by distance from median:")
    print("-" * 60)
    
    for i, idx in enumerate(sorted_indices):
        distance = distances[idx]
        name = names[idx]
        
        # Check if this is the cut line position
        if i == n_closest:
            print("=" * 60)
            print(f"====== CUT LINE: Above {n_closest} images selected ======")
            print("=" * 60)
        
        # Mark selected images
        marker = "[SELECTED]" if i < n_closest else "[NOT SELECTED]"
        print(f"  {i+1:4d}. {name:30s} distance={distance:8.3f}  {marker}")
    
    print("-" * 60)
    
    return sorted(selected_names), median_image_base  # Return in alphabetical order

def find_colmap_dir(input_dir: Path) -> Path:
    """Find the subfolder containing COLMAP files."""
    for path in input_dir.rglob("images.txt"):
        return path.parent
    return None

def generate_output_dir(input_dir: Path, mode: str, n_selected: int, median_image_base: str = "") -> Path:
    """
    Generate output directory name based on mode.
    For -list mode: input_dir_mini_{n_selected}
    For -dist mode: input_dir_mini_{median_image_base}_{n_selected}
    """
    parent_dir = input_dir.parent
    input_name = input_dir.name
    
    if mode == "list":
        output_name = f"{input_name}_mini_{n_selected}"
    elif mode == "dist":
        output_name = f"{input_name}_mini_{median_image_base}_{n_selected}"
    else:
        output_name = f"{input_name}_mini_{n_selected}"
    
    output_dir = parent_dir / output_name
    print(f"Output directory: {output_dir}")
    return output_dir

def filter_colmap_model(input_dir: Path, output_dir: Path, dir_img: str, select_fns: List[str]):
    """
    Create a filtered COLMAP model by keeping only specific images and
    rewriting images.txt and points3D.txt accordingly. Also copies cameras.txt.
    
    Args:
        input_dir (Path): Directory containing original COLMAP TXT model.
        output_dir (Path): Directory to write the filtered COLMAP TXT model.
        dir_img (str): Name of the images folder.
        select_fns (List[str]): List of image filenames to keep.
    """
    input_images_dir = input_dir / dir_img
    assert input_images_dir.is_dir(), f"Input images directory does not exist: {input_images_dir}"
    
    # Remove output directory if it exists and recreate it
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / dir_img
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Find the subfolder containing images.txt under input_dir
    input_colmap_dir = find_colmap_dir(input_dir)
    
    if input_colmap_dir is None:
        print(f"Could not find images.txt under {input_dir}")
        exit(1)
    
    # Calculate relative path from input_dir to input_colmap_dir
    relative_colmap_path = input_colmap_dir.relative_to(input_dir)
    
    # Create the same subfolder structure in output_dir
    output_colmap_dir = output_dir / relative_colmap_path
    output_colmap_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found COLMAP files in: {input_colmap_dir}")
    print(f"Output COLMAP files will be saved in: {output_colmap_dir}")
    print(f"Filtering for 'images.txt' and copying images") 
    
    # Convert list to set for faster lookup
    select_fns_set = set(select_fns)
    
    # Process images.txt (2 lines per image entry)
    lines = (input_colmap_dir / "images.txt").read_text().splitlines()
    filtered_images = []
    select_ids = []
    n_img = len(select_fns)
    i = 0
    i_done = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            filtered_images.append(line)
            i += 1
            continue
        tokens = line.strip().split()
        image_id = int(tokens[0])
        image_name = tokens[-1]
        if image_name in select_fns_set:
            i_done += 1
            print(f'\tProcessing {i_done} / {n_img} image : {image_name}')
            select_ids.append(image_id)
            # Keep this image's metadata and its next line (2D points)
            filtered_images.append(line)
            if i + 1 < len(lines):
                filtered_images.append(lines[i + 1])
            src = input_images_dir / image_name
            dst = output_images_dir / image_name
            if src.exists():
                print(f'\t\tCopying image from {src} to {dst}')
                os.system(f'cp {src} {dst}')
        i += 2  # Skip current and 2D points line

    (output_colmap_dir / "images.txt").write_text("\n".join(filtered_images) + "\n")

    print(f"Filtering for 'points3D.txt'") 
    # Process points3D.txt
    with open(input_colmap_dir / "points3D.txt", "r") as fin, open(output_colmap_dir / "points3D.txt", "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            tokens = line.strip().split()
            track_tokens = tokens[8:]
            track_pairs = list(zip(track_tokens[::2], track_tokens[1::2]))
            new_track = [
                f"{img_id} {pt2d_id}"
                for img_id, pt2d_id in track_pairs
                if int(img_id) in select_ids
            ]
            if new_track:
                new_line = " ".join(tokens[:8] + new_track)
                fout.write(new_line + "\n")

    # Copy cameras.txt without change
    cameras_src = input_colmap_dir / "cameras.txt"
    cameras_dst = output_colmap_dir / "cameras.txt"
    cameras_dst.write_text(cameras_src.read_text())

    print(f"Filtered model written to: {output_colmap_dir}")


def main():
    parser = argparse.ArgumentParser(description='Filter COLMAP model by image selection')
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    
    # List mode: specify images directly
    list_parser = subparsers.add_parser('list', help='Filter using explicit list of images')
    list_parser.add_argument('input_dir', help='Input directory containing COLMAP model')
    list_parser.add_argument('dir_img', help='Name of images folder')
    list_parser.add_argument('image_list', help='Space-separated list of image names')
    
    # Distance mode: select closest images to geometric median
    dist_parser = subparsers.add_parser('dist', help='Filter using distance from geometric median')
    dist_parser.add_argument('input_dir', help='Input directory containing COLMAP model')
    dist_parser.add_argument('dir_img', help='Name of images folder')
    dist_parser.add_argument('threshold', type=float, 
                           help='If >= 1: number of closest images. If < 1: fraction of total images')
    
    args = parser.parse_args()
    
    if args.mode is None:
        # Backward compatibility: if no mode specified, assume old format (deprecated)
        parser.print_help()
        exit(1)
    
    elif args.mode == 'list':
        input_dir = Path(args.input_dir)
        dir_img = args.dir_img
        select_fns = sorted(args.image_list.split())
        
        # Generate output directory name
        output_dir = generate_output_dir(input_dir, "list", len(select_fns))
        
        filter_colmap_model(input_dir, output_dir, dir_img, select_fns)
    
    elif args.mode == 'dist':
        input_dir = Path(args.input_dir)
        dir_img = args.dir_img
        threshold = args.threshold
        
        # Find COLMAP directory
        input_colmap_dir = find_colmap_dir(input_dir)
        if input_colmap_dir is None:
            print(f"Could not find images.txt under {input_dir}")
            exit(1)
        
        # Extract camera positions
        print(f"Extracting camera positions from {input_colmap_dir}")
        camera_positions = extract_camera_positions(input_colmap_dir)
        print(f"Found {len(camera_positions)} camera positions")
        
        # Find closest images
        select_fns, median_image_base = find_closest_images_by_distance(camera_positions, threshold)
        
        if not select_fns:
            print("No images selected")
            exit(1)
        
        # Generate output directory name
        output_dir = generate_output_dir(input_dir, "dist", len(select_fns), median_image_base)
        
        # Filter the model
        filter_colmap_model(input_dir, output_dir, dir_img, select_fns)


if __name__ == "__main__":
    main()
