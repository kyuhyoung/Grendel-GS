#!/bin/bash

# usage_dnq.sh - Divide and Conquer 3D Gaussian Splatting
# This script divides the entire image set into subsets and processes each subset with Grendel-GS,
# then merges all resulting PLY files into a single final PLY.

set -e

# Set up logging - redirect all output to both console and log file
LOG_FILE="usage_dnq.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "=== DNQ Script Started: $(date) ==="

# Required parameters
# Check if dataset name is provided as first argument
echo "DEBUG: First argument \$1 = '$1'"
echo "DEBUG: All arguments: $@"
if [ -n "$1" ]; then
    SOURCE_PATH="/data/$1"  # Use provided dataset name
    echo "DEBUG: Set SOURCE_PATH to: $SOURCE_PATH"
    shift  # Remove dataset name from arguments
else
    SOURCE_PATH="/data/Samsung_SN_30"  # Default path to COLMAP reconstruction
    echo "DEBUG: Using default SOURCE_PATH: $SOURCE_PATH"
fi
OUTPUT_PATH="./output/dnq_test"   # Output directory

# DNQ parameters
THRESHOLD_A=50000000  # Maximum pixel count for subset footprint union (50M pixels)
THRESHOLD_D=0.7       # Minimum ratio between min(C) and max(C)
DEBUG=false
DRY_RUN=false
MERGE_ONLY=false

# Grendel-GS parameters (inherit from usage_progressive.sh defaults)
ITERATIONS=30000
SH_DEGREE=3
BACKEND="gsplat"
DENSIFICATION_INTERVAL=100
DENSIFY_FROM_ITER=500
DENSIFY_MEMORY_LIMIT_PERCENTAGE=0.8

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

function print_usage() {
    echo "Usage: $0 [dataset_name] [options]"
    echo ""
    echo "Divide and Conquer 3D Gaussian Splatting"
    echo ""
    echo "Arguments:"
    echo "  dataset_name                Dataset name in /data/ directory (default: Samsung_SN_30)"
    echo ""
    echo "Options:"
    echo "  -o, --output_path PATH      Output directory (default: ./output/dnq_test)"
    echo "  -a, --threshold_a NUM       Max pixel count for subset union (default: 50M)"
    echo "  -d, --threshold_d RATIO     Min ratio between min/max subset sizes (default: 0.7)"
    echo "  --iterations NUM            Training iterations per subset (default: 30000)"
    echo "  --sh_degree NUM             SH degree (default: 3)"
    echo "  --backend NAME              Backend (gsplat/diff_gauss) (default: gsplat)"
    echo "  --debug                     Enable debug mode"
    echo "  --dry_run                   Only show subset division, don't run training"
    echo "  --merge_only                Only merge existing PLY files, skip training"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 Samsung_SN_30                  # Use Samsung_SN_30 dataset"
    echo "  $0 my_dataset -o ./output/result   # Use custom dataset with custom output"
    echo "  $0                                 # Use default Samsung_SN_30"
}

function log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

function log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

function log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function log_debug() {
    if [[ "$DEBUG" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Process remaining arguments after dataset name
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -o|--output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -a|--threshold_a)
            THRESHOLD_A="$2"
            shift 2
            ;;
        -d|--threshold_d)
            THRESHOLD_D="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --sh_degree)
            SH_DEGREE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --merge_only)
            MERGE_ONLY=true
            shift
            ;;
        *)
            # Unknown argument, add to EXTRA_ARGS
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Validate source path
if [[ "$MERGE_ONLY" != "true" ]]; then
    if [[ ! -d "$SOURCE_PATH" ]]; then
        log_error "Source path does not exist: $SOURCE_PATH"
        exit 1
    fi
    if [[ ! -d "$SOURCE_PATH/sparse" ]] && [[ ! -f "$SOURCE_PATH/cameras.txt" ]]; then
        log_error "No COLMAP data found in: $SOURCE_PATH"
        log_warning "Expected: sparse/ directory or cameras.txt file"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}Divide and Conquer 3D Gaussian Splatting${NC}"
echo -e "${CYAN}======================================${NC}"

if [[ "$MERGE_ONLY" != "true" ]]; then
    log_info "Configuration:"
    echo "  Source Path: $SOURCE_PATH"
    echo "  Output Path: $OUTPUT_PATH"
    echo "  Threshold A (Max pixels): $THRESHOLD_A"
    echo "  Threshold D (Min ratio): $THRESHOLD_D"
    echo "  Iterations: $ITERATIONS"
    echo "  SH Degree: $SH_DEGREE"
    echo "  Backend: $BACKEND"
    echo "  Debug: $DEBUG"
    echo "  Dry Run: $DRY_RUN"
else
    log_info "Merge-only mode enabled"
    echo "  Output Path: $OUTPUT_PATH"
fi

# Function to create subset division
function create_subsets() {
    local source_path="$1"
    local output_path="$2"

    log_info "Creating subset division..."

    # Use existing dnq_runner.py instead of creating inline script
    log_info "Using dnq_runner.py for subset creation and training..."
    
    # Build dnq_runner.py command
    local cmd="python3 dnq_runner.py"
    cmd="$cmd --source_path $source_path"
    cmd="$cmd --output_path $output_path"
    cmd="$cmd --pixel_threshold_a $THRESHOLD_A"
    cmd="$cmd --min_max_ratio_d $THRESHOLD_D"
    cmd="$cmd --max_subsets 8"
    cmd="$cmd --parallel_jobs 4"
    cmd="$cmd --iterations $ITERATIONS"
    cmd="$cmd --sh_degree $SH_DEGREE"
    cmd="$cmd --backend $BACKEND"
    cmd="$cmd --densification_interval $DENSIFICATION_INTERVAL"
    cmd="$cmd --densify_from_iter $DENSIFY_FROM_ITER"
    
    if [[ "$DEBUG" == "true" ]]; then
        cmd="$cmd --debug"
    fi
    
    log_info "Running DNQ command: $cmd"
    
    # Execute dnq_runner.py which will handle everything
    if eval "$cmd"; then
        log_success "DNQ runner completed successfully"
        return 0
    else
        log_error "DNQ runner failed"
        return 1
    fi
    
    # Old inline script creation - keeping as backup but not used
    return 0
    
    cat > "${output_path}/create_subsets_backup.py" << 'EOF'
#!/usr/bin/env python3
"""
Divide and Conquer Subset Creation for 3D Gaussian Splatting

This script divides camera images into subsets based on footprint union constraints.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import itertools

def load_colmap_data(colmap_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Load COLMAP cameras, images, and points3D data"""

    # Find sparse directory
    sparse_dirs = [
        colmap_path,
        colmap_path / "sparse",
        colmap_path / "sparse" / "0"
    ]

    sparse_dir = None
    for sdir in sparse_dirs:
        if (sdir / "cameras.txt").exists() and (sdir / "images.txt").exists():
            sparse_dir = sdir
            break

    if sparse_dir is None:
        raise FileNotFoundError(f"Could not find COLMAP files in {colmap_path}")

    print(f"Loading COLMAP data from: {sparse_dir}")

    # Load cameras
    cameras = {}
    with open(sparse_dir / "cameras.txt", 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(x) for x in parts[4:]]
                cameras[cam_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }

    # Load images
    images = {}
    with open(sparse_dir / "images.txt", 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 10:
                img_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                cam_id = int(parts[8])
                name = parts[9]
                images[img_id] = {
                    'quat': [qw, qx, qy, qz],
                    'trans': [tx, ty, tz],
                    'camera_id': cam_id,
                    'name': name
                }

    print(f"Loaded {len(cameras)} cameras, {len(images)} images")
    return cameras, images, {}

def calculate_camera_footprint(camera: Dict, image: Dict) -> Dict:
    """Calculate camera footprint as bounding box in world coordinates"""

    # For simplicity, we'll use a basic footprint calculation
    # In practice, this should project camera frustum to ground plane

    width = camera['width']
    height = camera['height']

    # Camera position
    tx, ty, tz = image['trans']

    # Simplified footprint: assume camera looks down and calculate approximate coverage
    # This is a placeholder - real implementation should use proper projection

    # Estimate footprint size based on height and camera parameters
    if len(camera['params']) >= 2:
        fx, fy = camera['params'][0], camera['params'][1]
        # Rough estimate of ground coverage
        ground_width = width * tz / fx if fx > 0 else width
        ground_height = height * tz / fy if fy > 0 else height
    else:
        # Fallback
        ground_width = width * 0.1
        ground_height = height * 0.1

    # Create bounding box
    footprint = {
        'min_x': tx - ground_width / 2,
        'max_x': tx + ground_width / 2,
        'min_y': ty - ground_height / 2,
        'max_y': ty + ground_height / 2,
        'center_x': tx,
        'center_y': ty,
        'width': ground_width,
        'height': ground_height,
        'pixels': width * height
    }

    return footprint

def calculate_union_footprint(footprints: List[Dict]) -> Dict:
    """Calculate union of multiple footprints"""
    if not footprints:
        return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0, 'width': 0, 'height': 0, 'pixels': 0}

    min_x = min(fp['min_x'] for fp in footprints)
    max_x = max(fp['max_x'] for fp in footprints)
    min_y = min(fp['min_y'] for fp in footprints)
    max_y = max(fp['max_y'] for fp in footprints)

    width = max_x - min_x
    height = max_y - min_y

    # Estimate total pixels in union (simplified)
    total_pixels = sum(fp['pixels'] for fp in footprints)

    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'width': width,
        'height': height,
        'pixels': total_pixels,
        'aspect_ratio': width / height if height > 0 else 1.0
    }

def greedy_subset_creation(image_footprints: Dict[int, Dict], threshold_a: int, threshold_d: float) -> List[List[int]]:
    """Create subsets using greedy algorithm with minimum 2 images per subset"""

    image_ids = list(image_footprints.keys())
    unassigned = set(image_ids)
    subsets = []

    print(f"Creating subsets for {len(image_ids)} images...")
    print(f"Threshold A (max pixels): {threshold_a:,}")
    print(f"Threshold D (min ratio): {threshold_d}")
    print(f"Minimum images per subset: 2")

    while unassigned:
        # Check if we have at least 2 images left to form a valid subset
        if len(unassigned) < 2:
            print(f"WARNING: Only {len(unassigned)} image(s) remaining, cannot form valid subset")
            # Add remaining image(s) to the smallest existing subset
            if unassigned and subsets:
                smallest_idx = min(range(len(subsets)), key=lambda i: len(subsets[i]))
                subsets[smallest_idx].extend(list(unassigned))
                print(f"  Added remaining image(s) to subset {smallest_idx + 1}")
                unassigned.clear()
            break
        
        # Start new subset with an unassigned image
        current_subset = [unassigned.pop()]
        current_footprints = [image_footprints[current_subset[0]]]
        current_union = calculate_union_footprint(current_footprints)

        print(f"\nStarting subset {len(subsets) + 1} with image {current_subset[0]}")
        
        # Ensure minimum 2 images - force add at least one more
        if unassigned:
            # Find closest image to ensure minimum subset size
            best_candidate = None
            best_distance = float('inf')
            for candidate_id in unassigned:
                # Simple distance metric based on footprint centers
                dist = ((image_footprints[candidate_id]['center_x'] - image_footprints[current_subset[0]]['center_x'])**2 +
                       (image_footprints[candidate_id]['center_y'] - image_footprints[current_subset[0]]['center_y'])**2)**0.5
                if dist < best_distance:
                    best_distance = dist
                    best_candidate = candidate_id
            
            if best_candidate is not None:
                current_subset.append(best_candidate)
                current_footprints.append(image_footprints[best_candidate])
                current_union = calculate_union_footprint(current_footprints)
                unassigned.remove(best_candidate)
                print(f"  Added image {best_candidate} to meet minimum size requirement")

        # Try to add more images to current subset
        improved = True
        while improved and unassigned:
            improved = False
            best_candidate = None
            best_union = None
            best_score = float('inf')

            for candidate_id in list(unassigned):
                candidate_footprint = image_footprints[candidate_id]
                test_footprints = current_footprints + [candidate_footprint]
                test_union = calculate_union_footprint(test_footprints)

                # Check constraints
                if test_union['pixels'] > threshold_a:
                    continue  # Exceeds pixel threshold

                # Score based on aspect ratio (prefer ratio close to 1.0)
                aspect_score = abs(1.0 - test_union['aspect_ratio'])

                if aspect_score < best_score:
                    best_score = aspect_score
                    best_candidate = candidate_id
                    best_union = test_union

            if best_candidate is not None:
                current_subset.append(best_candidate)
                current_footprints.append(image_footprints[best_candidate])
                current_union = best_union
                unassigned.remove(best_candidate)
                improved = True

                print(f"  Added image {best_candidate} (aspect ratio: {current_union['aspect_ratio']:.3f}, pixels: {current_union['pixels']:,})")

        subsets.append(current_subset)
        print(f"Completed subset {len(subsets)} with {len(current_subset)} images")
        print(f"  Union: {current_union['width']:.1f} x {current_union['height']:.1f}, aspect ratio: {current_union['aspect_ratio']:.3f}")
        print(f"  Total pixels: {current_union['pixels']:,}")
        # Show image IDs (first 10 if more than 10)
        if len(current_subset) <= 10:
            print(f"  Image IDs: {current_subset}")
        else:
            print(f"  Image IDs: {current_subset[:10]}... (and {len(current_subset)-10} more)")

    return subsets

def validate_subsets(subsets: List[List[int]], image_footprints: Dict[int, Dict], threshold_d: float) -> bool:
    """Validate subset constraints including minimum size"""

    print(f"\nValidating {len(subsets)} subsets...")
    
    # Check minimum size constraint (at least 2 images per subset)
    for i, subset in enumerate(subsets):
        if len(subset) < 2:
            print(f"ERROR: Subset {i+1} has only {len(subset)} image(s), minimum 2 required")
            return False

    # Check no image sharing
    all_images = set()
    for subset in subsets:
        subset_set = set(subset)
        if all_images & subset_set:
            print("ERROR: Subsets share images!")
            return False
        all_images.update(subset_set)

    # Check all images assigned
    expected_images = set(image_footprints.keys())
    if all_images != expected_images:
        missing = expected_images - all_images
        print(f"ERROR: {len(missing)} images not assigned: {list(missing)[:10]}...")
        return False

    # Calculate subset pixel counts
    pixel_counts = []
    for i, subset in enumerate(subsets):
        footprints = [image_footprints[img_id] for img_id in subset]
        union = calculate_union_footprint(footprints)
        pixel_counts.append(union['pixels'])
        print(f"  Subset {i+1}: {len(subset)} images, {union['pixels']:,} pixels, aspect ratio: {union['aspect_ratio']:.3f}")
        if len(subset) <= 10:
            print(f"    Image IDs: {subset}")
        else:
            print(f"    Image IDs: {subset[:10]}... (and {len(subset)-10} more)")

    # Check ratio constraint
    min_pixels = min(pixel_counts)
    max_pixels = max(pixel_counts)
    ratio = min_pixels / max_pixels if max_pixels > 0 else 0

    print(f"\nPixel count ratio: {ratio:.3f} (min: {min_pixels:,}, max: {max_pixels:,})")

    if ratio < threshold_d:
        print(f"WARNING: Ratio {ratio:.3f} is below threshold {threshold_d}")
        return False

    print("Subset validation passed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create subsets for divide and conquer 3DGS')
    parser.add_argument('source_path', help='Path to COLMAP reconstruction')
    parser.add_argument('output_path', help='Output directory')
    parser.add_argument('--threshold_a', type=int, default=50000000, help='Max pixels for subset union')
    parser.add_argument('--threshold_d', type=float, default=0.7, help='Min ratio between min/max subset sizes')

    args = parser.parse_args()

    source_path = Path(args.source_path)
    output_path = Path(args.output_path)

    # Load COLMAP data
    cameras, images, _ = load_colmap_data(source_path)

    # Calculate footprints for all images
    image_footprints = {}
    for img_id, image in images.items():
        camera = cameras[image['camera_id']]
        footprint = calculate_camera_footprint(camera, image)
        image_footprints[img_id] = footprint

    # Create subsets
    subsets = greedy_subset_creation(image_footprints, args.threshold_a, args.threshold_d)

    # Validate subsets
    valid = validate_subsets(subsets, image_footprints, args.threshold_d)

    # Save subsets
    subsets_data = {
        'subsets': subsets,
        'metadata': {
            'total_images': len(images),
            'num_subsets': len(subsets),
            'threshold_a': args.threshold_a,
            'threshold_d': args.threshold_d,
            'validation_passed': valid
        }
    }

    output_file = output_path / "subsets.json"
    with open(output_file, 'w') as f:
        json.dump(subsets_data, f, indent=2)

    print(f"\nSubsets saved to: {output_file}")
    print(f"Created {len(subsets)} subsets for {len(images)} images")
    
    # Print final subset summary
    print("\n=== FINAL SUBSET SUMMARY ===")
    for i, subset in enumerate(subsets):
        print(f"Subset {i+1}: {len(subset)} images")
        if len(subset) <= 20:
            print(f"  IDs: {subset}")
        else:
            print(f"  IDs: {subset[:20]}... (and {len(subset)-20} more)")
    print("============================")

    return 0 if valid else 1

if __name__ == "__main__":
    sys.exit(main())
EOF

    # Make script executable
    chmod +x "${output_path}/create_subsets.py"

    # Run subset creation
    log_info "Running subset division algorithm..."

    if ! python3 "${output_path}/create_subsets.py" "$source_path" "$output_path" \
        --threshold_a "$THRESHOLD_A" \
        --threshold_d "$THRESHOLD_D"; then
        log_error "Subset creation failed"
        return 1
    fi

    return 0
}

# Function to train individual subset
function train_subset() {
    local subset_id="$1"
    local source_path="$2"
    local output_path="$3"
    local image_ids="$4"  # Comma-separated image IDs

    local subset_output="${output_path}/subset_${subset_id}"
    mkdir -p "$subset_output"

    log_info "Training subset ${subset_id}..."
    log_debug "Images: $image_ids"

    # Create subset-specific training command
    local cmd="./train.py"
    cmd="$cmd --source_path $source_path"
    cmd="$cmd --model_path $subset_output"
    cmd="$cmd --iterations $ITERATIONS"
    cmd="$cmd --sh_degree $SH_DEGREE"
    cmd="$cmd --backend $BACKEND"
    cmd="$cmd --densification_interval $DENSIFICATION_INTERVAL"
    cmd="$cmd --densify_from_iter $DENSIFY_FROM_ITER"
    cmd="$cmd --densify_memory_limit_percentage $DENSIFY_MEMORY_LIMIT_PERCENTAGE"

    # Add image filtering if needed (this would require modifying train.py to accept image subset)
    # For now, we assume the full training but this should be extended

    log_debug "Training command: $cmd"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would execute: $cmd"
        # Create dummy PLY for testing
        echo "# Dummy PLY file for subset $subset_id" > "${subset_output}/point_cloud.ply"
        return 0
    fi

    # Execute training
    if eval "$cmd"; then
        log_success "Subset $subset_id training completed"
        return 0
    else
        log_error "Subset $subset_id training failed"
        return 1
    fi
}

# Function to merge all subset PLY files
function merge_subsets() {
    local output_path="$1"

    log_info "Merging all subset PLY files..."

    # Find all subset PLY files
    local ply_files=()
    for subset_dir in "${output_path}"/subset_*/; do
        if [[ -d "$subset_dir" ]]; then
            # Look for final PLY file
            local ply_candidates=(
                "${subset_dir}/point_cloud.ply"
                "${subset_dir}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
                "${subset_dir}/point_cloud/iteration_final/point_cloud.ply"
            )

            for ply_file in "${ply_candidates[@]}"; do
                if [[ -f "$ply_file" ]]; then
                    ply_files+=("$ply_file")
                    log_debug "Found PLY: $ply_file"
                    break
                fi
            done
        fi
    done

    if [[ ${#ply_files[@]} -eq 0 ]]; then
        log_error "No PLY files found to merge"
        return 1
    fi

    log_info "Found ${#ply_files[@]} PLY files to merge"

    # Create Python script for merging
    cat > "${output_path}/merge_ply.py" << 'EOF'
#!/usr/bin/env python3
"""
Merge multiple PLY files into a single PLY file
"""

import sys
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

def load_ply(ply_path):
    """Load PLY file and return vertex data"""
    print(f"Loading: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    vertex_array = np.array(vertex.data)
    print(f"  Loaded {len(vertex_array)} points")
    return vertex_array

def merge_ply_files(ply_files, output_ply):
    """Merge multiple PLY files"""
    print(f"Merging {len(ply_files)} PLY files...")

    all_vertices = []
    for ply_file in ply_files:
        vertices = load_ply(ply_file)
        all_vertices.append(vertices)

    # Concatenate all vertices
    if all_vertices:
        merged_vertices = np.concatenate(all_vertices)
        print(f"Total merged points: {len(merged_vertices)}")

        # Create PLY element
        vertex_element = PlyElement.describe(merged_vertices, 'vertex')

        # Write merged PLY
        PlyData([vertex_element], text=False).write(output_ply)

        file_size = Path(output_ply).stat().st_size / (1024 * 1024)  # MB
        print(f"Merged PLY saved: {output_ply}")
        print(f"File size: {file_size:.2f} MB")
        print(f"Total points: {len(merged_vertices):,}")
    else:
        print("No vertices to merge")
        return 1

    return 0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: merge_ply.py output.ply input1.ply input2.ply ...")
        sys.exit(1)

    output_ply = sys.argv[1]
    input_plys = sys.argv[2:]

    sys.exit(merge_ply_files(input_plys, output_ply))
EOF

    chmod +x "${output_path}/merge_ply.py"

    # Run merging
    local final_ply="${output_path}/final_merged.ply"

    if python3 "${output_path}/merge_ply.py" "$final_ply" "${ply_files[@]}"; then
        log_success "PLY files merged successfully: $final_ply"
        return 0
    else
        log_error "PLY merging failed"
        return 1
    fi
}

# Main execution
function main() {
    if [[ "$MERGE_ONLY" == "true" ]]; then
        # Only merge existing results
        if merge_subsets "$OUTPUT_PATH"; then
            log_success "Merge completed successfully"
            exit 0
        else
            log_error "Merge failed"
            exit 1
        fi
    fi

    # Use dnq_runner.py to handle the entire workflow
    log_info "Starting DNQ workflow using dnq_runner.py"
    if ! create_subsets "$SOURCE_PATH" "$OUTPUT_PATH"; then
        log_error "DNQ workflow failed"
        exit 1
    fi

    # Copy dnq_runner.log contents to our main log
    if [[ -f "dnq_runner.log" ]]; then
        echo "=== DNQ Runner Log Contents ===" >> "$LOG_FILE"
        cat "dnq_runner.log" >> "$LOG_FILE"
        echo "=== End DNQ Runner Log ===" >> "$LOG_FILE"
    fi

    log_success "Divide and Conquer 3DGS completed successfully!"
    log_info "Check ${OUTPUT_PATH}/final_merged.ply for results"
    echo "=== DNQ Script Completed: $(date) ===" >> "$LOG_FILE"
}

# Execute main function
main "$@"