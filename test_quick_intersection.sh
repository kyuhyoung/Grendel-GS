#!/bin/bash

# Quick test for footprint intersection filtering
# Uses minimal settings for fast execution

SOURCE_PATH="/data/sillim_ew_mini_100024_20"
OUTPUT_PATH="./output/test_quick_intersection"

echo "=========================================="
echo "Quick Intersection Filtering Test"
echo "=========================================="
echo ""
echo "Testing with FOOTPRINT_INTERSECTION_THRESHOLD=0.3"
echo ""

CUDA_VISIBLE_DEVICES=0 python3 progressive_learning/run_progressive.py \
  --source_path=$SOURCE_PATH \
  --output_path=$OUTPUT_PATH \
  --initial_cameras=2 \
  --iterations=100 \
  --iterations_per_window=2 \
  --sh_degree=0 \
  --resolution=4 \
  --backend=gsplat \
  --debug \
  --max_window_size=4 \
  --removal_strategy=fifo \
  --e_selection_strategy=balanced_smooth_trajectory \
  --e_outward_weight=3.0 \
  --e_compact_weight=3.5 \
  --e_smooth_window_weight=2.8 \
  --e_smooth_camera_weight=0.0 \
  --e_distance_weight=2.0 \
  --footprint_intersection_threshold=0.3 \
  --exit_after_first_removal \
  2>&1 | grep -E "Step 9.1|Step 16|Intersection threshold|Camera [0-9]+: intersection|Cameras intersecting|Current window D|Future window E|FIFO|INTERSECTION FILTERING TEST|Selected camera E|Initial window" | head -150

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
