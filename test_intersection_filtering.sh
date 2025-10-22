#!/bin/bash

# Test footprint intersection filtering
# This script tests different threshold values to verify filtering works correctly

SOURCE_PATH="/data/sillim_ew_mini_100024_20"
OUTPUT_PATH="./output/test_intersection"

echo "=========================================="
echo "Testing Footprint Intersection Filtering"
echo "=========================================="
echo ""

# Test 1: threshold = 0.0 (disabled, baseline)
echo "Test 1: FOOTPRINT_INTERSECTION_THRESHOLD=0.0 (disabled)"
echo "--------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 python3 progressive_learning/run_progressive.py \
  --source_path=$SOURCE_PATH \
  --output_path=${OUTPUT_PATH}_threshold_0.0 \
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
  --footprint_intersection_threshold=0.0 \
  --exit_after_first_removal \
  2>&1 | grep -E "Intersection threshold|Camera [0-9]+:|intersection ratio|Cameras intersecting|Selected camera E|Initial in D|Using future|Window [0-9]|FIFO" | head -100

echo ""
echo "=========================================="
echo ""

# Test 2: threshold = 0.3 (30% overlap required)
echo "Test 2: FOOTPRINT_INTERSECTION_THRESHOLD=0.3"
echo "--------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 python3 progressive_learning/run_progressive.py \
  --source_path=$SOURCE_PATH \
  --output_path=${OUTPUT_PATH}_threshold_0.3 \
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
  2>&1 | grep -E "Intersection threshold|Camera [0-9]+:|intersection ratio|Cameras intersecting|Selected camera E|Initial in D|Using future|Window [0-9]|FIFO" | head -100

echo ""
echo "=========================================="
echo ""

# Test 3: threshold = 0.5 (50% overlap required)
echo "Test 3: FOOTPRINT_INTERSECTION_THRESHOLD=0.5"
echo "--------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 python3 progressive_learning/run_progressive.py \
  --source_path=$SOURCE_PATH \
  --output_path=${OUTPUT_PATH}_threshold_0.5 \
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
  --footprint_intersection_threshold=0.5 \
  --exit_after_first_removal \
  2>&1 | grep -E "Intersection threshold|Camera [0-9]+:|intersection ratio|Cameras intersecting|Selected camera E|Initial in D|Using future|Window [0-9]|FIFO" | head -100

echo ""
echo "=========================================="
echo "Testing completed!"
echo "=========================================="
