#!/bin/bash

# COLMAP Visualizer Direct Execution Script
# colmap_visualizer.py를 직접 실행하는 스크립트

# 로그 파일 설정
LOG_FILE="run_colmap_visualizer.log"

# 로그 파일 초기화 및 tee 설정으로 콘솔과 로그에 동시 출력
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================" | tee -a "$LOG_FILE"
echo "Log started at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 색상 정의 (콘솔 출력용)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "COLMAP Visualizer with Footprints"
echo "========================================"
echo -e "${NC}"

# 기본 COLMAP 경로 설정 (실제 경로로 수정 필요)
DEFAULT_COLMAP_PATH="/data/sillim_ew_mini_100024_20/sparse/0"

# 인자 처리
if [ $# -eq 0 ]; then
    COLMAP_PATH=$DEFAULT_COLMAP_PATH
    echo -e "${YELLOW}Using default COLMAP path: $COLMAP_PATH${NC}"
else
    COLMAP_PATH=$1
    echo -e "${GREEN}Using provided COLMAP path: $COLMAP_PATH${NC}"
fi

# COLMAP 데이터 확인
echo -e "\n${YELLOW}Checking COLMAP data...${NC}"
if [ ! -f "$COLMAP_PATH/cameras.txt" ]; then
    echo -e "${RED}Error: cameras.txt not found in $COLMAP_PATH${NC}"
    exit 1
fi
if [ ! -f "$COLMAP_PATH/images.txt" ]; then
    echo -e "${RED}Error: images.txt not found in $COLMAP_PATH${NC}"
    exit 1
fi
if [ ! -f "$COLMAP_PATH/points3D.txt" ]; then
    echo -e "${RED}Error: points3D.txt not found in $COLMAP_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All required COLMAP files found${NC}"

# 출력 파일명 설정
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_3D="colmap_3d_scene_${TIMESTAMP}.png"
OUTPUT_ORTHO="orthographic_with_footprints_${TIMESTAMP}.png"

echo -e "\n${YELLOW}Output files:${NC}"
echo "  - 3D Scene: $OUTPUT_3D"
echo "  - Orthographic: $OUTPUT_ORTHO"

# Python 스크립트 작성 (임시)
TEMP_SCRIPT="/tmp/run_colmap_viz_${TIMESTAMP}.py"
cat > $TEMP_SCRIPT << 'EOF'
#!/usr/bin/env python3

import sys
import os

# Add project path
sys.path.insert(0, '/workspace/Grendel-GS')

from scripts.experiments.colmap_visualizer import COLMAPVisualizer
import numpy as np

def main():
    # Get COLMAP path from command line
    if len(sys.argv) < 2:
        print("Error: Please provide COLMAP path")
        sys.exit(1)

    colmap_path = sys.argv[1]
    output_3d = sys.argv[2] if len(sys.argv) > 2 else "colmap_3d_scene.png"
    output_ortho = sys.argv[3] if len(sys.argv) > 3 else "orthographic_with_footprints.png"

    print(f"Loading COLMAP data from: {colmap_path}")

    # Create visualizer
    viz = COLMAPVisualizer(colmap_path)

    try:
        # 1. Load COLMAP data
        print("Loading COLMAP data...")
        viz.read_cameras_txt()
        viz.read_images_txt()
        viz.read_points3d_txt()

        print(f"Loaded: {len(viz.cameras)} cameras, {len(viz.images)} images, {len(viz.points3d)} points")

        # 2. Create DTM
        print("Creating DTM...")
        viz.create_dtm(resolution=2.0)

        # 3. Create 3D visualization
        print("Creating 3D scene visualization...")
        scene_center = viz.visualize_3d_scene(save_path=output_3d)

        # 4. Create orthographic view with footprints
        print("Creating orthographic view with footprints...")
        viz.render_orthographic_view(scene_center, save_path=output_ortho)

        print(f"\n✓ Visualization completed successfully!")
        print(f"  - 3D Scene: {output_3d}")
        print(f"  - Orthographic: {output_ortho}")

        # Print footprint statistics
        print("\nFootprint Statistics:")
        print(f"  Total cameras: {len(viz.images)}")
        print(f"  Total points: {len(viz.points3d)}")
        print(f"  DTM resolution: 2.0 m")

        # Calculate coverage area (approximate)
        if hasattr(viz, 'dtm'):
            x_range = viz.dtm['x_grid'].max() - viz.dtm['x_grid'].min()
            y_range = viz.dtm['y_grid'].max() - viz.dtm['y_grid'].min()
            print(f"  DTM coverage: {x_range:.1f} x {y_range:.1f} m")
            print(f"  Total area: {x_range * y_range:.1f} m²")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Python 스크립트 실행
echo -e "\n${YELLOW}Running COLMAP visualizer...${NC}"
echo "----------------------------------------"

# Conda 환경 확인 및 활성화 (필요시)
if command -v conda &> /dev/null; then
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate Grendel 2>/dev/null || true
fi

# Python 실행
python3 $TEMP_SCRIPT "$COLMAP_PATH" "$OUTPUT_3D" "$OUTPUT_ORTHO"
RESULT=$?

# 임시 스크립트 삭제
rm -f $TEMP_SCRIPT

echo "----------------------------------------"

# 결과 확인
if [ $RESULT -eq 0 ]; then
    echo -e "\n${GREEN}✅ SUCCESS: Visualization completed${NC}"

    # 생성된 파일 확인
    echo -e "\n${YELLOW}Generated files:${NC}"
    if [ -f "$OUTPUT_3D" ]; then
        SIZE=$(ls -lh "$OUTPUT_3D" | awk '{print $5}')
        echo -e "  ${GREEN}✓${NC} $OUTPUT_3D ($SIZE)"
    else
        echo -e "  ${RED}✗${NC} $OUTPUT_3D not found"
    fi

    if [ -f "$OUTPUT_ORTHO" ]; then
        SIZE=$(ls -lh "$OUTPUT_ORTHO" | awk '{print $5}')
        echo -e "  ${GREEN}✓${NC} $OUTPUT_ORTHO ($SIZE)"
    else
        echo -e "  ${RED}✗${NC} $OUTPUT_ORTHO not found"
    fi
else
    echo -e "\n${RED}✗ FAILED: Visualization failed${NC}"
fi

echo -e "\n${BLUE}========================================"
echo "Completed at: $(date)"
echo -e "========================================${NC}"

# 로그 종료 메시지
echo "" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "Log ended at: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo -e "\n${YELLOW}Full log saved to: $LOG_FILE${NC}"

exit $RESULT