#!/bin/bash

# COLMAP Visualization Example Shell Script
# 콘솔에 모든 출력 표시

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 시작 시간 기록
echo "========================================"
echo "COLMAP Visualization Started at: $(date)"
echo "========================================"

# Python 환경 확인
echo -e "${YELLOW}Checking Python environment...${NC}"
python3 --version 2>&1

# 필요한 패키지 확인
echo -e "\n${YELLOW}Checking required packages...${NC}"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')" 2>&1
python3 -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')" 2>&1
python3 -c "import scipy; print(f'SciPy version: {scipy.__version__}')" 2>&1

# COLMAP 데이터 경로 찾기
echo -e "\n${YELLOW}Searching for COLMAP data...${NC}"
:<<END
COLMAP_PATHS=(
    "/data/sillim_ew_mini_100024_20/sparse/0"
    "/data/samsung_dong_mini_30/sparse/0"
    "/media2/4tb/aerial_photo_data/Sillim-dong/colmap_output/sparse/0"
    "./output/sillim_ew_mini_100024_20/sparse/0"
    "./output/samsung_dong_mini_30/sparse/0"
)
END
COLMAP_PATHS=(
    "/data/sillim_ew_mini_100024_20/sparse/0"
)

FOUND_PATH=""
for path in "${COLMAP_PATHS[@]}"; do
    echo "DEBUG: Checking path: $path"
    echo "DEBUG:   cameras.txt: $([ -f "$path/cameras.txt" ] && echo "exists" || echo "missing")"
    echo "DEBUG:   images.txt: $([ -f "$path/images.txt" ] && echo "exists" || echo "missing")"
    echo "DEBUG:   points3D.txt: $([ -f "$path/points3D.txt" ] && echo "exists" || echo "missing")"
    
    if [ -f "$path/cameras.txt" ] && [ -f "$path/images.txt" ] && [ -f "$path/points3D.txt" ]; then
        FOUND_PATH=$path
        echo -e "${GREEN}Found COLMAP data at: $path${NC}"
        break
    else
        echo "DEBUG: Path $path incomplete"
    fi
done

if [ -z "$FOUND_PATH" ]; then
    echo -e "${RED}No COLMAP data found in predefined paths${NC}"
    echo "Please specify COLMAP path as argument: $0 <path_to_colmap_sparse>"
    
    # 첫 번째 인자로 경로 받기
    if [ -n "$1" ]; then
        FOUND_PATH=$1
        echo "Using provided path: $FOUND_PATH"
    else
        echo -e "${RED}Exiting...${NC}"
        exit 1
    fi
fi

# 기존 출력 파일 백업
echo -e "\n${YELLOW}Backing up existing output files...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -f "colmap_3d_scene.png" ]; then
    mv colmap_3d_scene.png "colmap_3d_scene_${TIMESTAMP}.png"
    echo "Backed up: colmap_3d_scene.png -> colmap_3d_scene_${TIMESTAMP}.png"
fi
if [ -f "orthographic_nadir_view.png" ]; then
    mv orthographic_nadir_view.png "orthographic_nadir_view_${TIMESTAMP}.png"
    echo "Backed up: orthographic_nadir_view.png -> orthographic_nadir_view_${TIMESTAMP}.png"
fi

# Python 스크립트 실행
echo -e "\n${YELLOW}Running COLMAP visualization...${NC}"
echo "----------------------------------------"

# 환경 변수 설정 (matplotlib 백엔드)
export MPLBACKEND=Agg

# example_colmap_viz.py 파일이 존재하는지 확인
if [ ! -f "example_colmap_viz.py" ]; then
    echo -e "${RED}Error: example_colmap_viz.py not found${NC}"
    echo "Please make sure example_colmap_viz.py is in the current directory"
    exit 1
fi

# Python 환경 디버그
echo "DEBUG: Python environment info:"
which python3
python3 --version 2>&1
echo "DEBUG: Testing matplotlib import:"
python3 -c "import matplotlib; print('matplotlib OK')" 2>&1

# Python 스크립트 실행 및 결과 저장
echo "DEBUG: About to run Python script..."
echo "DEBUG: Current working directory: $(pwd)"
echo "DEBUG: Files in current directory:"
ls -la example_colmap_viz.py colmap_visualizer.py 2>&1

if [ -n "$FOUND_PATH" ]; then
    # COLMAP 경로가 자동으로 찾아진 경우, 환경변수로 전달
    export COLMAP_PATH="$FOUND_PATH"
    echo "DEBUG: Running with COLMAP_PATH=$COLMAP_PATH"
    echo "DEBUG: Executing: python3 example_colmap_viz.py"
    
    # Conda 환경 활성화 후 실행
    source /opt/conda/etc/profile.d/conda.sh
    conda activate Grendel
    python3 example_colmap_viz.py 2>&1
    PYTHON_EXIT_CODE=$?
    echo "DEBUG: Python exit code: $PYTHON_EXIT_CODE"
else
    # 인자로 경로가 전달된 경우
    echo "DEBUG: Running with argument $1"
    echo "DEBUG: Executing: python3 example_colmap_viz.py $1"
    
    # Conda 환경 활성화 후 실행
    source /opt/conda/etc/profile.d/conda.sh
    conda activate Grendel
    python3 example_colmap_viz.py "$1" 2>&1
    PYTHON_EXIT_CODE=$?
    echo "DEBUG: Python exit code: $PYTHON_EXIT_CODE"
fi
echo "DEBUG: Python script execution finished"

# 실행 결과 확인
RESULT=$?
echo "----------------------------------------"

if [ $RESULT -eq 0 ]; then
    echo -e "\n${GREEN}✅ SUCCESS: Visualization completed${NC}"
    
    # 생성된 파일 확인
    echo -e "\n${YELLOW}Generated files:${NC}"
    if [ -f "colmap_3d_scene.png" ]; then
        SIZE=$(ls -lh colmap_3d_scene.png | awk '{print $5}')
        echo -e "  ${GREEN}✓${NC} colmap_3d_scene.png ($SIZE)"
    else
        echo -e "  ${RED}✗${NC} colmap_3d_scene.png not found"
    fi
    
    if [ -f "orthographic_nadir_view.png" ]; then
        SIZE=$(ls -lh orthographic_nadir_view.png | awk '{print $5}')
        echo -e "  ${GREEN}✓${NC} orthographic_nadir_view.png ($SIZE)"
    else
        echo -e "  ${RED}✗${NC} orthographic_nadir_view.png not found"
    fi
else
    echo -e "\n${RED}✗ FAILED: Visualization failed with error code $RESULT${NC}"
fi

# 종료 시간 기록
echo -e "\n========================================"
echo "COLMAP Visualization Ended at: $(date)"
echo "========================================"

exit $RESULT
