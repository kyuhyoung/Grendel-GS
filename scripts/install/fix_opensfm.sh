#!/bin/bash

echo "=== Fixing OpenSfM Installation ==="

source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel

# Clean and reinstall OpenSfM
cd /opt
rm -rf OpenSfM

echo "Cloning OpenSfM..."
git clone --recursive https://github.com/mapillary/OpenSfM.git
cd OpenSfM

echo "Installing OpenSfM Python dependencies..."
pip install numpy scipy 
pip install opencv-python opencv-contrib-python
pip install PyYAML exifread gpxpy
pip install matplotlib networkx
pip install xmltodict cloudpickle repoze.lru psutil

echo "Setting up build environment..."
export OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
export CMAKE_PREFIX_PATH="/usr/lib/x86_64-linux-gnu/cmake:/opt/conda/envs/Grendel:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "Building OpenSfM..."
# Try different build methods
python setup.py build_ext --inplace || {
    echo "Standard build failed, trying pip install..."
    pip install -e . --no-build-isolation || {
        echo "pip install failed, trying minimal build..."
        mkdir -p build && cd build
        cmake .. -DOPENSFM_BUILD_TESTS=OFF -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
        make -j$(nproc)
        cd ..
    }
}

# Add to Python path
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
echo 'export PYTHONPATH="/opt/OpenSfM:$PYTHONPATH"' >> ~/.bashrc

# Test OpenSfM import
echo "Testing OpenSfM import..."
python3 -c "
import sys
sys.path.insert(0, '/opt/OpenSfM')
try:
    import opensfm
    print('✓ OpenSfM import successful')
    from opensfm.sensors import sensor_data
    print('✓ OpenSfM sensors import successful')
except Exception as e:
    print(f'✗ OpenSfM import failed: {e}')
"

# Test ODM with fixed path
echo ""
echo "Testing ODM with OpenSfM..."
export PYTHONPATH=/opt/OpenSfM:/opt/ODM:$PYTHONPATH
cd /opt/ODM
python3 -c "
import sys
sys.path.insert(0, '/opt/OpenSfM')
sys.path.insert(0, '/opt/ODM')
try:
    from opensfm.sensors import sensor_data
    print('✓ OpenSfM sensors available for ODM')
    from opendm.utils import get_processing_results_paths
    print('✓ ODM utils import successful')
except Exception as e:
    print(f'✗ Import failed: {e}')
"

echo ""
echo "If successful, run ODM with:"
echo "  export PYTHONPATH=/opt/OpenSfM:/opt/ODM:\$PYTHONPATH"
echo "  python3 /opt/ODM/run.py --help"