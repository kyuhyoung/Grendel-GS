#!/bin/bash

echo "=== ODM Python Installation ==="

# Clean up previous installations
echo "Cleaning up previous installations..."
rm -rf /opt/ODM 2>/dev/null || true
rm -rf /opt/OpenSfM 2>/dev/null || true
rm -f /opt/ODM/odm 2>/dev/null || true
rm -f /opt/ODM/odm-direct 2>/dev/null || true

# Clean up system-wide installations
rm -f /usr/local/bin/odm* 2>/dev/null || true
rm -f /usr/local/bin/opensfm* 2>/dev/null || true
rm -f /opt/conda/envs/Grendel/bin/odm* 2>/dev/null || true
rm -f /opt/conda/envs/Grendel/bin/opensfm* 2>/dev/null || true
rm -rf /opt/conda/envs/Grendel/lib/python*/site-packages/opensfm* 2>/dev/null || true
rm -rf /opt/conda/envs/Grendel/lib/python*/site-packages/odm* 2>/dev/null || true

# Uninstall pip packages
pip uninstall opensfm odm opendm -y 2>/dev/null || true

# Clean up any OpenSfM build artifacts
rm -rf /opt/OpenSfM/build 2>/dev/null || true
rm -rf /opt/ODM/build 2>/dev/null || true
rm -rf /opt/ODM/SuperBuild/build 2>/dev/null || true

# Activate Grendel environment
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel

# Clone fresh ODM
cd /opt
echo "Cloning fresh ODM..."
git clone https://github.com/OpenDroneMap/ODM.git
cd /opt/ODM

# Install Python dependencies from requirements.txt
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    # Skip problematic packages and use compatible versions
    pip install -r requirements.txt --upgrade --no-deps --force-reinstall || echo "Some packages failed, continuing..."
fi

# Install additional common dependencies
pip install opencv-python pillow numpy scipy matplotlib
pip install requests psutil lxml beautifulsoup4 xmltodict
pip install pyproj gdal rasterio

# Install system OpenCV for CMake support
echo "Installing system OpenCV for OpenSfM..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libopencv-dev libopencv-contrib-dev || echo "System OpenCV install failed, trying conda..."
conda install -c conda-forge opencv -y || echo "Conda OpenCV install failed, continuing..."

# Check if OpenSfM is included as submodule
echo "Checking for OpenSfM..."
OPENSFM_FOUND=0
if [ -d "SuperBuild/src/opensfm" ]; then
    echo "Found OpenSfM in SuperBuild, building..."
    cd SuperBuild/src/opensfm
    export PYTHONPATH=/opt/ODM/SuperBuild/src/opensfm:$PYTHONPATH
    OPENSFM_FOUND=1
elif [ -d "contrib/opensfm" ]; then
    echo "Found OpenSfM in contrib, building..."
    cd contrib/opensfm  
    export PYTHONPATH=/opt/ODM/contrib/opensfm:$PYTHONPATH
    OPENSFM_FOUND=1
else
    echo "OpenSfM not found in ODM, cloning separately..."
    cd /opt
    rm -rf OpenSfM
    git clone --recursive https://github.com/mapillary/OpenSfM.git
    cd OpenSfM
    export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
fi

# Build OpenSfM with multiple methods
echo "Building OpenSfM..."
export OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
export CMAKE_PREFIX_PATH="/usr/lib/x86_64-linux-gnu/cmake:/opt/conda/envs/Grendel:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install OpenSfM specific dependencies
pip install numpy scipy opencv-python opencv-contrib-python
pip install PyYAML exifread gpxpy matplotlib networkx
pip install xmltodict cloudpickle repoze.lru psutil

# Try different build methods
echo "Attempting OpenSfM build method 1: setup.py build_ext..."
python setup.py build_ext --inplace && echo "OpenSfM build successful!" || {
    echo "Method 1 failed, trying method 2: pip install -e..."
    pip install -e . --no-build-isolation && echo "OpenSfM pip install successful!" || {
        echo "Method 2 failed, trying method 3: cmake build..."
        mkdir -p build && cd build
        cmake .. -DOPENSFM_BUILD_TESTS=OFF -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 && \
        make -j$(nproc) && echo "OpenSfM cmake build successful!" || {
            echo "All build methods failed, OpenSfM may not work properly"
        }
        cd ..
    }
}

# Fix GLIBCXX version conflict and glog issues
echo "Fixing GLIBCXX version conflict and glog issues..."
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
conda install -c conda-forge libstdcxx-ng -y || echo "libstdcxx-ng install failed, continuing..."

# Fix glog flag duplication issue
echo "Fixing glog flag duplication..."
conda install -c conda-forge glog -y || echo "glog install failed, continuing..."
export GLOG_alsologtostderr=1
export GLOG_colorlogtostderr=1

# Set up environment
cd /opt/ODM
export PYTHONPATH=/opt/ODM:/opt/OpenSfM:$PYTHONPATH
export PATH=/opt/ODM:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Add to bashrc
echo 'export PYTHONPATH="/opt/ODM:/opt/OpenSfM:$PYTHONPATH"' >> ~/.bashrc
echo 'export PATH="/opt/ODM:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"' >> ~/.bashrc
echo 'export GLOG_alsologtostderr=1' >> ~/.bashrc
echo 'export GLOG_colorlogtostderr=1' >> ~/.bashrc

# Create wrapper script
cat > /opt/ODM/odm << 'EOF'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel
export PYTHONPATH="/opt/ODM:/opt/OpenSfM:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export GLOG_alsologtostderr=1
export GLOG_colorlogtostderr=1
cd /opt/ODM
python3 run.py "$@"
EOF
chmod +x /opt/ODM/odm

# Test OpenSfM import first
echo ""
echo "Testing OpenSfM import..."
python3 -c "
import sys
sys.path.insert(0, '/opt/OpenSfM')
sys.path.insert(0, '/opt/ODM')
try:
    import opensfm
    print('✓ OpenSfM import successful')
    from opensfm.sensors import sensor_data
    print('✓ OpenSfM sensors import successful')
except Exception as e:
    print(f'✗ OpenSfM import failed: {e}')
"

# Test ODM
echo ""
echo "Testing ODM installation..."
echo "Note: If wrapper script fails with glog error, use 'python3 /opt/ODM/run.py' directly"
export PYTHONPATH=/opt/OpenSfM:/opt/ODM:$PYTHONPATH
python3 run.py --help | head -20 || echo "Direct test failed, may need PYTHONPATH export"

# Create alternative direct runner without glog issues
cat > /opt/ODM/odm-direct << 'EOF'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel
export PYTHONPATH="/opt/ODM:/opt/OpenSfM:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
cd /opt/ODM
exec python3 run.py "$@"
EOF
chmod +x /opt/ODM/odm-direct

echo ""
echo "=== ODM Installation Complete ==="
echo "ODM installed at: /opt/ODM"  
echo ""
echo "Usage options:"
echo "  1. Direct Python (recommended): python3 /opt/ODM/run.py --help"
echo "  2. Alternative wrapper: /opt/ODM/odm-direct --help"
echo "  3. Standard wrapper: /opt/ODM/odm --help (may have glog error)"
echo ""
echo "If glog error occurs, use option 1 or 2"