#!/bin/bash

echo "=== ODM Installation without sudo ==="

# Activate Grendel environment
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel

# Explicitly exclude sudo from all installations
export DEBIAN_FRONTEND=noninteractive
echo "Updating package lists..."
apt-get update

echo "Installing basic packages one by one..."
apt-get install -y --no-install-recommends build-essential || true
apt-get install -y --no-install-recommends cmake || true  
apt-get install -y --no-install-recommends git || true
apt-get install -y --no-install-recommends python3-pip || true
apt-get install -y --no-install-recommends curl || true
apt-get install -y --no-install-recommends wget || true
apt-get install -y --no-install-recommends unzip || true
apt-get install -y --no-install-recommends pkg-config || true
apt-get install -y --no-install-recommends libeigen3-dev || true
apt-get install -y --no-install-recommends libboost-filesystem-dev || true
apt-get install -y --no-install-recommends libopencv-dev || true
apt-get install -y --no-install-recommends libgdal-dev || true
apt-get install -y --no-install-recommends gdal-bin || true
apt-get install -y --no-install-recommends python3-gdal || true
apt-get install -y --no-install-recommends libgeotiff-dev || true
apt-get install -y --no-install-recommends ninja-build || true

echo "Basic packages installed, continuing with Python setup..."

# Clone OpenSfM
cd /opt
echo "Cloning OpenSfM..."
if [ -d "OpenSfM" ]; then
    rm -rf OpenSfM
fi
git clone --recursive https://github.com/mapillary/OpenSfM.git
cd OpenSfM

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install --force-reinstall --no-deps opencv-python==4.8.1.78
pip install PyYAML exifread gpxpy pyproj matplotlib networkx pytest requests xmltodict cloudpickle repoze.lru psutil

# Build OpenSfM
echo "Building OpenSfM..."
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
python setup.py build

# Clone ODM
cd /opt
echo "Cloning ODM..."
if [ -d "ODM" ]; then
    rm -rf ODM
fi
git clone https://github.com/OpenDroneMap/ODM.git
cd ODM

# Install ODM dependencies
echo "Installing ODM dependencies..."
pip install fpdf2 joblib lxml beautifulsoup4 appsettings

# Set environment
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
export PATH=/opt/ODM:$PATH
echo 'export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH' >> ~/.bashrc
echo 'export PATH=/opt/ODM:$PATH' >> ~/.bashrc

# Create wrapper
cat > /opt/ODM/odm << 'EOF'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
cd /opt/ODM
python3 run.py "$@"
EOF
chmod +x /opt/ODM/odm

# Test
echo "Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '/opt/OpenSfM')
try:
    import opensfm
    print('✓ OpenSfM import successful')
except Exception as e:
    print(f'✗ OpenSfM import failed: {e}')

try:
    from opendm import context
    print('✓ ODM context import successful')
except Exception as e:
    print(f'✗ ODM context import failed: {e}')
"

echo "Installation completed!"