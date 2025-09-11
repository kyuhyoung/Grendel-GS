#!/bin/bash

echo "===== ODM Installation in Container ====="
echo "Run this command inside the container:"
echo ""
echo "cd /workspace/Grendel-GS && ./install_odm_container.sh"
echo ""
echo "Or copy and paste these commands directly:"
echo ""

cat << 'EOF'
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel
export DEBIAN_FRONTEND=noninteractive

# Install system dependencies
apt-get update && apt-get install -y \
    build-essential cmake git python3-pip curl wget unzip pkg-config \
    libeigen3-dev libboost-filesystem-dev libboost-iostreams-dev \
    libboost-regex-dev libboost-python-dev libboost-date-time-dev \
    libboost-thread-dev libopencv-dev libproj-dev libxerces-c-dev \
    libgdal-dev gdal-bin python3-gdal grass-dev libgeotiff-dev \
    libjsoncpp-dev python3-setuptools python3-dev python3-numpy \
    libimage-exiftool-perl ninja-build libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev python3-scipy python3-pyproj \
    python3-yaml python3-matplotlib

# Clone OpenSfM
cd /opt && rm -rf OpenSfM
git clone --recursive https://github.com/mapillary/OpenSfM.git && cd OpenSfM

# Install Python dependencies
pip install --upgrade pip
pip install --force-reinstall --no-deps opencv-python==4.8.1.78
pip install PyYAML==6.0.1 exifread==3.0.0 gpxpy==1.5.0
pip install "pyproj>=3.4.0" "matplotlib>=3.6.0" "networkx>=2.8"
pip install "pytest>=7.0" "requests>=2.28" xmltodict==0.12.0
pip install "cloudpickle>=2.2" repoze.lru==0.7 "psutil>=5.9"

# Build OpenSfM
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
python setup.py build

# Clone ODM
cd /opt && rm -rf ODM
git clone https://github.com/OpenDroneMap/ODM.git && cd ODM

# Install ODM dependencies
pip install "fpdf2>=2.7.0" "joblib>=1.2.0" "lxml>=4.9.0" "beautifulsoup4>=4.11.0" appsettings==0.2.5

# Set environment variables
echo 'export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH' >> ~/.bashrc
echo 'export PATH=/opt/ODM:$PATH' >> ~/.bashrc
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
export PATH=/opt/ODM:$PATH

# Create wrapper script
cat > /opt/ODM/odm << 'EOFODM'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
cd /opt/ODM
python3 run.py "$@"
EOFODM
chmod +x /opt/ODM/odm

# Test installation
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

echo "ODM installation completed!"
EOF