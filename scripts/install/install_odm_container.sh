#!/bin/bash

echo "=== Installing ODM in existing container ===" 
echo "Starting at $(date)"

# Check if we're in a container and Conda environment is available
if [ ! -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    echo "Error: This script must be run inside the Grendel-GS container"
    exit 1
fi

# Activate Grendel environment
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel

# Install ODM system dependencies
export DEBIAN_FRONTEND=noninteractive
echo "Installing system dependencies..."
apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    build-essential \
    cmake \
    git \
    python3-pip \
    curl \
    wget \
    unzip \
    pkg-config \
    libeigen3-dev \
    libboost-filesystem-dev \
    libboost-iostreams-dev \
    libboost-regex-dev \
    libboost-python-dev \
    libboost-date-time-dev \
    libboost-thread-dev \
    libopencv-dev \
    libproj-dev \
    libxerces-c-dev \
    libgdal-dev \
    gdal-bin \
    python3-gdal \
    grass-dev \
    libgeotiff-dev \
    libjsoncpp-dev \
    python3-setuptools \
    python3-dev \
    python3-numpy \
    libimage-exiftool-perl \
    ninja-build

# Install OpenSfM dependencies  
apt-get install -y --no-install-recommends --no-install-suggests \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    python3-scipy \
    python3-pyproj \
    python3-yaml \
    python3-matplotlib

echo "System dependencies installed successfully"

# Clone and build OpenSfM
cd /opt
echo "Cloning OpenSfM..."
if [ -d "OpenSfM" ]; then
    rm -rf OpenSfM
fi
git clone --recursive https://github.com/mapillary/OpenSfM.git
cd OpenSfM

# Install compatible Python dependencies
echo "Installing OpenSfM Python dependencies..."
pip install --upgrade pip

# Use --force-reinstall --no-deps to avoid conflicts
pip install --force-reinstall --no-deps opencv-python==4.8.1.78
pip install PyYAML==6.0.1
pip install exifread==3.0.0
pip install gpxpy==1.5.0
pip install pyproj>=3.4.0
pip install matplotlib>=3.6.0
pip install networkx>=2.8
pip install pytest>=7.0
pip install requests>=2.28
pip install xmltodict==0.12.0
pip install cloudpickle>=2.2
pip install repoze.lru==0.7
pip install psutil>=5.9

# Build OpenSfM
echo "Building OpenSfM..."
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
python setup.py build

echo "OpenSfM build completed"

# Clone and setup ODM
cd /opt
echo "Cloning ODM..."
if [ -d "ODM" ]; then
    rm -rf ODM
fi
git clone https://github.com/OpenDroneMap/ODM.git
cd ODM

# Install ODM Python requirements with flexible versions
echo "Installing ODM Python dependencies..."
pip install fpdf2>=2.7.0
pip install joblib>=1.2.0
pip install lxml>=4.9.0
pip install beautifulsoup4>=4.11.0
pip install appsettings==0.2.5

# Add environment variables
echo "export PYTHONPATH=/opt/OpenSfM:\$PYTHONPATH" >> ~/.bashrc
echo "export PATH=/opt/ODM:\$PATH" >> ~/.bashrc
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
export PATH=/opt/ODM:$PATH

# Create ODM wrapper script
cat > /opt/ODM/odm << 'EOF'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH
cd /opt/ODM
python3 run.py "$@"
EOF
chmod +x /opt/ODM/odm

# Test installation
echo ""
echo "=== Testing ODM Installation ==="
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

echo ""
echo "Testing ODM run.py..."
python3 run.py --help | head -20

echo ""
echo "=== Installation Complete ==="
echo "ODM installed at: /opt/ODM"
echo "OpenSfM installed at: /opt/OpenSfM"
echo "Usage: /opt/ODM/odm --help"
echo ""
echo "To run ODM: /opt/ODM/odm --project-path /data dataset_name"