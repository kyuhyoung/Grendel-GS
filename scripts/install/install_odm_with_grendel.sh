#!/bin/bash

echo "=== Installing ODM with Grendel-GS ==="
echo "Starting at $(date)"

# Activate Grendel environment
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel

# Install ODM system dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y \
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
apt-get install -y \
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
git clone --recursive https://github.com/mapillary/OpenSfM.git
cd OpenSfM

# Create conda environment for OpenSfM Python packages
echo "Installing OpenSfM Python dependencies in Grendel environment..."
# Install compatible versions that work with existing environment
pip install --no-deps opencv-python==4.8.1.78  # Compatible with existing opencv-contrib-python
pip install PyYAML==6.0.1
pip install exifread==3.0.0
pip install gpxpy==1.5.0
pip install pyproj==3.6.1  # Compatible with existing version
pip install matplotlib==3.7.4  # Use newer compatible version
pip install networkx==3.1
pip install pytest==7.4.3
pip install requests==2.31.0
pip install xmltodict==0.12.0
pip install cloudpickle==2.2.1
pip install repoze.lru==0.7
pip install psutil==5.9.6

# Build OpenSfM
echo "Building OpenSfM..."
python setup.py build

# Add OpenSfM to Python path
echo "export PYTHONPATH=/opt/OpenSfM:\$PYTHONPATH" >> ~/.bashrc
export PYTHONPATH=/opt/OpenSfM:$PYTHONPATH

# Clone and setup ODM
cd /opt
echo "Cloning ODM..."
if [ -d "ODM" ]; then
    rm -rf ODM
fi
git clone https://github.com/OpenDroneMap/ODM.git
cd ODM

# Install additional ODM Python requirements 
pip install fpdf2==2.7.9
pip install joblib==1.3.2  # Compatible with existing scikit-learn
pip install lxml==4.9.3
pip install beautifulsoup4==4.12.2
pip install appsettings==0.2.5
# xmltodict already installed above

# Create ODM configuration
echo "Configuring ODM..."
cat > settings.yaml << EOF
# ODM Settings
rerun-all: false
rerun-from: auto
end-with: auto
max-concurrency: 4
feature-type: sift
feature-quality: high
matcher-type: flann
min-num-features: 10000
EOF

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

# Add to PATH
echo 'export PATH="/opt/ODM:$PATH"' >> ~/.bashrc
echo 'export PYTHONPATH="/opt/OpenSfM:$PYTHONPATH"' >> ~/.bashrc
export PATH="/opt/ODM:$PATH"

# Test installation
echo ""
echo "=== Testing ODM Installation ==="
cd /opt/ODM
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
python3 run.py --help | head -20

echo ""
echo "=== Installation Complete ==="
echo "ODM installed at: /opt/ODM"
echo "OpenSfM installed at: /opt/OpenSfM"
echo "Usage: odm --help"
echo ""
echo "To run ODM: odm --project-path /data dataset_name"