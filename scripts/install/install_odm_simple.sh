#!/bin/bash

echo "=== Simple ODM Installation ==="

# Activate Grendel environment
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel

# Skip system packages for now, try with existing dependencies
echo "Skipping problematic system packages..."

# Clone OpenSfM
cd /opt
echo "Cloning OpenSfM..."
if [ -d "OpenSfM" ]; then
    rm -rf OpenSfM
fi
git clone --recursive https://github.com/mapillary/OpenSfM.git
cd OpenSfM

# Install minimal Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install --force-reinstall --no-deps opencv-python==4.8.1.78
pip install PyYAML exifread gpxpy pyproj matplotlib networkx pytest requests xmltodict cloudpickle repoze.lru psutil

# Try to build OpenSfM
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

# Install ODM Python dependencies
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