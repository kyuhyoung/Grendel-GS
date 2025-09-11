#!/bin/bash

echo "=== Official ODM Installation ==="

# Activate Grendel environment
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel

# Clone ODM first
cd /opt
echo "Cloning ODM..."
if [ -d "ODM" ]; then
    rm -rf ODM
fi
git clone https://github.com/OpenDroneMap/ODM.git
cd /opt/ODM

echo "Running ODM installation..."

# ODM has its own installation script
if [ -f "portable.sh" ]; then
    echo "Running ODM portable installation..."
    bash portable.sh
elif [ -f "install.sh" ]; then
    echo "Running ODM install script..."
    bash install.sh
else
    echo "No ODM install script found, trying manual setup..."
    
    # Install Python dependencies from requirements.txt if it exists
    if [ -f "requirements.txt" ]; then
        echo "Installing from requirements.txt..."
        pip install -r requirements.txt
    fi
    
    # Try to install using setup.py if it exists
    if [ -f "setup.py" ]; then
        echo "Installing using setup.py..."
        pip install -e .
    fi
fi

# Set environment variables
export PATH=/opt/ODM:$PATH
echo 'export PATH="/opt/ODM:$PATH"' >> ~/.bashrc

# Create wrapper script
cat > /opt/ODM/odm << 'EOF'
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel
cd /opt/ODM
python3 run.py "$@"
EOF
chmod +x /opt/ODM/odm

# Test ODM
echo ""
echo "Testing ODM installation..."
python3 run.py --help | head -20

echo ""
echo "=== ODM Installation Complete ==="
echo "ODM installed at: /opt/ODM"
echo "Usage: /opt/ODM/odm --help"
echo "Or: python3 /opt/ODM/run.py --help"