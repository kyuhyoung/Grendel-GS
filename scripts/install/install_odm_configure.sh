#!/bin/bash

echo "=== ODM Installation using configure.sh ==="

# Fix held packages issue
export DEBIAN_FRONTEND=noninteractive
echo "Fixing held packages and broken sudo..."
apt-mark unhold $(apt-mark showhold 2>/dev/null) 2>/dev/null || true

# Remove broken sudo package completely
dpkg --remove --force-remove-reinstreq sudo sudo-ldap libnss-sudo 2>/dev/null || true
dpkg --purge --force-remove-reinstreq sudo sudo-ldap libnss-sudo 2>/dev/null || true

# Block all sudo-related packages
echo "Blocking all sudo-related packages..."
mkdir -p /etc/apt/preferences.d/
cat > /etc/apt/preferences.d/block-sudo << 'EOF'
Package: sudo
Pin: release *
Pin-Priority: -1

Package: sudo-ldap
Pin: release *
Pin-Priority: -1

Package: libnss-sudo
Pin: release *
Pin-Priority: -1
EOF

apt-get update --allow-change-held-packages

# Activate Grendel environment
source /opt/conda/etc/profile.d/conda.sh
conda activate Grendel

# Clone ODM
cd /opt
echo "Cloning ODM..."
if [ -d "ODM" ]; then
    rm -rf ODM
fi
git clone https://github.com/OpenDroneMap/ODM.git
cd /opt/ODM

echo "Running ODM configure.sh install..."
bash configure.sh install

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