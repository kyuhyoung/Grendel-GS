#!/bin/bash

echo "=== Fixing OpenSfM glog issue ==="

# Create wrapper for OpenSfM to bypass glog error
cat > /opt/OpenSfM/bin/opensfm-wrapper << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/opt/OpenSfM')
os.environ['PYTHONPATH'] = '/opt/OpenSfM:' + os.environ.get('PYTHONPATH', '')

# Import and run OpenSfM main
from opensfm import main
main.main()
EOF

chmod +x /opt/OpenSfM/bin/opensfm-wrapper

# Backup original and replace with wrapper
if [ -f "/opt/OpenSfM/bin/opensfm" ]; then
    mv /opt/OpenSfM/bin/opensfm /opt/OpenSfM/bin/opensfm.original
fi
ln -sf /opt/OpenSfM/bin/opensfm-wrapper /opt/OpenSfM/bin/opensfm

# Also create in system path if needed
ln -sf /opt/OpenSfM/bin/opensfm-wrapper /usr/local/bin/opensfm 2>/dev/null || true

# Test the wrapper
echo "Testing OpenSfM wrapper..."
/opt/OpenSfM/bin/opensfm --help || echo "Wrapper may need adjustment"

echo ""
echo "OpenSfM wrapper created. ODM should now be able to call OpenSfM."
echo "Test with: python3 /opt/ODM/run.py --help"