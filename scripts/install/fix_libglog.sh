#!/bin/bash

echo "=== Diagnosing libglog issue ==="
echo ""

# 1. Check current LD_PRELOAD
echo "1. Current LD_PRELOAD value:"
echo "   $LD_PRELOAD"
echo ""

# 2. Find libglog library location
echo "2. Searching for libglog libraries:"
GLOG_LIBS=$(find /usr -name "libglog*" 2>/dev/null)
if [ -z "$GLOG_LIBS" ]; then
    echo "   No libglog libraries found!"
else
    echo "$GLOG_LIBS" | while read -r lib; do
        echo "   Found: $lib"
    done
fi
echo ""

# 3. Check ldconfig cache
echo "3. Checking ldconfig cache for glog:"
ldconfig -p | grep glog || echo "   No glog entries in ldconfig cache"
echo ""

# 4. Check package installation status
echo "4. Checking installed glog packages:"
dpkg -l | grep glog || echo "   No glog packages found via dpkg"
echo ""

# 5. Check the specific path that's causing the error
echo "5. Checking /usr/lib/x86_64-linux-gnu/ for libglog:"
ls -la /usr/lib/x86_64-linux-gnu/libglog* 2>/dev/null || echo "   Not found in /usr/lib/x86_64-linux-gnu/"
echo ""

echo "=== Attempting fixes ==="
echo ""

# Fix 1: Unset LD_PRELOAD
echo "Fix 1: Unsetting LD_PRELOAD..."
unset LD_PRELOAD
export LD_PRELOAD=""
echo "   LD_PRELOAD cleared"
echo ""

# Fix 2: Create symlink if libglog exists elsewhere
echo "Fix 2: Creating symlink if needed..."
GLOG_SO=$(find /usr -name "libglog.so.0" 2>/dev/null | head -1)
if [ -n "$GLOG_SO" ]; then
    if [ ! -e "/usr/lib/x86_64-linux-gnu/libglog.so.0" ]; then
        echo "   Found libglog at: $GLOG_SO"
        echo "   Creating symlink to /usr/lib/x86_64-linux-gnu/libglog.so.0"
        ln -s "$GLOG_SO" /usr/lib/x86_64-linux-gnu/libglog.so.0 2>/dev/null && \
            echo "   Symlink created successfully" || \
            echo "   Failed to create symlink (may need sudo)"
    else
        echo "   libglog.so.0 already exists in /usr/lib/x86_64-linux-gnu/"
    fi
else
    echo "   libglog.so.0 not found anywhere"
    
    # Try to find any libglog.so version
    GLOG_ANY=$(find /usr -name "libglog.so*" 2>/dev/null | head -1)
    if [ -n "$GLOG_ANY" ]; then
        echo "   Found alternative: $GLOG_ANY"
        echo "   Creating symlink as libglog.so.0"
        ln -s "$GLOG_ANY" /usr/lib/x86_64-linux-gnu/libglog.so.0 2>/dev/null && \
            echo "   Symlink created successfully" || \
            echo "   Failed to create symlink (may need sudo)"
    fi
fi
echo ""

# Fix 3: Update library cache
echo "Fix 3: Updating library cache..."
ldconfig 2>/dev/null && echo "   Library cache updated" || echo "   Failed to update (may need sudo)"
echo ""

# Fix 4: Add to bashrc to prevent future issues
echo "Fix 4: Adding permanent fix to ~/.bashrc..."
if ! grep -q "unset LD_PRELOAD" ~/.bashrc; then
    echo "# Fix for libglog LD_PRELOAD issue" >> ~/.bashrc
    echo "unset LD_PRELOAD" >> ~/.bashrc
    echo "   Added 'unset LD_PRELOAD' to ~/.bashrc"
else
    echo "   Fix already in ~/.bashrc"
fi
echo ""

echo "=== Testing fix ==="
# Test if the error still occurs
echo "Testing colmap command..."
colmap -h > /dev/null 2>&1 && echo "✓ COLMAP works without errors!" || echo "✗ COLMAP still has issues"
echo ""

echo "=== Summary ==="
echo "1. LD_PRELOAD has been cleared for this session"
echo "2. Added permanent fix to ~/.bashrc"
echo "3. If libglog was found, symlinks were created"
echo ""
echo "Please run 'source ~/.bashrc' or restart your shell to apply permanent fixes."
echo ""
echo "If issues persist, you may need to:"
echo "  - Install libglog: apt-get install libgoogle-glog-dev"
echo "  - Or rebuild the Docker image with proper libglog installation"