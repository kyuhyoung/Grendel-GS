#!/bin/bash
# Fix gsplat CUDA 12.4 compatibility issues

echo "Fixing gsplat CUDA 12.4 compatibility..."

# Method 1: Fix labeled_partition API calls
echo "Step 1: Patching gsplat source files..."

# Backup original files
cp submodules/gsplat/gsplat/cuda/csrc/fully_fused_projection_bwd.cu submodules/gsplat/gsplat/cuda/csrc/fully_fused_projection_bwd.cu.bak
cp submodules/gsplat/gsplat/cuda/csrc/fully_fused_projection_packed_bwd.cu submodules/gsplat/gsplat/cuda/csrc/fully_fused_projection_packed_bwd.cu.bak
cp submodules/gsplat/gsplat/cuda/csrc/world_to_cam_bwd.cu submodules/gsplat/gsplat/cuda/csrc/world_to_cam_bwd.cu.bak

# Replace labeled_partition with compatible code
sed -i 's/auto warp_group_g = cg::labeled_partition(warp, gid);/auto warp_group_g = cg::coalesced_threads();/g' \
    submodules/gsplat/gsplat/cuda/csrc/fully_fused_projection_bwd.cu

sed -i 's/auto warp_group_c = cg::labeled_partition(warp, cid);/auto warp_group_c = cg::coalesced_threads();/g' \
    submodules/gsplat/gsplat/cuda/csrc/fully_fused_projection_bwd.cu

sed -i 's/auto warp_group_g = cg::labeled_partition(warp, gid);/auto warp_group_g = cg::coalesced_threads();/g' \
    submodules/gsplat/gsplat/cuda/csrc/fully_fused_projection_packed_bwd.cu

sed -i 's/auto warp_group_c = cg::labeled_partition(warp, cid);/auto warp_group_c = cg::coalesced_threads();/g' \
    submodules/gsplat/gsplat/cuda/csrc/fully_fused_projection_packed_bwd.cu

sed -i 's/auto warp_group_g = cg::labeled_partition(warp, gid);/auto warp_group_g = cg::coalesced_threads();/g' \
    submodules/gsplat/gsplat/cuda/csrc/world_to_cam_bwd.cu

sed -i 's/auto warp_group_c = cg::labeled_partition(warp, cid);/auto warp_group_c = cg::coalesced_threads();/g' \
    submodules/gsplat/gsplat/cuda/csrc/world_to_cam_bwd.cu

echo "Step 2: Installing patched gsplat..."

# Clean previous build attempts
pip uninstall gsplat -y
rm -rf submodules/gsplat/build/
rm -rf submodules/gsplat/*.egg-info/

# Install with reduced parallelism to avoid memory issues
MAX_JOBS=4 pip install submodules/gsplat

echo "Step 3: Verifying installation..."
python -c "import gsplat; print('gsplat successfully imported:', gsplat.__version__)"

if [ $? -eq 0 ]; then
    echo "✓ gsplat fixed and installed successfully!"
else
    echo "✗ gsplat installation failed. Trying alternative method..."
    
    echo "Step 4: Alternative - Install pre-built gsplat..."
    pip install gsplat==1.0.0 --no-deps
    
    if [ $? -eq 0 ]; then
        echo "✓ Pre-built gsplat installed!"
    else
        echo "✗ Both methods failed. Manual intervention required."
        exit 1
    fi
fi

echo "Installation complete!"