#!/bin/bash
# Test visual debug only mode

echo "Testing --visual-debug-only mode..."
echo "This should generate debug images and exit immediately."
echo ""

# Create a simple test with minimal data requirement
./run_adaptive.sh \
    --source ./output/test_projection_fix \
    --output ./output/visual_debug_test \
    --gpu-ids "7" \
    --iterations 10 \
    --visual-debug-only

echo ""
echo "Checking output..."
if [ -d "./output/visual_debug_test/projection_debug" ]; then
    echo "✅ Debug images directory created"
    echo "Files generated:"
    ls -la ./output/visual_debug_test/projection_debug/*/  2>/dev/null | head -10
else
    echo "❌ No debug images directory found"
fi