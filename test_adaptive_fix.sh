#!/bin/bash
# Quick test to verify the projection fix works

set -e

echo "=========================================="
echo "Testing adaptive tile projection fix"
echo "=========================================="

# Create log file with timestamp
LOG_FILE="test_fix_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"

# Remove existing log file if owned by root
if [ -f "run_adaptive.log" ] && [ ! -w "run_adaptive.log" ]; then
    echo "Removing problematic log file..."
    sudo rm -f run_adaptive.log || true
fi

# Quick test with only 100 iterations
./run_adaptive.sh \
    --source /data/dabeeo/samsung_dong_mini_30 \
    --output ./output/test_projection_fix \
    --gpu-ids "5,6,7" \
    --iterations 10 \
    --debug-save-iters "1,10,50,100" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Check for issues:"
echo "=========================================="

# Check if "image is scalar" errors appear
echo "Checking for 'image is scalar' errors..."
if grep -q "image is scalar" "$LOG_FILE"; then
    echo "❌ STILL BROKEN: 'image is scalar' errors found"
    echo "Count: $(grep -c "image is scalar" "$LOG_FILE")"
else
    echo "✅ GOOD: No 'image is scalar' errors"
fi

# Check if Loss is decreasing
echo ""
echo "Checking if Loss is decreasing..."
grep "Loss=" "$LOG_FILE" | tail -10

echo ""
echo "=========================================="
echo "Test complete. Check $LOG_FILE for details"
echo "=========================================="