#!/bin/bash

# Run usage_progressive.sh and filter output to show intersection filtering results

echo "=========================================="
echo "Running usage_progressive.sh"
echo "Filtering output to show intersection filtering"
echo "=========================================="
echo ""

./usage_progressive.sh 2>&1 | grep -E "Intersection threshold|Camera [0-9]+: intersection ratio|Cameras intersecting|Selected camera E|Step 9.1|Step 16.4|Initial window selected|FIFO" --color=always

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
