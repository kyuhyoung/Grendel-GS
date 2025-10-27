#!/bin/bash

# Aggregate tile distribution statistics from JSON files
# Usage: ./aggregate_tile_stats.sh [output_directory]

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default output directory
OUTPUT_DIR="${1:-./output/progressive_test}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Tile Distribution Stats Aggregation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}Error: Directory does not exist: $OUTPUT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Output directory: $OUTPUT_DIR${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run Python aggregation script
python3 "$SCRIPT_DIR/aggregate_tile_stats.py" "$OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Aggregation completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Aggregation failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
