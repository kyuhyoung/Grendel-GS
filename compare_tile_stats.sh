#!/bin/bash

# Compare tile distribution statistics between heuristic and uniform modes
# Usage: ./compare_tile_stats.sh <heuristic_log> <uniform_log>

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ $# -ne 2 ]; then
    echo -e "${RED}Usage: $0 <heuristic_log> <uniform_log>${NC}"
    echo "Example: $0 tile_distribution_stats_heuristic.log tile_distribution_stats_uniform.log"
    exit 1
fi

HEURISTIC_LOG="$1"
UNIFORM_LOG="$2"

# Check if files exist
if [ ! -f "$HEURISTIC_LOG" ]; then
    echo -e "${RED}Error: Heuristic log file does not exist: $HEURISTIC_LOG${NC}"
    exit 1
fi

if [ ! -f "$UNIFORM_LOG" ]; then
    echo -e "${RED}Error: Uniform log file does not exist: $UNIFORM_LOG${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run Python comparison script
python3 "$SCRIPT_DIR/compare_tile_stats.py" "$HEURISTIC_LOG" "$UNIFORM_LOG"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}Comparison failed!${NC}"
    exit 1
fi
