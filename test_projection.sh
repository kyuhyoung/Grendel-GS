#!/bin/bash
set -e

# Test generalized projection implementation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run test script
python -u "${SCRIPT_DIR}/test_generalized_projection.py"