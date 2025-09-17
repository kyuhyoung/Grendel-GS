#!/bin/bash
# Progressive Training Script for Grendel-GS
# DTM-based incremental camera addition and Gaussian management

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
show_usage() {
    echo "Progressive Training Script for Grendel-GS"
    echo "=========================================="
    echo ""
    echo "Usage:"
    echo "  $0 -s SOURCE_PATH -o OUTPUT_PATH [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  -s, --source        Path to COLMAP reconstruction (containing sparse/)"
    echo "  -o, --output        Output directory for progressive training results"
    echo ""
    echo "Optional Arguments:"
    echo "  -m, --initial       Number of initial cameras (default: 4)"
    echo "  -t, --threshold     GPU memory threshold 0-1 (default: 0.9)"
    echo "  -d, --debug         Enable debug output"
    echo "  --dtm-module        Path to external DTM module (optional)"
    echo "  --iterations        Number of training iterations (default: 30000)"
    echo "  --sh-degree         Spherical harmonics degree (default: 3)"
    echo "  --resolution        Resolution downscaling factor (default: 1)"
    echo "  --backend           Rendering backend: default or gsplat (default: gsplat)"
    echo "  --deterministic     Enable deterministic training"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -s /data/colmap_scene -o ./output/progressive_results"
    echo "  $0 -s /data/colmap_scene -o ./output/test -m 6 -t 0.85 --debug"
    echo "  $0 -s /data/colmap_scene -o ./output/test --dtm-module /path/to/dtm"
    echo ""
}

# Default values
SOURCE_PATH=""
OUTPUT_PATH=""
INITIAL_CAMERAS=4
GPU_THRESHOLD=0.9
DEBUG=false
DTM_MODULE=""
ITERATIONS=30000
SH_DEGREE=3
RESOLUTION=1
BACKEND="gsplat"
DETERMINISTIC=""
EXTRA_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source)
            SOURCE_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -m|--initial)
            INITIAL_CAMERAS="$2"
            shift 2
            ;;
        -t|--threshold)
            GPU_THRESHOLD="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG=true
            shift
            ;;
        --only-actually-visible)
            ONLY_ACTUALLY_VISIBLE=true
            shift
            ;;
        --dtm-module)
            DTM_MODULE="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --sh-degree)
            SH_DEGREE="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --deterministic)
            DETERMINISTIC="--deterministic"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SOURCE_PATH" ]]; then
    print_colored $RED "❌ Error: Source path (-s) is required"
    show_usage
    exit 1
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    print_colored $RED "❌ Error: Output path (-o) is required"
    show_usage
    exit 1
fi

# Validate source path
if [[ ! -d "$SOURCE_PATH" ]]; then
    print_colored $RED "❌ Error: Source path does not exist: $SOURCE_PATH"
    exit 1
fi

if [[ ! -d "$SOURCE_PATH/sparse" ]] && [[ ! -f "$SOURCE_PATH/cameras.txt" ]]; then
    print_colored $RED "❌ Error: Source path must contain COLMAP data (sparse/ directory or cameras.txt)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"
if [[ $? -ne 0 ]]; then
    print_colored $RED "❌ Error: Cannot create output directory: $OUTPUT_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Set up logging
LOG_FILE="$OUTPUT_PATH/progressive_train.log"
rm -f "$LOG_FILE"  # Clear log file
exec > >(tee -a "$LOG_FILE") 2>&1

print_colored $BLUE "======================================"
print_colored $BLUE "Progressive Training for Grendel-GS"
print_colored $BLUE "======================================"

print_colored $GREEN "Configuration:"
echo "  Source Path: $SOURCE_PATH"
echo "  Output Path: $OUTPUT_PATH"
echo "  Initial Cameras: $INITIAL_CAMERAS"
echo "  GPU Threshold: $GPU_THRESHOLD"
echo "  Training Iterations: $ITERATIONS"
echo "  SH Degree: $SH_DEGREE"
echo "  Backend: $BACKEND"
echo "  Debug Mode: $DEBUG"
if [[ -n "$DTM_MODULE" ]]; then
    echo "  DTM Module: $DTM_MODULE"
fi
if [[ -n "$DETERMINISTIC" ]]; then
    echo "  Deterministic: Enabled"
fi
echo ""

# Check dependencies
print_colored $YELLOW "Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        print_colored $RED "❌ Python not found. Please install Python 3."
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Check if we're in the right directory
if [[ ! -f "train.py" ]]; then
    print_colored $RED "❌ train.py not found. Please run this script from the Grendel-GS root directory."
    exit 1
fi

# Check if progressive_learning module exists
if [[ ! -f "progressive_learning/progressive_trainer.py" ]]; then
    print_colored $RED "❌ progressive_trainer.py not found. Please ensure progressive_learning module is installed."
    exit 1
fi

# Check GPU availability
if $PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_COUNT=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    print_colored $GREEN "✓ CUDA available with $GPU_COUNT GPU(s)"
else
    print_colored $YELLOW "⚠️  CUDA not available. Training will use CPU (very slow)."
fi

print_colored $GREEN "✓ All dependencies check passed"
echo ""

# Set deterministic environment variables if requested
if [[ -n "$DETERMINISTIC" ]]; then
    print_colored $YELLOW "Setting up deterministic environment..."
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    export CUDA_LAUNCH_BLOCKING=1
    export PYTHONHASHSEED=0
fi

# Run progressive training
print_colored $BLUE "Starting Progressive Training..."
echo ""

# Build Python command
PYTHON_ARGS=""
PYTHON_ARGS="$PYTHON_ARGS --source_path=\"$SOURCE_PATH\""
PYTHON_ARGS="$PYTHON_ARGS --output_path=\"$OUTPUT_PATH\""
PYTHON_ARGS="$PYTHON_ARGS --initial_cameras=$INITIAL_CAMERAS"
PYTHON_ARGS="$PYTHON_ARGS --gpu_threshold=$GPU_THRESHOLD"
PYTHON_ARGS="$PYTHON_ARGS --iterations=$ITERATIONS"
PYTHON_ARGS="$PYTHON_ARGS --sh_degree=$SH_DEGREE"
PYTHON_ARGS="$PYTHON_ARGS --resolution=$RESOLUTION"
PYTHON_ARGS="$PYTHON_ARGS --backend=$BACKEND"

if [[ "$DEBUG" == true ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --debug"
fi

if [[ -n "$DTM_MODULE" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --dtm_module=\"$DTM_MODULE\""
fi

if [[ -n "$DETERMINISTIC" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --deterministic"
fi

if [[ -n "$ONLY_ACTUALLY_VISIBLE" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --only_actually_visible"
fi

# Add any extra arguments
PYTHON_ARGS="$PYTHON_ARGS $EXTRA_ARGS"

# Create the Python wrapper script
PYTHON_WRAPPER="$OUTPUT_PATH/run_progressive.py"
cat > "$PYTHON_WRAPPER" << EOF
#!/usr/bin/env python3
"""
Progressive Training Wrapper Script
Automatically generated by progressive_train.sh
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    from progressive_learning.progressive_trainer import ProgressiveTrainer
except ImportError as e:
    print(f"Error importing ProgressiveTrainer: {e}")
    print("Make sure you're running from the Grendel-GS root directory")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Progressive Training for Grendel-GS')
    parser.add_argument('--source_path', required=True, help='Path to COLMAP reconstruction')
    parser.add_argument('--output_path', required=True, help='Output directory')
    parser.add_argument('--initial_cameras', type=int, default=4, help='Number of initial cameras')
    parser.add_argument('--gpu_threshold', type=float, default=0.9, help='GPU memory threshold')
    parser.add_argument('--iterations', type=int, default=30000, help='Training iterations')
    parser.add_argument('--sh_degree', type=int, default=3, help='Spherical harmonics degree')
    parser.add_argument('--resolution', type=int, default=1, help='Resolution downscaling')
    parser.add_argument('--backend', default='gsplat', help='Rendering backend')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--dtm_module', help='Path to external DTM module')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic training')
    parser.add_argument('--only_actually_visible', action='store_true', help='Only keep points visible in camera frames')
    
    args = parser.parse_args()
    
    # Load DTM module if specified
    dtm_module = None
    if args.dtm_module:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("dtm_module", args.dtm_module)
            dtm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dtm_module)
            print(f"Loaded DTM module from {args.dtm_module}")
        except Exception as e:
            print(f"Warning: Could not load DTM module: {e}")
            print("Continuing with simplified footprint computation")
    
    # Create and run progressive trainer
    trainer = ProgressiveTrainer(
        colmap_path=args.source_path,
        output_path=args.output_path,
        dtm_module=dtm_module,
        initial_cameras=args.initial_cameras,
        gpu_memory_threshold=args.gpu_threshold,
        debug=args.debug,
        only_actually_visible=args.only_actually_visible
    )
    
    # Store additional training parameters for integration with Grendel-GS
    trainer.iterations = args.iterations
    trainer.sh_degree = args.sh_degree
    trainer.resolution = args.resolution
    trainer.backend = args.backend
    trainer.deterministic = args.deterministic
    
    # Run the progressive training pipeline
    try:
        trainer.run()
        print("\\n" + "="*60)
        print("Progressive Training Completed Successfully!")
        print(f"Results saved to: {args.output_path}")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\\nError during progressive training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Make wrapper executable
chmod +x "$PYTHON_WRAPPER" 2>/dev/null || echo "Warning: Could not make wrapper executable"

# Run the progressive training
print_colored $YELLOW "Executing progressive training..."
echo "Command: $PYTHON_CMD \"$PYTHON_WRAPPER\" $PYTHON_ARGS"
echo ""

eval "$PYTHON_CMD -u \"$PYTHON_WRAPPER\" $PYTHON_ARGS"
RESULT=$?

echo ""
if [[ $RESULT -eq 0 ]]; then
    print_colored $GREEN "✓ Progressive training completed successfully!"
    print_colored $GREEN "✓ Results saved to: $OUTPUT_PATH"
    print_colored $GREEN "✓ Log file: $LOG_FILE"
    
    # List output files
    echo ""
    print_colored $BLUE "Output files:"
    find "$OUTPUT_PATH" -type f -name "*.ply" -o -name "*.json" -o -name "*.txt" | head -10 | while read file; do
        echo "  $file"
    done
    
    if [[ $(find "$OUTPUT_PATH" -type f | wc -l) -gt 10 ]]; then
        echo "  ... and more"
    fi
else
    print_colored $RED "❌ Progressive training failed with exit code $RESULT"
    print_colored $RED "❌ Check the log file for details: $LOG_FILE"
fi

echo ""
print_colored $BLUE "======================================"
print_colored $BLUE "Progressive Training Script Complete"
print_colored $BLUE "======================================"

exit $RESULT