#!/bin/bash
# Visualize outward_spiral_compact E selection strategy
# Generates an animated GIF showing the algorithm in action

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Set up logging - redirect all output to log file
LOG_FILE="visualize_spiral.log"
rm -f "$LOG_FILE"  # Clear log file
exec > >(tee -a "$LOG_FILE") 2>&1

print_colored $CYAN "======================================"
print_colored $CYAN "Spiral Selection Visualization"
print_colored $CYAN "======================================"
echo ""

# Default parameters
N_CAMERAS=50
INITIAL_WINDOW=""  # Empty = n_cameras/10
MAX_STEPS=""  # Empty = all cameras
OUTPUT_FILE="spiral_selection.gif"
OUTWARD_WEIGHT=0.04
COMPACT_WEIGHT=2.5
SMOOTH_WINDOW_WEIGHT=2.8
SMOOTH_CAMERA_WEIGHT=0.7
DURATION=1.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_cameras)
            N_CAMERAS="$2"
            shift 2
            ;;
        --initial_window)
            INITIAL_WINDOW="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --outward_weight)
            OUTWARD_WEIGHT="$2"
            shift 2
            ;;
        --compact_weight)
            COMPACT_WEIGHT="$2"
            shift 2
            ;;
        --smooth_window_weight)
            SMOOTH_WINDOW_WEIGHT="$2"
            shift 2
            ;;
        --smooth_camera_weight)
            SMOOTH_CAMERA_WEIGHT="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n_cameras NUM        Number of cameras (default: 50)"
            echo "  --initial_window NUM   Initial window size (default: n_cameras/10)"
            echo "  --max_steps NUM        Maximum steps (default: all cameras)"
            echo "  --output FILE          Output GIF file (default: spiral_selection.gif)"
            echo "  --outward_weight VALUE        Weight for outward movement (default: 0.04)"
            echo "  --compact_weight VALUE        Weight for window compactness (default: 2.5)"
            echo "  --smooth_window_weight VALUE  Weight for smooth window trajectory (default: 2.8)"
            echo "  --smooth_camera_weight VALUE  Weight for smooth camera trajectory (default: 0.7)"
            echo "  --duration VALUE              Duration per frame in seconds (default: 1.0)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --n_cameras 100 --max_steps 30"
            echo "  $0 --alpha 1.5 --beta 0.5 --gamma 0.3 --output my_spiral.gif"
            exit 0
            ;;
        *)
            print_colored $RED "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
print_colored $GREEN "Configuration:"
echo "  Number of cameras: $N_CAMERAS"
if [[ -n "$INITIAL_WINDOW" ]]; then
    echo "  Initial window size: $INITIAL_WINDOW"
else
    echo "  Initial window size: Auto (n_cameras/10 = $(($N_CAMERAS / 10)))"
fi
if [[ -n "$MAX_STEPS" ]]; then
    echo "  Maximum steps: $MAX_STEPS"
else
    if [[ -n "$INITIAL_WINDOW" ]]; then
        echo "  Maximum steps: All cameras ($(($N_CAMERAS - $INITIAL_WINDOW)))"
    else
        echo "  Maximum steps: All cameras ($(($N_CAMERAS - $N_CAMERAS / 10)))"
    fi
fi
echo "  Output file: $OUTPUT_FILE"
echo "  Outward weight: $OUTWARD_WEIGHT"
echo "  Compact weight: $COMPACT_WEIGHT"
echo "  Smooth window weight: $SMOOTH_WINDOW_WEIGHT"
echo "  Smooth camera weight: $SMOOTH_CAMERA_WEIGHT"
echo "  Duration per frame: $DURATION seconds"
echo ""

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

print_colored $GREEN "✓ Python found: $PYTHON_CMD"

# Check if visualization script exists
if [[ ! -f "visualize_spiral_selection.py" ]]; then
    print_colored $RED "❌ Error: visualize_spiral_selection.py not found"
    print_colored $YELLOW "Please run this script from the Grendel-GS root directory"
    exit 1
fi

# Check/install required packages
print_colored $YELLOW "Checking required packages..."

PACKAGES_TO_CHECK=("numpy" "matplotlib" "imageio" "scipy")
MISSING_PACKAGES=()

for package in "${PACKAGES_TO_CHECK[@]}"; do
    if ! $PYTHON_CMD -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_colored $YELLOW "⚠️  Missing packages: ${MISSING_PACKAGES[*]}"
    print_colored $YELLOW "Installing missing packages..."

    if $PYTHON_CMD -m pip install "${MISSING_PACKAGES[@]}" --quiet; then
        print_colored $GREEN "✓ Packages installed successfully"
    else
        print_colored $RED "❌ Failed to install packages"
        print_colored $YELLOW "Please install manually: pip install ${MISSING_PACKAGES[*]}"
        exit 1
    fi
else
    print_colored $GREEN "✓ All required packages are installed"
fi

echo ""
print_colored $BLUE "Starting visualization..."
echo ""

# Run the visualization
CMD_ARGS="--n_cameras $N_CAMERAS"
if [[ -n "$INITIAL_WINDOW" ]]; then
    CMD_ARGS="$CMD_ARGS --initial_window $INITIAL_WINDOW"
fi
if [[ -n "$MAX_STEPS" ]]; then
    CMD_ARGS="$CMD_ARGS --max_steps $MAX_STEPS"
fi
CMD_ARGS="$CMD_ARGS --output \"$OUTPUT_FILE\" --outward_weight $OUTWARD_WEIGHT --compact_weight $COMPACT_WEIGHT --smooth_window_weight $SMOOTH_WINDOW_WEIGHT --smooth_camera_weight $SMOOTH_CAMERA_WEIGHT --duration $DURATION"

eval "$PYTHON_CMD visualize_spiral_selection.py $CMD_ARGS"

RESULT=$?

echo ""
if [[ $RESULT -eq 0 ]]; then
    print_colored $GREEN "✓ Visualization completed successfully!"
    print_colored $GREEN "✓ GIF saved to: $OUTPUT_FILE"

    # Check if file exists and show size
    if [[ -f "$OUTPUT_FILE" ]]; then
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        print_colored $CYAN "  File size: $FILE_SIZE"

        # Show absolute path
        ABS_PATH=$(realpath "$OUTPUT_FILE")
        print_colored $CYAN "  Full path: $ABS_PATH"
    fi
else
    print_colored $RED "❌ Visualization failed with exit code $RESULT"
    exit 1
fi

echo ""
print_colored $CYAN "======================================"
print_colored $CYAN "Visualization Complete"
print_colored $CYAN "======================================"

exit 0
