#!/bin/bash
# PLY File Comparison Script
# Wrapper for compare_ply.py with enhanced functionality

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
    echo "PLY File Comparison Tool"
    echo "========================"
    echo ""
    echo "Usage:"
    echo "  $0 file1.ply file2.ply                    # Compare two specific files"
    echo "  $0 dir                                    # Find identical PLY file pairs within directory"
    echo "  $0 -d dir1 dir2                          # Compare all PLY files between directories"
    echo "  $0 -p pattern1 pattern2                  # Compare files matching patterns"
    echo "  $0 -l dir                                # List all PLY files in directory"
    echo "  $0 -h                                    # Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 output/scene1/point_cloud.ply output/scene2/point_cloud.ply"
    echo "  $0 output/checkpoints                    # Find identical PLY pairs in directory"
    echo "  $0 -d output/run1 output/run2"
    echo "  $0 -p 'output/*/iteration_030000/*.ply' 'output2/*/iteration_030000/*.ply'"
    echo ""
}

# Function to check if file exists
check_file() {
    local file=$1
    if [[ ! -f "$file" ]]; then
        print_colored $RED "❌ File not found: $file"
        return 1
    fi
    return 0
}

# Function to check dependencies
check_dependencies() {
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        if ! command -v python &> /dev/null; then
            print_colored $RED "❌ Python not found. Please install Python 3."
            exit 1
        fi
        PYTHON_CMD="python"
    else
        PYTHON_CMD="python3"
    fi
    
    # Check if compare_ply.py exists
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    COMPARE_PY="$SCRIPT_DIR/compare_ply.py"
    
    if [[ ! -f "$COMPARE_PY" ]]; then
        print_colored $RED "❌ compare_ply.py not found in $SCRIPT_DIR"
        exit 1
    fi
    
    # Check Python dependencies
    if ! $PYTHON_CMD -c "import plyfile, numpy" 2>/dev/null; then
        print_colored $YELLOW "⚠️  Missing dependencies. Installing..."
        pip install plyfile numpy
        if [[ $? -ne 0 ]]; then
            print_colored $RED "❌ Failed to install dependencies"
            exit 1
        fi
    fi
}

# Function to compare two files
compare_files() {
    local file1=$1
    local file2=$2
    
    print_colored $BLUE "Comparing:"
    echo "  $file1"
    echo "  $file2"
    echo ""
    
    $PYTHON_CMD "$COMPARE_PY" "$file1" "$file2"
    local result=$?
    
    echo ""
    if [[ $result -eq 0 ]]; then
        print_colored $GREEN "✅ Files are identical"
    else
        print_colored $RED "❌ Files are different"
    fi
    
    return $result
}

# Function to compare directories
compare_directories() {
    local dir1=$1
    local dir2=$2
    
    if [[ ! -d "$dir1" ]]; then
        print_colored $RED "❌ Directory not found: $dir1"
        return 1
    fi
    
    if [[ ! -d "$dir2" ]]; then
        print_colored $RED "❌ Directory not found: $dir2"
        return 1
    fi
    
    print_colored $BLUE "Comparing directories:"
    echo "  $dir1"
    echo "  $dir2"
    echo ""
    
    # Find all PLY files in both directories
    local files1=($(find "$dir1" -name "*.ply" | sort))
    local files2=($(find "$dir2" -name "*.ply" | sort))
    
    if [[ ${#files1[@]} -eq 0 ]]; then
        print_colored $YELLOW "⚠️  No PLY files found in $dir1"
        return 1
    fi
    
    if [[ ${#files2[@]} -eq 0 ]]; then
        print_colored $YELLOW "⚠️  No PLY files found in $dir2"
        return 1
    fi
    
    # Compare files with matching relative paths
    local all_identical=true
    
    for file1 in "${files1[@]}"; do
        local rel_path=${file1#$dir1/}
        local file2="$dir2/$rel_path"
        
        if [[ -f "$file2" ]]; then
            echo "Comparing: $rel_path"
            compare_files "$file1" "$file2"
            if [[ $? -ne 0 ]]; then
                all_identical=false
            fi
            echo "----------------------------------------"
        else
            print_colored $YELLOW "⚠️  File $rel_path exists in $dir1 but not in $dir2"
            all_identical=false
        fi
    done
    
    # Check for files only in dir2
    for file2 in "${files2[@]}"; do
        local rel_path=${file2#$dir2/}
        local file1="$dir1/$rel_path"
        
        if [[ ! -f "$file1" ]]; then
            print_colored $YELLOW "⚠️  File $rel_path exists in $dir2 but not in $dir1"
            all_identical=false
        fi
    done
    
    if $all_identical; then
        print_colored $GREEN "✅ All files in directories are identical"
        return 0
    else
        print_colored $RED "❌ Some files differ between directories"
        return 1
    fi
}

# Function to find identical PLY file pairs within a single directory
compare_within_directory() {
    local dir=$1
    
    if [[ ! -d "$dir" ]]; then
        print_colored $RED "❌ Directory not found: $dir"
        return 1
    fi
    
    print_colored $BLUE "Finding identical PLY files in directory:"
    echo "  $dir"
    echo ""
    
    # Find all PLY files in the directory (not recursive)
    local files=($(find "$dir" -maxdepth 1 -name "*.ply" | sort))
    
    if [[ ${#files[@]} -eq 0 ]]; then
        print_colored $YELLOW "⚠️  No PLY files found in directory"
        return 1
    fi
    
    if [[ ${#files[@]} -eq 1 ]]; then
        local basename=$(basename "${files[0]}")
        print_colored $BLUE "Found 1 PLY file:"
        echo "  $basename"
        print_colored $YELLOW "Only one PLY file found. No pairs to compare."
        return 0
    fi
    
    print_colored $BLUE "Found ${#files[@]} PLY files:"
    for file in "${files[@]}"; do
        local basename=$(basename "$file")
        echo "  $basename"
    done
    echo ""
    
    # Find identical pairs
    local identical_pairs=()
    
    for ((i=0; i<${#files[@]}-1; i++)); do
        for ((j=i+1; j<${#files[@]}; j++)); do
            local file1="${files[i]}"
            local file2="${files[j]}"
            
            # Run comparison silently and check result
            $PYTHON_CMD "$COMPARE_PY" "$file1" "$file2" >/dev/null 2>&1
            local result=$?
            
            if [[ $result -eq 0 ]]; then
                local basename1=$(basename "$file1")
                local basename2=$(basename "$file2")
                identical_pairs+=("$basename1 = $basename2")
            fi
        done
    done
    
    # Report results
    if [[ ${#identical_pairs[@]} -gt 0 ]]; then
        print_colored $GREEN "✅ Found ${#identical_pairs[@]} identical PLY file pair(s):"
        for pair in "${identical_pairs[@]}"; do
            echo "  $pair"
        done
        return 0
    else
        print_colored $YELLOW "❌ No identical PLY files found."
        echo "All PLY files in this directory are different from each other."
        return 1
    fi
}

# Function to list PLY files
list_ply_files() {
    local dir=$1
    
    if [[ ! -d "$dir" ]]; then
        print_colored $RED "❌ Directory not found: $dir"
        return 1
    fi
    
    print_colored $BLUE "PLY files in $dir:"
    echo ""
    
    local files=($(find "$dir" -name "*.ply" | sort))
    
    if [[ ${#files[@]} -eq 0 ]]; then
        print_colored $YELLOW "⚠️  No PLY files found"
        return 1
    fi
    
    for file in "${files[@]}"; do
        local rel_path=${file#$dir/}
        local size=$(ls -lh "$file" | awk '{print $5}')
        echo "  $rel_path ($size)"
    done
    
    echo ""
    print_colored $GREEN "Total: ${#files[@]} PLY files"
}

# Main script
main() {
    # Check dependencies first
    check_dependencies
    
    # Parse arguments
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--directories)
            if [[ $# -ne 3 ]]; then
                print_colored $RED "❌ Directory comparison requires exactly 2 arguments"
                show_usage
                exit 1
            fi
            compare_directories "$2" "$3"
            exit $?
            ;;
        -l|--list)
            if [[ $# -ne 2 ]]; then
                print_colored $RED "❌ List option requires exactly 1 directory argument"
                show_usage
                exit 1
            fi
            list_ply_files "$2"
            exit $?
            ;;
        -p|--pattern)
            if [[ $# -ne 3 ]]; then
                print_colored $RED "❌ Pattern comparison requires exactly 2 pattern arguments"
                show_usage
                exit 1
            fi
            # Expand patterns
            files1=($2)
            files2=($3)
            
            if [[ ${#files1[@]} -ne ${#files2[@]} ]]; then
                print_colored $RED "❌ Pattern mismatch: ${#files1[@]} files vs ${#files2[@]} files"
                exit 1
            fi
            
            if [[ ${#files1[@]} -eq 0 ]]; then
                print_colored $YELLOW "⚠️  No files match the patterns"
                exit 1
            fi
            
            local all_identical=true
            for ((i=0; i<${#files1[@]}; i++)); do
                compare_files "${files1[i]}" "${files2[i]}"
                if [[ $? -ne 0 ]]; then
                    all_identical=false
                fi
                echo "----------------------------------------"
            done
            
            if $all_identical; then
                exit 0
            else
                exit 1
            fi
            ;;
        *)
            if [[ $# -eq 1 ]]; then
                # Check if it's a directory (single directory comparison)
                if [[ -d "$1" ]]; then
                    compare_within_directory "$1"
                    exit $?
                # Check if it's a file (error case - need 2 files)
                elif [[ -f "$1" ]]; then
                    print_colored $RED "❌ Single file provided. Please provide 2 PLY files to compare, or a directory to compare all PLY files within it."
                    show_usage
                    exit 1
                else
                    print_colored $RED "❌ Path not found: $1"
                    exit 1
                fi
            elif [[ $# -eq 2 ]]; then
                # Two arguments - compare two files
                check_file "$1" || exit 1
                check_file "$2" || exit 1
                
                compare_files "$1" "$2"
                exit $?
            else
                print_colored $RED "❌ Invalid number of arguments"
                show_usage
                exit 1
            fi
            ;;
    esac
}

# Run main function with all arguments
main "$@"