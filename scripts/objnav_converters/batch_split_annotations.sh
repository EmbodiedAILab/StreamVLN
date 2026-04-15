#!/bin/bash
# Batch split annotation files for all scenes in HM3D ObjectNav dataset

set -e

# Configuration
ANNOTATIONS_DIR="data/trajectory_data/objectnav/hm3d_v2_annotation"
NUM_PARTS=4
SCRIPT_PATH="scripts/objnav_converters/split_annotations.py"

# Counters
TOTAL_SCENES=0
PROCESSED=0
SKIPPED=0
FAILED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Batch Split Annotations Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Annotations directory: $ANNOTATIONS_DIR"
echo "Number of parts: $NUM_PARTS"
echo ""

# Check if directory exists
if [ ! -d "$ANNOTATIONS_DIR" ]; then
    echo -e "${RED}Error: Directory $ANNOTATIONS_DIR does not exist${NC}"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: Script $SCRIPT_PATH does not exist${NC}"
    exit 1
fi

# Count total scenes
TOTAL_SCENES=$(find "$ANNOTATIONS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo -e "${BLUE}Found $TOTAL_SCENES scenes to process${NC}"
echo ""

# Function to check if annotations are already split
is_already_split() {
    local scene_dir="$1"
    # Check if all expected split files exist
    for ((i=0; i<NUM_PARTS; i++)); do
        if [ ! -f "$scene_dir/annotation_$i.json" ]; then
            return 1  # Not split
        fi
    done
    return 0  # Already split
}

# Process each scene
CURRENT=0
for scene_dir in "$ANNOTATIONS_DIR"/*; do
    if [ ! -d "$scene_dir" ]; then
        continue
    fi

    CURRENT=$((CURRENT + 1))
    scene_name=$(basename "$scene_dir")
    annotations_json="$scene_dir/annotations.json"

    # Check if annotations.json exists
    if [ ! -f "$annotations_json" ]; then
        echo -e "${YELLOW}[$CURRENT/$TOTAL_SCENES]${NC} $scene_name - ${RED}No annotations.json found${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Check if already split
    if is_already_split "$scene_dir"; then
        echo -e "${YELLOW}[$CURRENT/$TOTAL_SCENES]${NC} $scene_name - ${GREEN}Already split, skipping${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Run split command
    echo -e "${YELLOW}[$CURRENT/$TOTAL_SCENES]${NC} $scene_name - ${BLUE}Processing...${NC}"
    if python "$SCRIPT_PATH" "$annotations_json" --num-parts "$NUM_PARTS" > /dev/null 2>&1; then
        echo -e "${YELLOW}[$CURRENT/$TOTAL_SCENES]${NC} $scene_name - ${GREEN}Successfully split${NC}"
        PROCESSED=$((PROCESSED + 1))
    else
        echo -e "${YELLOW}[$CURRENT/$TOTAL_SCENES]${NC} $scene_name - ${RED}Failed to split${NC}"
        FAILED=$((FAILED + 1))
    fi
done

# Print summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total scenes:        $TOTAL_SCENES"
echo -e "${GREEN}Successfully processed: $PROCESSED${NC}"
echo -e "${YELLOW}Already split (skipped): $SKIPPED${NC}"
echo -e "${RED}Failed:              $FAILED${NC}"
echo -e "${BLUE}========================================${NC}"
