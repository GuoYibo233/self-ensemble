#!/bin/bash
# 
# Generate FlexAttention results for all models on MyriadLAMA dataset
# 
# Usage:
#   bash scripts/generate_myriadlama_flexattention.sh [--rewrite] [--dry-run]
#
# Options:
#   --rewrite   Regenerate results even if they already exist
#   --dry-run   Show what would be done without actually running
#

set -e  # Exit on error

# Parse arguments
REWRITE_FLAG=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --rewrite)
            REWRITE_FLAG="--rewrite"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory for datasets
DATASET_ROOT="/net/tokyo100-10g/data/str01_01/y-guo/datasets"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================================================"
echo "Generate FlexAttention Results for MyriadLAMA Dataset"
echo "========================================================================"
echo "Dataset root: $DATASET_ROOT"
echo "Project root: $PROJECT_ROOT"
echo "Dataset: myriadlama"
echo "Rewrite: ${REWRITE_FLAG:-false}"
echo "Dry run: $DRY_RUN"
echo ""

# Track statistics
TOTAL_MODELS=0
SKIPPED_MODELS=0
GENERATED_MODELS=0
FAILED_MODELS=0

# ============================================================================
# Model List Configuration
# ============================================================================
# 配置要在 MyriadLAMA 上运行 FlexAttention 的模型列表
# ============================================================================

MODELS=(
    "llama3.2_1b"
    "llama3.2_1b_it"
    "llama3.2_3b"
    "llama3.2_3b_it"
    "qwen2.5_3b_it"
    "qwen2.5_7b_it"
    "qwen3_1.7b"
    "qwen3_4b"
    "qwen3_8b"
)

# ============================================================================

echo ""
echo "========================================================================"
echo "Processing Models on MyriadLAMA"
echo "========================================================================"

for model in "${MODELS[@]}"; do
    if [[ -z "$model" || "$model" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    ((TOTAL_MODELS++))
    
    model_dir="$DATASET_ROOT/myriadlama/$model"
    
    echo ""
    echo "--------------------------------------------------------------------"
    echo -e "${BLUE}Model: $model${NC}"
    echo "--------------------------------------------------------------------"
    
    # Check if model directory exists
    if [ ! -d "$model_dir" ]; then
        echo -e "${YELLOW}⚠ Model directory not found: $model_dir${NC}"
        echo "   Creating directory..."
        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$model_dir"
        fi
    fi
    
    # Check if flex_attention result already exists
    FLEX_ATTENTION_FILE="$model_dir/flex_attention-5.feather"
    
    if [ -f "$FLEX_ATTENTION_FILE" ] && [ -z "$REWRITE_FLAG" ]; then
        echo -e "${GREEN}✓ FlexAttention result already exists${NC}"
        echo "  - $FLEX_ATTENTION_FILE"
        echo "  Use --rewrite to regenerate"
        ((SKIPPED_MODELS++))
        continue
    fi
    
    # Build command
    CMD="python3 $PROJECT_ROOT/src/generate_flex_attention.py --dataset myriadlama --model $model --num_paraphrases 5 --lemmaize"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would execute:${NC}"
        echo "  $CMD"
        ((GENERATED_MODELS++))
    else
        echo "Executing: $CMD"
        echo ""
        
        # Run the command
        if $CMD; then
            echo ""
            echo -e "${GREEN}✅ Successfully generated FlexAttention result for $model${NC}"
            ((GENERATED_MODELS++))
        else
            echo ""
            echo -e "${RED}❌ Failed to generate FlexAttention result for $model${NC}"
            ((FAILED_MODELS++))
        fi
    fi
done

# Print summary
echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "Total models: $TOTAL_MODELS"
echo -e "${GREEN}Generated: $GENERATED_MODELS${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED_MODELS${NC}"
if [ $FAILED_MODELS -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED_MODELS${NC}"
fi
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}This was a dry run. Run without --dry-run to actually generate results.${NC}"
fi

if [ $FAILED_MODELS -gt 0 ]; then
    echo -e "${RED}Some models failed. Check the logs above for details.${NC}"
    exit 1
fi

echo "Done!"
echo "========================================================================"
