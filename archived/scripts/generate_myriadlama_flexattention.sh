#!/bin/bash
# 
# Generate FlexAttention results for all models on MyriadLAMA dataset
# 
# Usage:
#   bash scripts/generate_myriadlama_flexattention.sh [OPTIONS]
#
# Options:
#   --rewrite           Regenerate results even if they already exist
#   --dry-run           Show what would be done without actually running
#   --max-samples N     Process only N samples per model (default: all samples)
#   --generate-baseline Generate ONLY per-prompt baseline (skip FlexAttention)
#   --gpus GPUS         Specify GPUs to use (e.g., "4,5,6,7,8,9" or "0,1,2,3")
#                       (default: auto, no restriction)
#   --parallel N        Run N models in parallel (default: 1, sequential)
#                       Note: Requires sufficient GPUs for parallel execution
#

set -e  # Exit on error

# Parse arguments
REWRITE_FLAG=""
DRY_RUN=false
MAX_SAMPLES=""
GENERATE_BASELINE=false
GPUS=""  # Default: no GPU restriction (auto)
PARALLEL=1  # Default: sequential execution

while [[ $# -gt 0 ]]; do
    case $1 in
        --rewrite)
            REWRITE_FLAG="--rewrite"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --generate-baseline)
            GENERATE_BASELINE=true
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        *)
            shift
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
echo "Max samples per model: ${MAX_SAMPLES:-all}"
echo "Generate baseline: $GENERATE_BASELINE"
if [ "$GENERATE_BASELINE" = true ]; then
    echo "Mode: Baseline ONLY (skipping FlexAttention)"
fi
echo "GPUs to use: ${GPUS:-auto (no restriction)}"
echo "Parallel jobs: $PARALLEL"
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
    # Original models
    "llama3.2_3b"
    "llama3.2_3b_it"
    "qwen2.5_3b_it"
    "qwen2.5_7b_it"
    "qwen3_1.7b"
    "qwen3_4b"
    "qwen3_8b"
    
    # New models 
    "llama3.1_8b"
    # "llama3.1_70b"
    "deepseek_r1_distill_llama_8b"
    #"deepseek_r1_distill_qwen_32b"
    #"deepseek_r1_distill_qwen_14b"
    #"qwen2.5_7b"
    #"qwen2.5_14b"
)

# ============================================================================

echo ""
echo "========================================================================"
echo "Processing Models on MyriadLAMA"
echo "========================================================================"

# Function to process a single model
process_model() {
    local model="$1"
    local REWRITE_FLAG="$2"
    local MAX_SAMPLES="$3"
    local GENERATE_BASELINE="$4"
    local GPUS="$5"
    local DRY_RUN="$6"
    local PROJECT_ROOT="$7"
    local DATASET_ROOT="$8"
    
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
    
    # If baseline-only mode, skip FlexAttention generation check
    if [ "$GENERATE_BASELINE" = false ]; then
        # Check if flex_attention result already exists
        FLEX_ATTENTION_FILE="$model_dir/myriadlama_flex_5paras.feather"
        
        if [ -f "$FLEX_ATTENTION_FILE" ] && [ -z "$REWRITE_FLAG" ]; then
            echo -e "${GREEN}✓ FlexAttention result already exists${NC}"
            echo "  - $FLEX_ATTENTION_FILE"
            echo "  Use --rewrite to regenerate"
            return 2  # Special return code for skipped
        fi
    else
        # In baseline-only mode, check if baseline already exists
        BASELINE_FILE="$model_dir/myriadlama_baseline_5paras.feather"
        
        if [ -f "$BASELINE_FILE" ] && [ -z "$REWRITE_FLAG" ]; then
            echo -e "${GREEN}✓ Baseline result already exists${NC}"
            echo "  - $BASELINE_FILE"
            echo "  Use --rewrite to regenerate"
            return 2  # Special return code for skipped
        fi
    fi
    
    # Build command
    if [ -n "$GPUS" ]; then
        CMD="CUDA_VISIBLE_DEVICES=$GPUS python3 $PROJECT_ROOT/myriadlama_flex_attention_generate.py --model $model --num_paraphrases 5 --device auto --disable_p2p"
    else
        CMD="python3 $PROJECT_ROOT/myriadlama_flex_attention_generate.py --model $model --num_paraphrases 5 --device auto --disable_p2p"
    fi
    
    if [ -n "$REWRITE_FLAG" ]; then
        CMD="$CMD --rewrite"
    fi
    
    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi
    
    if [ "$GENERATE_BASELINE" = true ]; then
        CMD="$CMD --generate_baseline"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would execute:${NC}"
        echo "  $CMD"
        return 0
    else
        echo "Executing: $CMD"
        echo ""
        
        if eval $CMD; then
            echo ""
            if [ "$GENERATE_BASELINE" = true ]; then
                echo -e "${GREEN}✅ Successfully generated baseline result for $model${NC}"
            else
                echo -e "${GREEN}✅ Successfully generated FlexAttention result for $model${NC}"
            fi
            return 0
        else
            echo ""
            if [ "$GENERATE_BASELINE" = true ]; then
                echo -e "${RED}❌ Failed to generate baseline result for $model${NC}"
            else
                echo -e "${RED}❌ Failed to generate FlexAttention result for $model${NC}"
            fi
            return 1
        fi
    fi
}

# Export function and variables for parallel execution
export -f process_model
export RED GREEN YELLOW BLUE NC

# Process models
if [ "$PARALLEL" -eq 1 ]; then
    # Sequential execution
    for model in "${MODELS[@]}"; do
        if [[ -z "$model" || "$model" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        
        ((TOTAL_MODELS++)) || true
        
        process_model "$model" "$REWRITE_FLAG" "$MAX_SAMPLES" "$GENERATE_BASELINE" "$GPUS" "$DRY_RUN" "$PROJECT_ROOT" "$DATASET_ROOT"
        ret=$?
        
        if [ $ret -eq 0 ]; then
            ((GENERATED_MODELS++)) || true
        elif [ $ret -eq 2 ]; then
            ((SKIPPED_MODELS++)) || true
        else
            ((FAILED_MODELS++)) || true
        fi
    done
else
    # Parallel execution
    echo "Running $PARALLEL models in parallel..."
    echo ""
    
    # Create a temporary directory for job tracking
    TMP_DIR=$(mktemp -d)
    
    # Filter out empty lines and comments
    VALID_MODELS=()
    for model in "${MODELS[@]}"; do
        if [[ -n "$model" && ! "$model" =~ ^[[:space:]]*# ]]; then
            VALID_MODELS+=("$model")
        fi
    done
    
    TOTAL_MODELS=${#VALID_MODELS[@]}
    
    # Process models in parallel batches
    for ((i=0; i<${#VALID_MODELS[@]}; i+=$PARALLEL)); do
        BATCH_PIDS=()
        
        for ((j=0; j<$PARALLEL && i+j<${#VALID_MODELS[@]}; j++)); do
            model="${VALID_MODELS[$((i+j))]}"
            
            # Run model processing in background
            (
                process_model "$model" "$REWRITE_FLAG" "$MAX_SAMPLES" "$GENERATE_BASELINE" "$GPUS" "$DRY_RUN" "$PROJECT_ROOT" "$DATASET_ROOT"
                echo $? > "$TMP_DIR/${model}.status"
            ) &
            
            BATCH_PIDS+=($!)
        done
        
        # Wait for current batch to complete
        for pid in "${BATCH_PIDS[@]}"; do
            wait $pid
        done
    done
    
    # Collect results
    for model in "${VALID_MODELS[@]}"; do
        if [ -f "$TMP_DIR/${model}.status" ]; then
            ret=$(cat "$TMP_DIR/${model}.status")
            if [ $ret -eq 0 ]; then
                ((GENERATED_MODELS++)) || true
            elif [ $ret -eq 2 ]; then
                ((SKIPPED_MODELS++)) || true
            else
                ((FAILED_MODELS++)) || true
            fi
        fi
    done
    
    # Cleanup
    rm -rf "$TMP_DIR"
fi

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
