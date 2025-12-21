#!/bin/bash

# Ensemble Generation Batch Runner
# This script helps run generate_original.py with different parameter combinations

# =============================================================================
# Configuration
# =============================================================================

MODEL="llama3.2_3b_it"
DATASET="myriadlama"  # or "webqa"
DEVICE="cuda"

# =============================================================================
# Define parameter combinations to run
# Format: "method num_manual num_auto num_samples"
# =============================================================================

COMBINATIONS=(
    # Format: "method manual auto samples"
    # different ratio with 5 samples"
    "avg 0 15 5"
    "avg 5 5 5"
    "avg 5 0 5"
    "avg 5 10 5"
    "avg 5 15 5"
    # 1,3,5 total samples
    "avg 5 0 1"
    "avg 5 0 3"
    "avg 0 15 1"
    "avg 0 15 3"
    "avg 0 15 5"
    # 5 manual samples with more and more manual
    "avg 5 1 6"
    "avg 5 3 8"
    "avg 5 5 10"
    "avg 5 10 15"
    "avg 5 15 20"
    # 5 auto samples with more and more manual
    "avg 1 5 6"
    "avg 3 5 8"
    "avg 5 5 10"
    # Add more combinations here
)

# =============================================================================
# Run all combinations
# =============================================================================

echo "=========================================="
echo "Starting Ensemble Generation Batch Run"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Device: $DEVICE"
echo "Total combinations: ${#COMBINATIONS[@]}"
echo "=========================================="
echo ""

for i in "${!COMBINATIONS[@]}"; do
    combo="${COMBINATIONS[$i]}"
    read -r method manual auto samples <<< "$combo"
    
    echo "[$((i+1))/${#COMBINATIONS[@]}] Running: method=$method, manual=$manual, auto=$auto, samples=$samples"
    
    python generate_original.py \
        --method "$method" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --device "$DEVICE" \
        --num_manual "$manual" \
        --num_auto "$auto" \
        --num_samples "$samples"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Success"
    else
        echo "✗ Failed with exit code $exit_code"
    fi
    echo ""
done

echo "=========================================="
echo "Batch run completed!"
echo "Results saved in: ./results/$MODEL/"
echo "=========================================="
