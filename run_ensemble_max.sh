#!/bin/bash

# Ensemble generation with MAX method
# Usage: ./run_ensemble_max.sh [DEVICE]
# Example: ./run_ensemble_max.sh 0

DEVICE=${1:-0}
NUM_PARAPHRASES=5
NUM_SAMPLES=5
MAX_SAMPLES=10000
METHOD="max"

echo "======================================"
echo "Ensemble Generation - MAX Method"
echo "Device: $DEVICE"
echo "Parameters: ${NUM_PARAPHRASES} paras, ${NUM_SAMPLES} samples"
echo "======================================"
echo ""

for MODEL in llama3.2_3b_it llama3.2_1b_it llama3.1_8b_it llama3.2_3b llama3.2_1b llama3.1_8b qwen2.5_3b qwen2.5_7b qwen2.5_14b qwen2.5_3b_it qwen2.5_7b_it qwen2.5_14b_it; do
    echo "========================================"
    echo "Processing: $MODEL - $METHOD"
    echo "Started at: $(date)"
    echo "========================================"
    
    CUDA_VISIBLE_DEVICES=$DEVICE python3 g_ori_sample_sync_xzhao.py \
        --model $MODEL \
        --method $METHOD \
        --num_paraphrases $NUM_PARAPHRASES \
        --num_samples $NUM_SAMPLES \
        --max_samples $MAX_SAMPLES \
        --dataset myriadlama
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed: $MODEL"
    else
        echo "❌ Failed: $MODEL"
    fi
    echo ""
done

echo "======================================"
echo "All MAX experiments completed!"
echo "Finished at: $(date)"
echo "======================================"
