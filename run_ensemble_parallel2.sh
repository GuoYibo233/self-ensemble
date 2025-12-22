#!/bin/bash

# Parallel Ensemble Generation Script - Small Models
# Usage: ./run_ensemble_parallel2.sh
# Runs 8 tasks on 8 GPUs with smallest models
# For Quadro RTX 8000 machine with GPUs 0-7

NUM_PARAPHRASES=5
NUM_SAMPLES=5
MAX_SAMPLES=10000

# 8 GPUs for small models (Quadro RTX 8000: 0-7)
GPUS=(0 1 2 3 4 5 6 7)
GPU_INDEX=0

# Smallest 4 models × 2 methods = 8 tasks
for MODEL in llama3.2_1b llama3.2_1b_it llama3.2_3b llama3.2_3b_it; do
    for METHOD in avg max; do
        GPU_ID=${GPUS[$GPU_INDEX]}
        
        echo "Starting: $MODEL - $METHOD on GPU $GPU_ID"
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
            --model $MODEL \
            --method $METHOD \
            --num_paraphrases $NUM_PARAPHRASES \
            --num_samples $NUM_SAMPLES \
            --max_samples $MAX_SAMPLES \
            --dataset myriadlama &
        
        GPU_INDEX=$(( (GPU_INDEX + 1) ))
        sleep 2
    done
done

echo "Small model batch (8 tasks) submitted. Waiting for completion..."
wait
echo "Small model experiments completed!"

# ============================================================================
# Remaining small models - uncomment and run manually if needed
# ============================================================================

# # qwen2.5_3b and qwen2.5_3b_it (4 more tasks)
# for MODEL in qwen2.5_3b qwen2.5_3b_it; do
#     for METHOD in avg max; do
#         CUDA_VISIBLE_DEVICES=0 python3 g_ori_sample_sync_xzhao.py \
#             --model $MODEL --method $METHOD \
#             --num_paraphrases 5 --num_samples 5 --max_samples 10000 --dataset myriadlama &
#     done
# done
# 
# wait
# echo "All small model experiments completed!"
