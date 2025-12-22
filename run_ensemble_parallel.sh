#!/bin/bash

# Parallel Ensemble Generation Script
# Usage: ./run_ensemble_parallel.sh
# Note: GPU 4 is in use, using GPUs: 0,1,2,3,5,6,7,8,9

NUM_PARAPHRASES=5
NUM_SAMPLES=5
MAX_SAMPLES=10000

# Available GPUs (excluding GPU 4)
GPUS=(0 1 2 3 5 6 7 8 9)
GPU_INDEX=0

# First batch: 9 tasks (4 largest models × 2 methods + 1 model × 1 method)
for MODEL in qwen2.5_14b qwen2.5_14b_it llama3.1_8b_it llama3.1_8b; do
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
        
        GPU_INDEX=$(( (GPU_INDEX + 1) % ${#GPUS[@]} ))
        sleep 2
    done
done

# 9th task
GPU_ID=${GPUS[$GPU_INDEX]}
echo "Starting: qwen2.5_7b_it - avg on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
    --model qwen2.5_7b_it \
    --method avg \
    --num_paraphrases $NUM_PARAPHRASES \
    --num_samples $NUM_SAMPLES \
    --max_samples $MAX_SAMPLES \
    --dataset myriadlama &

echo "First batch (9 tasks) submitted. Waiting for completion..."
wait
echo "First batch completed!"

# ============================================================================
# Remaining tasks - uncomment and run manually after first batch completes
# ============================================================================

# # qwen2.5_7b_it - max
# CUDA_VISIBLE_DEVICES=0 python3 g_ori_sample_sync_xzhao.py \
#     --model qwen2.5_7b_it --method max \
#     --num_paraphrases 5 --num_samples 5 --max_samples 10000 --dataset myriadlama &
# 
# # qwen2.5_7b - avg, max
# CUDA_VISIBLE_DEVICES=1 python3 g_ori_sample_sync_xzhao.py \
#     --model qwen2.5_7b --method avg \
#     --num_paraphrases 5 --num_samples 5 --max_samples 10000 --dataset myriadlama &
# 
# CUDA_VISIBLE_DEVICES=2 python3 g_ori_sample_sync_xzhao.py \
#     --model qwen2.5_7b --method max \
#     --num_paraphrases 5 --num_samples 5 --max_samples 10000 --dataset myriadlama &
# 
# # Smaller models: llama3.2_3b_it, llama3.2_1b_it, llama3.2_3b, llama3.2_1b, qwen2.5_3b, qwen2.5_3b_it
# # (6 models × 2 methods = 12 tasks)
# 
# # for MODEL in llama3.2_3b_it llama3.2_1b_it llama3.2_3b llama3.2_1b qwen2.5_3b qwen2.5_3b_it; do
# #     for METHOD in avg max; do
# #         CUDA_VISIBLE_DEVICES=... python3 g_ori_sample_sync_xzhao.py \
# #             --model $MODEL --method $METHOD \
# #             --num_paraphrases 5 --num_samples 5 --max_samples 10000 --dataset myriadlama &
# #     done
# # done
# 
# wait
# echo "All experiments completed!"
