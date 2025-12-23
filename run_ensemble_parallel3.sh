#!/bin/bash

# Parallel Ensemble Generation Script - Remaining Tasks
# Usage: ./run_ensemble_parallel3.sh
# Runs 6 tasks on GPUs 0,1,2,3,8,9
# For Quadro RTX 8000 machine

NUM_PARAPHRASES=5
NUM_SAMPLES=5
MAX_SAMPLES=10000

# Using GPUs: 0, 1, 2, 3, 8, 9 (6 tasks)
GPUS=(0 1 2 3 8 9)
GPU_INDEX=0

# Remaining tasks: 12 tasks total
# Running: 6 tasks (llama3.1_8b_it avg/max, llama3.1_8b avg/max, qwen2.5_7b_it avg/max)
# TODO: 6 tasks (qwen2.5_7b avg/max, qwen2.5_3b avg/max, qwen2.5_3b_it avg/max)

# Tasks 1-2: llama3.1_8b_it (avg, max)
for METHOD in avg max; do
    GPU_ID=${GPUS[$GPU_INDEX]}
    echo "Starting: llama3.1_8b_it - $METHOD on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
        --model llama3.1_8b_it \
        --method $METHOD \
        --num_paraphrases $NUM_PARAPHRASES \
        --num_samples $NUM_SAMPLES \
        --max_samples $MAX_SAMPLES \
        --dataset myriadlama &
    GPU_INDEX=$(( (GPU_INDEX + 1) ))
    sleep 2
done

# Tasks 3-4: llama3.1_8b (avg, max)
for METHOD in avg max; do
    GPU_ID=${GPUS[$GPU_INDEX]}
    echo "Starting: llama3.1_8b - $METHOD on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
        --model llama3.1_8b \
        --method $METHOD \
        --num_paraphrases $NUM_PARAPHRASES \
        --num_samples $NUM_SAMPLES \
        --max_samples $MAX_SAMPLES \
        --dataset myriadlama &
    GPU_INDEX=$(( (GPU_INDEX + 1) ))
    sleep 2
done

# Tasks 5-6: qwen2.5_7b_it (avg, max)
for METHOD in avg max; do
    GPU_ID=${GPUS[$GPU_INDEX]}
    echo "Starting: qwen2.5_7b_it - $METHOD on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
        --model qwen2.5_7b_it \
        --method $METHOD \
        --num_paraphrases $NUM_PARAPHRASES \
        --num_samples $NUM_SAMPLES \
        --max_samples $MAX_SAMPLES \
        --dataset myriadlama &
    GPU_INDEX=$(( (GPU_INDEX + 1) ))
    sleep 2
done

echo "Current batch (6 tasks) submitted. Waiting for completion..."
wait
echo "Current batch completed!"
echo ""
echo "======================================"
echo "Summary: Batch 1 finished!"
echo "✅ Completed:"
echo "- llama3.1_8b_it: avg, max (GPUs 0,1)"
echo "- llama3.1_8b: avg, max (GPUs 2,3)"
echo "- qwen2.5_7b_it: avg, max (GPUs 8,9)"
echo ""
echo "❌ TODO - Not started yet:"
echo "- qwen2.5_7b: avg, max"
echo "- qwen2.5_3b: avg, max"
echo "- qwen2.5_3b_it: avg, max"
echo "======================================"

# ========================================
# TODO: Remaining 6 tasks (not running yet)
# ========================================

# # Tasks 7-8: qwen2.5_7b (avg, max) - TODO
# for METHOD in avg max; do
#     GPU_ID=${GPUS[$GPU_INDEX]}
#     echo "Starting: qwen2.5_7b - $METHOD on GPU $GPU_ID"
#     CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
#         --model qwen2.5_7b \
#         --method $METHOD \
#         --num_paraphrases $NUM_PARAPHRASES \
#         --num_samples $NUM_SAMPLES \
#         --max_samples $MAX_SAMPLES \
#         --dataset myriadlama &
#     GPU_INDEX=$(( (GPU_INDEX + 1) ))
#     sleep 2
# done

# # Tasks 9-10: qwen2.5_3b (avg, max) - TODO
# for METHOD in avg max; do
#     GPU_ID=${GPUS[$GPU_INDEX]}
#     echo "Starting: qwen2.5_3b - $METHOD on GPU $GPU_ID"
#     CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
#         --model qwen2.5_3b \
#         --method $METHOD \
#         --num_paraphrases $NUM_PARAPHRASES \
#         --num_samples $NUM_SAMPLES \
#         --max_samples $MAX_SAMPLES \
#         --dataset myriadlama &
#     GPU_INDEX=$(( (GPU_INDEX + 1) ))
#     sleep 2
# done

# # Tasks 11-12: qwen2.5_3b_it (avg, max) - TODO
# for METHOD in avg max; do
#     GPU_ID=${GPUS[$GPU_INDEX]}
#     echo "Starting: qwen2.5_3b_it - $METHOD on GPU $GPU_ID"
#     CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
#         --model qwen2.5_3b_it \
#         --method $METHOD \
#         --num_paraphrases $NUM_PARAPHRASES \
#         --num_samples $NUM_SAMPLES \
#         --max_samples $MAX_SAMPLES \
#         --dataset myriadlama &
#     GPU_INDEX=$(( (GPU_INDEX + 1) ))
#     sleep 2
# done
