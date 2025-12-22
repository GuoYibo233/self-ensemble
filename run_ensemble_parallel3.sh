#!/bin/bash

# Parallel Ensemble Generation Script - Remaining Tasks
# Usage: ./run_ensemble_parallel3.sh
# Runs remaining 7 tasks on available GPUs
# For Quadro RTX 8000 machine with GPUs 0-7

NUM_PARAPHRASES=5
NUM_SAMPLES=5
MAX_SAMPLES=10000

# 8 GPUs available (Quadro RTX 8000: 0-7)
GPUS=(0 1 2 3 4 5 6 7)
GPU_INDEX=0

# Remaining tasks: 7 tasks total
# qwen2.5_7b_it (max), qwen2.5_7b (avg, max), qwen2.5_3b (avg, max), qwen2.5_3b_it (avg, max)

# Task 1: qwen2.5_7b_it - max
GPU_ID=${GPUS[$GPU_INDEX]}
echo "Starting: qwen2.5_7b_it - max on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
    --model qwen2.5_7b_it \
    --method max \
    --num_paraphrases $NUM_PARAPHRASES \
    --num_samples $NUM_SAMPLES \
    --max_samples $MAX_SAMPLES \
    --dataset myriadlama &
GPU_INDEX=$(( (GPU_INDEX + 1) ))
sleep 2

# Tasks 2-3: qwen2.5_7b (avg, max)
for METHOD in avg max; do
    GPU_ID=${GPUS[$GPU_INDEX]}
    echo "Starting: qwen2.5_7b - $METHOD on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 g_ori_sample_sync_xzhao.py \
        --model qwen2.5_7b \
        --method $METHOD \
        --num_paraphrases $NUM_PARAPHRASES \
        --num_samples $NUM_SAMPLES \
        --max_samples $MAX_SAMPLES \
        --dataset myriadlama &
    GPU_INDEX=$(( (GPU_INDEX + 1) ))
    sleep 2
done

# Tasks 4-7: qwen2.5_3b and qwen2.5_3b_it (avg, max)
for MODEL in qwen2.5_3b qwen2.5_3b_it; do
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

echo "Remaining tasks (7 tasks) submitted. Waiting for completion..."
wait
echo "All remaining experiments completed!"
echo ""
echo "======================================"
echo "Summary: All 24 experiments finished!"
echo "- Batch 1 (run_ensemble_parallel.sh): 9 tasks"
echo "- Batch 2 (run_ensemble_parallel2.sh): 8 tasks"
echo "- Batch 3 (run_ensemble_parallel3.sh): 7 tasks"
echo "======================================"
