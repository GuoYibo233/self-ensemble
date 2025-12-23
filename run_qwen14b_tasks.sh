#!/bin/bash

# Run Qwen2.5-14B tasks on available GPUs
# Usage: ./run_qwen14b_tasks.sh

cd /home/y-guo/self-ensemble/self-ensemble

NUM_PARAPHRASES=5
NUM_SAMPLES=5
MAX_SAMPLES=10000

# Available GPUs: 0,1,2,3 (GPU 4 may be occupied, GPU 5-9 already running other tasks)
GPUS=(0 1 2 3)

echo "🚀 Starting 4 Qwen2.5-14B tasks..."
echo "==============================================="

# Task 1: qwen2.5_14b - avg on GPU 0
echo "Starting: qwen2.5_14b - avg on GPU ${GPUS[0]}"
CUDA_VISIBLE_DEVICES=${GPUS[0]} python3 g_ori_sample_sync_xzhao.py \
    --model qwen2.5_14b \
    --method avg \
    --num_paraphrases $NUM_PARAPHRASES \
    --num_samples $NUM_SAMPLES \
    --max_samples $MAX_SAMPLES \
    --dataset myriadlama &

sleep 5

# Task 2: qwen2.5_14b - max on GPU 1
echo "Starting: qwen2.5_14b - max on GPU ${GPUS[1]}"
CUDA_VISIBLE_DEVICES=${GPUS[1]} python3 g_ori_sample_sync_xzhao.py \
    --model qwen2.5_14b \
    --method max \
    --num_paraphrases $NUM_PARAPHRASES \
    --num_samples $NUM_SAMPLES \
    --max_samples $MAX_SAMPLES \
    --dataset myriadlama &

sleep 5

# Task 3: qwen2.5_14b_it - avg on GPU 2
echo "Starting: qwen2.5_14b_it - avg on GPU ${GPUS[2]}"
CUDA_VISIBLE_DEVICES=${GPUS[2]} python3 g_ori_sample_sync_xzhao.py \
    --model qwen2.5_14b_it \
    --method avg \
    --num_paraphrases $NUM_PARAPHRASES \
    --num_samples $NUM_SAMPLES \
    --max_samples $MAX_SAMPLES \
    --dataset myriadlama &

sleep 5

# Task 4: qwen2.5_14b_it - max on GPU 3
echo "Starting: qwen2.5_14b_it - max on GPU ${GPUS[3]}"
CUDA_VISIBLE_DEVICES=${GPUS[3]} python3 g_ori_sample_sync_xzhao.py \
    --model qwen2.5_14b_it \
    --method max \
    --num_paraphrases $NUM_PARAPHRASES \
    --num_samples $NUM_SAMPLES \
    --max_samples $MAX_SAMPLES \
    --dataset myriadlama &

echo ""
echo "✅ All 4 tasks submitted!"
echo "==============================================="
echo ""
echo "📊 Monitor with:"
echo "  watch -n 5 'ps aux | grep g_ori_sample_sync_xzhao.py | grep -v grep'"
echo "  nvidia-smi"
echo ""
echo "⚠️  Note: Tasks will run in background. Check tmux or use 'ps' to monitor."
