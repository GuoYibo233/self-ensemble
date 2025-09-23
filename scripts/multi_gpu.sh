#!/bin/bash

# Multi-GPU experiment script for 10 GPUs
MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name>"
    echo "Available models: qwen1.5_moe_a2.7b_chat, deepseek_moe_16b_chat, llada_moe_7b_instruct, qwen3_1.7b, etc."
    exit 1
fi

echo "Starting multi-GPU experiment with model: $MODEL_NAME"
echo "Using 10 GPUs (0-9)"

# Step 1: Generate paraphrases (GPU 0)
echo "Step 1: Generating paraphrases..."
CUDA_VISIBLE_DEVICES=0 python3 paraphrase.py --model $MODEL_NAME --dataset webqa &
PARAPHRASE_PID=$!
wait $PARAPHRASE_PID

# Step 2: Run different methods in parallel across multiple GPUs
echo "Step 2: Running generation methods in parallel..."

# WebQA dataset experiments
CUDA_VISIBLE_DEVICES=0 python3 generate.py --model $MODEL_NAME --dataset webqa --method per_prompt &
CUDA_VISIBLE_DEVICES=1 python3 generate.py --model $MODEL_NAME --dataset webqa --method max &
CUDA_VISIBLE_DEVICES=2 python3 generate.py --model $MODEL_NAME --dataset webqa --method avg &

# Confidence estimation
CUDA_VISIBLE_DEVICES=3 python3 confidence.py --model $MODEL_NAME --dataset webqa &

# MyriadLAMA dataset experiments (if dataset exists)
CUDA_VISIBLE_DEVICES=4 python3 generate.py --model $MODEL_NAME --dataset myriadlama --method per_prompt &
CUDA_VISIBLE_DEVICES=5 python3 generate.py --model $MODEL_NAME --dataset myriadlama --method max &
CUDA_VISIBLE_DEVICES=6 python3 generate.py --model $MODEL_NAME --dataset myriadlama --method avg &

# Additional experiments with different ensemble numbers
CUDA_VISIBLE_DEVICES=7 python3 generate.py --model $MODEL_NAME --dataset webqa --method max --num_ensemble 10 &
CUDA_VISIBLE_DEVICES=8 python3 generate.py --model $MODEL_NAME --dataset webqa --method avg --num_ensemble 10 &

# Confidence for MyriadLAMA
CUDA_VISIBLE_DEVICES=9 python3 confidence.py --model $MODEL_NAME --dataset myriadlama &

echo "All processes started. Waiting for completion..."
wait

echo "Multi-GPU experiment completed!"