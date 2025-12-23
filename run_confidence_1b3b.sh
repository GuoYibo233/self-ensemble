#!/bin/bash

# Script to generate confidence scores for non-IT 1b and 3b models
# Uses GPUs 0, 1, 2 for parallel processing

cd /home/y-guo/self-ensemble/self-ensemble

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flexattention

echo "Starting confidence generation for 1b and 3b models..."
echo "Using GPUs: 0, 1, 2"

# llama3.2_1b on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python confidence.py \
    --model llama3.2_1b \
    --dataset myriadlama \
    --device auto \
    > logs/confidence_llama3.2_1b.log 2>&1 &
PID1=$!
echo "Started llama3.2_1b on GPU 0 (PID: $PID1)"

# llama3.2_3b on GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python confidence.py \
    --model llama3.2_3b \
    --dataset myriadlama \
    --device auto \
    > logs/confidence_llama3.2_3b.log 2>&1 &
PID2=$!
echo "Started llama3.2_3b on GPU 1 (PID: $PID2)"

# qwen2.5_3b on GPU 2
CUDA_VISIBLE_DEVICES=2 nohup python confidence.py \
    --model qwen2.5_3b \
    --dataset myriadlama \
    --device auto \
    > logs/confidence_qwen2.5_3b.log 2>&1 &
PID3=$!
echo "Started qwen2.5_3b on GPU 2 (PID: $PID3)"

echo ""
echo "All tasks started!"
echo "PIDs: $PID1 (llama3.2_1b), $PID2 (llama3.2_3b), $PID3 (qwen2.5_3b)"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/confidence_llama3.2_1b.log"
echo "  tail -f logs/confidence_llama3.2_3b.log"
echo "  tail -f logs/confidence_qwen2.5_3b.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep confidence.py"
