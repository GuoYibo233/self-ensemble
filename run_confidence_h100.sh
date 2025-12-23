#!/bin/bash

# Script to run confidence generation on H100 machine with 3 GPUs
# Models: llama3.2_1b, qwen2.5_3b, llama3.2_3b (non-IT versions)
# GPUs: 0, 1, 2

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flexattention

# Set working directory
cd /home/y-guo/self-ensemble/self-ensemble

# Create logs directory if not exists
mkdir -p logs

# Models to run (non-IT versions only)
models=(
    "llama3.2_1b"
    "qwen2.5_3b"
    "llama3.2_3b"
)

# Run each model on a separate GPU
for i in "${!models[@]}"; do
    model="${models[$i]}"
    gpu=$i
    
    echo "=========================================="
    echo "Starting confidence generation for $model on GPU $gpu"
    echo "Time: $(date)"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=$gpu nohup python confidence.py \
        --model "$model" \
        --dataset myriadlama \
        --device auto \
        --rewrite \
        > "logs/confidence_${model}_gpu${gpu}.log" 2>&1 &
    
    pid=$!
    echo "Started $model on GPU $gpu with PID $pid"
    echo "$model,$gpu,$pid,$(date +%s)" >> logs/confidence_running.csv
    
    # Sleep a bit to avoid race conditions
    sleep 10
done

echo ""
echo "All confidence generation tasks started!"
echo "Monitor progress with:"
echo "  tail -f logs/confidence_*.log"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Check running processes:"
echo "  cat logs/confidence_running.csv"
