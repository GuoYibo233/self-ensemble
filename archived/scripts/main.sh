#!/bin/bash

MODEL_NAME=$1
GPU_COUNT=$2

GPU0=$((GPU_COUNT + 0))
GPU1=$((GPU_COUNT + 1))
GPU2=$((GPU_COUNT + 2))
GPU3=$((GPU_COUNT + 3))

CUDA_VISIBLE_DEVICES=$GPU0 python3 paraphrase.py --model $MODEL_NAME &
PARAPHRASE_PID=$!
    
wait $PARAPHRASE_PID
CUDA_VISIBLE_DEVICES=$GPU0 python3 confidence.py --model $MODEL_NAME &
CUDA_VISIBLE_DEVICES=$GPU1 python3 ../src/generate_original.py --model $MODEL_NAME --method per_prompt &
CUDA_VISIBLE_DEVICES=$GPU2 python3 ../src/generate_original.py --model $MODEL_NAME --method max &
CUDA_VISIBLE_DEVICES=$GPU3 python3 ../src/generate_original.py --model $MODEL_NAME --method avg
wait


