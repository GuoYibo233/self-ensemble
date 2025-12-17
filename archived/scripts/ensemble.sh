#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MODEL_NAME> <GPU>"
    exit 1
fi

MODEL_NAME=$1
DATASET=$2
GPU=$3
USE_CONF=$4

if [ "$DATASET" = "myriadlama" ]; then
    RANGE=10
elif [ "$DATASET" = "webqa" ]; then
    RANGE=6
fi

for num_ensemble in $(seq 2 $RANGE)
do
    CUDA_VISIBLE_DEVICES=$GPU python3 ../src/generate_original.py --model $MODEL_NAME --dataset $DATASET --num_ensemble $num_ensemble --method max 
    CUDA_VISIBLE_DEVICES=$GPU python3 ../src/generate_original.py --model $MODEL_NAME --dataset $DATASET --num_ensemble $num_ensemble --method avg 
    if [ "$USE_CONF" = "true" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python3 ../src/generate_original.py --model $MODEL_NAME --dataset $DATASET --num_ensemble $num_ensemble --method weighted_avg 
        CUDA_VISIBLE_DEVICES=$GPU python3 ../src/generate_original.py --model $MODEL_NAME --dataset $DATASET --num_ensemble $num_ensemble --method weighted_max 
    fi
done
