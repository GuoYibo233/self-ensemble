#!/bin/bash
DATASET=$1

if [ "$DATASET" = "myriadlama" ]; then
    RANGE=5
elif [ "$DATASET" = "webqa" ]; then
    RANGE=6
fi

for MODEL_NAME in "llama3.1_8b_it" "qwen2.5_3b_it" "qwen2.5_7b_it"
do
    echo "Processing model: $MODEL_NAME"
    
    # Generate outputs for each ensemble size
    for num_ensemble in $(seq 2 $RANGE)
    do
        python3 generate.py --model $MODEL_NAME --dataset $DATASET --num_ensemble $num_ensemble --lemmaize --method max 
        python3 generate.py --model $MODEL_NAME --dataset $DATASET --num_ensemble $num_ensemble --lemmaize --method avg 
        python3 generate.py --model $MODEL_NAME --dataset $DATASET --num_ensemble $num_ensemble --lemmaize --method weighted_avg 
        python3 generate.py --model $MODEL_NAME --dataset $DATASET --num_ensemble $num_ensemble --lemmaize --method weighted_max 
    done
done    