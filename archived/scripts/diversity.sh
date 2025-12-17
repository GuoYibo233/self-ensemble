#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 <MODEL_NAME> <DATASET> <GPU>"
    exit 1
fi

MODEL_NAME=$1
DATASET=$2
GPU=$3


generate_combinations() {
  local range=$1
  local combinations=()
  
  for ((i=0; i<=range; i++)); do
    for ((j=i+1; j<=range; j++)); do
      combinations+=("$i,$j")
    done
  done

  # Return the array elements
  echo "${combinations[@]}"
}

if [ "$DATASET" = "myriadlama" ]; then
    # INDEXS=("0,1,2" "0,1,3" "0,1,4" "0,2,3" "0,2,4" "0,3,4" "1,2,3" "1,2,4" "1,3,4" "2,3,4" "0,1,2,3" "0,1,2,4" "0,1,3,4" "0,2,3,4" "1,2,3,4")
    # ("0,1" "0,2" "0,3" "0,4" "1,2" "1,3" "1,4" "2,3" "2,4" "3,4" 
    read -a mylist <<< "$(generate_combinations 9)"
elif [ "$DATASET" = "webqa" ]; then
    # INDEXS=("0,1" "0,2" "0,3" "0,4" "0,5" "1,2" "1,3" "1,4" "1,5" "2,3" "2,4" "2,5" "3,4" "3,5" "4,5")
    read -a mylist <<< "$(generate_combinations 5)"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

for index in "${INDEXS[@]}"
do
    # CUDA_VISIBLE_DEVICES=$GPU python3 ../src/generate_original.py --model $MODEL_NAME --dataset $DATASET --indexs $index --method max 
    # CUDA_VISIBLE_DEVICES=$GPU python3 ../src/generate_original.py --model $MODEL_NAME --dataset $DATASET --indexs $index --method avg 
    CUDA_VISIBLE_DEVICES=$GPU python3 ../src/generate_original.py --model $MODEL_NAME --dataset $DATASET --indexs $index --method weighted_avg 
    # CUDA_VISIBLE_DEVICES=$GPU python3 ../src/generate_original.py --model $MODEL_NAME --dataset $DATASET --indexs $index --method weighted_max 
done
