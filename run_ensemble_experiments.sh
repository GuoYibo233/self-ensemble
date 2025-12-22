#!/bin/bash

# Ensemble generation experiments script
# Runs avg and max methods on multiple models
# Parameters: 5 paraphrases, 5 samples per question, 2000 questions (10000 total samples)

set -e  # Exit on error

# Define models
MODELS=(
    "llama3.2_3b_it"
    "llama3.2_1b_it"
    "llama3.1_8b_it"
    "llama3.2_3b"
    "llama3.2_1b"
    "llama3.1_8b"
    "qwen2.5_3b"
    "qwen2.5_7b"
    "qwen2.5_14b"
    "qwen2.5_3b_it"
    "qwen2.5_7b_it"
    "qwen2.5_14b_it"
)

# Define methods
METHODS=("avg" "max")

# Parameters
NUM_PARAPHRASES=5
NUM_SAMPLES=5
MAX_SAMPLES=10000  # 2000 questions × 5 samples = 10000 total samples
DATASET="myriadlama"

# Script path
SCRIPT="g_ori_sample_sync_xzhao.py"

# Log file
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/ensemble_experiments_$(date +%Y%m%d_%H%M%S).log"

echo "=====================================" | tee -a "$MAIN_LOG"
echo "Ensemble Generation Experiments" | tee -a "$MAIN_LOG"
echo "Started at: $(date)" | tee -a "$MAIN_LOG"
echo "=====================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Configuration:" | tee -a "$MAIN_LOG"
echo "  - Number of paraphrases: $NUM_PARAPHRASES" | tee -a "$MAIN_LOG"
echo "  - Samples per question: $NUM_SAMPLES" | tee -a "$MAIN_LOG"
echo "  - Total samples: $MAX_SAMPLES (2000 questions)" | tee -a "$MAIN_LOG"
echo "  - Dataset: $DATASET" | tee -a "$MAIN_LOG"
echo "  - Methods: ${METHODS[*]}" | tee -a "$MAIN_LOG"
echo "  - Models: ${#MODELS[@]} models" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Counter for progress
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#METHODS[@]}))
CURRENT=0

# Loop through methods and models
for METHOD in "${METHODS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo "=====================================" | tee -a "$MAIN_LOG"
        echo "Experiment [$CURRENT/$TOTAL_EXPERIMENTS]" | tee -a "$MAIN_LOG"
        echo "Model: $MODEL" | tee -a "$MAIN_LOG"
        echo "Method: $METHOD" | tee -a "$MAIN_LOG"
        echo "Started at: $(date)" | tee -a "$MAIN_LOG"
        echo "=====================================" | tee -a "$MAIN_LOG"
        
        # Create model-specific log file
        MODEL_LOG="$LOG_DIR/${MODEL}_${METHOD}_$(date +%Y%m%d_%H%M%S).log"
        
        # Check if output file already exists
        OUTPUT_FILE="./results/${MODEL}/syncedsample_ensemble_${METHOD}-${NUM_PARAPHRASES}paras-${NUM_SAMPLES}samples.feather"
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "⚠️  Output file already exists, skipping: $OUTPUT_FILE" | tee -a "$MAIN_LOG"
            echo "" | tee -a "$MAIN_LOG"
            continue
        fi
        
        # Run the experiment
        echo "Running command:" | tee -a "$MAIN_LOG"
        CMD="python3 $SCRIPT --model $MODEL --method $METHOD --num_paraphrases $NUM_PARAPHRASES --num_samples $NUM_SAMPLES --max_samples $MAX_SAMPLES --dataset $DATASET"
        echo "$CMD" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        
        # Execute with logging
        if $CMD 2>&1 | tee -a "$MODEL_LOG" "$MAIN_LOG"; then
            echo "✅ Completed successfully!" | tee -a "$MAIN_LOG"
        else
            echo "❌ Failed with error!" | tee -a "$MAIN_LOG"
            echo "Check log file: $MODEL_LOG" | tee -a "$MAIN_LOG"
        fi
        
        echo "Finished at: $(date)" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        
        # Optional: Add a small delay between experiments to prevent system overload
        sleep 2
    done
done

echo "=====================================" | tee -a "$MAIN_LOG"
echo "All experiments completed!" | tee -a "$MAIN_LOG"
echo "Finished at: $(date)" | tee -a "$MAIN_LOG"
echo "=====================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Summary:" | tee -a "$MAIN_LOG"
echo "  - Total experiments: $TOTAL_EXPERIMENTS" | tee -a "$MAIN_LOG"
echo "  - Log file: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Count successful outputs
SUCCESS_COUNT=0
for METHOD in "${METHODS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        OUTPUT_FILE="./results/${MODEL}/syncedsample_ensemble_${METHOD}-${NUM_PARAPHRASES}paras-${NUM_SAMPLES}samples.feather"
        if [ -f "$OUTPUT_FILE" ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    done
done

echo "Successful experiments: $SUCCESS_COUNT / $TOTAL_EXPERIMENTS" | tee -a "$MAIN_LOG"
