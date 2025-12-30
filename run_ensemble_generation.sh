#!/bin/bash

# Ensemble Generation Parallel Runner
# Distributes tasks across multiple GPUs

# =============================================================================
# Configuration
# =============================================================================

# Multiple models to run
MODELS=(
    "llama3.2_1b_it"
    "llama3.2_1b"
    "qwen2.5_3b"
    "qwen2.5_3b_it"
    "llama3.2_3b"
    "llama3.2_3b_it"
)

DATASET="myriadlama"  # or "webqa"

# GPUs to use
GPUS=(0 1 2)

# =============================================================================
# Define parameter combinations to run
# Format: "method num_manual num_auto num_samples"
# =============================================================================

COMBINATIONS=(
    "avg 1 0 1"
    "avg 2 0 2"
    "avg 3 0 3"
    "avg 4 0 4"
    "avg 5 0 5"
    "avg 0 1 1"
    "avg 0 2 2"
    "avg 0 3 3"
    "avg 0 4 4"
    "avg 0 5 5"
    "avg 4 1 5"
    "avg 3 2 5"
    "avg 2 3 5"
    "avg 1 4 5"
    "avg 5 5 2"
    "avg 5 5 3"
    "avg 5 5 4"
    "avg 5 5 6"
    "avg 5 5 7"
    "avg 5 5 8"
    "avg 5 5 9"
    "avg 5 5 10"
)

# =============================================================================
# Generate all tasks (model + combination pairs)
# =============================================================================

ALL_TASKS=()
for model in "${MODELS[@]}"; do
    for combo in "${COMBINATIONS[@]}"; do
        ALL_TASKS+=("$model|$combo")
    done
done

TOTAL_TASKS=${#ALL_TASKS[@]}
NUM_GPUS=${#GPUS[@]}
TASKS_PER_GPU=$((TOTAL_TASKS / NUM_GPUS))

# Create main log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="./logs/run_ensemble_${TIMESTAMP}.log"
mkdir -p ./logs

echo "=========================================="
echo "Starting Parallel Ensemble Generation"
echo "Time: $(date)"
echo "Models: ${MODELS[@]}"
echo "Dataset: $DATASET"
echo "GPUs: ${GPUS[@]}"
echo "Total tasks: $TOTAL_TASKS"
echo "Tasks per GPU: $TASKS_PER_GPU"
echo "Main log: $MAIN_LOG"
echo "=========================================="
echo ""

# Also save to main log
{
    echo "=========================================="
    echo "Starting Parallel Ensemble Generation"
    echo "Time: $(date)"
    echo "Models: ${MODELS[@]}"
    echo "Dataset: $DATASET"
    echo "GPUs: ${GPUS[@]}"
    echo "Total tasks: $TOTAL_TASKS"
    echo "Tasks per GPU: $TASKS_PER_GPU"
    echo "=========================================="
    echo ""
} > "$MAIN_LOG"

# =============================================================================
# Function to run tasks on a specific GPU
# =============================================================================

run_gpu_tasks() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    
    # Activate conda environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate flexattention
    
    mkdir -p ./logs
    
    echo "[GPU $gpu_id] Starting tasks $start_idx to $((end_idx-1))" | tee -a "$MAIN_LOG"
    
    for ((i=start_idx; i<end_idx; i++)); do
        task="${ALL_TASKS[$i]}"
        IFS='|' read -r model combo <<< "$task"
        read -r method manual auto samples <<< "$combo"
        
        # Create log file with parameters in name
        local log_file="./logs/${model}_${method}_m${manual}a${auto}s${samples}_gpu${gpu_id}.log"
        
        echo "[GPU $gpu_id] [$(date +"%H:%M:%S")] Task $((i-start_idx+1))/$((end_idx-start_idx)): model=$model, method=$method, manual=$manual, auto=$auto, samples=$samples" | tee "$log_file" | tee -a "$MAIN_LOG"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python generate_original.py \
            --method "$method" \
            --model "$model" \
            --dataset "$DATASET" \
            --device "cuda" \
            --num_manual "$manual" \
            --num_auto "$auto" \
            --num_samples "$samples" \
            >> "$log_file" 2>&1
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "[GPU $gpu_id] [$(date +"%H:%M:%S")] ✓ Success" | tee -a "$log_file" | tee -a "$MAIN_LOG"
        else
            echo "[GPU $gpu_id] [$(date +"%H:%M:%S")] ✗ Failed with exit code $exit_code" | tee -a "$log_file" | tee -a "$MAIN_LOG"
        fi
        echo "" | tee -a "$log_file"
    done
    
    echo "[GPU $gpu_id] [$(date +"%H:%M:%S")] All tasks completed!" | tee -a "$MAIN_LOG"
}

# =============================================================================
# Launch parallel jobs on each GPU
# =============================================================================

# Calculate how many GPUs get extra tasks
# 28 tasks / 10 GPUs = 2 base + 8 GPUs get 1 extra
base_tasks_per_gpu=$((TOTAL_TASKS / NUM_GPUS))
extra_tasks=$((TOTAL_TASKS % NUM_GPUS))

current_idx=0
for gpu_idx in "${!GPUS[@]}"; do
    gpu_id=${GPUS[$gpu_idx]}
    start_idx=$current_idx
    
    # First 'extra_tasks' GPUs get one more task
    if [ $gpu_idx -lt $extra_tasks ]; then
        end_idx=$((start_idx + base_tasks_per_gpu + 1))
    else
        end_idx=$((start_idx + base_tasks_per_gpu))
    fi
    
    current_idx=$end_idx
    
    # Run in background
    run_gpu_tasks $gpu_id $start_idx $end_idx &
done

# Wait for all background jobs to complete
wait

echo "=========================================="
echo "All parallel jobs completed!"
echo "Completion time: $(date)"
echo "Results saved in: ./results/<model_name>/"
echo "Main log: $MAIN_LOG"
echo "Task logs: ./logs/*_gpu*.log"
echo "=========================================="

# Also save to main log
{
    echo ""
    echo "=========================================="
    echo "All parallel jobs completed!"
    echo "Completion time: $(date)"
    echo "Results saved in: ./results/<model_name>/"
    echo "=========================================="
} >> "$MAIN_LOG"
