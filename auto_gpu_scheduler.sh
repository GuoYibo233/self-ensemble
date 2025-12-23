#!/bin/bash

# Auto GPU Scheduler
# Automatically runs commands on available GPUs
# Usage: ./auto_gpu_scheduler.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flexattention

# Configuration
CHECK_INTERVAL=30  # Check GPU status every 30 seconds
GPU_MEMORY_THRESHOLD=1000  # GPU is considered free if memory usage < 1000 MiB
GPU_UTIL_THRESHOLD=10  # GPU is considered free if utilization < 10%

# Task queue file (one command per line)
TASK_QUEUE="gpu_task_queue.txt"
PROCESSED_TASKS="logs/processed_tasks.log"  # Track processed task line numbers

# Log file
LOG_FILE="logs/auto_gpu_scheduler_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Function to check if a GPU is free
is_gpu_free() {
    local gpu_id=$1
    
    # Get GPU memory usage and utilization
    local gpu_info=$(nvidia-smi -i $gpu_id --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    local mem_used=$(echo $gpu_info | awk -F', ' '{print $1}')
    local util=$(echo $gpu_info | awk -F', ' '{print $2}')
    
    # Check if both memory and utilization are below threshold
    if [ "$mem_used" -lt "$GPU_MEMORY_THRESHOLD" ] && [ "$util" -lt "$GPU_UTIL_THRESHOLD" ]; then
        return 0  # GPU is free
    else
        return 1  # GPU is busy
    fi
}

# Function to get list of all GPU IDs
get_all_gpus() {
    nvidia-smi --query-gpu=index --format=csv,noheader
}

# Function to find a free GPU
find_free_gpu() {
    local all_gpus=$(get_all_gpus)
    
    for gpu_id in $all_gpus; do
        if is_gpu_free $gpu_id; then
            echo $gpu_id
            return 0
        fi
    done
    
    return 1  # No free GPU found
}

# Function to run a task on a specific GPU
run_task_on_gpu() {
    local gpu_id=$1
    local task=$2
    
    echo "[$(date)] Starting task on GPU $gpu_id: $task" | tee -a "$LOG_FILE"
    
    # Extract model name and method from task for log filename
    local model_name=$(echo "$task" | grep -oP '(?<=--model )\S+')
    local method=$(echo "$task" | grep -oP '(?<=--method )\S+')
    local task_log="logs/task_${model_name}_${method}_gpu${gpu_id}.log"
    
    echo "[$(date)] Task log will be saved to: $task_log" | tee -a "$LOG_FILE"
    
    # Run the task in background with GPU specified and output to log file
    CUDA_VISIBLE_DEVICES=$gpu_id bash -c "$task" > "$task_log" 2>&1 &
    local pid=$!
    
    echo "[$(date)] Task started on GPU $gpu_id with PID $pid" | tee -a "$LOG_FILE"
    
    # Store the GPU-PID mapping with line number for later commenting
    echo "$gpu_id:$pid:$task:$CURRENT_LINE_NUM" >> logs/running_tasks.tmp
    
    return 0
}

# Function to remove completed tasks from tracking
cleanup_completed_tasks() {
    if [ ! -f logs/running_tasks.tmp ]; then
        return
    fi
    
    while IFS=: read -r gpu_id pid task; do
        if ! ps -p $pid > /dev/null 2>&1; then
            echo "[$(date)] Task completed on GPU $gpu_id (PID $pid)" | tee -a "$LOG_FILE"
            
            # Find and comment out the completed task in queue file
            # Extract the line number from our tracking
            local completed_line=$(grep ":$pid:" logs/running_tasks.tmp | cut -d':' -f4)
            if [ ! -z "$completed_line" ]; then
                comment_task_in_queue "$completed_line"
            fi
        else
            echo "$gpu_id:$pid:$task" >> logs/running_tasks.tmp.new
        fi
    done < logs/running_tasks.tmp
    
    if [ -f logs/running_tasks.tmp.new ]; then
        mv logs/running_tasks.tmp.new logs/running_tasks.tmp
    else
        rm -f logs/running_tasks.tmp
    fi
}

# Function to check if a task was already processed
is_task_processed() {
    local line_num=$1
    if [ -f "$PROCESSED_TASKS" ]; then
        grep -q "^$line_num$" "$PROCESSED_TASKS"
        return $?
    fi
    return 1
}

# Function to mark task as processed
mark_task_processed() {
    local line_num=$1
    echo "$line_num" >> "$PROCESSED_TASKS"
}

# Function to comment out completed task in queue file
comment_task_in_queue() {
    local line_num=$1
    
    # Create a temporary file
    local temp_file="${TASK_QUEUE}.tmp"
    
    # Read the file, add comment to specific line
    awk -v line="$line_num" 'NR==line && !/^#/ {print "# [COMPLETED] " $0; next} {print}' "$TASK_QUEUE" > "$temp_file"
    
    # Replace original file
    mv "$temp_file" "$TASK_QUEUE"
    
    echo "[$(date)] Marked line $line_num as completed in queue file" | tee -a "$LOG_FILE"
}

# Function to get next unprocessed task
get_next_task() {
    if [ ! -f "$TASK_QUEUE" ]; then
        return 1
    fi
    
    local line_num=0
    while IFS= read -r task; do
        ((line_num++))
        
        # Skip if already processed
        if is_task_processed $line_num; then
            continue
        fi
        
        # Skip empty lines and comments
        if [[ -z "$task" ]] || [[ "$task" =~ ^# ]]; then
            mark_task_processed $line_num
            continue
        fi
        
        # Return the task and line number
        echo "$line_num|$task"
        return 0
    done < "$TASK_QUEUE"
    
    return 1  # No more tasks
}

# Main loop
main() {
    echo "=====================================" | tee -a "$LOG_FILE"
    echo "Auto GPU Scheduler Started (Dynamic Mode)" | tee -a "$LOG_FILE"
    echo "Time: $(date)" | tee -a "$LOG_FILE"
    echo "=====================================" | tee -a "$LOG_FILE"
    echo ""
    echo "💡 You can add new tasks anytime by editing: $TASK_QUEUE" | tee -a "$LOG_FILE"
    echo "💡 The scheduler will automatically pick up new tasks" | tee -a "$LOG_FILE"
    echo "💡 Use Ctrl+C to stop the scheduler" | tee -a "$LOG_FILE"
    echo ""
    
    # Check if task queue exists
    if [ ! -f "$TASK_QUEUE" ]; then
        echo "Error: Task queue file '$TASK_QUEUE' not found!" | tee -a "$LOG_FILE"
        echo "Please create the file with one command per line." | tee -a "$LOG_FILE"
        exit 1
    fi
    
    # Clean up old tracking files on fresh start
    rm -f logs/running_tasks.tmp
    
    # If this is a fresh start, clear processed tasks log
    if [ ! -f "$PROCESSED_TASKS" ]; then
        touch "$PROCESSED_TASKS"
        echo "[$(date)] Starting fresh - no previous tasks tracked" | tee -a "$LOG_FILE"
    else
        local processed_count=$(wc -l < "$PROCESSED_TASKS")
        echo "[$(date)] Resuming - $processed_count tasks already processed" | tee -a "$LOG_FILE"
    fi
    echo ""
    
    local tasks_submitted=0
    local last_check_time=0
    
    # Continuous loop - keep checking for new tasks
    while true; do
        cleanup_completed_tasks
        
        # Try to get next unprocessed task
        task_info=$(get_next_task)
        
        if [ $? -eq 0 ]; then
            # Extract line number and task
            local line_num=$(echo "$task_info" | cut -d'|' -f1)
            local task=$(echo "$task_info" | cut -d'|' -f2-)
            
            # Store line number globally for later use
            CURRENT_LINE_NUM=$line_num
            
            echo "[$(date)] Found new task (line $line_num): ${task:0:80}..." | tee -a "$LOG_FILE"
            ((tasks_submitted++))
            
            # Wait for a free GPU
            echo "[$(date)] Waiting for free GPU..." | tee -a "$LOG_FILE"
            while true; do
                cleanup_completed_tasks
                
                free_gpu=$(find_free_gpu)
                if [ $? -eq 0 ]; then
                    echo "[$(date)] Found free GPU: $free_gpu" | tee -a "$LOG_FILE"
                    run_task_on_gpu $free_gpu "$task"
                    mark_task_processed $line_num
                    echo "[$(date)] Total tasks submitted: $tasks_submitted" | tee -a "$LOG_FILE"
                    echo ""
                    break
                fi
                
                sleep $CHECK_INTERVAL
            done
            
            # Small delay before checking for next task
            sleep 2
        else
            # No pending tasks, wait and check again
            current_time=$(date +%s)
            if [ $((current_time - last_check_time)) -ge 60 ]; then
                echo "[$(date)] No pending tasks. Monitoring for new tasks... (Submitted: $tasks_submitted)" | tee -a "$LOG_FILE"
                last_check_time=$current_time
            fi
            
            sleep $CHECK_INTERVAL
        fi
    done
}

# Run main function
main
