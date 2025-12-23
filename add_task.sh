#!/bin/bash

# Quick script to add tasks to the queue
# Usage: ./add_task.sh "your command here"
# Or: ./add_task.sh  (interactive mode)

TASK_QUEUE="gpu_task_queue.txt"

if [ $# -eq 0 ]; then
    # Interactive mode
    echo "Enter task command (or press Ctrl+C to cancel):"
    read -r task
else
    # Command line argument
    task="$*"
fi

if [ -z "$task" ]; then
    echo "Error: Task cannot be empty"
    exit 1
fi

# Add task to queue
echo "$task" >> "$TASK_QUEUE"
echo "✅ Task added to queue: $TASK_QUEUE"
echo "📝 Task: $task"
echo ""
echo "Current queue status:"
echo "-------------------"
tail -5 "$TASK_QUEUE"
