#!/bin/bash

# View task queue status
# Usage: ./view_queue.sh

TASK_QUEUE="gpu_task_queue.txt"

if [ ! -f "$TASK_QUEUE" ]; then
    echo "❌ Task queue file not found: $TASK_QUEUE"
    exit 1
fi

echo "======================================"
echo "📋 Task Queue Status"
echo "======================================"
echo ""

# Count statistics
total_lines=$(wc -l < "$TASK_QUEUE")
completed=$(grep -c "^# \[COMPLETED\]" "$TASK_QUEUE" || echo 0)
pending=$(grep -v "^#" "$TASK_QUEUE" | grep -v "^$" | wc -l)
comments=$(grep -c "^#" "$TASK_QUEUE" | awk -v comp="$completed" '{print $1 - comp}')

echo "📊 Statistics:"
echo "   Total lines:      $total_lines"
echo "   ✅ Completed:     $completed"
echo "   ⏳ Pending:       $pending"
echo "   💬 Comments:      $comments"
echo ""
echo "======================================"
echo "📝 Queue Contents:"
echo "======================================"
echo ""

# Display with line numbers and status
line_num=0
while IFS= read -r line; do
    ((line_num++))
    
    if [[ "$line" =~ ^#\ \[COMPLETED\] ]]; then
        # Completed task
        echo -e "✅ Line $line_num: \e[90m$line\e[0m"
    elif [[ "$line" =~ ^# ]]; then
        # Comment
        echo -e "💬 Line $line_num: \e[90m$line\e[0m"
    elif [[ -z "$line" ]]; then
        # Empty line
        continue
    else
        # Pending task
        echo -e "⏳ Line $line_num: \e[1m$line\e[0m"
    fi
done < "$TASK_QUEUE"

echo ""
echo "======================================"
echo "💡 Tips:"
echo "   - Add tasks: ./add_task.sh \"command\""
echo "   - Edit queue: nano $TASK_QUEUE"
echo "   - View logs: tail -f logs/auto_gpu_scheduler_*.log"
echo "======================================"
