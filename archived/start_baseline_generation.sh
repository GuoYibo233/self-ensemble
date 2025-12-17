#!/bin/bash
#
# Quick starter script for baseline generation
# Usage: bash start_baseline_generation.sh
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "========================================================================"
echo "  批量生成Baseline - 快速启动脚本"
echo "========================================================================"
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "baseline_generate.py" ]; then
    echo -e "${YELLOW}错误: 请在项目根目录运行此脚本${NC}"
    echo "正确的目录: /home/y-guo/self-ensemble/self-ensemble"
    exit 1
fi

echo "当前目录: $(pwd)"
echo ""

# Show what will be done
echo -e "${GREEN}步骤1: 查看将要处理的模型${NC}"
echo "----------------------------------------"
python3 scripts/generate_all_baselines.py --dry-run | grep -A 20 "Scanning for Existing Models"
echo ""

# Ask for confirmation
echo -e "${YELLOW}预计总时间: 6-12小时（顺序执行）${NC}"
echo ""
read -p "是否在tmux中启动baseline生成? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消。"
    echo ""
    echo "你可以手动运行:"
    echo "  bash scripts/generate_all_baselines.sh"
    exit 0
fi

# Check if tmux session already exists
if tmux has-session -t baseline_gen 2>/dev/null; then
    echo -e "${YELLOW}tmux session 'baseline_gen' 已存在${NC}"
    read -p "是否连接到现有session? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "已取消。"
        exit 0
    fi
    tmux attach -t baseline_gen
    exit 0
fi

# Create tmux session and start generation
echo -e "${GREEN}创建tmux session: baseline_gen${NC}"
echo "----------------------------------------"
echo ""
echo "操作提示:"
echo "  - 分离session: 按 Ctrl+B 然后按 D"
echo "  - 重新连接: tmux attach -t baseline_gen"
echo "  - 查看进度: 在另一个终端运行 'watch -n 60 ls -lh /net/.../baseline_*.feather'"
echo ""
echo "按任意键继续..."
read -n 1 -s

tmux new-session -s baseline_gen "bash scripts/generate_all_baselines.sh; echo ''; echo '✅ 所有baseline生成完成！'; echo '按任意键关闭...'; read"
