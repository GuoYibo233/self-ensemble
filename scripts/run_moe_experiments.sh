#!/bin/bash
# 实验室服务器MoE实验运行脚本
# run_moe_experiments.sh

set -e  # 遇到错误立即退出

echo "🧪 Starting MoE Experiments on Lab Server"
echo "========================================"

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate self-ensemble-moe

# 检查环境
echo "🔍 Environment Check:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际可用GPU调整
export TOKENIZERS_PARALLELISM=false  # 避免多进程冲突

# 创建结果目录
mkdir -p /data/results/moe-experiments
mkdir -p /data/logs/moe-experiments

# 实验1: Qwen1.5-MoE-A2.7B-Chat - AVG方法
echo "🧠 Experiment 1: Qwen1.5-MoE-A2.7B-Chat with AVG method"
python train_moe_lab.py \
    --model_name qwen1.5_moe_a2.7b_chat \
    --dataset myriadlama \
    --max_samples 500 \
    --batch_size 4 \
    --max_steps 100 \
    --experiment_name qwen_moe_chat_avg \
    --output_dir /data/results/moe-experiments \
    2>&1 | tee /data/logs/moe-experiments/qwen_moe_chat_avg.log

# 实验2: Qwen1.5-MoE-A2.7B - 基础版本对比
echo "🧠 Experiment 2: Qwen1.5-MoE-A2.7B base model"
python train_moe_lab.py \
    --model_name qwen1.5_moe_a2.7b \
    --dataset myriadlama \
    --max_samples 500 \
    --batch_size 4 \
    --max_steps 100 \
    --experiment_name qwen_moe_base \
    --output_dir /data/results/moe-experiments \
    2>&1 | tee /data/logs/moe-experiments/qwen_moe_base.log

# 如果GPU资源充足，可以运行更大的模型
# 实验3: Mixtral-8x7B (需要更多GPU内存)
if [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -gt 20000 ]; then
    echo "🧠 Experiment 3: Mixtral-8x7B-Instruct (Large MoE)"
    python train_moe_lab.py \
        --model_name mixtral_7b_instruct \
        --dataset myriadlama \
        --max_samples 200 \
        --batch_size 2 \
        --max_steps 50 \
        --experiment_name mixtral_8x7b \
        --output_dir /data/results/moe-experiments \
        --use_multi_gpu \
        2>&1 | tee /data/logs/moe-experiments/mixtral_8x7b.log
else
    echo "⚠️  Skipping Mixtral experiment (insufficient GPU memory)"
fi

# 生成实验报告
echo "📊 Generating experiment report..."
python -c "
import os
import json
from pathlib import Path

results_dir = Path('/data/results/moe-experiments')
experiments = []

for exp_dir in results_dir.iterdir():
    if exp_dir.is_dir():
        config_file = exp_dir / 'config.json'
        results_file = exp_dir / 'results.json'
        
        if config_file.exists() and results_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            with open(results_file) as f:
                results = json.load(f)
            
            experiments.append({
                'name': config['experiment_name'],
                'model': config['model_name'],
                'accuracy': results['accuracy'],
                'samples': results['total_samples']
            })

print('\\n📋 Experiment Summary:')
print('=' * 60)
for exp in sorted(experiments, key=lambda x: x['accuracy'], reverse=True):
    print(f'{exp[\"name\"]:30} | {exp[\"model\"]:20} | Acc: {exp[\"accuracy\"]:.3f} | Samples: {exp[\"samples\"]}')
"

echo "✅ All MoE experiments completed!"
echo "📁 Results saved in: /data/results/moe-experiments"
echo "📝 Logs saved in: /data/logs/moe-experiments"