#!/bin/bash
# 实验室服务器部署脚本
# deploy_to_lab_server.sh

echo "🚀 开始部署Self-Ensemble + MoE到实验室服务器"

# 1. 创建项目目录
LAB_PROJECT_DIR="/home/[your_username]/self-ensemble-moe"
mkdir -p $LAB_PROJECT_DIR
cd $LAB_PROJECT_DIR

# 2. 克隆项目代码
git clone https://github.com/GuoYibo233/self-ensemble.git .
# 或者从本地上传代码

# 3. 创建conda环境
conda create -n self-ensemble-moe python=3.9 -y
conda activate self-ensemble-moe

# 4. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install pandas feather-format tqdm
pip install spacy
python -m spacy download en_core_web_lg

# 5. 检查GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"

# 6. 创建模型存储目录
mkdir -p /data/models/moe  # 根据实验室服务器的存储结构调整

echo "✅ 环境部署完成"