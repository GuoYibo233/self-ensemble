#!/bin/bash
# 实验室服务器部署脚本
# deploy_to_lab_server.sh

set -e  # 遇到错误立即退出

echo "🚀 开始部署Self-Ensemble + MoE到实验室服务器"
echo "================================================"

# 获取当前用户名
USERNAME=$(whoami)
echo "👤 当前用户: $USERNAME"

# 1. 创建项目目录
LAB_PROJECT_DIR="$HOME/self-ensemble-moe"
echo "📁 创建项目目录: $LAB_PROJECT_DIR"
mkdir -p $LAB_PROJECT_DIR
cd $LAB_PROJECT_DIR

# 2. 检查代码是否已存在
if [ -d ".git" ]; then
    echo "📦 更新现有代码库..."
    git pull origin main
else
    echo "📦 克隆项目代码..."
    git clone https://github.com/GuoYibo233/self-ensemble.git .
fi

# 3. 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "⚠️  Conda未安装，正在安装Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    rm /tmp/miniconda.sh
else
    echo "✅ Conda已安装"
fi

# 4. 创建或激活conda环境
echo "🐍 设置Python环境..."
if conda env list | grep -q "self-ensemble-moe"; then
    echo "📦 激活现有环境: self-ensemble-moe"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate self-ensemble-moe
else
    echo "📦 创建新环境: self-ensemble-moe"
    conda create -n self-ensemble-moe python=3.9 -y
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate self-ensemble-moe
fi

# 5. 检测CUDA版本并安装PyTorch
echo "🔍 检测CUDA环境..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
    echo "✅ 检测到CUDA版本: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "🔧 安装PyTorch for CUDA 12.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        echo "🔧 安装PyTorch for CUDA 11.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "⚠️  未知CUDA版本，安装CPU版本PyTorch"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "⚠️  未检测到CUDA，安装CPU版本PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 6. 安装其他依赖
echo "📦 安装项目依赖..."
pip install transformers datasets accelerate
pip install pandas feather-format tqdm psutil
pip install spacy

# 下载spacy模型
echo "🔤 下载spacy语言模型..."
python -m spacy download en_core_web_lg

# 7. 创建数据目录
echo "📁 创建数据目录..."
# 首先尝试创建在/data下，如果没有权限则使用用户目录
if [ -w "/data" ]; then
    DATA_ROOT="/data"
else
    DATA_ROOT="$HOME/data"
fi

mkdir -p $DATA_ROOT/models/moe
mkdir -p $DATA_ROOT/datasets/myriadlama  
mkdir -p $DATA_ROOT/results/self-ensemble-moe
mkdir -p $DATA_ROOT/logs/self-ensemble-moe

echo "📂 数据目录创建在: $DATA_ROOT"

# 8. 更新配置文件中的路径
echo "🔧 更新配置文件路径..."
sed -i "s|/home/\[your_username\]|$HOME|g" lab_moe_config.py
sed -i "s|/data/|$DATA_ROOT/|g" lab_moe_config.py

# 9. 测试环境
echo "🧪 测试环境配置..."
python -c "
import sys
print(f'Python版本: {sys.version}')

try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError as e:
    print(f'PyTorch导入失败: {e}')

try:
    import transformers
    print(f'Transformers版本: {transformers.__version__}')
except ImportError as e:
    print(f'Transformers导入失败: {e}')

try:
    import spacy
    nlp = spacy.load('en_core_web_lg')
    print('Spacy模型加载成功')
except Exception as e:
    print(f'Spacy模型加载失败: {e}')
"

# 10. 创建快速测试脚本
echo "📝 创建快速测试脚本..."
cat > quick_test_lab.py << 'EOF'
#!/usr/bin/env python3
"""快速测试实验室环境"""

import torch
import sys
from pathlib import Path

def test_environment():
    print("🧪 实验室环境测试")
    print("=" * 40)
    
    # 测试基础环境
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # 测试CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # 测试GPU内存
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i} 内存: {total_memory:.1f} GB")
    else:
        print("⚠️  CUDA不可用")
    
    # 测试目录权限
    directories = [
        "$DATA_ROOT/models/moe",
        "$DATA_ROOT/datasets/myriadlama", 
        "$DATA_ROOT/results/self-ensemble-moe",
        "$DATA_ROOT/logs/self-ensemble-moe"
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"✅ 目录可访问: {dir_path}")
        else:
            print(f"❌ 目录不存在: {dir_path}")

if __name__ == "__main__":
    test_environment()
EOF

chmod +x quick_test_lab.py

# 11. 显示GPU信息
echo ""
echo "🖥️  GPU信息:"
nvidia-smi

echo ""
echo "✅ 环境部署完成！"
echo "================="
echo "📁 项目目录: $LAB_PROJECT_DIR"
echo "📂 数据目录: $DATA_ROOT" 
echo "🐍 Conda环境: self-ensemble-moe"
echo ""
echo "🚀 下一步操作:"
echo "1. 运行快速测试: python quick_test_lab.py"
echo "2. 测试基础模型: python test_environment.py"
echo "3. 开始MoE训练: python train_moe_lab.py --help"
echo "4. 批量实验: ./scripts/run_moe_experiments.sh"