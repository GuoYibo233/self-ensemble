# 🚀 实验室服务器环境迁移完整指南

## 📋 迁移前准备清单

### 1. 本地环境检查
```powershell
# 确认当前项目状态
cd D:\Codes\Research\SelfE\self-ensemble
git status
git add .
git commit -m "Ready for lab deployment"
git push origin main
```

### 2. 实验室服务器信息确认
- [ ] 服务器地址: `tokyo106` (已配置SSH)
- [ ] 用户名: 需要确认具体用户名
- [ ] GPU配置: 需要确认可用GPU型号和数量
- [ ] 存储路径: 需要确认数据存储目录

## 🔗 SSH连接设置

### 1. 测试SSH连接
```powershell
# 测试连接
ssh tokyo106

# 如果需要指定用户名
ssh username@tokyo106
```

### 2. 设置SSH密钥（如果还没有）
```powershell
# 生成SSH密钥（你已经执行过了）
ssh-keygen -t ecdsa -b 521 -C "y-guo@tkl_2"

# 复制公钥到服务器
ssh-copy-id tokyo106
# 或手动复制 ~/.ssh/id_ecdsa.pub 到服务器
```

## 📦 代码迁移方案

### 方案1: Git克隆（推荐）
```bash
# 在实验室服务器上执行
ssh tokyo106

# 创建项目目录
mkdir -p ~/self-ensemble-moe
cd ~/self-ensemble-moe

# 克隆代码
git clone https://github.com/GuoYibo233/self-ensemble.git .

# 检查文件
ls -la
```

### 方案2: SCP直接传输
```powershell
# 在本地Windows PowerShell中执行
cd D:\Codes\Research\SelfE\self-ensemble

# 打包项目（排除不必要的文件）
tar -czf self-ensemble.tar.gz --exclude=__pycache__ --exclude=.git --exclude=mock_data .

# 传输到服务器
scp self-ensemble.tar.gz tokyo106:~/

# 在服务器上解压
ssh tokyo106 "cd ~ && tar -xzf self-ensemble.tar.gz && mv self-ensemble self-ensemble-moe"
```

### 方案3: rsync同步（最佳选择）
```powershell
# 使用rsync同步（需要在WSL或Git Bash中）
rsync -avz --exclude='__pycache__' --exclude='.git' --exclude='mock_data' \
    /d/Codes/Research/SelfE/self-ensemble/ \
    tokyo106:~/self-ensemble-moe/
```

## 🛠️ 服务器环境设置

### 1. 登录服务器并设置环境
```bash
ssh tokyo106

# 更新系统包
sudo apt update

# 检查CUDA版本
nvidia-smi
nvcc --version
```

### 2. 安装Miniconda（如果没有）
```bash
# 下载Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 重新加载shell
source ~/.bashrc
```

### 3. 创建项目环境
```bash
cd ~/self-ensemble-moe

# 创建conda环境
conda create -n self-ensemble-moe python=3.9 -y
conda activate self-ensemble-moe

# 安装PyTorch（根据CUDA版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install transformers datasets accelerate
pip install pandas feather-format tqdm psutil
pip install spacy

# 下载spacy模型
python -m spacy download en_core_web_lg
```

## 🔧 配置文件修改

### 1. 更新实验室配置
```bash
# 编辑lab_moe_config.py
nano lab_moe_config.py

# 需要修改的部分：
# - 用户名路径
# - 数据存储路径
# - 模型缓存路径
```

### 2. 创建必要目录
```bash
# 创建数据目录（根据实际路径调整）
mkdir -p /data/models/moe
mkdir -p /data/datasets/myriadlama
mkdir -p /data/results/self-ensemble-moe
mkdir -p /data/logs/self-ensemble-moe

# 如果没有/data权限，使用用户目录
mkdir -p ~/data/models/moe
mkdir -p ~/data/datasets/myriadlama
mkdir -p ~/data/results/self-ensemble-moe
mkdir -p ~/data/logs/self-ensemble-moe
```

## 🧪 环境验证

### 1. 基础环境测试
```bash
# 激活环境
conda activate self-ensemble-moe

# 测试Python环境
python test_environment.py

# 测试GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 2. 快速功能测试
```bash
# 小规模测试
python test_real_model.py --max_samples 10 --model_name qwen2.5_1.5b_it

# 检查结果
ls -la results/
```

## 🚀 开始MoE训练

### 1. 单个实验测试
```bash
# 运行小规模MoE实验
python train_moe_lab.py \
    --model_name qwen1.5_moe_a2.7b_chat \
    --max_samples 50 \
    --batch_size 2 \
    --experiment_name test_moe \
    --output_dir ~/data/results
```

### 2. 批量实验
```bash
# 运行完整实验套件
chmod +x scripts/run_moe_experiments.sh
./scripts/run_moe_experiments.sh
```

### 3. 监控训练
```bash
# 在另一个终端中监控
python monitor_moe_training.py --action monitor --interval 30
```

## 🐛 常见问题解决

### 问题1: 权限不足
```bash
# 如果无法访问/data目录
sudo chown -R $USER:$USER /data/
# 或使用用户目录 ~/data/
```

### 问题2: CUDA版本不匹配
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装匹配的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题3: 网络连接问题
```bash
# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或下载到本地
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B-Chat
```

## 📊 迁移检查清单

- [ ] SSH连接正常
- [ ] 代码成功传输
- [ ] Conda环境创建
- [ ] PyTorch + CUDA安装
- [ ] 依赖包安装完成
- [ ] 目录权限设置
- [ ] GPU可用性确认
- [ ] 基础功能测试通过
- [ ] MoE模型测试成功

## 💡 优化建议

1. **使用screen/tmux**: 避免SSH断连影响训练
```bash
# 安装tmux
sudo apt install tmux

# 创建会话
tmux new-session -d -s moe-training
tmux attach-session -t moe-training
```

2. **设置环境变量**:
```bash
# 添加到 ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0,1,2,3' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

3. **定期备份结果**:
```bash
# 设置自动备份
crontab -e
# 添加: 0 2 * * * tar -czf ~/backup/results_$(date +\%Y\%m\%d).tar.gz ~/data/results/
```

---

按照这个指南，你就能完整地将环境迁移到实验室服务器上了！🚀