# 🚀 一键迁移命令参考

## 方案1: 完整Git迁移（推荐）

### 在本地准备（Windows PowerShell）
```powershell
# 1. 提交当前改动
cd D:\Codes\Research\SelfE\self-ensemble
git add .
git commit -m "准备实验室部署 - MoE训练框架"
git push origin main

# 2. 连接到服务器
ssh tokyo106
```

### 在服务器执行
```bash
# 3. 克隆项目并部署
git clone https://github.com/GuoYibo233/self-ensemble.git ~/self-ensemble-moe
cd ~/self-ensemble-moe
chmod +x scripts/deploy_to_lab_server.sh
./scripts/deploy_to_lab_server.sh
```

## 方案2: 直接文件传输

### 使用SCP传输（在本地PowerShell中）
```powershell
# 打包项目
cd D:\Codes\Research\SelfE\self-ensemble
tar -czf self-ensemble.tar.gz --exclude=__pycache__ --exclude=.git .

# 传输到服务器
scp self-ensemble.tar.gz tokyo106:~/

# 在服务器上解压并部署
ssh tokyo106 "
cd ~ && 
tar -xzf self-ensemble.tar.gz -C self-ensemble-moe --strip-components=0 && 
cd self-ensemble-moe && 
chmod +x scripts/deploy_to_lab_server.sh && 
./scripts/deploy_to_lab_server.sh
"
```

## 方案3: 逐步手动操作

### 第一步：连接服务器
```bash
ssh tokyo106
```

### 第二步：创建环境
```bash
# 创建项目目录
mkdir -p ~/self-ensemble-moe
cd ~/self-ensemble-moe

# 克隆代码
git clone https://github.com/GuoYibo233/self-ensemble.git .

# 创建conda环境
conda create -n self-ensemble-moe python=3.9 -y
conda activate self-ensemble-moe
```

### 第三步：安装依赖
```bash
# 检查CUDA版本
nvidia-smi

# 安装PyTorch（根据CUDA版本选择）
# For CUDA 11.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install transformers datasets accelerate pandas feather-format tqdm psutil spacy
python -m spacy download en_core_web_lg
```

### 第四步：创建目录
```bash
# 创建数据目录
mkdir -p ~/data/{models/moe,datasets/myriadlama,results/self-ensemble-moe,logs/self-ensemble-moe}
```

### 第五步：测试环境
```bash
# 快速测试
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 完整测试
python test_environment.py
```

## 🎯 一键迁移命令（最简单）

### 选择你的情况：

#### 情况A：服务器已有Git和Conda
```bash
ssh tokyo106 'bash -s' << 'EOF'
git clone https://github.com/GuoYibo233/self-ensemble.git ~/self-ensemble-moe
cd ~/self-ensemble-moe
chmod +x scripts/deploy_to_lab_server.sh
./scripts/deploy_to_lab_server.sh
EOF
```

#### 情况B：需要完整设置
```bash
ssh tokyo106 'bash -s' << 'EOF'
# 安装Miniconda（如果需要）
if ! command -v conda &> /dev/null; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# 克隆项目
git clone https://github.com/GuoYibo233/self-ensemble.git ~/self-ensemble-moe
cd ~/self-ensemble-moe

# 运行部署脚本
chmod +x scripts/deploy_to_lab_server.sh
./scripts/deploy_to_lab_server.sh
EOF
```

## 🧪 验证迁移成功

```bash
# 1. 连接服务器
ssh tokyo106

# 2. 激活环境
conda activate self-ensemble-moe
cd ~/self-ensemble-moe

# 3. 运行测试
python quick_test_lab.py

# 4. 小规模MoE测试
python train_moe_lab.py \
    --model_name qwen1.5_moe_a2.7b_chat \
    --max_samples 10 \
    --batch_size 1 \
    --experiment_name quick_test

# 5. 检查结果
ls -la ~/data/results/self-ensemble-moe/
```

## 🐛 常见问题解决

### 问题1: SSH连接失败
```powershell
# 检查SSH配置
ssh -v tokyo106

# 如果需要指定用户名
ssh username@tokyo106
```

### 问题2: Git克隆失败
```bash
# 使用HTTPS替代SSH
git clone https://github.com/GuoYibo233/self-ensemble.git ~/self-ensemble-moe
```

### 问题3: Conda环境问题
```bash
# 重新初始化conda
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 问题4: CUDA不匹配
```bash
# 检查CUDA版本
nvidia-smi | grep "CUDA Version"

# 重新安装对应的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📞 下一步操作

迁移完成后，你可以：

1. **运行快速测试**: `python quick_test_lab.py`
2. **开始MoE训练**: `python train_moe_lab.py --help`
3. **批量实验**: `./scripts/run_moe_experiments.sh`
4. **监控训练**: `python monitor_moe_training.py --action monitor`

---

选择最适合你当前情况的方案即可！ 🚀